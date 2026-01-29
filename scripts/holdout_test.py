#!/usr/bin/env python3
"""
Strict Holdout Test - 500 most recent events with complete isolation.

Ensures NO data leakage from holdout set into training/predictions.
Reports pooled most likely numbers and holdout accuracy.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from datetime import datetime
import json
from pathlib import Path

print('=' * 60)
print('  STRICT HOLDOUT TEST - 500 Most Recent Events')
print('=' * 60)
print()

# Configuration
CONFIG = {
    'holdout_size': 500,
    'window_length': 64,
    'latent_dim': 128,
    'batch_size': 32,
    'n_exclude': 12,  # Exclude 12, keep 27 most likely
    'n_keep': 27,
    'max_epochs': 30,
    'patience': 5,
    'device': 'cpu'
}


class TimeFreqEncoder(nn.Module):
    def __init__(self, n_features=39, latent_dim=128):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=7, padding=3),
            nn.SiLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=5, padding=2, stride=2),
            nn.SiLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=3, padding=1, stride=2),
            nn.SiLU(),
            nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(1)
        )
        self.freq = nn.Sequential(
            nn.Linear(n_features * 2, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(128 + 64, latent_dim),
            nn.SiLU()
        )

    def forward(self, x):
        x_t = x.permute(0, 2, 1)
        h_temporal = self.temporal(x_t).squeeze(-1)
        x_mean = x.mean(dim=1)
        x_fft = torch.fft.fft(x_mean.to(torch.complex64))
        h_freq = self.freq(torch.cat([torch.abs(x_fft), torch.angle(x_fft)], dim=1))
        return self.fusion(torch.cat([h_temporal, h_freq], dim=1))


class Classifier(nn.Module):
    def __init__(self, latent_dim=128, n_classes=39):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes)
        )

    def forward(self, h):
        return torch.sigmoid(self.net(h))


def load_data():
    """Load full dataset."""
    df = pd.read_csv('data/raw/CA5_quantum.csv')
    qv_cols = [c for c in df.columns if c.startswith('QV_')]
    data = df[qv_cols].values.astype(np.float32)
    dates = df['event-ID'].values

    # Normalize
    data_norm = data / (data.sum(axis=1, keepdims=True) + 1e-8)

    return data, data_norm, dates


def get_actual_top5(row):
    """Get the 5 QVs with highest values (1-indexed)."""
    top5_idx = np.argsort(row)[-5:]
    return set(idx + 1 for idx in top5_idx)


def create_windows(data, window_length, stride=4):
    """Create training windows."""
    windows, labels = [], []
    for i in range(0, len(data) - window_length, stride):
        window = data[i:i + window_length]
        if i + window_length < len(data):
            label = (data[i + window_length] > 0).astype(np.float32)
            windows.append(window)
            labels.append(label)
    return np.array(windows, dtype=np.float32), np.array(labels, dtype=np.float32)


def train_model(train_data, device):
    """Train model on training data only (no holdout contamination)."""
    windows, labels = create_windows(train_data, CONFIG['window_length'])

    # Split into train/val
    n_train = int(0.9 * len(windows))
    train_dataset = TensorDataset(
        torch.from_numpy(windows[:n_train]),
        torch.from_numpy(labels[:n_train])
    )
    val_dataset = TensorDataset(
        torch.from_numpy(windows[n_train:]),
        torch.from_numpy(labels[n_train:])
    )

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])

    encoder = TimeFreqEncoder(latent_dim=CONFIG['latent_dim']).to(device)
    classifier = Classifier(latent_dim=CONFIG['latent_dim']).to(device)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=1e-3, weight_decay=1e-4
    )
    criterion = nn.BCELoss()

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(CONFIG['max_epochs']):
        encoder.train()
        classifier.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(classifier(encoder(X)), y)
            loss.backward()
            optimizer.step()

        encoder.eval()
        classifier.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                val_loss += criterion(classifier(encoder(X)), y).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {
                'encoder': encoder.state_dict(),
                'classifier': classifier.state_dict()
            }
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                break

    encoder.load_state_dict(best_state['encoder'])
    classifier.load_state_dict(best_state['classifier'])

    return encoder, classifier


def predict_most_likely(encoder, classifier, window, history, device):
    """Predict most likely QVs (returns indices of top N_KEEP)."""
    encoder.eval()
    classifier.eval()

    with torch.no_grad():
        X = torch.from_numpy(window).unsqueeze(0).to(device)
        p_occur = classifier(encoder(X)).squeeze().cpu().numpy()

    # Add rarity bonus from recent history
    if len(history) > 0:
        recent_freq = history[-100:].mean(axis=0) if len(history) >= 100 else history.mean(axis=0)
        # Combine model prediction with frequency (higher freq = more likely)
        combined = 0.7 * p_occur + 0.3 * recent_freq
    else:
        combined = p_occur

    # Get top 27 most likely (1-indexed)
    top_indices = np.argsort(combined)[-CONFIG['n_keep']:]
    return set(idx + 1 for idx in top_indices), combined


def run_holdout_test():
    """Run strict holdout test on 500 most recent events."""
    device = CONFIG['device']

    # Load data
    print('Loading data...')
    data, data_norm, dates = load_data()
    total_events = len(data)

    # STRICT ISOLATION: Split data
    holdout_start = total_events - CONFIG['holdout_size']
    train_data = data_norm[:holdout_start]
    holdout_data = data_norm[holdout_start:]
    holdout_raw = data[holdout_start:]
    holdout_dates = dates[holdout_start:]

    print(f'Total events: {total_events}')
    print(f'Training data: events 0-{holdout_start-1} ({holdout_start} events)')
    print(f'Holdout data: events {holdout_start}-{total_events-1} ({CONFIG["holdout_size"]} events)')
    print(f'Holdout period: {holdout_dates[0]} to {holdout_dates[-1]}')
    print()
    print('ISOLATION CHECK: Training uses ZERO holdout data')
    print()

    # Train model on training data ONLY
    print('Training model on pre-holdout data only...')
    encoder, classifier = train_model(train_data, device)
    print('Model trained.')
    print()

    # Run predictions on holdout set
    print(f'Running predictions on {CONFIG["holdout_size"]} holdout events...')

    results = []
    wrong_counts = Counter()
    all_predictions = []

    window_len = CONFIG['window_length']

    for i in range(CONFIG['holdout_size']):
        # Get window ENDING just before this holdout event
        # Use only data available BEFORE this event
        available_data = np.vstack([train_data, holdout_data[:i]]) if i > 0 else train_data

        if len(available_data) < window_len:
            continue

        window = available_data[-window_len:].astype(np.float32)
        history = available_data[:-window_len] if len(available_data) > window_len else available_data

        # Predict
        predicted_likely, scores = predict_most_likely(encoder, classifier, window, history, device)

        # Get actual winners for this event
        actual_qvs = get_actual_top5(holdout_raw[i])

        # Count wrong (actual winners NOT in predicted likely set)
        wrong = len(actual_qvs - predicted_likely)
        wrong_counts[wrong] += 1

        results.append({
            'event_idx': holdout_start + i,
            'date': str(holdout_dates[i]),
            'predicted_likely': sorted(predicted_likely),
            'actual': sorted(actual_qvs),
            'wrong': wrong
        })
        all_predictions.append(scores)

        if (i + 1) % 100 == 0:
            print(f'  Processed {i+1}/{CONFIG["holdout_size"]} events...')

    # Compute pooled prediction for NEXT event
    print()
    print('Computing pooled prediction for next event...')

    # Use all available data for final prediction
    all_data = data_norm
    final_window = all_data[-window_len:].astype(np.float32)
    final_history = all_data[:-window_len]
    next_likely, next_scores = predict_most_likely(encoder, classifier, final_window, final_history, device)

    # Get excluded QVs
    excluded = set(range(1, 40)) - next_likely

    return results, wrong_counts, sorted(next_likely), sorted(excluded), next_scores


def main():
    results, wrong_counts, next_likely, excluded, scores = run_holdout_test()

    total = sum(wrong_counts.values())

    print()
    print('=' * 60)
    print(f'  Pooled {CONFIG["n_keep"]} Most Likely numbers next prediction')
    print(f'  {sorted(next_likely)}')
    print()
    print(f'  Excluded {CONFIG["n_exclude"]} numbers:')
    print(f'  {sorted(excluded)}')
    print()
    print('  HOLDOUT TEST SUMMARY - 500 most recent events')
    print('-' * 60)

    for wrong in range(6):
        count = wrong_counts.get(wrong, 0)
        pct = count / total * 100 if total > 0 else 0
        print(f'  {wrong} wrong: {count:3d} events ({pct:5.2f}%)')

    print()

    # Additional stats
    avg_wrong = sum(w * c for w, c in wrong_counts.items()) / total if total > 0 else 0
    perfect_rate = wrong_counts.get(0, 0) / total * 100 if total > 0 else 0
    good_rate = (wrong_counts.get(0, 0) + wrong_counts.get(1, 0)) / total * 100 if total > 0 else 0

    print(f'  Average wrong: {avg_wrong:.2f}')
    print(f'  Perfect (0 wrong): {perfect_rate:.1f}%')
    print(f'  Good (0-1 wrong): {good_rate:.1f}%')
    print('=' * 60)

    # Save results
    output_dir = Path('holdout_tests')
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    summary = {
        'config': CONFIG,
        'holdout_size': total,
        'wrong_distribution': {str(k): v for k, v in sorted(wrong_counts.items())},
        'avg_wrong': avg_wrong,
        'perfect_rate': perfect_rate,
        'good_rate': good_rate,
        'next_prediction': {
            'likely_27': next_likely,
            'excluded_12': excluded
        },
        'timestamp': datetime.now().isoformat()
    }

    with open(output_dir / f'holdout_test_{timestamp}.json', 'w') as f:
        json.dump(summary, f, indent=2)

    with open(output_dir / 'holdout_test_latest.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print()
    print(f'Results saved to: holdout_tests/holdout_test_{timestamp}.json')


if __name__ == '__main__':
    main()
