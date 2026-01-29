#!/usr/bin/env python3
"""
Backtest Inverse Prediction - Walk-forward validation.

Validates the RAINCOAT model by running predictions on historical data
and computing FP@20 (false positives in the exclusion set).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json
from datetime import datetime
from pathlib import Path
from collections import Counter

print('=== BACKTEST INVERSE PREDICTION ===')
print()

# Configuration
CONFIG = {
    'window_length': 64,
    'latent_dim': 128,
    'n_exclusions': 20,
    'backtest_start': 500,  # Start after enough history
    'backtest_step': 10,    # Predict every N events (for speed)
    'fusion_weights': {
        'base': 0.50,
        'rarity': 0.25,
        'spectral': 0.15,
        'recency': 0.10
    }
}


class TimeFreqEncoder(nn.Module):
    """Time-frequency encoder (must match training)."""

    def __init__(self, n_features=39, latent_dim=128):
        super().__init__()
        self.n_features = n_features

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
        batch_size = x.shape[0]
        x_t = x.permute(0, 2, 1)
        h_temporal = self.temporal(x_t).squeeze(-1)

        x_mean = x.mean(dim=1)
        x_fft = torch.fft.fft(x_mean.to(torch.complex64))
        amp = torch.abs(x_fft)
        phase = torch.angle(x_fft)
        h_freq = self.freq(torch.cat([amp, phase], dim=1))

        h = self.fusion(torch.cat([h_temporal, h_freq], dim=1))
        return h


class Classifier(nn.Module):
    """Multi-label classifier."""

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


def load_model():
    """Load trained RAINCOAT model."""
    checkpoint = torch.load('models/c5_raincoat_latest.pth', map_location='cpu')

    encoder = TimeFreqEncoder(n_features=39, latent_dim=CONFIG['latent_dim'])
    classifier = Classifier(latent_dim=CONFIG['latent_dim'], n_classes=39)

    encoder.load_state_dict(checkpoint['encoder_state'])
    classifier.load_state_dict(checkpoint['classifier_state'])

    encoder.eval()
    classifier.eval()

    return encoder, classifier


def load_full_data():
    """Load full dataset for backtesting."""
    # Load original data
    df = pd.read_csv('data/raw/CA5_quantum.csv')
    qv_cols = [c for c in df.columns if c.startswith('QV_')]

    data = df[qv_cols].values.astype(np.float32)
    dates = df['event-ID'].values

    # Normalize by row sum
    data_norm = data / (data.sum(axis=1, keepdims=True) + 1e-8)

    return data, data_norm, dates, qv_cols


def get_actual_qvs(row):
    """Get actual QVs that occurred (non-zero values)."""
    # For count data, any QV with count > 0 is considered "occurred"
    # But we need to identify which were the "selected" ones
    # In aggregated data, we look at top-5 by probability
    probs = row / (row.sum() + 1e-8)
    top5_indices = np.argsort(probs)[-5:]
    return set(top5_indices + 1)  # 1-indexed


def predict_exclusions(encoder, classifier, window, history_for_rarity):
    """Generate exclusion prediction for a single window."""
    with torch.no_grad():
        X = torch.from_numpy(window).unsqueeze(0)
        h = encoder(X)
        p_occur = classifier(h).squeeze().numpy()

    # Base non-occurrence score
    s_base = 1.0 - p_occur

    # Rarity from recent history
    if len(history_for_rarity) > 0:
        mean_prob = history_for_rarity.mean(axis=0)
        s_rarity = 1.0 - mean_prob
        s_rarity = (s_rarity - s_rarity.min()) / (s_rarity.max() - s_rarity.min() + 1e-8)
    else:
        s_rarity = np.zeros(39)

    # Simple fusion (base + rarity)
    w = CONFIG['fusion_weights']
    s_fused = w['base'] * s_base + w['rarity'] * s_rarity

    # Get top-20 exclusions
    exclusion_indices = np.argsort(s_fused)[::-1][:CONFIG['n_exclusions']]
    excluded_qvs = set(i + 1 for i in exclusion_indices)

    return excluded_qvs, s_fused


def run_backtest(encoder, classifier, data_norm, dates):
    """Run walk-forward backtest."""
    n_total = len(data_norm)
    window_length = CONFIG['window_length']
    start_idx = CONFIG['backtest_start']
    step = CONFIG['backtest_step']

    results = []
    fp_counts = []

    # Calculate number of predictions
    n_predictions = (n_total - start_idx - 1) // step
    print(f'Backtest range: event {start_idx} to {n_total-1}')
    print(f'Step size: {step}')
    print(f'Total predictions: {n_predictions}')
    print()

    print('Running backtest...')
    for i, pred_idx in enumerate(range(start_idx, n_total - 1, step)):
        # Get window ending at pred_idx
        window_start = max(0, pred_idx - window_length)
        window = data_norm[window_start:pred_idx]

        # Pad if needed
        if len(window) < window_length:
            padding = np.zeros((window_length - len(window), 39), dtype=np.float32)
            window = np.vstack([padding, window])

        # History for rarity (last 50 events before window)
        rarity_start = max(0, window_start - 50)
        history_for_rarity = data_norm[rarity_start:window_start]

        # Make prediction
        excluded_qvs, scores = predict_exclusions(
            encoder, classifier, window.astype(np.float32), history_for_rarity
        )

        # Get actual result (next event)
        actual_row = data_norm[pred_idx]
        actual_qvs = get_actual_qvs(actual_row)

        # Compute FP@20 (actual QVs that were incorrectly excluded)
        false_positives = actual_qvs & excluded_qvs
        fp = len(false_positives)

        results.append({
            'pred_idx': pred_idx,
            'date': str(dates[pred_idx]),
            'excluded': list(excluded_qvs),
            'actual': list(actual_qvs),
            'fp': fp,
            'false_positives': list(false_positives)
        })
        fp_counts.append(fp)

        # Progress
        if (i + 1) % 100 == 0:
            current_mean = np.mean(fp_counts)
            print(f'  Progress: {i+1}/{n_predictions} predictions, running FP@20: {current_mean:.3f}')

    return results, fp_counts


def analyze_results(results, fp_counts):
    """Analyze backtest results."""
    fp_array = np.array(fp_counts)

    metrics = {
        'n_predictions': len(fp_counts),
        'fp_at_20_mean': float(fp_array.mean()),
        'fp_at_20_std': float(fp_array.std()),
        'fp_at_20_median': float(np.median(fp_array)),
        'perfect_rate': float((fp_array == 0).mean() * 100),
        'one_fp_rate': float((fp_array == 1).mean() * 100),
        'two_plus_fp_rate': float((fp_array >= 2).mean() * 100),
        'worst_case_fp': int(fp_array.max()),
        'hit_at_5_rate': float((fp_array == 5).mean() * 100),
    }

    # FP distribution
    fp_distribution = {
        f'fp_{i}': int((fp_array == i).sum()) for i in range(6)
    }

    # Random baseline: expected FP = 20/39 * 5 = 2.56
    random_baseline = 20 * 5 / 39

    return metrics, fp_distribution, random_baseline


def main():
    print('## Loading Model')
    encoder, classifier = load_model()
    print('  Model loaded')
    print()

    print('## Loading Data')
    data, data_norm, dates, qv_cols = load_full_data()
    print(f'  Total events: {len(data)}')
    print(f'  Date range: {dates[0]} to {dates[-1]}')
    print()

    print('## Running Backtest')
    results, fp_counts = run_backtest(encoder, classifier, data_norm, dates)
    print()

    print('## Analyzing Results')
    metrics, fp_distribution, random_baseline = analyze_results(results, fp_counts)
    print()

    # Print report
    print('=' * 60)
    print('BACKTEST RESULTS')
    print('=' * 60)
    print()

    print('## Primary Metrics')
    print(f"  Predictions:        {metrics['n_predictions']}")
    print(f"  FP@20 Mean:         {metrics['fp_at_20_mean']:.3f} (random baseline: {random_baseline:.2f})")
    print(f"  FP@20 Std:          {metrics['fp_at_20_std']:.3f}")
    print(f"  FP@20 Median:       {metrics['fp_at_20_median']:.1f}")
    print()

    print('## Success Rates')
    print(f"  Perfect (FP=0):     {metrics['perfect_rate']:.1f}%")
    print(f"  One FP (FP=1):      {metrics['one_fp_rate']:.1f}%")
    print(f"  Two+ FP (FP>=2):    {metrics['two_plus_fp_rate']:.1f}%")
    print(f"  Worst case:         {metrics['worst_case_fp']} FP")
    print()

    print('## FP Distribution')
    total = metrics['n_predictions']
    for i in range(6):
        count = fp_distribution[f'fp_{i}']
        pct = count / total * 100
        bar = '#' * int(pct / 2)
        label = 'Perfect!' if i == 0 else f'{i} wrong'
        print(f"  FP={i}: {count:4d} ({pct:5.1f}%) {bar} {label}")
    print()

    print('## Comparison to Baseline')
    improvement = (random_baseline - metrics['fp_at_20_mean']) / random_baseline * 100
    print(f"  Random baseline:    {random_baseline:.2f} FP")
    print(f"  Model performance:  {metrics['fp_at_20_mean']:.2f} FP")
    print(f"  Improvement:        {improvement:.1f}%")
    print()

    # Save results
    print('## Saving Results')
    output_dir = Path('backtests')
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Full results
    full_output = {
        'config': CONFIG,
        'metrics': metrics,
        'fp_distribution': fp_distribution,
        'random_baseline': random_baseline,
        'improvement_pct': improvement,
        'timestamp': datetime.now().isoformat()
    }

    with open(output_dir / f'backtest_results_{timestamp}.json', 'w') as f:
        json.dump(full_output, f, indent=2)
    print(f'  Saved: backtests/backtest_results_{timestamp}.json')

    # Also save as latest
    with open(output_dir / 'backtest_results_latest.json', 'w') as f:
        json.dump(full_output, f, indent=2)
    print(f'  Saved: backtests/backtest_results_latest.json')

    # Save detailed predictions (sample)
    sample_results = results[:100] + results[-100:]  # First and last 100
    with open(output_dir / f'backtest_details_{timestamp}.json', 'w') as f:
        json.dump(sample_results, f, indent=2)
    print(f'  Saved: backtests/backtest_details_{timestamp}.json')
    print()

    print('=' * 60)
    print('STATUS: Backtest complete!')
    print()

    # Interpretation
    if metrics['fp_at_20_mean'] < random_baseline * 0.8:
        print('INTERPRETATION: Model shows meaningful improvement over random.')
    elif metrics['fp_at_20_mean'] < random_baseline:
        print('INTERPRETATION: Model shows modest improvement over random.')
    else:
        print('INTERPRETATION: Model needs improvement - performing near random baseline.')


if __name__ == '__main__':
    main()
