#!/usr/bin/env python3
"""
Score Exclusions - Generate inverse prediction for C5.

Produces the 20 least likely QVs for the next event using:
- RAINCOAT model predictions
- Target rarity prior
- Spectral mismatch
- Risk controls
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json
from datetime import datetime, timedelta
from pathlib import Path

print('=== SCORE EXCLUSIONS ===')
print()

# Load configuration
CONFIG = {
    'window_length': 64,
    'latent_dim': 128,
    'n_exclusions': 20,
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
    print('## Loading Model')

    checkpoint = torch.load('models/c5_raincoat_latest.pth', map_location='cpu')

    encoder = TimeFreqEncoder(n_features=39, latent_dim=CONFIG['latent_dim'])
    classifier = Classifier(latent_dim=CONFIG['latent_dim'], n_classes=39)

    encoder.load_state_dict(checkpoint['encoder_state'])
    classifier.load_state_dict(checkpoint['classifier_state'])

    encoder.eval()
    classifier.eval()

    print(f"  Model loaded: {checkpoint['timestamp']}")
    print(f"  Training metrics: val_loss={checkpoint['metrics']['best_val_loss']:.4f}")
    print()

    return encoder, classifier, checkpoint


def load_data():
    """Load source and target domain data."""
    print('## Loading Data')

    source_df = pd.read_csv('data/prepared/source_domain.csv')
    target_df = pd.read_csv('data/prepared/target_domain.csv')

    qv_cols = [c for c in source_df.columns if c.startswith('QV_')]

    source_data = source_df[qv_cols].values.astype(np.float32)
    target_data = target_df[qv_cols].values.astype(np.float32)

    # Normalize
    source_norm = source_data / (source_data.sum(axis=1, keepdims=True) + 1e-8)
    target_norm = target_data / (target_data.sum(axis=1, keepdims=True) + 1e-8)

    # Get dates
    source_dates = source_df['event-ID'].values
    target_dates = target_df['event-ID'].values

    print(f"  Source: {source_data.shape}, dates: {source_dates[0]} to {source_dates[-1]}")
    print(f"  Target: {target_data.shape}, dates: {target_dates[0]} to {target_dates[-1]}")
    print()

    return source_norm, target_norm, source_dates, target_dates, qv_cols


def prepare_input_window(target_data, window_length):
    """Prepare the most recent window for prediction."""
    # Use the last window_length events
    if len(target_data) >= window_length:
        window = target_data[-window_length:]
    else:
        # Pad with zeros if not enough data
        padding = np.zeros((window_length - len(target_data), target_data.shape[1]))
        window = np.vstack([padding, target_data])

    return window.astype(np.float32)


def compute_rarity_scores(target_data):
    """Compute target domain rarity (non-occurrence) scores."""
    # How often each QV did NOT have high probability in target
    mean_prob = target_data.mean(axis=0)
    rarity = 1.0 - mean_prob
    # Normalize to [0, 1]
    rarity = (rarity - rarity.min()) / (rarity.max() - rarity.min() + 1e-8)
    return rarity


def compute_spectral_mismatch(source_data, target_data):
    """Compute spectral mismatch between source and target."""
    # FFT of mean distributions
    source_mean = source_data.mean(axis=0)
    target_mean = target_data.mean(axis=0)

    source_fft = np.fft.fft(source_mean)
    target_fft = np.fft.fft(target_mean)

    # Amplitude difference
    amp_diff = np.abs(np.abs(target_fft) - np.abs(source_fft))

    # Normalize
    amp_diff = (amp_diff - amp_diff.min()) / (amp_diff.max() - amp_diff.min() + 1e-8)

    return amp_diff


def compute_recency_scores(target_data):
    """Score based on recent non-occurrence (last few events)."""
    # Look at last 10 events (or all if less)
    recent = target_data[-10:] if len(target_data) >= 10 else target_data

    # QVs that haven't appeared recently get higher scores
    recent_occurrence = (recent > 0.05).any(axis=0).astype(np.float32)
    recency = 1.0 - recent_occurrence

    return recency


def apply_risk_controls(fused_scores, exclusion_indices):
    """Apply risk controls to the exclusion set."""
    controls_applied = []

    # 1. Gap check - warn if margin is small
    sorted_scores = np.sort(fused_scores)[::-1]
    if len(sorted_scores) > 20:
        gap = sorted_scores[19] - sorted_scores[20]
        if gap < 0.05:
            controls_applied.append(f"gap_warning (gap={gap:.4f})")

    # 2. Diversity check - warn if too many adjacent QVs
    exclusion_set = set(exclusion_indices)
    adjacent_count = 0
    for idx in exclusion_indices:
        if (idx - 1) in exclusion_set or (idx + 1) in exclusion_set:
            adjacent_count += 1

    if adjacent_count > 10:
        controls_applied.append(f"adjacency_warning ({adjacent_count} adjacent)")

    return controls_applied if controls_applied else ["none"]


def main():
    # Load model and data
    encoder, classifier, checkpoint = load_model()
    source_data, target_data, source_dates, target_dates, qv_cols = load_data()

    # Prepare input window
    print('## Preparing Input')
    window = prepare_input_window(target_data, CONFIG['window_length'])
    print(f"  Input window: {window.shape}")
    print(f"  Window covers: last {CONFIG['window_length']} events")
    print()

    # Get model predictions
    print('## Model Prediction')
    with torch.no_grad():
        X = torch.from_numpy(window).unsqueeze(0)  # (1, seq, features)
        h = encoder(X)
        p_occur = classifier(h).squeeze().numpy()  # (39,)

    print(f"  Occurrence probabilities: min={p_occur.min():.4f}, max={p_occur.max():.4f}")

    # Base non-occurrence score
    s_base = 1.0 - p_occur
    print(f"  Base exclusion scores: min={s_base.min():.4f}, max={s_base.max():.4f}")
    print()

    # Compute fusion components
    print('## Computing Fusion Scores')

    s_rarity = compute_rarity_scores(target_data)
    print(f"  Rarity scores computed")

    s_spectral = compute_spectral_mismatch(source_data, target_data)
    print(f"  Spectral mismatch computed")

    s_recency = compute_recency_scores(target_data)
    print(f"  Recency scores computed")
    print()

    # Fused score
    w = CONFIG['fusion_weights']
    s_fused = (
        w['base'] * s_base +
        w['rarity'] * s_rarity +
        w['spectral'] * s_spectral +
        w['recency'] * s_recency
    )

    # Normalize
    s_fused = (s_fused - s_fused.min()) / (s_fused.max() - s_fused.min() + 1e-8)

    print('## Ranking Exclusions')

    # Get top-20 exclusions (highest scores = least likely to occur)
    exclusion_indices = np.argsort(s_fused)[::-1][:CONFIG['n_exclusions']]
    excluded_qvs = [i + 1 for i in exclusion_indices]  # 1-indexed

    # Apply risk controls
    risk_controls = apply_risk_controls(s_fused, exclusion_indices)

    # Compute confidence metrics
    sorted_scores = np.sort(s_fused)[::-1]
    mean_exclusion_score = sorted_scores[:20].mean()
    score_gap = sorted_scores[19] - sorted_scores[20] if len(sorted_scores) > 20 else 0

    print(f"  Mean exclusion score: {mean_exclusion_score:.4f}")
    print(f"  Score gap at cutoff: {score_gap:.4f}")
    print(f"  Risk controls: {risk_controls}")
    print()

    # Generate output
    print('## Exclusion Results')
    print()
    print(f"Prediction for next event after: {target_dates[-1]}")
    print()
    print('Top 20 Excluded QVs (Least Likely to Occur):')
    print('-' * 70)
    print(f"{'Rank':<6}{'QV':<8}{'Fused':<10}{'Base':<10}{'Rarity':<10}{'Spectral':<10}")
    print('-' * 70)

    for rank, idx in enumerate(exclusion_indices):
        qv = idx + 1
        print(f"{rank+1:<6}QV_{qv:<4}{s_fused[idx]:<10.4f}{s_base[idx]:<10.4f}{s_rarity[idx]:<10.4f}{s_spectral[idx]:<10.4f}")

    print('-' * 70)
    print()

    # Excluded QVs summary
    print(f"EXCLUDED QVs: {sorted(excluded_qvs)}")
    print()

    # Remaining QVs (potential winners)
    all_qvs = set(range(1, 40))
    remaining_qvs = sorted(all_qvs - set(excluded_qvs))
    print(f"REMAINING QVs (19 candidates): {remaining_qvs}")
    print()

    # Save output
    print('## Saving Output')

    output = {
        'prediction_date': str(datetime.now().date() + timedelta(days=1)),
        'input_window_end': str(target_dates[-1]),
        'model_timestamp': checkpoint['timestamp'],
        'excluded_qvs': sorted(excluded_qvs),
        'remaining_qvs': remaining_qvs,
        'scores': {
            f'QV_{i+1}': {
                'fused': float(s_fused[i]),
                'base': float(s_base[i]),
                'rarity': float(s_rarity[i]),
                'spectral': float(s_spectral[i]),
                'recency': float(s_recency[i])
            }
            for i in range(39)
        },
        'fusion_weights': CONFIG['fusion_weights'],
        'confidence': {
            'mean_exclusion_score': float(mean_exclusion_score),
            'score_gap_at_cutoff': float(score_gap),
        },
        'risk_controls': risk_controls,
        'created_at': datetime.now().isoformat()
    }

    output_path = Path('predictions')
    output_path.mkdir(exist_ok=True)

    output_file = output_path / f"exclusion_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {output_file}")

    # Also save as latest
    latest_file = output_path / 'exclusion_scores_latest.json'
    with open(latest_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {latest_file}")
    print()

    print('=' * 70)
    print('STATUS: Exclusion scoring complete!')
    print()
    print('INVERSE PREDICTION SUMMARY:')
    print(f"  Exclude these 20 QVs: {sorted(excluded_qvs)}")
    print(f"  Candidates (19 QVs):  {remaining_qvs}")
    print()
    print('The 5 actual winning QVs should be among the 19 candidates, NOT in the excluded 20.')


if __name__ == '__main__':
    main()
