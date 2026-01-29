#!/usr/bin/env python3
"""
RAINCOAT-style training for C5 inverse prediction.

Three-stage domain adaptation:
- Stage A: Supervised training on source domain
- Stage B: Domain alignment with MMD
- Stage C: Target reconstruction/correction
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import json
from datetime import datetime
from pathlib import Path

# Configuration
CONFIG = {
    'window_length': 64,       # Shorter windows for count data
    'latent_dim': 128,
    'batch_size': 32,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'max_epochs_a': 50,
    'max_epochs_b': 30,
    'max_epochs_c': 20,
    'patience': 10,
    'lambda_mmd': 0.1,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print('=== TRAIN RAINCOAT ===')
print()
print('## Configuration')
for k, v in CONFIG.items():
    print(f'  {k}: {v}')
print()


class TimeFreqEncoder(nn.Module):
    """Time-frequency encoder with temporal CNN and FFT branch."""

    def __init__(self, n_features=39, latent_dim=128):
        super().__init__()
        self.n_features = n_features

        # Temporal branch - 1D CNN
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

        # Frequency branch - process FFT features
        self.freq = nn.Sequential(
            nn.Linear(n_features * 2, 128),  # amplitude + phase
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU()
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(128 + 64, latent_dim),
            nn.SiLU()
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        batch_size = x.shape[0]

        # Temporal branch
        x_t = x.permute(0, 2, 1)  # (batch, features, seq_len)
        h_temporal = self.temporal(x_t).squeeze(-1)  # (batch, 128)

        # Frequency branch - FFT of mean signal
        x_mean = x.mean(dim=1)  # (batch, n_features)
        x_fft = torch.fft.fft(x_mean.to(torch.complex64))
        amp = torch.abs(x_fft)
        phase = torch.angle(x_fft)
        h_freq = self.freq(torch.cat([amp, phase], dim=1))  # (batch, 64)

        # Fusion
        h = self.fusion(torch.cat([h_temporal, h_freq], dim=1))
        return h


class Classifier(nn.Module):
    """Multi-label classifier for 39 QVs."""

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


class Decoder(nn.Module):
    """Decoder for target reconstruction."""

    def __init__(self, latent_dim=128, seq_len=64, n_features=39):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.SiLU(),
            nn.Linear(256, seq_len * n_features)
        )

    def forward(self, h):
        out = self.net(h)
        return out.view(-1, self.seq_len, self.n_features)


def mmd_loss(source_features, target_features):
    """Maximum Mean Discrepancy loss for domain alignment."""
    # Gaussian kernel MMD
    def gaussian_kernel(x, y, sigma=1.0):
        dist = torch.cdist(x, y, p=2)
        return torch.exp(-dist ** 2 / (2 * sigma ** 2))

    n_s = source_features.shape[0]
    n_t = target_features.shape[0]

    k_ss = gaussian_kernel(source_features, source_features)
    k_tt = gaussian_kernel(target_features, target_features)
    k_st = gaussian_kernel(source_features, target_features)

    mmd = k_ss.sum() / (n_s * n_s) + k_tt.sum() / (n_t * n_t) - 2 * k_st.sum() / (n_s * n_t)
    return mmd


def create_windows(data, window_length, stride=1):
    """Create overlapping windows from time series data."""
    windows = []
    labels = []
    n_rows = len(data)

    for i in range(0, n_rows - window_length, stride):
        window = data[i:i + window_length]
        # Label is the next row after window (for prediction)
        if i + window_length < n_rows:
            label = data[i + window_length]
            # Convert to binary (occurred or not)
            label_binary = (label > 0).astype(np.float32)
            windows.append(window)
            labels.append(label_binary)

    return np.array(windows, dtype=np.float32), np.array(labels, dtype=np.float32)


def load_data():
    """Load and prepare training data."""
    print('## Loading Data')

    # Load domain data
    source_df = pd.read_csv('data/prepared/source_domain.csv')
    target_df = pd.read_csv('data/prepared/target_domain.csv')

    qv_cols = [c for c in source_df.columns if c.startswith('QV_')]

    source_data = source_df[qv_cols].values.astype(np.float32)
    target_data = target_df[qv_cols].values.astype(np.float32)

    print(f'  Source: {source_data.shape}')
    print(f'  Target: {target_data.shape}')

    # Normalize by row sum (convert to probabilities)
    source_data = source_data / (source_data.sum(axis=1, keepdims=True) + 1e-8)
    target_data = target_data / (target_data.sum(axis=1, keepdims=True) + 1e-8)

    # Create windows
    window_length = CONFIG['window_length']

    # For source: create windows with labels
    source_windows, source_labels = create_windows(source_data, window_length, stride=4)
    print(f'  Source windows: {source_windows.shape}, labels: {source_labels.shape}')

    # For target: just windows (unsupervised)
    target_windows, _ = create_windows(target_data, window_length, stride=1)
    print(f'  Target windows: {target_windows.shape}')

    # Split source into train/val
    n_train = int(0.9 * len(source_windows))
    train_windows = source_windows[:n_train]
    train_labels = source_labels[:n_train]
    val_windows = source_windows[n_train:]
    val_labels = source_labels[n_train:]

    print(f'  Train: {train_windows.shape}, Val: {val_windows.shape}')
    print()

    return train_windows, train_labels, val_windows, val_labels, target_windows


def train_stage_a(encoder, classifier, train_loader, val_loader, device):
    """Stage A: Supervised training on source domain."""
    print('## Stage A: Supervised Training')

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=CONFIG['lr'],
        weight_decay=CONFIG['weight_decay']
    )

    criterion = nn.BCELoss()
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(CONFIG['max_epochs_a']):
        # Training
        encoder.train()
        classifier.train()
        train_loss = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            h = encoder(X)
            pred = classifier(h)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        encoder.eval()
        classifier.eval()
        val_loss = 0

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                h = encoder(X)
                pred = classifier(h)
                loss = criterion(pred, y)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {
                'encoder': encoder.state_dict(),
                'classifier': classifier.state_dict()
            }
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'  Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}')

        if patience_counter >= CONFIG['patience']:
            print(f'  Early stopping at epoch {epoch+1}')
            break

    # Restore best
    encoder.load_state_dict(best_state['encoder'])
    classifier.load_state_dict(best_state['classifier'])

    print(f'  Best val_loss: {best_val_loss:.4f}')
    print()

    return history, best_val_loss


def train_stage_b(encoder, classifier, train_loader, target_loader, device):
    """Stage B: Domain alignment with MMD."""
    print('## Stage B: Domain Alignment')

    optimizer = torch.optim.AdamW(
        encoder.parameters(),
        lr=CONFIG['lr'] * 0.1,  # Lower LR for fine-tuning
        weight_decay=CONFIG['weight_decay']
    )

    criterion = nn.BCELoss()
    history = {'bce_loss': [], 'mmd_loss': [], 'total_loss': []}

    target_iter = iter(target_loader)

    for epoch in range(CONFIG['max_epochs_b']):
        encoder.train()
        classifier.eval()  # Keep classifier frozen in alignment

        epoch_bce = 0
        epoch_mmd = 0

        for X_s, y_s in train_loader:
            # Get target batch
            try:
                X_t = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                X_t = next(target_iter)

            if isinstance(X_t, (list, tuple)):
                X_t = X_t[0]

            X_s, y_s = X_s.to(device), y_s.to(device)
            X_t = X_t.to(device)

            optimizer.zero_grad()

            # Source forward
            h_s = encoder(X_s)
            pred_s = classifier(h_s)
            loss_bce = criterion(pred_s, y_s)

            # Target forward
            h_t = encoder(X_t)

            # MMD alignment
            loss_mmd = mmd_loss(h_s, h_t)

            # Combined loss
            loss = loss_bce + CONFIG['lambda_mmd'] * loss_mmd
            loss.backward()
            optimizer.step()

            epoch_bce += loss_bce.item()
            epoch_mmd += loss_mmd.item()

        epoch_bce /= len(train_loader)
        epoch_mmd /= len(train_loader)

        history['bce_loss'].append(epoch_bce)
        history['mmd_loss'].append(epoch_mmd)
        history['total_loss'].append(epoch_bce + CONFIG['lambda_mmd'] * epoch_mmd)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'  Epoch {epoch+1}: BCE={epoch_bce:.4f}, MMD={epoch_mmd:.4f}')

    print(f'  Final MMD: {history["mmd_loss"][-1]:.4f}')
    print()

    return history


def train_stage_c(encoder, decoder, target_loader, device):
    """Stage C: Target reconstruction/correction."""
    print('## Stage C: Target Correction')

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=CONFIG['lr'] * 0.1,
        weight_decay=CONFIG['weight_decay']
    )

    history = {'recon_loss': []}

    # Track embedding movement
    encoder.eval()
    with torch.no_grad():
        all_targets = []
        for X_t in target_loader:
            if isinstance(X_t, (list, tuple)):
                X_t = X_t[0]
            all_targets.append(X_t)
        all_targets = torch.cat(all_targets, dim=0).to(device)
        initial_embeddings = encoder(all_targets).cpu().numpy()

    encoder.train()
    decoder.train()

    for epoch in range(CONFIG['max_epochs_c']):
        epoch_loss = 0

        for X_t in target_loader:
            if isinstance(X_t, (list, tuple)):
                X_t = X_t[0]
            X_t = X_t.to(device)

            optimizer.zero_grad()

            h = encoder(X_t)
            recon = decoder(h)
            loss = F.mse_loss(recon, X_t)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(target_loader)
        history['recon_loss'].append(epoch_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'  Epoch {epoch+1}: recon_loss={epoch_loss:.4f}')

    # Compute embedding movement
    encoder.eval()
    with torch.no_grad():
        final_embeddings = encoder(all_targets).cpu().numpy()

    movement = np.linalg.norm(final_embeddings - initial_embeddings) / len(initial_embeddings)
    movement_pct = movement / np.linalg.norm(initial_embeddings.mean(axis=0)) * 100

    print(f'  Embedding movement: {movement:.4f} ({movement_pct:.1f}%)')
    print()

    return history, movement_pct


def main():
    device = CONFIG['device']
    print(f'Using device: {device}')
    print()

    # Load data
    train_windows, train_labels, val_windows, val_labels, target_windows = load_data()

    # Create data loaders
    train_dataset = TensorDataset(
        torch.from_numpy(train_windows),
        torch.from_numpy(train_labels)
    )
    val_dataset = TensorDataset(
        torch.from_numpy(val_windows),
        torch.from_numpy(val_labels)
    )
    target_dataset = TensorDataset(torch.from_numpy(target_windows))

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
    target_loader = DataLoader(target_dataset, batch_size=CONFIG['batch_size'], shuffle=True)

    # Create models
    encoder = TimeFreqEncoder(n_features=39, latent_dim=CONFIG['latent_dim']).to(device)
    classifier = Classifier(latent_dim=CONFIG['latent_dim'], n_classes=39).to(device)
    decoder = Decoder(latent_dim=CONFIG['latent_dim'], seq_len=CONFIG['window_length'], n_features=39).to(device)

    print(f'Encoder params: {sum(p.numel() for p in encoder.parameters()):,}')
    print(f'Classifier params: {sum(p.numel() for p in classifier.parameters()):,}')
    print(f'Decoder params: {sum(p.numel() for p in decoder.parameters()):,}')
    print()

    # Stage A: Supervised training
    history_a, best_val_loss = train_stage_a(encoder, classifier, train_loader, val_loader, device)

    # Stage B: Domain alignment
    history_b = train_stage_b(encoder, classifier, train_loader, target_loader, device)

    # Stage C: Target correction
    history_c, movement_pct = train_stage_c(encoder, decoder, target_loader, device)

    # Save checkpoint
    print('## Saving Checkpoint')
    checkpoint = {
        'encoder_state': encoder.state_dict(),
        'classifier_state': classifier.state_dict(),
        'decoder_state': decoder.state_dict(),
        'config': CONFIG,
        'history': {
            'stage_a': history_a,
            'stage_b': history_b,
            'stage_c': history_c
        },
        'metrics': {
            'best_val_loss': best_val_loss,
            'final_mmd': history_b['mmd_loss'][-1],
            'final_recon': history_c['recon_loss'][-1],
            'embedding_movement_pct': movement_pct
        },
        'timestamp': datetime.now().isoformat()
    }

    model_path = Path('models') / f'c5_raincoat_v{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
    torch.save(checkpoint, model_path)
    print(f'  Saved: {model_path}')

    # Also save as latest
    latest_path = Path('models') / 'c5_raincoat_latest.pth'
    torch.save(checkpoint, latest_path)
    print(f'  Saved: {latest_path}')

    # Save training log
    log = {
        'config': CONFIG,
        'metrics': checkpoint['metrics'],
        'model_path': str(model_path),
        'timestamp': checkpoint['timestamp']
    }
    log_path = Path('logs') / f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f'  Saved: {log_path}')
    print()

    # Summary
    print('=' * 50)
    print('## Training Summary')
    print()
    print('Stage A (Supervised):')
    print(f'  Best validation loss: {best_val_loss:.4f}')
    print()
    print('Stage B (Alignment):')
    print(f'  Final MMD: {history_b["mmd_loss"][-1]:.4f}')
    print()
    print('Stage C (Correction):')
    print(f'  Final reconstruction loss: {history_c["recon_loss"][-1]:.4f}')
    print(f'  Embedding movement: {movement_pct:.1f}%')
    print()
    print('STATUS: RAINCOAT training complete!')


if __name__ == '__main__':
    main()
