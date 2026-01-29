#!/usr/bin/env python3
"""Run quantumization pipeline on prepared domain data."""

import numpy as np
import pandas as pd
import json
from datetime import datetime
from scipy.linalg import expm
import math
import os

print('=== QUANTUMIZE WORKFLOW ===')
print()

# Configuration
CONFIG = {
    'taus': [0.15, 0.50, 1.25],
    'top_k_edges': 12,
    'phase_a': 0.70,
    'phase_b': 1.30,
    'phase_c': 0.11,
    'embedding_dim': 128,
    'seed': 7,
    'include_full_distributions': False
}

EPS = 1e-12
OUTPUT_DIR = 'data/quantumized'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print('## Configuration')
print(f"Taus (quantum walk): {CONFIG['taus']}")
print(f"Top-k edges: {CONFIG['top_k_edges']}")
print(f"Embedding dim: {CONFIG['embedding_dim']}")
print()

# Helper functions
def row_normalize(X):
    row_sum = X.sum(axis=1).astype(np.float64)
    row_sum = np.where(row_sum <= 0, 1.0, row_sum)
    return X.astype(np.float64) / row_sum[:, None], row_sum

def make_phases(n_rows, n_parts, phase_a, phase_b, phase_c):
    part_idx = np.arange(n_parts, dtype=np.float64)[None, :]
    time_idx = np.arange(n_rows, dtype=np.float64)[:, None]
    denom_part = max(1.0, float(n_parts))
    denom_time = max(1.0, float(n_rows))
    raw = (phase_a * (part_idx / denom_part) +
           phase_b * (time_idx / denom_time) +
           phase_c * ((part_idx * time_idx) / (denom_part * denom_time)))
    return 2.0 * math.pi * (raw % 1.0)

def make_state_vector(p, phi):
    amp = np.sqrt(np.clip(p, 0.0, 1.0))
    psi = amp * np.exp(1j * phi)
    norms = np.linalg.norm(psi, axis=1, keepdims=True)
    norms = np.where(norms <= 0, 1.0, norms)
    return psi / norms

def unitary_dft(psi):
    d = psi.shape[1]
    return np.fft.fft(psi, axis=1) / math.sqrt(float(d))

def build_adjacency(X, top_k):
    b = (X > 0).astype(np.float64)
    cooc = b.T @ b
    diag = np.sqrt(np.clip(np.diag(cooc), EPS, None))
    denom = diag[:, None] * diag[None, :]
    w = cooc / np.clip(denom, EPS, None)
    np.fill_diagonal(w, 0.0)
    d = w.shape[0]
    k = max(1, min(top_k, d - 1))
    mask = np.zeros_like(w, dtype=bool)
    for i in range(d):
        idx = np.argpartition(w[i], -k)[-k:]
        mask[i, idx] = True
    w_sparse = np.where(mask, w, 0.0)
    w_sym = 0.5 * (w_sparse + w_sparse.T)
    np.fill_diagonal(w_sym, 0.0)
    return w_sym

def quantum_walk_unitary(L, tau):
    return expm((-1j) * L * float(tau))

def safe_entropy(p):
    p_clip = np.clip(p, EPS, 1.0)
    return -np.sum(p_clip * np.log(p_clip), axis=1)

def topk_mass(p, k):
    k = max(1, min(k, p.shape[1]))
    part = np.partition(p, -k, axis=1)[:, -k:]
    return np.sum(part, axis=1)

def spectral_centroid(q):
    d = q.shape[1]
    idx = np.arange(d, dtype=np.float64)[None, :]
    return np.sum(q * idx, axis=1)

def l1_coherence(p):
    s = np.sum(np.sqrt(np.clip(p, 0.0, 1.0)), axis=1)
    return (s * s) - 1.0

def quantumize(df, name):
    print(f'Processing: {name}')

    date_col = 'event-ID'
    qv_cols = [c for c in df.columns if c.startswith('QV_')]
    n_parts = len(qv_cols)

    X = df[qv_cols].values.astype(np.float64)
    n_rows = X.shape[0]
    print(f'  Rows: {n_rows}, Parts: {n_parts}')

    # Step 1: Normalize to probabilities
    p, row_sum = row_normalize(X)
    print(f'  [1/6] Amplitude encoding...')

    # Step 2: Generate phases
    phi = make_phases(n_rows, n_parts, CONFIG['phase_a'], CONFIG['phase_b'], CONFIG['phase_c'])
    print(f'  [2/6] Phase generation...')

    # Step 3: Create state vectors
    psi = make_state_vector(p, phi)
    print(f'  [3/6] State vectors (norm check: {np.linalg.norm(psi[0]):.6f})')

    # Step 4: Unitary DFT
    psi_fft = unitary_dft(psi)
    q_fft = np.abs(psi_fft) ** 2
    print(f'  [4/6] Unitary DFT...')

    # Step 5: Quantum walks
    W = build_adjacency(X, CONFIG['top_k_edges'])
    D = np.diag(W.sum(axis=1))
    L = D - W

    q_walks = {}
    for tau in CONFIG['taus']:
        U = quantum_walk_unitary(L, tau)
        psi_tau = psi @ U.T
        q_walks[tau] = np.abs(psi_tau) ** 2
    print(f"  [5/6] Quantum walks (taus: {CONFIG['taus']})...")

    # Step 6: Extract features
    features = {
        'row_total': row_sum,
        'entropy_p': safe_entropy(p),
        'purity': np.sum(p * p, axis=1),
        'coherence_l1': l1_coherence(p),
        'top3_mass_p': topk_mass(p, 3),
        'top5_mass_p': topk_mass(p, 5),
        'entropy_fft': safe_entropy(q_fft),
        'spectral_centroid_fft': spectral_centroid(q_fft),
        'top3_mass_fft': topk_mass(q_fft, 3),
    }

    for tau, q_tau in q_walks.items():
        tag = f'{tau:.2f}'.replace('.', 'p')
        features[f'entropy_qwalk_{tag}'] = safe_entropy(q_tau)
        features[f'spectral_centroid_qwalk_{tag}'] = spectral_centroid(q_tau)
        features[f'top3_mass_qwalk_{tag}'] = topk_mass(q_tau, 3)

    print(f'  [6/6] Feature extraction ({len(features)} features)...')

    # Build output dataframe
    out_df = pd.DataFrame()
    out_df['row_id'] = range(n_rows)
    out_df['date'] = df[date_col].values
    for k, v in features.items():
        out_df[k] = v

    # Generate embeddings
    print(f'  Generating embeddings...')
    tau0 = sorted(q_walks.keys())[0]
    F = np.concatenate([p, q_fft, q_walks[tau0]], axis=1).astype(np.float32)

    rng = np.random.default_rng(CONFIG['seed'])
    proj = rng.normal(0, 1, (F.shape[1], CONFIG['embedding_dim'])).astype(np.float32)
    proj /= math.sqrt(CONFIG['embedding_dim'])

    emb = (F @ proj).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms <= 0, 1.0, norms).astype(np.float32)
    embeddings = emb / norms

    print(f'  Done! Features: {len(out_df.columns)}, Embeddings: {embeddings.shape}')
    return out_df, embeddings


if __name__ == '__main__':
    # Process source domain
    print()
    print('## Processing Source Domain')
    source_df = pd.read_csv('data/prepared/source_domain.csv')
    source_features, source_embeddings = quantumize(source_df, 'source_domain')

    # Process target domain
    print()
    print('## Processing Target Domain')
    target_df = pd.read_csv('data/prepared/target_domain.csv')
    target_features, target_embeddings = quantumize(target_df, 'target_domain')

    # Save outputs
    print()
    print('## Saving Outputs')

    source_features.to_csv(f'{OUTPUT_DIR}/source_quantumized.csv', index=False)
    print(f'  Saved: {OUTPUT_DIR}/source_quantumized.csv')

    target_features.to_csv(f'{OUTPUT_DIR}/target_quantumized.csv', index=False)
    print(f'  Saved: {OUTPUT_DIR}/target_quantumized.csv')

    np.save(f'{OUTPUT_DIR}/source_embeddings.npy', source_embeddings)
    print(f'  Saved: {OUTPUT_DIR}/source_embeddings.npy ({source_embeddings.shape})')

    np.save(f'{OUTPUT_DIR}/target_embeddings.npy', target_embeddings)
    print(f'  Saved: {OUTPUT_DIR}/target_embeddings.npy ({target_embeddings.shape})')

    # Save combined embeddings for FAISS
    all_embeddings = np.vstack([source_embeddings, target_embeddings])
    np.save(f'{OUTPUT_DIR}/all_embeddings.npy', all_embeddings)
    print(f'  Saved: {OUTPUT_DIR}/all_embeddings.npy ({all_embeddings.shape})')

    # Metadata
    metadata = pd.concat([
        source_features[['row_id', 'date']].assign(domain='source'),
        target_features[['row_id', 'date']].assign(domain='target')
    ], ignore_index=True)
    metadata.to_csv(f'{OUTPUT_DIR}/embedding_metadata.csv', index=False)
    print(f'  Saved: {OUTPUT_DIR}/embedding_metadata.csv')

    # Config
    config_out = {
        **CONFIG,
        'source_rows': len(source_features),
        'target_rows': len(target_features),
        'n_features': len(source_features.columns) - 2,
        'created_at': datetime.now().isoformat()
    }
    with open(f'{OUTPUT_DIR}/quantumize_config.json', 'w') as f:
        json.dump(config_out, f, indent=2)
    print(f'  Saved: {OUTPUT_DIR}/quantumize_config.json')

    # Summary statistics
    print()
    print('## Feature Statistics')
    print(f"Entropy (p) range: [{source_features['entropy_p'].min():.3f}, {source_features['entropy_p'].max():.3f}]")
    print(f"Purity range: [{source_features['purity'].min():.3f}, {source_features['purity'].max():.3f}]")
    print(f"Coherence range: [{source_features['coherence_l1'].min():.3f}, {source_features['coherence_l1'].max():.3f}]")

    print()
    print('=' * 50)
    print('STATUS: Quantumization complete!')
