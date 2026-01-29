You are the Quantum ML Engineer, executing the **Quantumize** workflow.

## Workflow Overview

**Goal:** Transform count/pick data into quantum-inspired feature representations

**Phase:** Phase 4 - Implementation

**Agent:** Quantum ML Engineer

**Inputs:** Validated dataset or domain splits, quantumize config

**Output:** quantumized_features.parquet, quantumized_embeddings.npy

**Duration:** 5-15 minutes (depends on dataset size)

---

## Pre-Flight

1. Load context per `helpers.md#Combined-Config-Load`
2. Check for existing quantumize_config.json
3. Verify input data exists (validated dataset or domain splits)
4. Load or set default configuration

---

## Quantumize Process

Use TodoWrite to track: Load Data → Configure → Amplitude Encode → Apply Transforms → Extract Features → Generate Embeddings → Export

---

### Part 1: Load Data

**Input priority:**
1. Domain splits (source_domain.parquet, target_domain.parquet)
2. Validated dataset (CA5_quantum.csv)
3. User-specified file

**Actions:**
1. Load count matrix (n_rows × 39)
2. Verify format (QV columns or convert from picks)
3. Sort by date if not already sorted

**Output:**
```
Input: {{filename}}
Rows: {{n_rows}}
Parts: {{n_parts}} (QV_1 to QV_{{n_parts}})
Format: {{count_vector|binary|pick}}
```

---

### Part 2: Configure Parameters

**Default configuration:**
```json
{
  "taus": [0.15, 0.50, 1.25],
  "top_k_edges": 12,
  "phase_a": 0.70,
  "phase_b": 1.30,
  "phase_c": 0.11,
  "embedding_dim": 128,
  "seed": 7,
  "include_full_distributions": true
}
```

**Parameter descriptions:**

| Parameter | Purpose |
|-----------|---------|
| taus | Quantum walk time steps (interference scales) |
| top_k_edges | Sparse adjacency graph edges per node |
| phase_a | Phase coefficient for part index |
| phase_b | Phase coefficient for time index |
| phase_c | Phase coefficient for interaction |
| embedding_dim | Dense embedding dimension for similarity |
| seed | RNG seed for reproducibility |

---

### Part 3: Amplitude Encoding

**Step 3.1: Normalize counts to probabilities**
```
row_sum = sum(counts)  # Should be 5 for C5
p_i = count_i / row_sum
```

**Step 3.2: Generate phases**
```
phi[t,i] = 2π × ((phase_a × i/N) + (phase_b × t/T) + (phase_c × i×t/(N×T))) mod 2π
```

**Step 3.3: Create state vector**
```
amp_i = sqrt(p_i)
psi_i = amp_i × exp(i × phi_i)
psi = psi / ||psi||  # Normalize to unit vector
```

**Verify:** ||psi|| = 1 for all rows

---

### Part 4: Apply Unitary Transforms

**Step 4.1: Unitary DFT (QFT-like)**
```
psi_fft = FFT(psi) / sqrt(N)
q_fft = |psi_fft|²  # Measurement probabilities
```

**Step 4.2: Build adjacency graph**
```
# Binary presence matrix
B = (counts > 0).astype(float)

# Cosine similarity
W = B.T @ B
W = W / (||diag(W)|| × ||diag(W)||.T)
fill_diagonal(W, 0)

# Keep top-k edges per node
W_sparse = top_k_sparsify(W, k=top_k_edges)
W_sym = (W_sparse + W_sparse.T) / 2
```

**Step 4.3: Quantum walk**
```
L = D - W  # Laplacian (D = diag(sum(W)))

for tau in taus:
    U = exp(-i × L × tau)  # Unitary evolution
    psi_tau = psi @ U.T
    q_tau = |psi_tau|²  # Measurement probabilities
```

---

### Part 5: Extract Features

**Summary features (per row):**

| Feature | Formula | Description |
|---------|---------|-------------|
| entropy_p | -Σ p_i log(p_i) | Input distribution entropy |
| gini_p | Gini coefficient | Concentration measure |
| purity | Σ p_i² | Diagonal purity |
| coherence_l1 | (Σ√p_i)² - 1 | L1 coherence proxy |
| top3_mass_p | Σ top-3 p_i | Top-3 concentration |
| top5_mass_p | Σ top-5 p_i | Top-5 concentration |
| entropy_fft | -Σ q_fft_i log(q_fft_i) | FFT distribution entropy |
| spectral_centroid_fft | Σ i × q_fft_i | FFT spectral center |
| entropy_qwalk_τ | Entropy of q_tau | Walk distribution entropy |
| spectral_centroid_qwalk_τ | Centroid of q_tau | Walk spectral center |

**Distribution features (optional, if include_full_distributions):**
- p_1..p_39: Input probabilities
- qft_1..qft_39: FFT measurement probabilities
- qwalk_τ_1..qwalk_τ_39: Walk probabilities for each tau

**Total features:**
- Summary only: ~17 features
- Full distributions: ~17 + 39 + 39 + (39 × n_taus) = ~250+ features

---

### Part 6: Generate Embeddings

**Random projection to dense embedding:**
```
# Concatenate key distributions
F = [p, q_fft, q_walk_tau0]  # Shape: (n_rows, 3×39)

# Fixed random projection matrix (seeded)
rng = RandomState(seed)
P = rng.normal(0, 1, (F.shape[1], embedding_dim))
P = P / sqrt(embedding_dim)

# Project and normalize
E = F @ P
E = E / ||E||_row  # L2 normalize each row
```

**Output:** embeddings (n_rows × 128), L2-normalized for cosine similarity

---

### Part 7: Export

**Output directory:** `./data/quantumized/`

**Files:**

1. **quantumized_features.parquet** (or .csv)
   - row_id, date columns
   - All extracted features

2. **quantumized_embeddings.npy**
   - Shape: (n_rows, embedding_dim)
   - dtype: float32
   - L2-normalized

3. **quantumized_metadata.csv**
   - row_id, date for embedding lookup

4. **quantumize_config.json**
   - All parameters used
   - Input/output paths
   - Timestamps

---

## Generate Output

**Display:**
```
## Quantumization Complete

**Input:** {{filename}} ({{n_rows}} rows)

**Features Generated:**
- Summary features: 17
- Distribution features: {{dist_count}}
- Total columns: {{total_cols}}

**Embeddings:**
- Shape: ({{n_rows}}, {{embedding_dim}})
- L2-normalized: Yes

**Output Files:**
- ./data/quantumized/quantumized_features.parquet
- ./data/quantumized/quantumized_embeddings.npy
- ./data/quantumized/quantumize_config.json

**Feature Statistics:**
- Entropy range: [{{min_ent}}, {{max_ent}}]
- Purity range: [{{min_pur}}, {{max_pur}}]
- Coherence range: [{{min_coh}}, {{max_coh}}]
```

---

## Update Status

Per `helpers.md#Update-Workflow-Status`:
- Mark quantumization complete
- Record config and output paths
- Store feature statistics

---

## Recommend Next Steps

```
✓ Quantumization complete!

Next steps:
1. /build-faiss-index - Create similarity index from embeddings
2. /validate-quantum-features - Check feature distributions
3. /train-raincoat - Use features for domain adaptation

Quantum features ready for RAINCOAT Specialist.
```

---

## CLI Usage (via quantumize_ca5.py)

```bash
python quantumize_ca5.py \
  --input data/source_domain.csv \
  --output-dir data/quantumized \
  --output-format parquet \
  --taus 0.15,0.50,1.25 \
  --embedding-dim 128 \
  --save-config
```

---

## Helper References

- Load config: `helpers.md#Combined-Config-Load`
- Update status: `helpers.md#Update-Workflow-Status`
- Save document: `helpers.md#Save-Output-Document`
- Reference implementation: `docs-imported/quantumize_ca5.py`

---

## Notes for LLMs

- Use TodoWrite to track 7 quantumization stages
- Reference quantumize_ca5.py for exact implementation
- Verify ||psi|| = 1 after state construction
- Check entropy bounds: 0 ≤ H ≤ log(39) ≈ 3.66 nats
- Embeddings MUST be L2-normalized for cosine similarity
- Use scipy.linalg.expm for matrix exponential (quantum walk)
- Seed ensures reproducibility—same config = same output

**Remember:** Quantum-inspired features impose structured representations—they enable interference patterns without requiring quantum hardware.
