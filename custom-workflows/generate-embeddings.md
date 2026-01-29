You are the Quantum ML Engineer, executing the **Generate Embeddings** workflow.

## Workflow Overview

**Goal:** Create dense vector embeddings from quantum features for similarity search

**Phase:** Phase 4 - Implementation

**Agent:** Quantum ML Engineer

**Inputs:** Quantumized features (or raw count data), embedding_dim

**Output:** quantumized_embeddings.npy, embedding_metadata.csv

**Duration:** 2-5 minutes

---

## Pre-Flight

1. Load context per `helpers.md#Combined-Config-Load`
2. Check if quantumized features exist (prefer these)
3. If not, check for validated dataset (will quantumize on-the-fly)
4. Load embedding configuration

---

## Generate Embeddings Process

Use TodoWrite to track: Load Features → Configure → Project → Normalize → Export

---

### Part 1: Load Features

**Option A: Use existing quantumized features**
```
Load: ./data/quantumized/quantumized_features.parquet
Extract: p_*, qft_*, qwalk_* columns
```

**Option B: Generate features on-the-fly**
```
Load: validated dataset
Run: /quantumize (abbreviated, distributions only)
Extract: p, q_fft, q_walk distributions
```

**Feature matrix F:**
- Shape: (n_rows, n_features)
- Typical: (11691, 117) for p + q_fft + q_walk_tau0

---

### Part 2: Configure Projection

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| embedding_dim | 128 | Output dimension |
| seed | 7 | RNG seed for projection matrix |
| method | random_projection | Projection method |

**Alternative methods (future):**
- PCA: Data-dependent, requires fitting
- Autoencoder: Learned, requires training
- Random projection: Fast, reproducible, seed-controlled

---

### Part 3: Random Projection

**Create projection matrix:**
```python
rng = np.random.default_rng(seed)
P = rng.normal(loc=0.0, scale=1.0, size=(n_features, embedding_dim))
P = P / np.sqrt(embedding_dim)  # Scale for variance preservation
```

**Project features:**
```python
E = F.astype(np.float32) @ P.astype(np.float32)
```

**Properties:**
- Johnson-Lindenstrauss: Distances approximately preserved
- Reproducible: Same seed = same projection
- Fast: Single matrix multiply

---

### Part 4: L2 Normalize

**Normalize for cosine similarity:**
```python
norms = np.linalg.norm(E, axis=1, keepdims=True)
norms = np.where(norms <= 0, 1.0, norms)  # Avoid division by zero
E_normalized = E / norms
```

**Verify:**
```python
assert np.allclose(np.linalg.norm(E_normalized, axis=1), 1.0)
```

**Why normalize:**
- Cosine similarity = dot product for unit vectors
- FAISS IndexFlatIP uses inner product
- Consistent similarity scale [−1, 1]

---

### Part 5: Export

**Output directory:** `./data/quantumized/`

**Files:**

1. **quantumized_embeddings.npy**
```python
np.save(path, E_normalized.astype(np.float32))
```
- Shape: (n_rows, embedding_dim)
- dtype: float32
- L2-normalized

2. **quantumized_embedding_metadata.csv**
```csv
row_id,date
0,1992-02-24
1,1992-02-25
...
```

3. **embedding_config.json**
```json
{
  "embedding_dim": 128,
  "seed": 7,
  "method": "random_projection",
  "n_rows": 11691,
  "n_input_features": 117,
  "l2_normalized": true,
  "created_at": "2026-01-29T12:00:00Z"
}
```

---

## Generate Output

**Display:**
```
## Embeddings Generated

**Configuration:**
- Input features: {{n_features}}
- Output dimension: {{embedding_dim}}
- Method: random_projection (seed={{seed}})

**Output:**
- Shape: ({{n_rows}}, {{embedding_dim}})
- Normalized: L2 (unit vectors)
- File: ./data/quantumized/quantumized_embeddings.npy

**Memory:**
- Size: {{n_rows * embedding_dim * 4 / 1024 / 1024:.1f}} MB

**Sample similarities (first 5 rows):**
| Row | Most Similar | Similarity |
|-----|--------------|------------|
{{for i in range(5)}}
| {{i}} | {{nearest[i]}} | {{sim[i]:.4f}} |
{{endfor}}
```

---

## Update Status

Per `helpers.md#Update-Workflow-Status`:
- Mark embedding generation complete
- Record configuration
- Store output paths

---

## Recommend Next Steps

```
✓ Embeddings generated!

Next steps:
1. /build-faiss-index - Create searchable index
2. /query-similar - Find similar events
3. /train-raincoat - Use embeddings for regime detection

Embeddings ready for similarity search.
```

---

## Usage Examples

**Load embeddings in Python:**
```python
import numpy as np
import pandas as pd

embeddings = np.load('./data/quantumized/quantumized_embeddings.npy')
metadata = pd.read_csv('./data/quantumized/quantumized_embedding_metadata.csv')

# Find similar events to row 100
query = embeddings[100:101]
similarities = embeddings @ query.T
top_k = np.argsort(similarities.flatten())[-10:][::-1]
```

**Query with FAISS (after /build-faiss-index):**
```python
import faiss

index = faiss.read_index('./data/quantumized/quantumized_faiss.index')
D, I = index.search(query, k=10)  # D=distances, I=indices
```

---

## Helper References

- Load config: `helpers.md#Combined-Config-Load`
- Update status: `helpers.md#Update-Workflow-Status`
- Quantumize workflow: `custom-workflows/quantumize.md`

---

## Notes for LLMs

- Use TodoWrite to track 5 embedding steps
- Random projection preserves distances (Johnson-Lindenstrauss lemma)
- L2 normalization is REQUIRED for cosine similarity
- Same seed guarantees identical embeddings
- float32 is sufficient precision, saves memory
- Metadata CSV enables date-based queries

**Remember:** Embeddings compress quantum features into a dense space optimized for similarity search—the seed ensures reproducibility.
