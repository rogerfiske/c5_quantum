You are the Quantum ML Engineer, executing the **Build FAISS Index** workflow.

## Workflow Overview

**Goal:** Create a FAISS index for fast similarity search on quantum embeddings

**Phase:** Phase 4 - Implementation

**Agent:** Quantum ML Engineer

**Inputs:** quantumized_embeddings.npy (L2-normalized)

**Output:** quantumized_faiss.index

**Duration:** 1-3 minutes

---

## Pre-Flight

1. Load context per `helpers.md#Combined-Config-Load`
2. Verify embeddings exist (`quantumized_embeddings.npy`)
3. Check FAISS is installed (`pip install faiss-cpu`)
4. Load embedding metadata

---

## Build FAISS Index Process

Use TodoWrite to track: Load Embeddings → Select Index Type → Build Index → Verify → Export

---

### Part 1: Load Embeddings

**Load:**
```python
import numpy as np

embeddings = np.load('./data/quantumized/quantumized_embeddings.npy')
print(f"Shape: {embeddings.shape}")  # (n_rows, embedding_dim)
print(f"dtype: {embeddings.dtype}")  # float32
```

**Verify L2 normalization:**
```python
norms = np.linalg.norm(embeddings, axis=1)
assert np.allclose(norms, 1.0), "Embeddings must be L2-normalized"
```

---

### Part 2: Select Index Type

**For cosine similarity with normalized vectors:**

| Index Type | Use Case | Speed | Memory |
|------------|----------|-------|--------|
| IndexFlatIP | Exact search, < 100k vectors | O(n) | Low |
| IndexIVFFlat | Approximate, 100k-1M vectors | O(√n) | Medium |
| IndexHNSW | Approximate, any size | O(log n) | High |

**Default: IndexFlatIP**
- Inner product = cosine similarity for unit vectors
- Exact results (no approximation)
- Suitable for C5 dataset (~12k vectors)

---

### Part 3: Build Index

**IndexFlatIP (exact):**
```python
import faiss

dim = embeddings.shape[1]  # 128
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

print(f"Index size: {index.ntotal} vectors")
```

**IndexIVFFlat (approximate, for larger datasets):**
```python
nlist = 100  # Number of clusters
quantizer = faiss.IndexFlatIP(dim)
index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
index.train(embeddings)
index.add(embeddings)
index.nprobe = 10  # Search 10 clusters (tradeoff: speed vs accuracy)
```

---

### Part 4: Verify Index

**Test query:**
```python
# Query with first 5 vectors
query = embeddings[:5]
k = 10  # Top-10 neighbors

D, I = index.search(query, k)
# D: distances (similarities for IP)
# I: indices of neighbors

# First result should be the query itself (similarity ~1.0)
for i in range(5):
    print(f"Query {i}: top match = {I[i,0]}, similarity = {D[i,0]:.4f}")
    assert I[i, 0] == i, "Self should be top match"
    assert D[i, 0] > 0.99, "Self-similarity should be ~1.0"
```

**Sample output:**
```
Query 0: top match = 0, similarity = 1.0000
Query 1: top match = 1, similarity = 1.0000
...
```

---

### Part 5: Export

**Save index:**
```python
faiss.write_index(index, './data/quantumized/quantumized_faiss.index')
```

**Index metadata (faiss_config.json):**
```json
{
  "index_type": "IndexFlatIP",
  "dimension": 128,
  "ntotal": 11691,
  "metric": "inner_product",
  "normalized": true,
  "created_at": "2026-01-29T12:00:00Z"
}
```

---

## Generate Output

**Display:**
```
## FAISS Index Built

**Configuration:**
- Index type: IndexFlatIP
- Dimension: {{dim}}
- Vectors: {{ntotal}}
- Metric: Inner Product (cosine for normalized vectors)

**Verification:**
- Self-similarity test: PASSED
- Sample query time: {{query_time_ms:.2f}} ms

**Output:**
- ./data/quantumized/quantumized_faiss.index
- ./data/quantumized/faiss_config.json

**Usage:**
```python
import faiss
index = faiss.read_index('./data/quantumized/quantumized_faiss.index')
D, I = index.search(query_vectors, k=10)
```
```

---

## Update Status

Per `helpers.md#Update-Workflow-Status`:
- Mark FAISS index built
- Record index configuration
- Store output path

---

## Recommend Next Steps

```
✓ FAISS index built!

Next steps:
1. /query-similar - Find similar events by date or index
2. /train-raincoat - Use similarity for regime detection

Index ready for similarity queries.
```

---

## Query Examples

**Find similar events:**
```python
import faiss
import numpy as np
import pandas as pd

# Load
index = faiss.read_index('./data/quantumized/quantumized_faiss.index')
embeddings = np.load('./data/quantumized/quantumized_embeddings.npy')
metadata = pd.read_csv('./data/quantumized/quantumized_embedding_metadata.csv')

# Query: events similar to the most recent
query_idx = -1  # Last event
query = embeddings[query_idx:query_idx+1]

D, I = index.search(query, k=10)

print("Events most similar to latest:")
for rank, (idx, sim) in enumerate(zip(I[0], D[0])):
    date = metadata.iloc[idx]['date']
    print(f"  {rank+1}. Event {idx} ({date}): {sim:.4f}")
```

**Find regime clusters:**
```python
# Get all pairwise similarities (expensive for large n)
all_sims = embeddings @ embeddings.T

# Or use FAISS for k-NN graph
k = 20
D, I = index.search(embeddings, k)
# I is now a k-NN graph: I[i] = indices of k nearest neighbors of i
```

---

## Helper References

- Load config: `helpers.md#Combined-Config-Load`
- Update status: `helpers.md#Update-Workflow-Status`
- Embeddings workflow: `custom-workflows/generate-embeddings.md`

---

## Notes for LLMs

- Use TodoWrite to track 5 index building steps
- IndexFlatIP is sufficient for ~12k vectors
- Embeddings MUST be L2-normalized for cosine similarity
- Self-similarity test validates normalization
- faiss-cpu is pure Python; faiss-gpu requires CUDA
- Index file is portable across machines

**Remember:** FAISS enables millisecond similarity queries on thousands of vectors—essential for retrieval-augmented prediction.
