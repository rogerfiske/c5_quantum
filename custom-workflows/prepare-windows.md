You are the Data Pipeline Engineer, executing the **Prepare Windows** workflow.

## Workflow Overview

**Goal:** Slice time series data into overlapping windows for RAINCOAT training

**Phase:** Phase 3 - Solutioning

**Agent:** Data Pipeline Engineer

**Inputs:** Validated dataset (or domain split), window length L, stride

**Output:** Windowed dataset with labels, window_config.json

**Duration:** 5-10 minutes

---

## Pre-Flight

1. Load context per `helpers.md#Combined-Config-Load`
2. Verify dataset has been validated
3. Check if domain split exists (use source_domain if so)
4. Parse window parameters (L, stride)

---

## Prepare Windows Process

Use TodoWrite to track: Load Data → Configure Windows → Slice Data → Generate Labels → Export

---

### Part 1: Load Data

**Priority order:**
1. Use source_domain.parquet if exists (from /split-domains)
2. Use validated dataset otherwise

**Load and verify:**
- Chronological order
- No missing values in QV columns
- Total row count sufficient for window length

---

### Part 2: Configure Windows

**Parameters:**

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| L (window_length) | 128 | 64, 128, 256 | Events per window |
| stride | 1 | 1, L/2, L | Step between windows |
| label_offset | 1 | 1 | Steps ahead for label |

**Calculate:**
```
n_windows = (n_rows - L - label_offset) // stride + 1
```

**Validate:**
- L < n_rows (window fits in data)
- stride > 0
- n_windows > 0

**Output:**
```
Window configuration:
- Length (L): {{L}}
- Stride: {{stride}}
- Label offset: {{label_offset}}
- Total windows: {{n_windows}}
- Data utilization: {{(n_windows * stride + L) / n_rows * 100}}%
```

---

### Part 3: Slice Data

**Window structure:**
```
Window i:
  start_idx = i * stride
  end_idx = start_idx + L
  X[i] = data[start_idx:end_idx]  # Shape: (L, 39)

Label:
  label_idx = end_idx + label_offset - 1
  y[i] = data[label_idx]  # Shape: (39,) binary
```

**Example (L=128, stride=1):**
```
Window 0: rows [0:128], label = row 128
Window 1: rows [1:129], label = row 129
Window 2: rows [2:130], label = row 130
...
```

**Store as:**
- X: numpy array (n_windows, L, 39)
- y: numpy array (n_windows, 39)
- metadata: DataFrame with window_id, start_date, end_date, label_date

---

### Part 4: Generate Labels

**Label format (39-dim binary):**
- 1 if QV_i occurred in the next event
- 0 otherwise
- Exactly 5 ones per label vector

**Verify:**
```python
assert (y.sum(axis=1) == 5).all(), "All labels must have exactly 5 ones"
```

**Optional: Class weights**
Calculate class imbalance weights for training:
```python
pos_freq = y.mean(axis=0)  # Per-QV occurrence frequency
weights = 1.0 / (pos_freq + eps)
weights = weights / weights.sum() * 39  # Normalize
```

---

### Part 5: Export

**Output directory:** `./data/windows/`

**Files:**

1. **windows_X.npy** - Window features (n_windows, L, 39)
2. **windows_y.npy** - Labels (n_windows, 39)
3. **windows_metadata.csv** - Window metadata
4. **window_config.json**:
```json
{
  "window_length": 128,
  "stride": 1,
  "label_offset": 1,
  "n_windows": 11562,
  "n_features": 39,
  "source_file": "source_domain.parquet",
  "date_range": ["1992-02-24", "2024-11-10"],
  "class_weights": [1.02, 0.98, ...],
  "created_at": "2026-01-29T12:00:00Z"
}
```

---

## Generate Output

**Display:**
```
## Windows Prepared

**Configuration:**
- Window length: {{L}}
- Stride: {{stride}}
- Windows created: {{n_windows}}

**Shapes:**
- X: ({{n_windows}}, {{L}}, 39)
- y: ({{n_windows}}, 39)

**Class Distribution:**
- Most frequent QV: QV_{{max_qv}} ({{max_freq}}%)
- Least frequent QV: QV_{{min_qv}} ({{min_freq}}%)

**Files:**
- ./data/windows/windows_X.npy
- ./data/windows/windows_y.npy
- ./data/windows/windows_metadata.csv
- ./data/windows/window_config.json
```

---

## Update Status

Per `helpers.md#Update-Workflow-Status`:
- Mark window preparation complete
- Record window configuration
- Store output paths

---

## Recommend Next Steps

```
✓ Windows prepared!

Next steps:
1. /quantumize - Apply quantum features to windows (optional)
2. /train-raincoat - Train time-frequency encoder on windows

Data ready for RAINCOAT Specialist.
```

---

## Memory Optimization

For large datasets, support chunked processing:

```
/prepare-windows --chunk-size 1000

Processes windows in chunks to avoid memory issues.
Outputs: windows_X_chunk_001.npy, windows_X_chunk_002.npy, etc.
```

---

## Helper References

- Load config: `helpers.md#Combined-Config-Load`
- Update status: `helpers.md#Update-Workflow-Status`
- Save document: `helpers.md#Save-Output-Document`

---

## Notes for LLMs

- Use TodoWrite to track 5 preparation steps
- Verify label constraint (exactly 5 ones) on every label
- stride=1 maximizes training data but increases storage
- Class weights help with imbalanced QV frequencies
- Memory: (11000 windows × 128 length × 39 features × 4 bytes) ≈ 220MB
- Reference RAINCOAT_C5_Concept_Brief.md for window size recommendations

**Remember:** Windows must maintain strict temporal order—the label always comes AFTER the window, never overlapping.
