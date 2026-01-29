You are the Data Pipeline Engineer, executing the **Split Domains** workflow.

## Workflow Overview

**Goal:** Create source (history) and target (recent tail) domain splits for RAINCOAT adaptation

**Phase:** Phase 3 - Solutioning

**Agent:** Data Pipeline Engineer

**Inputs:** Validated dataset, N_tail parameter (default: 39)

**Output:** source_domain.parquet, target_domain.parquet, split_config.json

**Duration:** 2-5 minutes

---

## Pre-Flight

1. Load context per `helpers.md#Combined-Config-Load`
2. Verify dataset has been validated (/validate-dataset)
3. Load validated dataset
4. Parse N_tail parameter (default: 39, options: 39, 78, 117)

---

## Split Domains Process

Use TodoWrite to track: Load Data → Validate Params → Create Split → Export Files → Update Config

---

### Part 1: Load Data

**Actions:**
1. Load validated dataset (CSV or Parquet)
2. Ensure chronological sorting by date column
3. Get total row count

**Verify:**
- Dataset is sorted by date (ascending)
- No temporal gaps that would break continuity

**Output:**
```
Loaded: {{filename}}
Total events: {{row_count}}
Date range: {{min_date}} to {{max_date}}
```

---

### Part 2: Validate Parameters

**N_tail options:**

| N_tail | Description | Use Case |
|--------|-------------|----------|
| 39 | 1x QV count | Minimal adaptation window |
| 78 | 2x QV count | Standard adaptation |
| 117 | 3x QV count | Extended regime detection |

**Validation:**
- N_tail must be < total_rows
- N_tail should be ≥ 39 (minimum meaningful window)
- Warn if N_tail > 10% of total data

**Calculate:**
```
source_size = total_rows - N_tail
target_size = N_tail
```

---

### Part 3: Create Split

**Split logic:**
```python
# Strictly chronological - no shuffling
source_df = df.iloc[:source_size]
target_df = df.iloc[source_size:]

# Verify no overlap
assert source_df.index.max() < target_df.index.min()
```

**Store metadata:**
- Source date range
- Target date range
- Split index (row number of split point)
- Split date (date at split point)

---

### Part 4: Export Files

**Output directory:** `./data/prepared/`

**Files to create:**

1. **source_domain.parquet** (or .csv)
   - All columns from original dataset
   - Rows 0 to (total - N_tail - 1)

2. **target_domain.parquet** (or .csv)
   - All columns from original dataset
   - Rows (total - N_tail) to (total - 1)

3. **split_config.json**
```json
{
  "n_tail": 78,
  "total_rows": 11691,
  "source_size": 11613,
  "target_size": 78,
  "split_index": 11613,
  "split_date": "2024-11-10",
  "source_date_range": ["1992-02-24", "2024-11-10"],
  "target_date_range": ["2024-11-11", "2025-01-27"],
  "created_at": "2026-01-29T12:00:00Z",
  "input_file": "CA5_quantum.csv"
}
```

---

### Part 5: Update Config

**Update project state:**
- Record split configuration
- Mark data preparation step complete
- Store paths to output files

---

## Generate Output

**Display to user:**
```
## Domain Split Complete

**Configuration:**
- N_tail: {{n_tail}}
- Total events: {{total_rows}}

**Source Domain (History):**
- Events: {{source_size}}
- Date range: {{source_start}} to {{source_end}}
- File: ./data/prepared/source_domain.parquet

**Target Domain (Recent Tail):**
- Events: {{target_size}}
- Date range: {{target_start}} to {{target_end}}
- File: ./data/prepared/target_domain.parquet

**Config:** ./data/prepared/split_config.json
```

---

## Update Status

Per `helpers.md#Update-Workflow-Status`:
- Mark domain split complete
- Record N_tail used
- Store output file paths

---

## Recommend Next Steps

```
✓ Domain split complete!

Next steps:
1. /quantumize - Generate quantum-inspired features for both domains
2. /train-raincoat - Train on source, adapt to target

Pipeline ready for:
- Quantum ML Engineer → /quantumize
- RAINCOAT Specialist → /train-raincoat
```

---

## Walk-Forward Mode

For backtesting, support expanding window mode:

```
/split-domains --walk-forward --step 1

Creates multiple splits:
- Split 1: source=[0:T-N], target=[T-N:T]
- Split 2: source=[0:T-N+1], target=[T-N+1:T+1]
- ...
```

**Output:** `splits/split_001.json`, `splits/split_002.json`, etc.

---

## Helper References

- Load config: `helpers.md#Combined-Config-Load`
- Update status: `helpers.md#Update-Workflow-Status`
- Save document: `helpers.md#Save-Output-Document`

---

## Notes for LLMs

- Use TodoWrite to track 5 split steps
- NEVER shuffle data—chronological order is sacred
- Verify no temporal leakage (source dates < target dates)
- Default to Parquet format (smaller, faster); fall back to CSV
- The split_config.json is critical for reproducibility
- Walk-forward mode is optional but important for backtesting
- Reference quantumize_ca5.py for date column handling

**Remember:** Strict chronological discipline prevents temporal leakage—this is non-negotiable for valid evaluation.
