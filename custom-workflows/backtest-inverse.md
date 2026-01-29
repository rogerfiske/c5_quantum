You are the RAINCOAT Specialist, executing the **Backtest Inverse** workflow.

## Workflow Overview

**Goal:** Validate inverse prediction model with walk-forward backtesting

**Phase:** Phase 4 - Implementation

**Agent:** RAINCOAT Specialist

**Inputs:** Trained model, full dataset, backtest config

**Output:** Backtest results, FP@20 distribution, performance report

**Duration:** 30-120 minutes (depends on backtest range)

---

## Pre-Flight

1. Load context per `helpers.md#Combined-Config-Load`
2. Load trained RAINCOAT model
3. Load full dataset (chronologically sorted)
4. Configure backtest parameters (range, step size)

---

## Backtest Inverse Process

Use TodoWrite to track: Configure → Initialize → Walk-Forward Loop → Aggregate Metrics → Generate Report

---

### Part 1: Configure Backtest

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| start_idx | L + N_tail | First prediction index |
| end_idx | len(data) - 1 | Last prediction index |
| step | 1 | Prediction interval |
| retrain_freq | 0 | Retrain every N steps (0=no retrain) |
| adapt_freq | 39 | Re-adapt every N steps |

**Example configuration:**
```json
{
  "backtest_range": {
    "start": "2020-01-01",
    "end": "2025-01-27"
  },
  "step": 1,
  "retrain_freq": 0,
  "adapt_freq": 39,
  "n_tail": 78,
  "window_length": 128
}
```

**Calculate:**
```python
n_predictions = (end_idx - start_idx) // step + 1
print(f"Backtest will make {n_predictions} predictions")
```

---

### Part 2: Initialize

**Data structures:**
```python
results = []
fp_counts = []  # False positives per prediction
actual_qvs = []  # Actual QVs for each event
predicted_exclusions = []  # Our exclusion sets
```

**Model state:**
```python
encoder.eval()
classifier.eval()

# Track adaptation history
last_adapt_idx = 0
```

---

### Part 3: Walk-Forward Loop

**For each prediction step:**

```python
for pred_idx in range(start_idx, end_idx + 1, step):
    # 1. Define domains at this point in time
    current_history = data[:pred_idx]
    source_end = pred_idx - n_tail
    target_start = source_end
    target_end = pred_idx

    # 2. Re-adapt if needed
    if (pred_idx - last_adapt_idx) >= adapt_freq:
        source_data = data[max(0, source_end - source_size):source_end]
        target_data = data[target_start:target_end]

        # Quick adaptation (Stage B + C, fewer epochs)
        quick_adapt(encoder, source_data, target_data)
        last_adapt_idx = pred_idx

    # 3. Prepare input window
    window = data[pred_idx - L:pred_idx]
    X = torch.tensor(window).unsqueeze(0)

    # 4. Get predictions
    with torch.no_grad():
        latent = encoder(X)
        p_occur = classifier(latent).squeeze()

    # 5. Compute exclusion scores
    s_fused = compute_fusion_scores(p_occur, target_data, ...)

    # 6. Get exclusion set
    exclusion_set = get_top_k_exclusions(s_fused, k=20)

    # 7. Get actual result
    actual = data[pred_idx]  # Next event's QVs
    actual_qv_indices = np.where(actual == 1)[0]

    # 8. Compute FP@20
    fp = len(set(actual_qv_indices) & set(exclusion_set))

    # 9. Store results
    results.append({
        'pred_idx': pred_idx,
        'date': dates[pred_idx],
        'exclusion_set': exclusion_set.tolist(),
        'actual_qvs': actual_qv_indices.tolist(),
        'fp_at_20': fp,
        'scores': s_fused.tolist(),
    })
    fp_counts.append(fp)
```

**Progress tracking:**
```
Backtest progress: [####------] 40% (400/1000)
Current FP@20: mean=0.42, last_10_mean=0.30
```

---

### Part 4: Aggregate Metrics

**Primary Metrics:**

| Metric | Formula | Target |
|--------|---------|--------|
| FP@20 | Mean false positives | 0 (ideal) |
| Perfect Rate | % predictions with FP=0 | 100% (ideal) |
| Worst Case | Max FP in any prediction | < 3 |
| Hit@5 | % where all 5 actuals in exclusions | 0% (ideal) |

**Compute:**
```python
fp_array = np.array(fp_counts)

metrics = {
    'fp_at_20_mean': fp_array.mean(),
    'fp_at_20_std': fp_array.std(),
    'fp_at_20_median': np.median(fp_array),
    'perfect_rate': (fp_array == 0).mean() * 100,
    'one_fp_rate': (fp_array == 1).mean() * 100,
    'two_plus_fp_rate': (fp_array >= 2).mean() * 100,
    'worst_case_fp': fp_array.max(),
    'hit_at_5_rate': (fp_array == 5).mean() * 100,
}
```

**FP Distribution:**
```python
fp_distribution = {
    'fp_0': (fp_array == 0).sum(),
    'fp_1': (fp_array == 1).sum(),
    'fp_2': (fp_array == 2).sum(),
    'fp_3': (fp_array == 3).sum(),
    'fp_4': (fp_array == 4).sum(),
    'fp_5': (fp_array == 5).sum(),
}
```

**Coverage Gap Analysis:**
```python
gaps = [r['scores'][r['exclusion_set'][19]] - r['scores'][r['exclusion_set'][20]]
        for r in results if len(r['scores']) > 20]
metrics['mean_gap'] = np.mean(gaps)
metrics['min_gap'] = np.min(gaps)
```

**Middle-Outcome Suppression:**
```python
# Per project doctrine: prefer extremes (0 or 5 wrong) over middle (2-3 wrong)
middle_outcomes = ((fp_array >= 2) & (fp_array <= 3)).sum()
extreme_outcomes = ((fp_array == 0) | (fp_array >= 4)).sum()
metrics['middle_suppression_ratio'] = extreme_outcomes / (middle_outcomes + 1)
```

---

### Part 5: Generate Report

**Backtest Report:**

```markdown
## Backtest Results

**Period:** {{start_date}} to {{end_date}}
**Predictions:** {{n_predictions}}
**Adaptation frequency:** Every {{adapt_freq}} events

### Primary Metrics

| Metric | Value | Target |
|--------|-------|--------|
| FP@20 Mean | {{fp_mean:.3f}} | 0.0 |
| FP@20 Std | {{fp_std:.3f}} | - |
| Perfect Rate (FP=0) | {{perfect_rate:.1f}}% | 100% |
| Worst Case | {{worst_case}} | <3 |
| Hit@5 Rate | {{hit_5:.2f}}% | 0% |

### FP Distribution

| FP Count | Predictions | Percentage |
|----------|-------------|------------|
| 0 (Perfect) | {{fp_0}} | {{fp_0_pct:.1f}}% |
| 1 | {{fp_1}} | {{fp_1_pct:.1f}}% |
| 2 | {{fp_2}} | {{fp_2_pct:.1f}}% |
| 3 | {{fp_3}} | {{fp_3_pct:.1f}}% |
| 4 | {{fp_4}} | {{fp_4_pct:.1f}}% |
| 5 (Worst) | {{fp_5}} | {{fp_5_pct:.1f}}% |

### Histogram

```
FP=0: ████████████████████ (45%)
FP=1: ██████████████ (30%)
FP=2: ████████ (15%)
FP=3: ███ (6%)
FP=4: █ (3%)
FP=5: ▏ (1%)
```

### Time Series Analysis

**Rolling FP@20 (50-event window):**
- Best period: {{best_period}} (mean FP: {{best_fp:.2f}})
- Worst period: {{worst_period}} (mean FP: {{worst_fp:.2f}})

**Regime Detection:**
- Identified {{n_regimes}} regime shifts
- Performance varies by regime

### Confidence Analysis

**Coverage Gap at Cutoff:**
- Mean gap: {{mean_gap:.3f}}
- Min gap: {{min_gap:.3f}} (least confident)
- Predictions with gap < 0.05: {{low_gap_count}} ({{low_gap_pct:.1f}}%)

### Comparison to Baseline

| Method | FP@20 Mean | Perfect Rate |
|--------|------------|--------------|
| RAINCOAT (this) | {{raincoat_fp:.3f}} | {{raincoat_perfect:.1f}}% |
| Random exclusion | 2.56 | 0.8% |
| Frequency-only | {{freq_fp:.3f}} | {{freq_perfect:.1f}}% |
```

---

## Output Files

**Save results:**
```python
# Detailed results
with open(f'./backtests/backtest_{date}.json', 'w') as f:
    json.dump({
        'config': config,
        'metrics': metrics,
        'fp_distribution': fp_distribution,
        'results': results,  # All predictions
    }, f, indent=2)

# Summary report
with open(f'./backtests/backtest_report_{date}.md', 'w') as f:
    f.write(report_markdown)

# Rolling metrics CSV
pd.DataFrame([{
    'date': r['date'],
    'fp': r['fp_at_20'],
    'gap': r['gap'],
} for r in results]).to_csv(f'./backtests/rolling_fp_{date}.csv', index=False)
```

---

## Update Status

Per `helpers.md#Update-Workflow-Status`:
- Mark backtest complete
- Record key metrics
- Store output paths

---

## Recommend Next Steps

```
✓ Backtest complete!

**Summary:** FP@20 = {{fp_mean:.3f}}, Perfect Rate = {{perfect_rate:.1f}}%

Next steps based on results:

{{if perfect_rate > 50}}
✓ Good performance! Ready for production.
- Continue with /score-exclusions for live predictions
{{elif perfect_rate > 30}}
△ Moderate performance. Consider:
- /ablation-report - Identify weak components
- Adjust fusion weights
- Increase adaptation frequency
{{else}}
✗ Needs improvement:
- Review training data quality
- Try different hyperparameters
- Consider longer adaptation windows
{{endif}}
```

---

## Baseline Comparison

**Random baseline:**
```python
# Random exclusion: pick 20 random QVs
# Expected FP = 20/39 * 5 ≈ 2.56
random_fp = 20 * 5 / 39
```

**Frequency baseline:**
```python
# Exclude 20 least frequent QVs historically
historical_freq = data[:source_end].mean(axis=0)
least_freq_20 = np.argsort(historical_freq)[:20]
```

---

## Helper References

- Load config: `helpers.md#Combined-Config-Load`
- Update status: `helpers.md#Update-Workflow-Status`
- Scoring workflow: `custom-workflows/score-exclusions.md`

---

## Notes for LLMs

- Use TodoWrite to track 5 backtest stages
- STRICT chronological discipline—no look-ahead ever
- Quick adaptation is faster than full retraining
- FP@20 = 0 is perfect; random baseline ≈ 2.56
- Middle-outcome suppression is a project-specific metric
- Store all predictions for post-hoc analysis

**Remember:** Walk-forward backtesting is the gold standard—it simulates real-world deployment where you only know the past.
