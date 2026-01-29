# C5 Quantum Predictions

This directory stores daily inverse predictions for future analysis and accuracy testing.

## Directory Structure

```
predictions/
├── README.md
└── daily/
    └── prediction_YYYY-MM-DD.json
```

## Prediction File Format

Each prediction file contains:

| Field | Description |
|-------|-------------|
| `prediction_date` | Date the prediction is for |
| `prediction_made_at` | Timestamp when prediction was generated |
| `model_version` | Model checkpoint used |
| `excluded_qvs` | 20 QVs predicted NOT to occur |
| `remaining_qvs` | 19 candidate QVs (winners should be here) |
| `exclusion_ranking` | Full ranking with scores |
| `confidence_metrics` | Prediction confidence indicators |
| `validation` | Filled in after actual results are known |

## Validation Process

After the actual event results are known:

1. Update `validation.actual_qvs` with the 5 winning QVs
2. Calculate `validation.false_positives` (actual QVs that were excluded)
3. Set `validation.fp_count` (0-5, lower is better)
4. Add `validation.validated_at` timestamp
5. Set `validation.result`: "perfect" (0), "good" (1), "fair" (2), "poor" (3+)

## Metrics

- **FP@20 = 0**: Perfect prediction (none of 5 winners in exclusion set)
- **FP@20 = 1**: Good prediction (1 winner incorrectly excluded)
- **FP@20 = 2-3**: Fair prediction
- **FP@20 = 4-5**: Poor prediction

## Historical Performance

Based on backtest (1,120 predictions):
- Mean FP@20: 2.28 (vs 2.56 random baseline)
- Perfect rate: 4.1%
- Improvement: 11% over random
