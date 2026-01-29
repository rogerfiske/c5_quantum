You are the RAINCOAT Specialist, executing the **Score Exclusions** workflow.

## Workflow Overview

**Goal:** Generate fused inverse prediction scores to identify 20 least likely QVs

**Phase:** Phase 4 - Implementation

**Agent:** RAINCOAT Specialist

**Inputs:** Trained RAINCOAT model, target domain data, scoring config

**Output:** exclusion_scores.json, ranked QV exclusions

**Duration:** 2-5 minutes

---

## Pre-Flight

1. Load context per `helpers.md#Combined-Config-Load`
2. Load trained model checkpoint
3. Load target domain data (most recent window)
4. Load scoring configuration (fusion weights)

---

## Score Exclusions Process

Use TodoWrite to track: Load Model → Prepare Input → Get Predictions → Compute Fusion Scores → Apply Risk Controls → Output Rankings

---

### Part 1: Load Model

**Load checkpoint:**
```python
checkpoint = torch.load('./models/c5_raincoat_v{date}.ckpt')
encoder.load_state_dict(checkpoint['encoder_state'])
classifier.load_state_dict(checkpoint['classifier_state'])
encoder.eval()
classifier.eval()
```

**Verify model:**
- Check architecture matches config
- Verify checkpoint includes all components
- Note training metrics (AUC, OT divergence)

---

### Part 2: Prepare Input

**For next-event prediction:**
```python
# Get most recent L events as input window
latest_window = data[-L:]  # Shape: (L, 39)
X = torch.tensor(latest_window).unsqueeze(0)  # Shape: (1, L, 39)
```

**Context features (optional):**
- Recent QV frequencies (last N events)
- Source vs target spectral differences
- OT transport residuals

---

### Part 3: Get Predictions

**Forward pass:**
```python
with torch.no_grad():
    latent = encoder(X)
    p_occur = classifier(latent).squeeze()  # Shape: (39,)
```

**p_occur[i]** = probability that QV_i occurs in next event

**Verify:**
- All probabilities in [0, 1]
- No NaN or Inf values

---

### Part 4: Compute Fusion Scores

**Base non-occurrence score:**
```python
s_base = 1.0 - p_occur  # Higher = less likely to occur
```

**Fusion components:**

| Component | Weight | Description |
|-----------|--------|-------------|
| s_base | w1 (0.5) | Model non-occurrence probability |
| s_rarity | w2 (0.2) | Target domain rarity prior |
| s_spectral | w3 (0.15) | Spectral mismatch score |
| s_ot | w4 (0.15) | OT transport residual |

**1. Target Rarity Prior:**
```python
# Empirical non-occurrence in target domain
target_freq = target_data.mean(axis=0)  # Per-QV frequency in target
s_rarity = 1.0 - target_freq
```

**2. Spectral Mismatch:**
```python
# Amplitude/phase distance between source and target
source_fft = np.fft.fft(source_data.mean(axis=0))
target_fft = np.fft.fft(target_data.mean(axis=0))

amp_diff = np.abs(np.abs(target_fft) - np.abs(source_fft))
phase_diff = np.abs(np.angle(target_fft) - np.angle(source_fft))
s_spectral = (amp_diff + phase_diff) / 2
s_spectral = s_spectral / s_spectral.max()  # Normalize to [0, 1]
```

**3. OT Transport Residual:**
```python
# Per-channel transport cost from alignment
# (Stored during training or computed post-hoc)
s_ot = ot_residuals / ot_residuals.max()  # Normalize
```

**Fused score:**
```python
s_fused = w1 * s_base + w2 * s_rarity + w3 * s_spectral + w4 * s_ot
```

---

### Part 5: Apply Risk Controls

**5.1 Diversity Filter:**
```python
# Discourage adjacent QVs if they historically co-occur
adjacency_penalty = compute_adjacency_penalty(exclusion_set, cooccurrence_matrix)
# Apply penalty to scores of adjacent QVs already in set
```

**5.2 Prototype Gap Check:**
```python
# If margin between 20th and 21st is small, prefer safer exclusions
ranked = np.argsort(s_fused)[::-1]  # Descending
gap = s_fused[ranked[19]] - s_fused[ranked[20]]

if gap < threshold:
    # Replace marginal exclusions with those having higher spectral/OT
    for i in range(19, -1, -1):
        if s_fused[ranked[i]] - s_fused[ranked[20]] < margin:
            # Find safer alternative
            alt = find_safer_alternative(ranked[i], s_spectral, s_ot)
            if alt is not None:
                ranked[i] = alt
```

**5.3 Bimodality Check:**
```python
# If score distribution is bimodal, set threshold between modes
from scipy.stats import gaussian_kde

kde = gaussian_kde(s_fused)
# Find local minimum between modes
threshold = find_bimodal_threshold(kde, s_fused)

if threshold is not None:
    n_exclude = (s_fused > threshold).sum()
    # Adjust exclusion count based on natural threshold
```

---

### Part 6: Output Rankings

**Final exclusion set:**
```python
ranked_indices = np.argsort(s_fused)[::-1]  # Descending by exclusion score
exclusion_set = ranked_indices[:20]  # Top 20 least likely

# Map to QV numbers (1-indexed)
excluded_qvs = [i + 1 for i in exclusion_set]
```

**Output JSON:**
```json
{
  "prediction_date": "2026-01-30",
  "input_window_end": "2026-01-29",
  "excluded_qvs": [3, 7, 12, 15, 18, ...],
  "scores": {
    "QV_1": 0.42,
    "QV_2": 0.38,
    ...
  },
  "fusion_weights": {
    "base": 0.5,
    "rarity": 0.2,
    "spectral": 0.15,
    "ot": 0.15
  },
  "risk_controls": {
    "diversity_applied": true,
    "gap_check_applied": false,
    "bimodal_threshold": null
  },
  "confidence": {
    "mean_exclusion_score": 0.72,
    "score_gap_at_20": 0.08,
    "bimodal": false
  }
}
```

---

## Generate Output

**Display:**
```
## Exclusion Scores Generated

**Prediction for:** {{next_date}}
**Input window:** {{window_start}} to {{window_end}}

**Top 20 Exclusions (Least Likely QVs):**

| Rank | QV | Fused Score | Base | Rarity | Spectral | OT |
|------|-----|-------------|------|--------|----------|-----|
{{for rank, qv in enumerate(excluded_qvs[:20])}}
| {{rank+1}} | QV_{{qv}} | {{s_fused[qv-1]:.3f}} | {{s_base[qv-1]:.3f}} | {{s_rarity[qv-1]:.3f}} | {{s_spectral[qv-1]:.3f}} | {{s_ot[qv-1]:.3f}} |
{{endfor}}

**Confidence Metrics:**
- Mean exclusion score: {{mean_score:.3f}}
- Score gap at cutoff: {{gap:.3f}}
- Distribution: {{unimodal|bimodal}}

**Risk Controls Applied:**
{{list applied controls}}

**Output:** ./predictions/exclusion_scores_{{date}}.json
```

---

## Update Status

Per `helpers.md#Update-Workflow-Status`:
- Mark scoring complete
- Record prediction date
- Store output path

---

## Recommend Next Steps

```
✓ Exclusion scores generated!

Next steps:
1. Wait for actual result to validate
2. /backtest-inverse - Run historical validation
3. Re-run /score-exclusions after next event

Prediction ready for evaluation.
```

---

## Interpretation Guide

**High exclusion score (>0.8):**
- Strong confidence this QV will NOT occur
- Safe to include in exclusion set

**Medium exclusion score (0.5-0.8):**
- Moderate confidence
- Check fusion components for insight

**Low exclusion score (<0.5):**
- Model thinks this QV may occur
- Should NOT be in exclusion set

**Score gap analysis:**
- Gap > 0.1: Clear separation, confident cutoff
- Gap < 0.05: Marginal, consider risk controls
- Gap ≈ 0: Uncertain, use bimodality check

---

## Helper References

- Load config: `helpers.md#Combined-Config-Load`
- Update status: `helpers.md#Update-Workflow-Status`
- RAINCOAT concept: `docs-imported/RAINCOAT_C5_Concept_Brief.md`

---

## Notes for LLMs

- Use TodoWrite to track 6 scoring steps
- NEVER include high-probability QVs in exclusion set
- Fusion weights should sum to 1.0
- Risk controls are safety nets, not primary drivers
- Output must be exactly 20 QVs (per task definition)
- Store predictions for backtest validation

**Remember:** Inverse prediction succeeds when NONE of the 5 actual QVs fall in our excluded 20—optimize for zero false positives.
