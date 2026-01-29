# RAINCOAT-Style Frequency-Aware Domain Adaptation for Inverse Prediction on the C-5 Binary Matrix
**Concept Brief — v1.0**  
Prepared: 2025-09-24 17:17 

---

## 1) Goal (Inverse Prediction)
Predict 20 “least likely” Quantum Values (QVs) for the next event (QV range 1–39), such that none of the 5 actual next-event values fall into the predicted 20. This is an inverse formulation: we rank non-occurrence likelihoods and choose the bottom-20 as our output set. The pipeline must avoid standard time-series forecasting models (ARIMA/TFT/LSTM/Prophet/etc.) and instead leverage frequency-aware domain adaptation (RAINCOAT-style) with optimal transport alignment in a time–frequency latent space.

Dataset: c5_binary.csv — wide one-hot rows per event with columns QV_1..QV_39 (exactly five 1s per row).  fileciteturn0file0

Reference article: RAINCOAT: Frequency-Aware Domain Adaptation for Time Series (align-then-correct, time–frequency encoder, Sinkhorn divergence). fileciteturn0file1  Also see the ICML’23 paper and resources. citeturn0search3turn0search4turn0search6turn0search5turn0search2

---

## 2) Why RAINCOAT-style for C-5?
- Regime shifts: The “recent tail” of events often behaves differently from the long history. RAINCOAT explicitly aligns source↔target distributions in a learned latent space using Sinkhorn divergence (regularized optimal transport), which is robust when supports differ in frequency space. citeturn0search3turn0search6
- Time–frequency robustness: Amplitude/phase features derived from DFT complement temporal encoders and are empirically more domain-invariant; RAINCOAT is the first method to bring frequency features into time-series DA at scale. citeturn0search3turn0search13
- Unknown/novel behavior: The align-then-correct stage reconstructs target data to surface target-specific deviations (akin to unknown classes), which suits our inverse task where we want to exclude values most inconsistent with recent dynamics. citeturn0search3turn0search14

---

## 3) Problem to Model Mapping
### 3.1 Domains
- Source domain (S): Earlier history (e.g., all events up to T-N).
- Target domain (T): Recent window (e.g., last 39–117 events; start with 39 as a “tail horizon” and validate 2–3x variants).
The binary column for each QV (1–39) yields 39 parallel binary sequences; the row’s five 1s enforce the “five-active-per-event” structure.  fileciteturn0file0

### 3.2 Task Recast
- Train an encoder–classifier on S (labels = which of the 39 QVs are active per event; multi-label per row with exactly 5 positives).
- Adapt representations to T with Sinkhorn alignment in the time–frequency latent space, then perform target reconstruction (correction) to emphasize T-specific traits. citeturn0search3turn0search6
- For the next event (T+1), produce per-QV scores interpreted as probability of occurrence. Our inverse output is the 20 with the lowest occurrence scores (with risk controls below).

---

## 4) End-to-End Method Overview (RAINCOAT-style, non-standard TS)
1) Windows & labeling:
   - Slice each QV column into overlapping windows (e.g., length L in {64, 128, 256} events; stride 1).
   - The event-level label is the 39-dim binary vector; within-window supervision uses the next-step occurrence indicators.
2) Time–Frequency Encoder (G_TF):
   - Temporal branch: lightweight 1D-CNN stacks (no RNNs/TFT/Transformers) to encode recent patterns.
   - Frequency branch: smoothed DFT (cosine/Hann) per QV-channel -> amplitude + phase features (polar). Concatenate with temporal branch. citeturn0search3
3) Align (Sinkhorn):
   - Minimize Sinkhorn divergence between source and target feature distributions (mini-batch OT). This handles disjoint support and provides stable gradients. citeturn0search6
4) Correct (Target Reconstruction):
   - Freeze the classifier; fine-tune encoder + auxiliary decoder to reconstruct unlabeled target windows. Increased movement in embedding vs class prototypes highlights target-specific deviations. citeturn0search3
5) Scoring for Inverse Prediction:
   - Obtain per-QV logits/probabilities for next event p(QV_i occurs).
   - Convert to non-occurrence scores: s_i = 1 - p(QV_i occurs).
   - Exclusion fusion: combine s_i with: (a) Target rarity prior (recent empirical non-occurrence), (b) Spectral mismatch (target vs source amplitude/phase distances), (c) OT residuals (post-alignment transport cost by channel). Weighted sum (learn weights on validation).
   - Select bottom-20 by fused score. (See risk controls.)
6) Risk Controls (minimize false positives):
   - Diversity filter: discourage picking too many adjacent QVs if adjacency historically co-occurs (optional constraint).
   - Prototype gap check: if any chosen QV’s score is marginally above the 20th, replace with next-lowest whose spectral mismatch & OT residual are higher (safer exclusions).
   - Bimodality check: if next-event score distribution is bimodal, set the cut between modes as the exclusion threshold (RAINCOAT-inspired unknown detection). citeturn0search3

---

## 5) Data Preparation & Splits
- Integrity: Verify each row has exactly five 1s across QV_1..QV_39. Drop/flag anomalies.  fileciteturn0file0
- Chronological walk-forward: Refit/adapt on expanding history; target window = last N_tail events; predict T+1; slide forward.
- Windows: {L in {64, 128, 256}} x {N_tail in {39, 78, 117}} grid for sensitivity.

---

## 6) Model Details (non-standard TS)
- Encoders:
  - Temporal 1D-CNN (3–5 blocks; kernel {3,5,7}; gated linear units or SiLU).
  - Frequency branch: DFT(+window), amplitude/phase pooling across bands; optional learned spectral bins or wavelet ablation (still non-AR/Transformer). citeturn0search3
- Alignment: Sinkhorn divergence with epsilon regularization; cosine distance in latent space; per-channel or grouped channels to respect QV semantics. citeturn0search6
- Correction: Target-only reconstruction loss (L2) with movement-based re-weighting for channels that shift most. citeturn0search3
- Classifier: Simple linear or shallow MLP head -> 39 logits (sigmoid). (We keep the head simple to focus on representation + adaptation.)

---

## 7) Training Schedule
1) Stage A (Supervised on Source): BCE loss on next-step occurrence; early stopping via AUC-PR averaged over 39 QVs.
2) Stage B (Align): Add Sinkhorn term lambda_OT; alternate mini-batches S/T.
3) Stage C (Correct): Freeze head; train encoder+decoder on T for K epochs; monitor reconstruction & embedding movement.
4) Inference: Score next event; fuse exclusions; output bottom-20 QVs.

---

## 8) Metrics — Inverse Prediction
- Primary: FP@20 = count of actual next-event QVs inside predicted 20 (target = 0; lower is better).
- Secondary:
  - Hit@5_on_exclusions (should be 0),
  - Coverage gap = s_20 - s_21 (margin at the cut),
  - Middle-outcome suppression index: frequency of “2–3 wrong” vs “0 or 5 wrong” across backtests (we prefer extremes per project doctrine).
- Backtesting: Rolling horizon; report distributions over thousands of steps.

---

## 9) Ablations & Sensitivity
- Drop frequency branch -> expect significant degradation (RAINCOAT ablations show this). citeturn0search3
- Replace Sinkhorn with MMD -> expect weaker alignment. citeturn0search13
- Remove correction stage -> worse exclusion safety on regime shifts. citeturn0search3

---

## 10) Risks & Mitigations
- Binary sparsity: Only five 1s per row; mitigate via windowed context + spectral pooling to densify features.
- Over-confidence: Calibrate logits with temperature scaling on validation windows.
- Concept drift faster than N_tail: Increase adaptation cadence; shrink L; raise lambda_OT.
- Evaluation leakage: Enforce strict chronology; freeze thresholds during walk-forward.

---

## 11) Artifacts & Deliverables (for PRD/Architecture)
- Code modules:
  - encoders/time_cnn.py, encoders/freq_encoder.py, losses/sinkhorn.py, models/raincoat_c5.py, scoring/exclusion_fusion.py, train_align_correct.py, backtest_runner.py
- Configs: YAML for data paths, window sizes, lambda_OT, L, fusion weights.
- Reports: Rolling FP@20 curves, exclusion distributions, spectral shift heatmaps, OT transport maps.
- Releases: c5_raincoat_exclusion_v{date}.ckpt, backtests_{range}.csv, and predictions_next_event.json.

---

## 12) Compute & Environment
- GPU optional (CNNs/DFT are light); CPU works for prototyping. For large grids, 1x A100/H100 accelerates OT & training.
- Dependencies: PyTorch or JAX, POT or geomloss for Sinkhorn; numpy/scipy for DFT; pandas for IO.

---

## 13) References
- He et al., RAINCOAT — ICML 2023 paper & PDF; Harvard page; GitHub. citeturn0search3turn0search4turn0search6turn0search2turn0search5
- Benchmarks & follow-ups citing frequency features + Sinkhorn for TS DA. citeturn0search7turn0search10turn0search11turn0search13turn0search18
- Dataset description: C-5 Binary Matrix (structure/columns). fileciteturn0file0
- RAINCOAT explainer (align–then–correct summary). fileciteturn0file1

---

## 14) Next Steps (toward PRD.md & architecture.md)
1) Lock N_tail, L, lambda_OT via short grid on a held-out historical span.
2) Implement exclusion fusion + bimodality threshold.
3) Build walk-forward backtester that logs FP@20 and margin statistics.
4) Package reproducible configs + CLI; emit JSON for the 20 least-likely QVs per step.
