You are the RAINCOAT Specialist, executing the **Train RAINCOAT** workflow.

## Workflow Overview

**Goal:** Train time-frequency encoder with domain adaptation for C5 inverse prediction

**Phase:** Phase 4 - Implementation

**Agent:** RAINCOAT Specialist

**Inputs:** Windowed data, domain splits, quantumized features (optional)

**Output:** Trained model checkpoint, training logs

**Duration:** 30-60 minutes (depends on data size and hardware)

---

## Pre-Flight

1. Load context per `helpers.md#Combined-Config-Load`
2. Verify windowed data exists (`windows_X.npy`, `windows_y.npy`)
3. Verify domain split exists (`split_config.json`)
4. Check PyTorch and dependencies available
5. Load training configuration

---

## Train RAINCOAT Process

Use TodoWrite to track: Load Data → Configure Model → Stage A (Supervised) → Stage B (Align) → Stage C (Correct) → Save Checkpoint

---

### Part 1: Load Data

**Required files:**
- `./data/windows/windows_X.npy` - Window features
- `./data/windows/windows_y.npy` - Labels (next-step occurrence)
- `./data/prepared/split_config.json` - Domain split info

**Create data loaders:**
```python
# Source domain: history windows
source_X = windows_X[:source_split_idx]
source_y = windows_y[:source_split_idx]

# Target domain: recent tail windows
target_X = windows_X[source_split_idx:]
target_y = windows_y[source_split_idx:]  # May be None for unsupervised adaptation

source_loader = DataLoader((source_X, source_y), batch_size=64, shuffle=True)
target_loader = DataLoader((target_X,), batch_size=64, shuffle=True)
```

---

### Part 2: Configure Model

**Architecture:**

```
Time-Frequency Encoder (G_TF)
├── Temporal Branch
│   ├── Conv1D(39, 64, kernel=7, stride=1)
│   ├── SiLU + BatchNorm
│   ├── Conv1D(64, 128, kernel=5, stride=2)
│   ├── SiLU + BatchNorm
│   ├── Conv1D(128, 256, kernel=3, stride=2)
│   ├── SiLU + BatchNorm
│   └── AdaptiveAvgPool1D → (256,)
│
├── Frequency Branch
│   ├── DFT (Hann window, per channel)
│   ├── Amplitude + Phase extraction
│   ├── Conv1D(78, 128, kernel=3)  # 78 = 39 amp + 39 phase
│   ├── SiLU + BatchNorm
│   └── AdaptiveAvgPool1D → (128,)
│
└── Fusion
    ├── Concat(temporal, frequency) → (384,)
    ├── Linear(384, 256)
    └── SiLU → latent (256,)

Classifier Head
├── Linear(256, 128)
├── SiLU
├── Linear(128, 39)
└── Sigmoid → probabilities (39,)

Decoder (for correction stage)
├── Linear(256, 384)
├── SiLU
├── Linear(384, L × 39)
└── Reshape → reconstructed window
```

**Hyperparameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| L | 128 | Window length |
| latent_dim | 256 | Encoder output dimension |
| batch_size | 64 | Training batch size |
| lr | 1e-3 | Learning rate |
| weight_decay | 1e-4 | L2 regularization |
| lambda_ot | 0.1 | Sinkhorn loss weight |
| epsilon | 0.1 | Sinkhorn regularization |

---

### Part 3: Stage A - Supervised Training

**Objective:** Train encoder + classifier on source domain

**Loss:**
```
L_A = BCE(predictions, labels)
     = -Σ[y_i log(p_i) + (1-y_i) log(1-p_i)]
```

**Training loop:**
```python
for epoch in range(max_epochs_A):
    for X_batch, y_batch in source_loader:
        latent = encoder(X_batch)
        pred = classifier(latent)
        loss = bce_loss(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation: AUC-PR averaged over 39 QVs
    val_auc = compute_auc_pr(val_pred, val_labels)

    # Early stopping
    if val_auc > best_auc:
        best_auc = val_auc
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break
```

**Stopping criterion:** AUC-PR on validation set (early stopping, patience=10)

**Output:**
```
Stage A Complete
- Epochs: {{n_epochs}}
- Final BCE loss: {{loss:.4f}}
- Validation AUC-PR: {{auc:.4f}}
```

---

### Part 4: Stage B - Domain Alignment

**Objective:** Align source and target distributions using Sinkhorn divergence

**Loss:**
```
L_B = L_A + λ_OT × D_sinkhorn(f_source, f_target)
```

**Sinkhorn divergence:**
```python
from geomloss import SamplesLoss

sinkhorn = SamplesLoss(
    loss="sinkhorn",
    p=2,
    blur=epsilon,
    scaling=0.9,
    backend="tensorized"
)

def alignment_loss(source_features, target_features):
    return sinkhorn(source_features, target_features)
```

**Training loop:**
```python
for epoch in range(max_epochs_B):
    for (X_s, y_s), (X_t,) in zip(source_loader, cycle(target_loader)):
        # Source forward pass
        latent_s = encoder(X_s)
        pred_s = classifier(latent_s)
        loss_bce = bce_loss(pred_s, y_s)

        # Target forward pass (no labels)
        latent_t = encoder(X_t)

        # Sinkhorn alignment
        loss_ot = sinkhorn(latent_s, latent_t)

        # Combined loss
        loss = loss_bce + lambda_ot * loss_ot

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Monitor OT divergence
    print(f"Epoch {epoch}: BCE={loss_bce:.4f}, OT={loss_ot:.4f}")
```

**Convergence:** OT divergence should decrease; stop when plateau

**Output:**
```
Stage B Complete
- Epochs: {{n_epochs}}
- Final OT divergence: {{ot:.4f}}
- BCE maintained: {{bce:.4f}}
```

---

### Part 5: Stage C - Target Correction

**Objective:** Fine-tune encoder with target reconstruction to capture target-specific traits

**Setup:**
- FREEZE classifier head
- TRAIN encoder + decoder
- USE target domain only

**Loss:**
```
L_C = ||X_target - Decoder(Encoder(X_target))||²
```

**Training loop:**
```python
# Freeze classifier
for param in classifier.parameters():
    param.requires_grad = False

for epoch in range(max_epochs_C):
    for X_t in target_loader:
        latent = encoder(X_t)
        reconstructed = decoder(latent)
        loss = mse_loss(reconstructed, X_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Monitor embedding movement
    with torch.no_grad():
        new_latent = encoder(target_X)
        movement = torch.norm(new_latent - prev_latent) / len(target_X)
        prev_latent = new_latent.clone()

    print(f"Epoch {epoch}: Recon={loss:.4f}, Movement={movement:.4f}")
```

**Embedding movement:** Large movement indicates significant target-specific learning

**Output:**
```
Stage C Complete
- Epochs: {{n_epochs}}
- Final reconstruction loss: {{recon:.4f}}
- Embedding movement: {{movement:.4f}} ({{movement_pct:.1f}}%)
```

---

### Part 6: Save Checkpoint

**Save model state:**
```python
checkpoint = {
    'encoder_state': encoder.state_dict(),
    'classifier_state': classifier.state_dict(),
    'decoder_state': decoder.state_dict(),
    'config': {
        'L': L,
        'latent_dim': latent_dim,
        'lambda_ot': lambda_ot,
        'epsilon': epsilon,
    },
    'training_history': {
        'stage_a': {'epochs': n_a, 'auc': best_auc},
        'stage_b': {'epochs': n_b, 'final_ot': final_ot},
        'stage_c': {'epochs': n_c, 'movement': movement},
    },
    'timestamp': datetime.now().isoformat(),
}

torch.save(checkpoint, f'./models/c5_raincoat_v{date}.ckpt')
```

**Training log:**
```json
{
  "model_path": "./models/c5_raincoat_v20260129.ckpt",
  "stage_a": {"epochs": 45, "best_auc": 0.73, "final_bce": 0.42},
  "stage_b": {"epochs": 30, "final_ot": 0.18, "lambda_ot": 0.1},
  "stage_c": {"epochs": 20, "recon_loss": 0.31, "movement": 0.123},
  "total_time_minutes": 47,
  "hardware": "CPU / CUDA"
}
```

---

## Generate Output

**Display:**
```
## RAINCOAT Training Complete

**Model:** c5_raincoat_v{{date}}.ckpt

**Stage A (Supervised):**
- Epochs: {{n_a}}
- Best AUC-PR: {{auc:.4f}}

**Stage B (Alignment):**
- Epochs: {{n_b}}
- OT divergence: {{start_ot:.4f}} → {{end_ot:.4f}}

**Stage C (Correction):**
- Epochs: {{n_c}}
- Reconstruction loss: {{recon:.4f}}
- Embedding movement: {{movement:.1f}}%

**Total training time:** {{time}} minutes

**Files:**
- ./models/c5_raincoat_v{{date}}.ckpt
- ./logs/training_log_{{date}}.json
```

---

## Update Status

Per `helpers.md#Update-Workflow-Status`:
- Mark training complete
- Record model checkpoint path
- Store training metrics

---

## Recommend Next Steps

```
✓ RAINCOAT model trained!

Next steps:
1. /score-exclusions - Generate inverse prediction scores
2. /backtest-inverse - Validate with walk-forward testing
3. /ablation-report - Compare with/without components

Model ready for inference.
```

---

## Hardware Notes

| Hardware | Expected Time | Notes |
|----------|---------------|-------|
| CPU | 45-90 min | Feasible for C5 size |
| GPU (RTX 3060) | 15-30 min | Recommended |
| GPU (A100) | 5-15 min | Overkill but fast |

---

## Helper References

- Load config: `helpers.md#Combined-Config-Load`
- Update status: `helpers.md#Update-Workflow-Status`
- RAINCOAT concept: `docs-imported/RAINCOAT_C5_Concept_Brief.md`

---

## Notes for LLMs

- Use TodoWrite to track 6 training stages
- Monitor AUC-PR (not accuracy) for imbalanced multi-label
- Sinkhorn convergence is critical before Stage C
- Embedding movement > 10% indicates meaningful adaptation
- NEVER use standard TS models (ARIMA, LSTM, etc.) per project doctrine
- Reference geomloss or POT for Sinkhorn implementation

**Remember:** RAINCOAT's power comes from align-then-correct—skipping either stage degrades performance on regime shifts.
