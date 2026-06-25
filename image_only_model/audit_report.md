
# Audit Report: biomedclip-image-classifier-final.ipynb

## Summary

The primary notebook (`biomedclip-image-classifier-new (1).ipynb` — 24 cells) was fully audited,
merged with the reference notebook (`biomedclip-image-classifier.ipynb` — 59 cells), corrected,
and rebuilt into a production-ready **33-cell** Kaggle notebook:

**Output**: `biomedclip-image-classifier-final.ipynb`

---

## Audit Findings & Fixes

### 1. CRITICAL — `configure_encoder` Fallback Bug (FIXED)

**Problem**: The original notebook always fell back to:
> `⚠️ Fallback: unfroze last 25% of encoder tensors`

This happened because the function tried `model.encoder.visual.transformer.resblocks` but did not
handle the actual open_clip ViT attribute layout robustly.

**Fix**: Rewrote `configure_encoder` with a new `_get_resblocks()` helper that tries 3 attribute
paths in order:
1. `encoder.visual.transformer.resblocks`  (standard open_clip ViT-B/16)
2. `encoder.visual.blocks`                  (timm-style ViT)
3. `encoder.transformer.resblocks`          (flat layout)

Added explicit encoder structure inspection (Cell 13) that prints the actual attribute path found
at runtime, so the user can verify before training.

Also correctly unfreeze:
- Last N transformer blocks (default: 4)
- `ln_post` (layer norm before projection)
- `proj` (projection layer) — handles both `nn.Parameter` and `nn.Module`

**DataParallel safety**: `configure_encoder()` is now called BEFORE `nn.DataParallel` wrapping.

---

### 2. CRITICAL — Duplicate Preview Cells Removed

**Problem**: Primary notebook had 3 overlapping data preview cells (cells 21, 22, 23) with
inconsistent variable dependencies (`cfg` undefined, `train_df` vs `train_split_df`).

**Fix**: Replaced with a single unified Cell 12 that uses the correct training variables
(`train_split_df`, `cfg`) and only runs after the split + subset are created.

---

### 3. CRITICAL — Notebook Truncated (9 missing major sections)

**Problem**: Primary had 24 cells, stopping after training loop definition. It was missing
ALL post-training functionality:

| Missing Section | Status |
|---|---|
| Main training loop | ADDED (Cell 21) |
| Training history plots | ADDED (Cell 22) |
| Load best model | ADDED (Cell 23) |
| Threshold optimization | ADDED (Cell 24) |
| Final evaluation (test set) | ADDED (Cell 25) |
| ROC curves + confusion matrices + AUROC bar | ADDED (Cell 26) |
| GradCAM | ADDED (Cell 27) |
| Single-image inference | ADDED (Cell 28) |
| Feature extraction (512-d embeddings) | ADDED (Cell 29) |
| Save all artifacts + config.json | ADDED (Cell 30) |
| Package & Download ZIP | ADDED (Cell 31) |
| Final summary | ADDED (Cell 32) |

---

### 4. HIGH — Missing EMA in Primary Notebook

**Problem**: EMA class was referenced but not defined in the primary notebook's training loop.

**Fix**: Added complete `EMA` class (Cell 16) with `apply_shadow()`, `restore()`,
`state_dict()`, `load_state_dict()` — fully integrated into training loop.

---

### 5. HIGH — Fixed Threshold Optimization (Never 0.5)

**Problem**: Validation and test evaluation used fixed `0.5` threshold throughout.

**Fix**: Added `optimize_thresholds()` (Cell 24) that grid-searches 0.1→0.9 (step=0.01)
independently for each of the 6 disease classes, optimizing F1. Saved to `thresholds.json`.
All evaluation (validation, test, inference) uses optimized thresholds.

---

### 6. HIGH — TTA Properly Implemented

**Problem**: TTA was declared in config but never actually implemented in the dataset or inference.

**Fix**:
- `MIMICDataset.__getitem__()` averages original + horizontal flip when `tta=True`
- `predict_single_image()` applies TTA for single-image inference
- Val/test datasets use `tta=cfg.tta_enabled`

---

### 7. HIGH — Dataset Path Resolution Robustness

**Problem**: Original `resolve_image_path()` only tried one path format. Many rows would be
silently dropped if the CSV stored paths in a different format.

**Fix**: New resolver (Cell 6) tries:
1. Already-absolute path
2. Strips known prefixes ("files", "official_data_iccv_final")
3. Joins directly with IMAGE_ROOT
4. Reports number of missing images skipped

---

### 8. MEDIUM — Patient Leakage Prevention Strengthened

**Problem**: MultilabelStratifiedKFold split could create patient-level overlap between
train and val sets (same subject_id in both).

**Fix**: Cell 7 explicitly checks for `subject_id` overlap after splitting and re-splits
at the patient level if any overlap is found.

---

### 9. MEDIUM — Missing Subset Label Distribution Plot

**Problem**: No visual confirmation of subset label distribution was provided.

**Fix**: Cell 8 generates a 3-panel bar chart (Train / Val / Test) showing positive
counts per disease label, saved to `plots/label_distribution.png`.

---

### 10. MEDIUM — ROC Curves Plot Bug

**Problem**: Reference notebook's ROC plot used `fpr` and `tpr` from outer scope inside
exception handler, causing `NameError` if the first label's `roc_curve()` call fails.

**Fix**: Local `fpr`, `tpr` variables inside each loop iteration, with proper try/except
fallback per disease.

---

### 11. MINOR — Warning Suppression Order

**Problem**: `warnings.filterwarnings("ignore")` appeared after some imports that could
already trigger deprecation warnings.

**Fix**: Moved `warnings.filterwarnings("ignore")` to the very first line of Cell 2,
before any other import.

---

### 12. MINOR — Empty Last Cell

**Problem**: Primary notebook had an empty cell [23] at the end.

**Fix**: Removed; replaced with complete final summary cell.

---

## Features Added (vs Primary)

| Feature | Implementation |
|---|---|
| Threshold Optimization | Grid 0.1→0.9, step 0.01, per-class F1 |
| TTA | Original + HFlip average |
| GradCAM for ViT | reshape_transform + last resblock target |
| Single-image inference | `predict_single_image()` with TTA + GradCAM |
| Feature extraction | `extract_embeddings()` → `image_embeddings.npy` |
| EMA (Exponential Moving Average) | decay=0.999, shadow copy |
| Per-class optimal thresholds | `thresholds.json` |
| Label distribution plots | Saved to `plots/label_distribution.png` |
| Training history plots | Loss, AUROC, F1, EM-Acc |
| ROC curves (2×3 grid) | Per disease, saved to PNG |
| Confusion matrices | Seaborn heatmap, 2×3 grid |
| AUROC bar chart | Horizontal, with macro line |
| Checkpoint resume | `checkpoint_last.pth` auto-loaded |
| Config save | `config.json` with all hyperparameters |
| ZIP packaging | All outputs in one downloadable ZIP |
| DataParallel fix | `configure_encoder` before wrapping |
| BiomedCLIP inspection | Prints actual attribute path found |

---

## Notebook Structure (33 cells)

| Cell | Type | Section |
|---|---|---|
| 00 | markdown | Title & TOC |
| 01 | code | Install Dependencies |
| 02 | code | Imports |
| 03 | code | Configuration (CFG dataclass) |
| 04 | code | Utility Functions |
| 05 | code | Dataset Discovery |
| 06 | code | Data Loading + Path Resolution |
| 07 | code | Train/Val Split (patient-level) |
| 08 | code | Subset Mode |
| 09 | code | Image Preprocessing |
| 10 | code | MIMICDataset class + TTA |
| 11 | code | DataLoaders |
| 12 | code | Data Preview (5 × 6 labels) |
| 13 | code | Load BiomedCLIP + inspect encoder |
| 14 | code | ImageClassifier + configure_encoder (FIXED) |
| 15 | code | Loss Function (BCEWithLogitsLoss + class weights) |
| 16 | code | EMA |
| 17 | code | Optimizer + Scheduler + GradScaler |
| 18 | code | Metrics functions |
| 19 | code | Checkpoint + EarlyStopping |
| 20 | code | train_one_epoch + validate_one_epoch |
| 21 | code | Main Training Loop |
| 22 | code | Training History Plots |
| 23 | code | Load Best Model (EMA weights) |
| 24 | code | Threshold Optimization |
| 25 | code | Final Evaluation (test set) |
| 26 | code | Evaluation Plots (ROC, CM, AUROC bar) |
| 27 | code | GradCAM |
| 28 | code | Single-Image Inference |
| 29 | code | Feature Extraction (512-d embeddings) |
| 30 | code | Save All Artifacts |
| 31 | code | Package & Download |
| 32 | code | Final Summary |

---

## Saved Artifacts (at /kaggle/working/)

| File | Description |
|---|---|
| `best_model.pth` | Best model (raw weights) |
| `best_model_ema.pth` | Best EMA model |
| `checkpoint_last.pth` | Full training state (resume) |
| `thresholds.json` | Per-class optimal thresholds |
| `training_history.pkl` | Full loss/metric history |
| `image_embeddings.npy` | (N, 512) embeddings for fusion |
| `study_ids.csv` | study_id + split label per row |
| `final_metrics.json` | All test metrics |
| `config.json` | Full config + thresholds |
| `plots/training_history.png` | Training curves |
| `plots/label_distribution.png` | Label distribution |
| `plots/roc_curves.png` | Per-disease ROC |
| `plots/confusion_matrices.png` | Per-disease CM |
| `plots/auroc_bar.png` | AUROC bar chart |
| `gradcam_examples/*.png` | GradCAM per disease |
| `mimic_cxr_image_branch_outputs.zip` | All outputs zipped |

---

## Remaining Notes

- **`iterstrat`**: If not installed on Kaggle, the notebook falls back to `train_test_split`.
  Cell 1 installs it via pip; the import cell handles both cases gracefully.
- **`grad-cam`**: GradCAM cell wraps everything in `if GRADCAM_AVAILABLE` so it skips
  cleanly if the package fails to install.
- **BiomedCLIP model cache**: First run downloads ~350 MB. Kaggle caches it; subsequent
  runs are instant.
- **T4 × 2 DataParallel**: Model is wrapped in `nn.DataParallel` if 2 GPUs are detected.
  EMA and checkpoint loading always unwrap via `.module` before accessing raw weights.
