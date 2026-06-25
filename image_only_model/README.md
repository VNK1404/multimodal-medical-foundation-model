# Image-Only Model — BiomedCLIP Image Classifier

This folder contains the **image-only** branch of the Multimodal Medical Foundation Model project.  
It trains and evaluates a BiomedCLIP-based classifier on chest X-ray (MIMIC-CXR) data using **image features alone**, without any text/report input.

---

## 📁 Folder Structure

```
image_only_model/
├── README.md                           ← this file
├── biomedclip-image-classifier.ipynb   ← main training & evaluation notebook
├── data_preprocessing.py               ← data loading, augmentation pipeline
├── data_preprocessing_explanation.txt  ← detailed explanation of preprocessing steps
├── audit_report.md                     ← model audit: fairness, performance, caveats
├── requirements.txt                    ← Python dependencies
├── .gitignore                          ← ignores large model weights (*.pth)
│
├── configs/
│   └── training_config.json            ← hyperparameters & training settings
│
├── output/
│   ├── final_metrics.json              ← final test-set metrics (AUROC, F1, etc.)
│   ├── thresholds.json                 ← per-class classification thresholds
│   ├── training_history.pkl            ← epoch-level loss & metric curves
│   ├── tta_metrics.json                ← test-time augmentation metrics
│   └── plots/
│       ├── auroc_bar.png               ← per-class AUROC bar chart
│       ├── confusion_matrices.png      ← per-class confusion matrices
│       ├── roc_curves.png              ← ROC curves for all classes
│       └── training_history.png        ← train/val loss & accuracy over epochs
│
├── src/
│   ├── __init__.py
│   ├── eval/                           ← evaluation scripts
│   ├── rag/                            ← retrieval-augmented generation utilities
│   └── train/                          ← training scripts (baselines, fusion)
│
└── tests/
    ├── __init__.py
    └── test_smoke.py                   ← smoke tests for pipeline integrity
```

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/VNK1404/multimodal-medical-foundation-model.git
cd multimodal-medical-foundation-model/image_only_model

# 2. Install dependencies
pip install -r requirements.txt

# 3. Open the notebook
jupyter notebook biomedclip-image-classifier.ipynb
```

---

## ⚙️ Training Configuration

Key settings in `configs/training_config.json`:

| Parameter | Value |
|-----------|-------|
| Base model | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` |
| Image size | 224 × 224 |
| Batch size | 32 |
| Epochs | 30 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |

Edit `configs/training_config.json` to adjust any setting before running.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Mean AUROC | See `output/final_metrics.json` |
| TTA Mean AUROC | See `output/tta_metrics.json` |

Plots are available under `output/plots/`.

> **Note:** Model weight files (`best_model.pth`, `best_model_ema.pth`) are not stored in this repository due to their size (~1.4 GB each). They are available on request.

---

## 🔬 Data

- Dataset: [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) (requires PhysioNet credentialed access)
- Preprocessing: see `data_preprocessing.py` and `data_preprocessing_explanation.txt`

---

## 📄 License

MIT License — see the root `LICENSE` file for details.
