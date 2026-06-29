# 🏥 Image+Text Multimodal Model (Chest X-ray Disease Classifier)

This folder contains the **Image+Text multimodal branch** of the Multimodal Medical Foundation Model project. It trains and evaluates a late-fusion disease classification system that leverages both visual features from chest X-ray images and clinical context from radiology reports.

---

## 📁 Directory Structure

```
image+text_model/
├── README.md                                 ← this file
├── image-text-model.ipynb                    ← Jupyter notebook for model training & evaluation
├── multimodal_medical_foundation_model.ipynb ← Jupyter notebook for model development & prototyping
└── output/                                   ← training artifacts, metrics, and visualization plots
    ├── config.json                           ← experiment hyperparameters & environment configuration
    ├── baseline_metrics.json                 ← performance metrics for the baseline model
    ├── final_metrics.json                    ← final test-set metrics (baseline, EMA, EMA+TTA)
    ├── model_comparison.json                 ← comparison of different evaluation pipelines
    ├── thresholds.json                       ← optimized class decision thresholds
    ├── training_history.json                 ← training and validation metrics history per epoch
    ├── auroc_bar_chart.png                   ← bar chart comparing per-class AUROCs
    ├── confusion_matrices.png                ← confusion matrices for all 6 pathologies
    ├── roc_curves.png                        ← ROC curves for all classes
    ├── training_history.png                  ← train/validation loss & AUROC curves over epochs
    └── gradcam_examples/                     ← attention heatmaps visualizing model explainability
        ├── gradcam_sample_1.png
        ├── gradcam_sample_2.png
        └── gradcam_sample_3.png
```

> **Note:** Large model weight files (e.g., `best_model.pth`, `best_model_ema.pth`, `checkpoint_last.pth`) are ignored by Git due to their size (~1.2 GB to ~3.2 GB) but are saved in `output/` locally upon running the training pipeline.

---

## 🚀 Quick Start

Ensure you have installed the top-level repository dependencies first:

```bash
# 1. Clone the repository
git clone https://github.com/VNK1404/multimodal-medical-foundation-model.git
cd multimodal-medical-foundation-model/image+text_model

# 2. Open the main development notebook
jupyter notebook multimodal_medical_foundation_model.ipynb
```

---

## ⚙️ Model Architecture & Features

The model fuses visual and textual modalities using a late fusion strategy:

1. **Visual Encoder:** Fine-tuned [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) (ViT-B/16) with the first **6 Transformer blocks frozen** to preserve robust pre-trained clinical visual features.
2. **Text Encoder:** Fine-tuned [BioClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) with the first **10 layers frozen** to encode specialized clinical semantics from radiology reports.
3. **Late Fusion Head:** Linear projections map both modality embeddings (512-dim image, 768-dim text) to a shared 512-dim space. They are concatenated into a 1024-dim fused representation, followed by a deep fully connected head with LayerNorm, GELU activation, and Dropout for classification.
4. **Optimized Training Objectives:**
   - **Focal Loss** ($\gamma=2.0$) with class-weight balancing to address high class imbalance.
   - **Mixed Precision (AMP)** for faster training and reduced GPU memory footprint.
   - **Exponential Moving Average (EMA)** of weights ($\text{decay}=0.999$) to stabilize validation performance.
   - **Early Stopping** with validation loss/AUROC monitoring.

---

## 📊 Results & Evaluation

The model performs **multi-label classification** across 6 pathologies on chest X-rays. Below is the summary of results achieved on the validation set using the **EMA weights** (our best performing pipeline, which achieves a **Macro AUROC of 0.9383**).

### Aggregate Performance

| Pipeline | Macro AUROC | Micro AUROC | Macro F1 | Micro F1 | Mean Per-Class Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 0.9365 | 0.9127 | 0.7573 | 0.8257 | 90.24% |
| **EMA (Best)** | **0.9383** | **0.9209** | **0.7646** | **0.8298** | **90.19%** |
| **EMA + TTA** | 0.9381 | 0.9209 | 0.7647 | 0.8240 | 89.74% |

### Per-Pathology Performance (EMA Model)

Per-class thresholds were optimized during training to maximize the F1-score:

| Pathology | AUROC | F1-Score | Precision | Recall (Sensitivity) | AUPRC | Optimized Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **No Finding** | 0.9621 | 0.9139 | 0.8565 | 0.9795 | 0.9726 | 0.25 |
| **Cardiomegaly** | 0.9632 | 0.7651 | 0.6706 | 0.8906 | 0.8640 | 0.59 |
| **Edema** | 0.9511 | 0.8095 | 0.7234 | 0.9189 | 0.8458 | 0.58 |
| **Pleural Effusion** | 0.9621 | 0.8603 | 0.8652 | 0.8556 | 0.9068 | 0.70 |
| **Pneumonia** | 0.9301 | 0.7650 | 0.6587 | 0.9121 | 0.8209 | 0.62 |
| **Pneumothorax** | 0.8609 | 0.4737 | 0.5000 | 0.4500 | 0.4514 | 0.70 |

*Performance plots are available in `output/`.*

---

## 🔍 Interpretability & Clinical Explainability

To ensure transparency in clinical decision support, the model supports two explainability methods:

* **GradCAM (Visual Attribution):** Visualizes the visual attention maps of the BiomedCLIP ViT-B/16 backbone, highlighting the specific regions of the chest X-ray image that influenced the model's predictions.
* **Integrated Gradients (Textual Attribution):** Attributes class predictions back to individual tokens and terms within the radiology reports using the Captum library, highlighting which sentences or words drove the decisions.

*Example GradCAM outputs are saved in `output/gradcam_examples/`.*

---

## 🔬 Dataset

The model is trained on the [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) database (v2.0.0), a large publicly available dataset of chest radiographs with free-text radiology reports. Accessing this data requires credentialing via PhysioNet.

---

## 📄 License

This module is licensed under the MIT License. See the root `LICENSE` file for details.
