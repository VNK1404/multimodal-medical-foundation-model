# 🏥 Multimodal Medical Foundation Model

A research project building a **Multimodal Medical Foundation Model** for automated chest X-ray analysis using deep learning. This repository houses multiple branches of development, exploring both unimodal (image-only) and multimodal (image+text) disease classification pipelines on the clinical chest radiograph dataset (MIMIC-CXR).

---

## 📁 Repository Structure

```
multimodal-medical-foundation-model/
├── README.md               ← this file (global overview)
├── requirements.txt        ← top-level Python dependencies
├── src/                    ← core source library
├── tests/                  ← test suite
│
├── image_only_model/       ← Unimodal classifier (Image-Only)
│   ├── README.md           ← detailed documentation for image-only branch
│   ├── biomedclip-image-classifier.ipynb
│   ├── configs/
│   └── output/             ← metrics & plots
│
└── image+text_model/       ← Multimodal classifier (Image + Text)
    ├── README.md           ← detailed documentation for image+text branch
    ├── image-text-model.ipynb
    └── output/             ← metrics & plots
```

---

## 🗂️ Project Modules

### 1. 🖼️ [Image-Only Model (BiomedCLIP Classifier)](./image_only_model/)
*   **Description:** Implements a unimodal classification system leveraging **BiomedCLIP** (ViT-B/16 image encoder) to predict 14 chest pathologies.
*   **Key Features:** Custom medical preprocessing, test-time augmentation (TTA), threshold optimization, and GradCAM visualization for visual interpretability.
*   **Documentation:** See the dedicated [image_only_model/README.md](./image_only_model/README.md) for full details.

### 2. 📝 [Image+Text Multimodal Model (Late Fusion Classifier)](./image+text_model/)
*   **Description:** Implements a multimodal classification system that jointly learns from chest X-ray images (encoded via **BiomedCLIP**) and clinical radiology reports (encoded via **BioClinicalBERT**).
*   **Key Features:** Late fusion projection architecture, class-weighted Focal Loss to address imbalance, Exponential Moving Average (EMA) weights, and dual explainability methods (GradCAM for vision + Integrated Gradients for text).
*   **Performance:** Achieves a state-of-the-art **Macro AUROC of 0.9383** on validation.
*   **Documentation:** See the dedicated [image+text_model/README.md](./image+text_model/README.md) for full details.

---

## 🚀 Getting Started

To get started with the project, clone the repository and install the dependencies:

```bash
# Clone the repository
git clone https://github.com/VNK1404/multimodal-medical-foundation-model.git
cd multimodal-medical-foundation-model

# Install dependencies
pip install -r requirements.txt
```

Navigate to the module you want to explore and run the notebooks:

### For the Image-Only Model:
```bash
cd image_only_model
jupyter notebook biomedclip-image-classifier.ipynb
```

### For the Image+Text Multimodal Model:
```bash
cd image+text_model
jupyter notebook image-text-model.ipynb
```

---

## 📄 License

This repository is licensed under the MIT License — see the root `LICENSE` file for details.
