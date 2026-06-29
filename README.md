# Multimodal Medical Foundation Model

A research project building a **Multimodal Medical Foundation Model** for automated chest X-ray analysis using deep learning.

---

## 📁 Repository Structure

```
multimodal-medical-foundation-model/
├── README.md               ← this file
├── requirements.txt        ← top-level Python dependencies
├── src/                    ← core source library
├── tests/                  ← test suite
└── image_only_model/       ← Image-Only BiomedCLIP classifier (main module)
    ├── README.md
    ├── biomedclip-image-classifier.ipynb
    ├── data_preprocessing.py
    ├── configs/
    ├── output/             ← metrics & plots
    ├── src/
    └── tests/
```

---

## 🗂️ Modules

### [`image_only_model/`](./image_only_model/)
The image-only classifier trained on MIMIC-CXR using **BiomedCLIP** (ViT-base + PubMedBERT).  
It classifies 14 chest pathologies from X-ray images alone.  
See the dedicated [image_only_model/README.md](./image_only_model/README.md) for full details.

---

## 🚀 Getting Started

```bash
git clone https://github.com/VNK1404/multimodal-medical-foundation-model.git
cd multimodal-medical-foundation-model
pip install -r requirements.txt
```

Then navigate to the module you want to explore:
```bash
cd image_only_model
jupyter notebook biomedclip-image-classifier.ipynb
```

---

## 📄 License

MIT License
