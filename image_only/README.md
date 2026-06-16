# BiomedCLIP Chest X-Ray Classifier (Image-Only Model)

This folder contains the source code, notebooks, and configurations for the **Image-Only Branch** of the **Multimodal Medical Foundation Model**. This model leverages a pretrained **BiomedCLIP** vision-language foundation model to classify chest X-ray images from the **MIMIC-CXR** dataset across 6 clinical labels.

---

## 🌟 Key Features

* **BiomedCLIP Backbone**: Built upon `biomedclip-vit-b-16` pretrained on millions of biomedical image-text pairs.
* **Flexible Fine-Tuning**: Features a robust `configure_encoder` module that dynamically resolves transformer block layers across different ViT architectures to unfreeze specific top layers (default: last 4 blocks, `ln_post`, and projection layers).
* **EMA (Exponential Moving Average)**: Standardized shadow weight updates (decay rate `0.999`) to improve model robustness and generalization.
* **Threshold Optimization**: Replaces static thresholds with dynamic per-class F1-optimized thresholds (grid search `0.1` to `0.9` at `0.01` resolution).
* **Test-Time Augmentation (TTA)**: Averages predictions on original and horizontally flipped images during validation and test phases to boost performance.
* **GradCAM for ViT**: Integrated explainability via Visual Attribution Maps customized for the Vision Transformer layout.
* **Multimodal Embedding Extractor**: Extracts and exports 512-dimensional visual feature vectors for subsequent cross-modal late fusion.

---

## 📂 Project Directory Structure

* [biomedclip-image-classifier-final.ipynb](file:///d:/proejects/Multimodal%20Medical%20Foundation%20Model/image_only/biomedclip-image-classifier-final.ipynb): Main training, evaluation, validation, and inference Jupyter notebook.
* [data_preprocessing.py](file:///d:/proejects/Multimodal%20Medical%20Foundation%20Model/image_only/data_preprocessing.py): Preprocessing logic for subject-level splits, data validation, and leakage detection.
* [data_preprocessing_explanation.txt](file:///d:/proejects/Multimodal%20Medical%20Foundation%20Model/image_only/data_preprocessing_explanation.txt): Detailed explanation of split logic and class balances.
* [audit_report.md](file:///d:/proejects/Multimodal%20Medical%20Foundation%20Model/image_only/audit_report.md): Summary of updates, fixes, and architectural adjustments made during code integration.

---

## 📊 Dataset Specifications

The model processes chest radiographs from the **MIMIC-CXR** dataset.

* **Image Root Directory**: `/kaggle/input/datasets/simhadrisadaram/mimic-cxr-dataset/official_data_iccv_final/files`
* **CSV Labels Directory**: `/kaggle/input/datasets/vedantkulkarni14/labeled-mimic-cxr/`
  * Train: `mimic_final_labeled_train.csv`
  * Validate: `mimic_final_labeled_validate.csv`
* **Disease Categories Classified**:
  1. Atelectasis
  2. Cardiomegaly
  3. Consolidation
  4. Edema
  5. Pleural Effusion
  6. Support Devices

---

## 🛠️ Usage Instructions

### 1. Model Configuration
All hyper-parameters and setup options are defined under the `CFG` class in the main notebook:
* `backbone`: `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
* `epochs`: `10`
* `batch_size`: `64`
* `lr`: `5e-5`
* `unfreeze_blocks`: `4` (last 4 Transformer blocks)
* `ema_decay`: `0.999`
* `tta_enabled`: `True`

### 2. Run Training & Evaluation
Open [biomedclip-image-classifier-final.ipynb](file:///d:/proejects/Multimodal%20Medical%20Foundation%20Model/image_only/biomedclip-image-classifier-final.ipynb) inside a Kaggle / Jupyter environment containing a GPU (Tesla T4 × 2 recommended) and run all cells sequentially. The workflow will:
1. Validate datasets and resolve local file paths.
2. Apply stratified splitting and check for patient-level leakage.
3. Configure the custom model head and unfreeze the selected encoder layers.
4. Run training with early stopping, saving periodic checkpoints.
5. Search for optimized F1 score thresholds.
6. Evaluate the test partition, saving ROC, CM, and AUROC plots.

---

## 💾 Generated Output Artifacts

All training and evaluation outputs are compiled into `/kaggle/working/` and packaged as a single ZIP file:

* `best_model.pth`: The raw parameters representing the lowest validation loss state.
* `best_model_ema.pth`: Model weights after application of the Exponential Moving Average shadow weights.
* `thresholds.json`: Optimized classification thresholds per target disease.
* `image_embeddings.npy`: Exported `(N, 512)` tensor arrays representing extraction embeddings.
* `plots/`: Contains visual charts for:
  - `training_history.png`
  - `label_distribution.png`
  - `roc_curves.png`
  - `confusion_matrices.png`
  - `auroc_bar.png`
* `gradcam_examples/`: Visual attention heatmap exports for individual chest X-ray samples.
* `mimic_cxr_image_branch_outputs.zip`: A unified zip file containing all of the above artifacts.
