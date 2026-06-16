# MIMIC-CXR Label Generation Pipeline

This directory contains the source code, outputs, and validation reports for the **Label Generation** component of the **Multimodal Medical Foundation Model**. This pipeline processes row-level radiology reports from the **MIMIC-CXR** dataset and extracts study-level disease labels across 6 target clinical categories using rule-based Clinical NLP (Negation and Uncertainty detection).

---

## 🌟 Key Features

* **Rule-Based Clinical NLP Labeller**: A deterministic keyword-matching engine combined with localized context window analysis to identify findings in free-text radiology reports.
* **Context-Aware Negation Detection**: Scans a 10-word left-context window for negation cues (e.g., *"no evidence of"*, *"without"*, *"free of"*) to correctly assign negative labels.
* **Uncertainty Policy Resolution**: Scans an 8-word left-context window for uncertainty phrases (e.g., *"possible"*, *"suggestive of"*, *"cannot exclude"*). Resolves uncertainty using the conservative `ones` policy (treating uncertain mentions as positive).
* **Optimal Image Selection**: Resolves multiple images per clinical study by filtering out lateral views and ranking frontal views by preference: **PA (Posterior-Anterior)** $\rightarrow$ **AP (Anterior-Posterior)** $\rightarrow$ first available.
* **Validation & Manual Inspection**: Features an inspection pipeline that exports a report of 100 random studies for manual quality assurance and verification.

---

## 📂 Directory Structure

* [label-generator.ipynb](file:///d:/proejects/Multimodal%20Medical%20Foundation%20Model/label_generation/label-generator.ipynb): Complete Jupyter notebook demonstrating text cleaning, keyword-based label generation, image selection, validation, and statistical visualization.
* [labeled_mimic_cxr/](file:///d:/proejects/Multimodal%20Medical%20Foundation%20Model/label_generation/labeled_mimic_cxr): Compiled visualization charts, statistics, and text audits.
  * [label_statistics.json](file:///d:/proejects/Multimodal%20Medical%20Foundation%20Model/label_generation/labeled_mimic_cxr/label_statistics.json): Detailed summary of dataset counts, study volumes, and disease prevalence.
  * [manual_inspection_report.txt](file:///d:/proejects/Multimodal%20Medical%20Foundation%20Model/label_generation/labeled_mimic_cxr/manual_inspection_report.txt): The audit trail of 100 randomly sampled studies showing raw report text mapped to generated labels for verification.
  * `cooccurrence_train.png` / `cooccurrence_validate.png`: Pearson correlation matrices demonstrating disease co-occurrence.
  * `label_count_hist_train.png` / `label_count_hist_validate.png`: Histograms of study count distributions by positive label density.
  * `label_distribution.png`: Comparative bar chart of disease prevalence across splits.
  * `view_distribution.png`: Distribution of frontal X-ray views (PA vs. AP) selected for the final dataset.

---

## 🔒 Data Privacy & Compliance Notice

> [!IMPORTANT]
> The generated label files (`mimic_final_labeled_train.csv` and `mimic_final_labeled_validate.csv`) contain raw patient identifier IDs (`subject_id`, `study_id`) and **raw clinical radiology reports**.
> 
> In accordance with the **MIMIC-CXR Data Use Agreement (DUA)**, HIPAA privacy regulations, and the project guidelines, **these raw CSV files are excluded from Git tracking** (`.gitignore`) and must not be pushed to public repositories.
> 
> You can easily regenerate these CSVs locally by following the **Usage Instructions** below.

---

## ⚙️ NLP Configuration & Rules

The pipeline targets **6 clinical labels** using the following rules:

### 1. Target Disease Categories & Keywords
* **Cardiomegaly**: `cardiomegaly`, `enlarged cardiac silhouette`, `cardiac enlargement`, `enlarged heart`, `prominent cardiac silhouette`, `increased cardiac size`, `cardiac silhouette is enlarged`.
* **Edema**: `pulmonary edema`, `interstitial edema`, `alveolar edema`, `vascular congestion`, `fluid overload`, `pulmonary vascular congestion`, `congestive heart failure`, `chf`, `pulmonary venous hypertension`, `interstitial opacities`, `Kerley b lines`, `haziness`.
* **Pleural Effusion**: `pleural effusion`, `bilateral effusions`, `bilateral pleural effusions`, `small effusion`, `moderate effusion`, `large effusion`, `pleural fluid`, `blunting of the costophrenic angle`, `blunting of costophrenic`, `costophrenic blunting`, `pleural space`, `layering effusion`.
* **Pneumonia**: `pneumonia`, `airspace opacity`, `airspace disease`, `lobar consolidation`, `consolidation`, `infiltrate`, `infiltration`, `infectious process`, `focal opacity`, `parenchymal opacity`, `atelectasis with pneumonia`, `bronchopneumonia`, `community acquired pneumonia`.
* **Pneumothorax**: `pneumothorax`, `collapsed lung`, `lung collapse`, `apical pneumothorax`, `tension pneumothorax`, `small pneumothorax`, `iatrogenic pneumothorax`.
* **No Finding**: Automatically set to `1` if and only if **all other 5 diseases are 0** (absent).

### 2. Context Windows
* **Negation Window (10 tokens)**: Checks the 10 words preceding a keyword match for negation cues: `no`, `without`, `no evidence of`, `no sign of`, `without evidence of`, `negative for`, `absence of`, `free of`, `not seen`, `not identified`, `not present`, `not demonstrated`, `not detected`, `not visualized`, `no new`, `no acute`.
* **Uncertainty Window (8 tokens)**: Checks the 8 words preceding a keyword match for uncertainty triggers: `possible`, `possibly`, `cannot exclude`, `cannot rule out`, `can not exclude`, `cannot be excluded`, `may represent`, `could represent`, `suspicious for`, `consistent with`, `suggestive of`, `question of`, `uncertain`, `equivocal`, `cannot be determined`.

---

## 📊 Dataset Statistics & Characteristics

Below is a summary of the generated dataset statistics (available in `label_statistics.json`):

| Metric / Disease Label | Train Split ($N=45,224$ studies) | Validate Split ($N=338$ studies) |
| :--- | :---: | :---: |
| **No Finding** (Prevalence) | 55.20% ($24,965$) | 57.69% ($195$) |
| **Pneumonia** (Prevalence) | 27.37% ($12,380$) | 26.92% ($91$) |
| **Pleural Effusion** (Prevalence) | 25.55% ($11,553$) | 26.63% ($90$) |
| **Edema** (Prevalence) | 21.73% ($9,825$) | 21.89% ($74$) |
| **Cardiomegaly** (Prevalence) | 19.72% ($8,919$) | 18.93% ($64$) |
| **Pneumothorax** (Prevalence) | 5.44% ($2,461$) | 5.92% ($20$) |

---

## 🛠️ Usage & Local Execution

To run the pipeline and generate the dataset CSVs locally:

1. **Prerequisites**: Ensure you have Python with the following libraries:
   ```bash
   pip install numpy pandas matplotlib seaborn
   ```
2. **Kaggle Setup**: Mount the official MIMIC-CXR dataset at `/kaggle/input/datasets/simhadrisadaram/mimic-cxr-dataset`.
3. **Execute**: Open [label-generator.ipynb](file:///d:/proejects/Multimodal%20Medical%20Foundation%20Model/label_generation/label-generator.ipynb) in a Jupyter server or upload it to a Kaggle notebook and click **Run All**.
4. **Outputs**: The generated CSV files (`mimic_final_labeled_train.csv` and `mimic_final_labeled_validate.csv`) will be saved directly into `/kaggle/working` (which corresponds to your local `labeled_mimic_cxr/` directory during integration).
