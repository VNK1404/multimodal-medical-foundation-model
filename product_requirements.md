# Product Requirements Document (PRD)
## Project: Multimodal Medical Foundation Model

### 1. Summary

We will build a **multimodal medical foundation model** that can:

- Understand chest X‑ray images.
- Understand clinical text (radiology reports, doctor notes).
- Use basic structured EHR data (age, sex, view position, simple vitals).
- Retrieve external medical knowledge using RAG.
- Handle missing modalities at inference time.
- Provide explainable predictions and reports. [file:252]

The system is a **research prototype** and portfolio project, not a clinical device.

---

### 2. Problem & Motivation

Most current healthcare AI systems are:

- Image → Diagnosis (e.g., chest X‑ray classifier), or
- Image + Text → Diagnosis (vision–language models). [file:252]

Real clinical decision‑making uses:

- X‑rays, CT, MRI.
- Radiology reports and clinical notes.
- Patient demographics, vitals, labs, history.
- Clinical guidelines and research papers. [file:252]

Key gaps:

- Limited to image+text, poor integration of full EHR.
- Fragile when one modality is missing (e.g., report not yet written).
- Weak explainability (no clear “why”).
- Little use of external knowledge during inference. [file:252]

---

### 3. Goals

- G1: Strong **image‑only** chest X‑ray baseline on CheXpert‑v1.0‑small.
- G2: Multimodal model combining image + report + simple metadata.
- G3: Missing‑modality robustness (train with all; test with subsets).
- G4: Explainable outputs (Grad‑CAM for images, highlighted phrases for text, feature importances for metadata).
- G5: RAG pipeline that injects external medical knowledge into explanations and draft reports. [file:252]
- G6: Web demo for:
  - “Generate Draft Report” from image (+metadata).
  - “Analyze Existing Report” from image + report.

---

### 4. Non‑Goals

- Clinical deployment or regulatory approval.
- Commercial use (datasets are non‑commercial research only).
- Full EHR integration (we focus on a small, research‑friendly subset).
- Handling all imaging modalities (scope: chest X‑ray first). [file:252]

---

### 5. Target Users

- Student / early‑career AIML engineers (you + team).
- Academic researchers exploring multimodal medical AI.
- Recruiters/professors reviewing your portfolio.

---

### 6. Key User Stories

1. **Researcher** trains an image‑only CheXpert classifier, inspects AUROC/AUPRC and Grad‑CAM maps.
2. **Researcher** trains a multimodal fusion model, compares performance vs image‑only and under missing modality scenarios.
3. **User** uploads X‑ray (and optional metadata) and gets:
   - Predicted findings.
   - Draft radiology report (Findings, Impression).
   - Heatmaps + highlighted support sentences + knowledge snippets.
4. **User** uploads X‑ray + existing report and gets:
   - Agreement/disagreement analysis.
   - Explanations and evidence for discrepancies.

---

### 7. High‑Level Features

- F1: Image‑only classifier (DenseNet/EfficientNet).
- F2: Text encoder for radiology reports.
- F3: Structured metadata encoder (age, sex, basic tags).
- F4: RadGraph‑style entity/relation features (where available).
- F5: Fusion module that supports missing‑modality masks.
- F6: RAG engine over curated medical text (guidelines, textbooks, open articles). [file:252]
- F7: Explainability tools (visual + textual).
- F8: FastAPI backend + simple web UI.

---

### 8. Success Metrics

- Image‑only model: competitive AUROC on CheXpert 5 competition labels.
- Multimodal model: improved macro AUROC vs image‑only.
- Missing‑modality experiments: clearly quantified performance drop and robustness.
- Qualitative: explanations and knowledge snippets look clinically reasonable.

---

### 9. Dataset Strategy

Concept paper suggests **MIMIC‑CXR as primary multimodal dataset**. [file:252]

Practical plan:

- Stage 1 (confirmed): CheXpert‑v1.0‑small (Kaggle) as main image dataset.
- Stage 2 (optional if access/time allow): partial MIMIC‑CXR for image+report+metadata, or use CheXpert Plus + RadGraph for multimodal. [file:252]
- External text for RAG from open medical resources (respecting each source’s license).