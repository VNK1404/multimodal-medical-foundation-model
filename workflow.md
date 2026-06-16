# Project Workflows (Aligned with Concept Note)

## 1. Research & Data Workflow

1. **Literature review**
   - Multimodal medical models (image+text+EHR).
   - Missing‑modality learning.
   - Medical RAG and clinical LLMs. [file:252]

2. **Data access & preprocessing**
   - Phase 1: CheXpert‑v1.0‑small from Kaggle (image+labels).
   - Phase 2: CheXpert Plus / MIMIC‑CXR (if available) for image+report+metadata.

3. **Index building**
   - Build a unified `study_index` linking image, labels, optional text, metadata, RadGraph‑like annotations.

4. **External knowledge curation**
   - Collect open‑access text (guidelines, articles, knowledge bases). [file:252]
   - Clean, chunk, embed, and index.

---

## 2. Model Development Workflow

1. Implement and validate **image‑only model**.
2. Implement text encoder and metadata encoder; run small experiments.
3. Design fusion transformer and implement missing‑modality masking.
4. Integrate RadGraph‑style features where available.
5. Train and compare:
   - Image‑only
   - Image+Text
   - Image+Text+Metadata
   - Full + RAG‑assisted reasoning. [file:252]

---

## 3. Inference Workflows

### A. Generate Draft Report

1. User uploads X‑ray (+optional metadata).
2. Backend:
   - Image encoder → predictions.
   - Create context → retrieve medical knowledge.
   - Generate report sections (Findings, Impression) using fusion + RAG.
   - Produce Grad‑CAM maps and textual highlights.
3. Frontend shows results and allows downloading a summary.

### B. Analyze Existing Report

1. User uploads X‑ray + report.
2. Backend:
   - Encode both modalities.
   - Predict pathologies and compare with report statements.
   - Use RAG to fetch supporting/contradicting evidence.
   - Generate a discrepancy summary.
3. Frontend visualizes agreement/disagreement and explanations.

---

## 4. Team & Dev Workflow

- GitHub with `main`, `dev`, and `feature/*`.
- Issues grouped by:
  - Data.
  - Image model.
  - Multimodal.
  - RAG.
  - API/UI.
- Short experiment notes per run; results summarized in a shared doc.