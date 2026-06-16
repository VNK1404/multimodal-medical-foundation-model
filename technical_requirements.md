# Technical Requirements Document (TRD)
## Multimodal Medical Foundation Model

### 1. Modalities

The system supports four modality types inspired by the concept note: [file:252]

1. **Medical Images**
   - Chest X‑rays (PNG/JPEG, optionally DICOM samples).

2. **Clinical Text**
   - Radiology reports, doctor notes.

3. **Structured Data**
   - Age, sex, view position, simple vitals or flags (e.g., ICU/not).

4. **External Medical Knowledge**
   - Retrieved on‑the‑fly via RAG from:
     - Guidelines.
     - Open‑access articles.
     - Curated knowledge bases. [file:252]

---

### 2. Architecture Overview

Layers:

- Encoders:
  - Image encoder (CNN or vision transformer).
  - Text encoder (Bio/clinical transformer).
  - Metadata encoder (MLP).
  - Optional graph encoder for RadGraph‑style entities/relations. [file:252]

- Fusion:
  - Transformer‑based fusion block taking modality embeddings + modality masks (for missing‑modality reasoning).

- Heads:
  - Classification head (multi‑label pathology prediction).
  - Report generation head (encoder‑decoder or instruction‑tuned LLM).
  - Explanation head aggregating attribution signals.

- RAG:
  - Text encoder for knowledge chunks.
  - Vector index and retriever.

- API:
  - FastAPI service exposing inference endpoints.

---

### 3. Tech Stack

- **Python:** 3.10+
- **DL Framework:** PyTorch + optionally Lightning.
- **NLP:** Hugging Face transformers (BioClinicalBERT / BioGPT or similar).
- **Vision:** torchvision models, timm, or specialized medical models.
- **RAG:** FAISS or ChromaDB for vector search.
- **Backend:** FastAPI + Uvicorn.
- **Frontend:** React/Next.js or minimal HTML/CSS/JS.
- **Metrics:** scikit‑learn; wandb or TensorBoard for logging.

---

### 4. Image‑Only Model Requirements

- Input: frontal chest X‑ray images from CheXpert‑v1.0‑small.
- Preprocessing:
  - Resize to 224×224 or 320×320.
  - Normalize with dataset stats.
  - Augmentations: horizontal flip, small rotation, slight crop, brightness/contrast adjustments.

- Model:
  - DenseNet121 or EfficientNet baseline, pretrained.
  - Output: logits for 5 or 14 labels.

- Training:
  - BCEWithLogitsLoss with optional class weighting.
  - AdamW optimizer, cosine or step LR scheduler.
  - Mixed precision when GPU available.

- Metrics:
  - Per‑label AUROC, macro AUROC, macro AUPRC, loss curves.

- Explainability:
  - Grad‑CAM (or Grad‑CAM++) for selected labels.

---

### 5. Multimodal Model Requirements

Inputs:

- `z_img`: image embedding.
- `z_txt`: text embedding (report).
- `z_meta`: metadata embedding.
- `z_graph` (optional): entity/relation embedding.

Fusion module:

- Accepts a set of tokens:
  - [IMG], [TXT_1..N], [META], [GRAPH] with modality masks.
- Uses cross‑attention / transformer layers.
- Outputs fused representation for classification and generation.

Missing‑modality handling:

- Modality presence vector (e.g., `[1,0,1,0]`).
- Fusion can be trained with random modality dropout to learn robustness. [file:252]

---

### 6. RAG Requirements

- Corpus: curated external medical text (diseases relevant to CheXpert labels, radiology patterns, guidelines). [file:252]
- Chunking: 256–512 tokens per chunk.
- Embedding: domain‑appropriate sentence embedding model.
- Index: FAISS/Chroma stored on disk.
- Retrieval:
  - Use patient context (findings text, predictions) to form a query.
  - Retrieve top‑k passages (k≈3–5).
- Integration:
  - Provide retrieved passages as additional context to the generation head and explanations.

---

### 7. API & Security

- All endpoints validate file size & type.
- No raw dataset or PHI exposure.
- Timeouts/logging for long‑running operations.
- Rate limiting (basic) to avoid misuse.

Endpoints:

- `/v1/image/predict`
- `/v1/multimodal/predict`
- `/v1/report/generate`
- `/v1/report/analyze`

All return probabilities, structured outputs, and explanation artifacts.

---

### 8. Non‑Functional Requirements

- Reproducibility: seed control and config files.
- Modularity: separate `data/`, `models/`, `fusion/`, `rag/`, `api/`.
- Tests: unit tests for key components.
- Performance: batch inference optimized, GPU support.