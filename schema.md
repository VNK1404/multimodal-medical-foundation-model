# Schemas (Data, RAG, API) – Foundation Model

## 1. Study Index Schema

Table: `study_index` (per image/study)

- `patient_id`: string
- `study_id`: string
- `image_path`: string
- `labels`: vector<float>[n_labels]
- `report_text`: string (nullable)
- `radgraph_json_path`: string (nullable)
- `age`: int (nullable)
- `sex`: string (nullable)
- `view_position`: string (e.g., AP/PA)
- `split`: string {train, val, test}
- `source_dataset`: string {"CheXpert", "MIMIC-CXR", "Other"} [file:252]

Stored as Parquet or CSV.

---

## 2. Knowledge Chunk Schema (RAG)

Table: `knowledge_chunks`

- `chunk_id`: string
- `source`: string (e.g., guideline name, PubMed ID).
- `section`: string (e.g., “Etiology”, “Imaging Findings”).
- `text`: string
- `embedding`: vector<float>[d]
- `tags`: list<string> (e.g., ["Edema", "Pleural Effusion"]). [file:252]

---

## 3. Model Config Schema (YAML/JSON)

Example fields:

- `image_model`:
  - `backbone`: "densenet121"
  - `pretrained`: true
  - `input_size`: 224
- `multimodal`:
  - `use_text`: true
  - `use_metadata`: true
  - `use_graph`: false
  - `fusion_layers`: 4
  - `hidden_dim`: 512
- `training`:
  - `epochs`: 20
  - `batch_size`: 32
  - `lr`: 1e-4
  - `missing_modality_dropout`: 0.2

---

## 4. API Schemas (Simplified)

### `/v1/image/predict`

**Request:**

```json
{
  "image": "<file>",
  "return_explanations": true
}
```

**Response:**

```json
{
  "predictions": [
    {"label": "Edema", "probability": 0.81},
    {"label": "Pleural Effusion", "probability": 0.35}
  ],
  "grad_cam": "<image_or_url>",
  "meta": {"model_version": "img_v1.0"}
}
```

---

### `/v1/multimodal/predict`

**Request:**

```json
{
  "image": "<file>",
  "report_text": "string (optional)",
  "metadata": {"age": 65, "sex": "M", "view": "PA"},
  "return_rag": true
}
```

**Response:**

```json
{
  "predictions": [...],
  "explanations": {
    "grad_cam": "<image_or_url>",
    "highlighted_report_html": "<html>",
    "metadata_importance": {...}
  },
  "knowledge": [
    {"source": "GuidelineX", "text": "snippet...", "score": 0.92}
  ]
}
```