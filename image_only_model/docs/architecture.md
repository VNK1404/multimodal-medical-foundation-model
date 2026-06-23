# Architecture

## High-Level System Diagram

```mermaid
flowchart LR
    A[Chest X-ray Images] --> B[Image Encoder]
    C[Radiology Reports] --> D[Text Encoder]
    E[Structured EHR Metadata] --> F[Metadata Projection]
    B --> G[Multimodal Fusion]
    D --> G
    F --> G
    G --> H[Classifier / Risk Head]
    G --> I[RAG Module]
    I --> J[Knowledge Retrieval]
    J --> K[Report Generation]
```

The system ingests chest X-ray images, radiology reports, and structured EHR metadata through dedicated encoders. These features are fused into a shared representation that feeds a classifier, retrieval-augmented generation pipeline, and report generation components.

## Planned Phases

- Data ingestion, cleaning, and indexing.
- Baseline unimodal and simple fusion experiments.
- Multimodal fusion training with missing-modality robustness.
- RAG integration for knowledge-grounded report generation.
- Explainability, calibration, and uncertainty analysis.
- Evaluation, packaging, and deployment preparation.
