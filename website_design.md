# Website / UI Design (Foundation Model Demo)

## 1. IA & Pages

1. **Home**
   - Hero: project name + short tagline about multimodal medical foundation model. [file:252]
   - Sections:
     - “Why multimodal?” (diagram: image + text + structured + knowledge).
     - “Research gaps this project addresses” (missing modality, explainability, external knowledge).

2. **Demos**
   - Tab 1: “Generate Draft Report”.
   - Tab 2: “Analyze Existing Report”.

3. **Models & Research**
   - High‑level description of:
     - Image‑only model.
     - Multimodal fusion model.
     - RAG pipeline.
   - Plots and tables: AUROCs, ablations.

4. **Architecture**
   - Diagrams showing modalities and fusion (pipes: X‑ray, report, metadata, knowledge). [file:252]

5. **About / Team / Contact**

---

## 2. Key Screen: Home

- Clean hero:
  - Title: “Multimodal Medical Foundation Model”.
  - Subtitle: “Vision + Language + Structured Data + Knowledge with missing‑modality reasoning and explainability.” [file:252]
  - Buttons:
    - “Try Draft Report Demo”
    - “View Architecture”

- Explainer section:
  - Side‑by‑side panels:
    - “Image‑only models” vs
    - “Our proposed foundation model: image + report + metadata + knowledge” (small table). [file:252]

---

## 3. Key Screen: Draft Report Demo

Layout:

- Left card:
  - X‑ray upload.
  - Simple metadata (age, sex, view).
  - “Run model” button.

- Right panel (tabs):
  - Predictions (list + bar chart).
  - Grad‑CAM overlay image.
  - Draft report text.
  - RAG knowledge snippets with citations.

---

## 4. Visual Style

- Professional, research‑lab style:
  - Neutral background.
  - One accent color (teal/blue).
  - Emphasis on diagrams and heatmaps.
- Typography:
  - Simple, readable sans‑serif.
- Mobile‑friendly layout.