# Project Rules & Conventions (Multimodal Foundation Model)

## 1. Data & Licensing

- Use MIMIC‑CXR, CheXpert, CheXpert Plus, and RadGraph‑style annotations **only** under their research‑only, non‑commercial licenses. [file:252]
- Do not publish or commit raw data or direct download links.
- All examples in reports must be anonymized and non‑identifiable.

## 2. Scope Discipline

- Mandatory:
  - Image‑only baseline.
  - Image+text+metadata fusion with missing‑modality experiments.
  - Basic RAG and explanation demo.
- Optional (time‑dependent):
  - Full RadGraph graph encoder.
  - Fancy UI styling.
- Avoid scope creep (e.g., adding CT, MRI, or complex EHR).

## 3. Coding & Structure

- Use clear module separation:
  - `data/`, `models/`, `fusion/`, `rag/`, `api/`, `ui/`.
- Use configs instead of hardcoding hyperparameters.
- Follow PEP8 and type hints where feasible.

## 4. Experiments

- Every experiment:
  - Has a config file in `configs/`.
  - Logs metrics and plots into `outputs/<exp_name>/`.
  - Includes a short markdown note of findings.

## 5. Reviews & PRs

- All significant changes go through PRs.
- At least one teammate reviews each PR.
- Tests for core functionality must pass before merging.

## 6. Safety & Ethics

- Make it clear in README and UI:
  - “Research prototype, not for clinical use.”
- No clinical decisions should be based on this system.
- Use RAG sources that are reputable and permitted for such use (guidelines, textbooks, open‑access papers). [file:252]

## 7. Documentation

- Keep PRD, TRD, and schema files updated as design evolves.
- Major design decisions should be recorded in a simple `DECISIONS.md` log.