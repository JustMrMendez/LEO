# Repository Guidelines

This repository contains the self-contained research notebook `HW3.ipynb` plus the two datasets and cached artefacts required to reproduce the CAP5610 HW3 experiments on tree ensembles and SHAP interpretation.

## Contents
- `HW3.ipynb` – primary notebook with installation cell, experiment tracker, optimisation workflow, and reporting blocks.
- `lncRNA_5_Cancers.csv` – classification dataset used in Tasks 1–2.
- `hw3-drug-screening-data.csv` – regression dataset (alias `GDSC2_13drugs.csv`) used in Tasks 3–4.
- `hw3_outputs/` – cached checkpoints, SHAP results, and generated CSV/HTML artefacts from the latest run (can be deleted for a cold start).
- `catboost_info/` – CatBoost training logs produced during notebook execution.
- `docs/` – assignment PDF and supporting material for reference.

## Getting Started
1. Launch Jupyter (or a compatible environment such as Google Colab or Vertex Workbench).
2. Open `HW3.ipynb` and run the cells sequentially. The first code cell installs all Python dependencies via `%pip`, so no additional setup is required.
3. Ensure both CSV datasets remain alongside the notebook; the data loaders resolve paths relative to the notebook directory.

## Notebook Features
- **Experiment Automation:** A progress tracker with checkpoints allows you to resume long runs and logs per-step timing/memory metrics.
- **Optimised Data & Model Pipeline:** Uses Polars-backed ingestion (when available), NumPy variance pruning, and joblib parallelism for classifier/regressor sweeps.
- **Robust SHAP Analysis:** Caches sampled SHAP backgrounds, gracefully falls back between TreeExplainer perturbation modes, and renders all required force plots directly in the notebook.
- **Reporting:** Generates CSV summaries and HTML visuals in `hw3_outputs/` while the final section surfaces runtime tables and compliance with assignment deliverables.

## Regenerating Artefacts
If you need a fresh run without cached speed-ups, delete `hw3_outputs/` before executing the notebook. The experiment tracker will recreate the directory, checkpoints, and reports from scratch.

## Support
Refer to `docs/Fall2025-HW3.pdf` for the original assignment brief. For questions about the optimisation framework or extending the notebook (e.g., GPU acceleration, feature store integration), start from the “Literature & Benchmark Survey” section inside the notebook.
