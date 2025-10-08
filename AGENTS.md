# Repository Guidelines

## Project Structure & Module Organization
Keep the raw datasets `lncRNA_5_Cancers.csv` and `hw3-drug-screening-data.csv` tracked under `data/raw/`; store the assignment brief `Fall2025-HW3.pdf` in `docs/` for quick reference. Derive features and train/test splits into `data/processed/` using descriptive suffixes like `*_fold1.parquet`. Implementation modules should live in `src/leo/` (add `__init__.py` for package imports), while exploratory notebooks belong in `notebooks/` and must rely on helper loaders in `src/leo/data.py` instead of hard-coded paths. Log figures and SHAP visualizations to `reports/figures/`, and persist trained models as timestamped artifacts inside `models/`.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` creates the local environment shared across experiments.
- `pip install -r requirements.txt` installs dependencies; keep scikit-learn, shap, xgboost, lightgbm, catboost, pandas, and seaborn pinned here.
- `jupyter lab` launches the exploratory workspace used for classifier/regressor prototyping.
- `python -m src.leo.experiments tree --config configs/baseline.yaml` runs scripted pipelines; prefer module execution so relative imports resolve cleanly.
- `pytest` executes all automated tests; combine with `pytest -k shap --maxfail=1` while iterating on interpretability utilities.

## Coding Style & Naming Conventions
Target Python 3.11, four-space indentation, and `black` formatting (`black src tests`). Lint with `ruff check` and run static type checks using `mypy src`. Name modules and files in snake_case (`src/leo/models/random_forest.py`), classes in CapWords, and experiment configs as `configs/{task}_{model}.yaml`. Prefer dataclasses for parameter groups and add concise docstrings that cite homework task numbers.

## Testing Guidelines
Author tests with `pytest`, mirroring module names (`tests/test_random_forest.py`). Use fixtures to load small excerpts of the CSVs and avoid loading entire files in unit tests. Ensure new models define deterministic seeds and include SHAP regression tests via `pytest tests/test_shap_regression.py --cov=src/leo`. Maintain coverage at or above 85% by default.

## Commit & Pull Request Guidelines
Follow Conventional Commits (`feat(model): add lightgbm regressor`) and keep subjects under 72 characters. Each PR should include a short task summary, references to Canvas requirements, and screenshots of key SHAP plots when visuals change. Request review only after linting and `pytest` pass locally, and note any large file additions or data provenance updates in the description.

## Reproducibility & Data Handling
Never commit new raw patient data without confirming anonymization and license compliance. Record preprocessing steps in `docs/changelog.md`, including random seeds and library versions. When models or notebooks depend on external storage, provide `.env.example` entries and keep secrets out of version control.
