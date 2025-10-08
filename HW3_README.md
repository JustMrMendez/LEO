
# HW3 Runner (CAP5610 • Fall 2025)

## Files
- `HW3_Mendez.ipynb` — ready-to-run notebook (single cell). Open it and run.
- `HW3_Mendez.py` — same logic as a script.
- Outputs folder: `hw3_outputs/` — model tables, confusion matrix, SHAP CSVs, and SHAP force-plot HTMLs.

> **Before submitting to Canvas:** rename files to match the spec, e.g. `HW3_<YourLastName>.ipynb` and `HW3_<YourLastName>.py`.

## What you need locally
- Python 3.9+
- Packages: `pandas`, `numpy`, `scikit-learn`.
- Optional (recommended for best results): `xgboost`, `lightgbm`, `catboost`, `shap`.

Install (example):
```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost shap
```

## Data placement
- Put the classification CSV next to the notebook/script and name it: `lncRNA_5_Cancers.csv`.
- Put the regression CSV next to it and name it: `GDSC2_13drugs.csv` (download from your course Module 2 link).

## How to run
**Notebook:**
1. Open `HW3_Mendez.ipynb`.
2. Edit `Author` comment / rename file as needed.
3. If you have more memory, optionally bump `MAX_FEATURES_CLASSIF` and `MAX_FEATURES_REGRESS`.
4. Run the single cell. Artifacts land in `hw3_outputs/`.

**Script:**
```bash
python HW3_Mendez.py
```

## What the code does
- **Task 1:** Trains DecisionTree, RandomForest, GBM (+ XGBoost/LightGBM/CatBoost if installed) on a memory-capped subset of features. Reports Accuracy & F1-macro. Saves confusion matrix + classification report.
- **Task 2:** Uses SHAP on the best classifier to produce:
  - (a) Top-10 features per cancer via mean(|SHAP|) aggregation.
  - (b) 5 interactive *force plots* (HTML) for the requested patient `TCGA-39-5011-01A` (or a fallback if missing).
- **Task 3:** Trains the same family of regressors on GDSC2 (target: `LN_IC50`). Reports MAE, MSE, RMSE, R².
- **Task 4:** SHAP on the best regressor:
  - (a) Top-10 features per drug (by mean |SHAP|).
  - (b) Top-10 features for the drug–cell-line pair with the **least absolute prediction error**.

## Report guidance (what to include)
Follow the Canvas instructions exactly:
1. **Algorithms/tools used** — what/why/how for each model, SHAP basics, and when to use TreeExplainer.
2. **Results** — numbered Figures/Tables with titles; describe and interpret the metrics & plots; state takeaways and conclusions (e.g., which model wins and why).
3. **Repro** — note your environment, versions, random seed, and any caps you used (`MAX_FEATURES_*`).

## Troubleshooting
- **Out of memory?** Lower `MAX_FEATURES_CLASSIF/REGRESS` (e.g., 2000 or 1000). Tree models don't need scaling.
- **SHAP too slow?** Lower `SHAP_SAMPLES_PER_CLASS` (e.g., 20). Force plots are still instant for a single row.
- **Missing XGBoost/LightGBM/CatBoost/SHAP?** The code runs without them and still compares the others; install later and re-run to upgrade results.
