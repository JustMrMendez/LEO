
# HW3 Report — CAP5610 (Fall 2025)
**Student:** <Your Name Here>  
**File Name (per spec):** `HW3_<YourLastName>`

## 1. Algorithms / Approaches / Tools
For each algorithm (Decision Tree, Random Forest, GBM, XGBoost, LightGBM, CatBoost) and SHAP:
- **What it is / What it does**
- **How it works (high-level mechanics)**
- **Application in this assignment**

*Example (Random Forest):*
- **What/Does:** Ensemble of decision trees voting on class; reduces variance.
- **How:** Bagging + random feature subspace; majority vote (classification).
- **Application:** Baseline strong classifier for high-dimensional gene expression; robust to noise; little preprocessing.

*(Repeat similarly for others.)*

## 2. Results

### 2.1 Classifiers (Task 1)
**Table 1.** Model comparison on held-out test set (Accuracy & F1-macro).  
_Insert `task1_model_comparison.csv` as a table._

**Figure 1.** Confusion matrix (test set).  
_Insert `task1_confusion_matrix.csv` (render as figure)._

**Table 2.** Classification report (per-class precision/recall/F1).  
_Insert `task1_classification_report.csv`._

**Observations & Takeaways:**
- Which model wins and by how much?
- Any class imbalance behavior you notice (e.g., macro F1 vs Accuracy)?
- Possible reasons (bias-variance trade-off, overfitting, depth/leaf size, etc.).

### 2.2 SHAP on Best Classifier (Task 2)
**Table 3.** Top-10 features per cancer (mean |SHAP|).  
_Insert `task2a_top10_features_per_cancer.csv`._

**Figure 2a–2e.** Force plots for patient `TCGA-39-5011-01A` for KIRC, LUAD, LUSC, PRAD, THCA.  
_Attach 5 HTMLs or screenshots from `task2b_forceplot_*.html`._

**Observations & Biological Intuition (if any):**
- Feature patterns that distinguish cancers?
- Are top features shared or cancer-specific?

### 2.3 Regressors (Task 3)
**Table 4.** Model comparison (MAE, MSE, RMSE, R²).  
_Insert `task3_regressor_comparison.csv`._

**Observation:** Which model fits dose-response best? Any signs of over/underfitting?

### 2.4 SHAP on Best Regressor (Task 4)
**Table 5.** Top-10 features per drug (mean |SHAP|).  
_Insert `task4a_top10_features_per_drug.csv`._

**Table 6.** Top-10 features for least-error drug–cell-line pair.  
_Insert `task4b_top10_features_least_error_<pair>.csv`._

**Observations:** Consistent biomarkers across drugs? Any drug-specific signatures?

## 3. Conclusion
- One-paragraph summary: best classifier & regressor; key insights from SHAP; practical implications.

## 4. Reproducibility
- Environment: Python version, package versions.
- Random seed: 42.
- Any caps/tweaks: `MAX_FEATURES_*`, `SHAP_SAMPLES_*`.
- Command(s) to run.

---

> **Submission checklist (Canvas):**
> - Report (this file as PDF or Word).
> - Source code (`.py` or notebook).
> - File names follow `HW3_<LastName>`.
> - Files are uploaded separately (no ZIP).
