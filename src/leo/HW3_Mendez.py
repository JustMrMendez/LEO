
# HW3 — CAP5610 Fall 2025
# Tree-based ML for Classification/Regression + SHAP
# Author: <Your Last Name Here>
# File name convention: HW3_lastName  (adjust the file names before submission)
#
# Tasks (from assignment PDF):
#  Task 1: Best tree-based classifier among: DecisionTree, RandomForest, GBM, XGBoost, LightGBM, CatBoost.
#          Metrics: Accuracy, F1.
#  Task 2: SHAP on best classifier: (a) per-cancer top-10 features, (b) force plots for ID=TCGA-39-5011-01A across 5 cancer types.
#  Task 3: Best tree-based regressor among the same algorithms on GDSC2 13 drugs dataset (LN_IC50 target).
#          Metrics: MAE, MSE, RMSE, R2.
#  Task 4: SHAP on best regressor: (a) per-drug top-10 features, (b) top-10 features for the least-error drug–cell-line pair.
#
# NOTE: This script is optimized for memory. For very wide matrices (many genes), set MAX_FEATURES to a safe cap.
#       For XGBoost/LightGBM/CatBoost you need those packages installed in your environment.

import os
import re
import json
import math
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, classification_report, confusion_matrix,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor


# Optional libs (install if missing):
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAVE_LGBM = True
except Exception:
    HAVE_LGBM = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAVE_CAT = True
except Exception:
    HAVE_CAT = False

try:
    import shap
    HAVE_SHAP = True
except Exception:
    HAVE_SHAP = False


# -------------------- CONFIG --------------------
RANDOM_STATE = 42
CANCER_SET = {"KIRC","LUAD","LUSC","PRAD","THCA"}
PATIENT_ID_TO_PLOT = "TCGA-39-5011-01A"
CANCER_CSV = "lncRNA_5_Cancers.csv"        # Put the file alongside this script/notebook or change to absolute path
GDSC2_CSV   = "GDSC2_13drugs.csv"          # Provide locally (download from the course Module 2 / link)
OUT_DIR = Path("hw3_outputs")               # Created if missing
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Memory safety knobs
MAX_FEATURES_CLASSIF = 4000   # cap feature count (columns) to avoid OOM on laptops
MAX_FEATURES_REGRESS = 4000
SHAP_SAMPLES_PER_CLASS = 50   # number of rows per cancer for SHAP mean|SHAP| aggregation (Task 2a)
SHAP_SAMPLES_REG = 200        # number of rows for regressor SHAP aggregation (Task 4a)

# -------------------- UTILITIES --------------------
def detect_id_and_target(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """Detect a TCGA-like ID column and a 5-class cancer target column from a small sample."""
    id_col = None
    target_col = None
    for c in df.select_dtypes(include=['object']).columns:
        if df[c].astype(str).str.contains(r"^TCGA-", na=False).any():
            id_col = c
            break
    if id_col is None:
        for c in df.columns:
            if re.search(r"(id|patient|sample)", c, re.I):
                if df[c].nunique(dropna=True) > 10:
                    id_col = c
                    break
    for c in df.columns:
        vals = set(map(str, df[c].dropna().unique()))
        if vals.issubset(CANCER_SET) and len(vals) == 5:
            target_col = c
            break
    if target_col is None:
        for c in df.columns:
            if re.search(r"(cancer|type|label|class)", c, re.I):
                target_col = c
                break
    return id_col, target_col


def memory_savvy_read_cancers(csv_path: str, max_features: int) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series], str, Optional[str]]:
    """Read only header and a tiny sample to detect id/target; then load a capped subset of features."""
    header_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    sample_df = pd.read_csv(csv_path, nrows=200)
    id_col, target_col = detect_id_and_target(sample_df)
    if target_col is None:
        raise RuntimeError("Could not detect the target column from sample.")

    feature_cols = [c for c in header_cols if c not in {id_col, target_col}]
    selected_feature_cols = feature_cols[:max_features]
    usecols = [target_col] + ([id_col] if id_col else []) + selected_feature_cols

    df_part = pd.read_csv(csv_path, usecols=usecols)
    y = df_part[target_col].astype(str)
    X = df_part.drop(columns=[target_col])

    ids = None
    if id_col and id_col in X.columns:
        ids = X[id_col].astype(str)
        X = X.drop(columns=[id_col])

    # Coerce to numeric and drop all-NaN columns
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.astype(np.float32)
    X = X.loc[:, X.notna().any(axis=0)]
    return X, y, ids, target_col, id_col


def train_compare_classifiers(X: pd.DataFrame, y: pd.Series, random_state=RANDOM_STATE) -> Tuple[pd.DataFrame, Dict[str, Pipeline], Dict[int, str]]:
    """Train all required classifiers with efficient defaults; return test metrics and fitted models.

    Returns
    -------
    metrics_df : pd.DataFrame
        Test-set comparison table.
    fitted : Dict[str, Pipeline]
        Pipelines keyed by model name.
    idx_to_class : Dict[int, str]
        Mapping from encoded integer labels back to original class names.
    """

    class_names = sorted(y.astype(str).unique())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    y_encoded = y.map(class_to_idx).astype(int)

    X_train, X_test, y_train_enc, y_test_enc, y_train_lbl, y_test_lbl = train_test_split(
        X,
        y_encoded,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    preprocess = ColumnTransformer([("num", SimpleImputer(strategy="median"), X.columns)], remainder="drop")
    models = {}

    models["DecisionTree"] = Pipeline([
        ("prep", preprocess),
        ("clf", DecisionTreeClassifier(random_state=random_state, min_samples_leaf=2, class_weight="balanced"))
    ])

    models["RandomForest"] = Pipeline([
        ("prep", preprocess),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1, class_weight="balanced_subsample"))
    ])

    models["GBM"] = Pipeline([
        ("prep", preprocess),
        ("clf", GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=random_state))
    ])

    if HAVE_XGB:
        models["XGBoost"] = Pipeline([
            ("prep", preprocess),
            ("clf", XGBClassifier(
                objective="multi:softprob", eval_metric="mlogloss",
                n_estimators=500, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=random_state
            ))
        ])

    if HAVE_LGBM:
        models["LightGBM"] = Pipeline([
            ("prep", preprocess),
            ("clf", LGBMClassifier(n_estimators=700, learning_rate=0.05, num_leaves=63, subsample=0.8, colsample_bytree=0.8, random_state=random_state))
        ])

    if HAVE_CAT:
        models["CatBoost"] = Pipeline([
            ("prep", preprocess),
            ("clf", CatBoostClassifier(iterations=600, learning_rate=0.05, depth=6, loss_function="MultiClass", random_seed=random_state, verbose=False))
        ])

    rows = []
    fitted = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train_enc)
        y_pred_enc = pipe.predict(X_test)
        y_pred_enc = np.asarray(y_pred_enc, dtype=int)
        y_pred = [idx_to_class[int(i)] for i in y_pred_enc]
        acc = accuracy_score(y_test_lbl, y_pred)
        f1m = f1_score(y_test_lbl, y_pred, average="macro")
        rows.append({"Model": name, "Test_Accuracy": acc, "Test_F1_Macro": f1m})
        fitted[name] = pipe

    res = pd.DataFrame(rows).sort_values(by=["Test_F1_Macro","Test_Accuracy"], ascending=False).reset_index(drop=True)

    # Confusion matrix of best
    best_name = res.iloc[0]["Model"]
    best_model = fitted[best_name]
    y_pred_best_enc = best_model.predict(X_test)
    y_pred_best_enc = np.asarray(y_pred_best_enc, dtype=int)
    y_pred_best = [idx_to_class[int(i)] for i in y_pred_best_enc]
    cm = confusion_matrix(y_test_lbl, y_pred_best, labels=class_names)
    cm_df = pd.DataFrame(
        cm,
        index=[f"True_{c}" for c in class_names],
        columns=[f"Pred_{c}" for c in class_names],
    )
    cm_df.to_csv(OUT_DIR/"task1_confusion_matrix.csv", index=True)
    pd.DataFrame(
        classification_report(
            y_test_lbl,
            y_pred_best,
            output_dict=True,
            zero_division=0,
            labels=class_names,
            target_names=class_names,
        )
    ).T.to_csv(OUT_DIR/"task1_classification_report.csv")

    res.to_csv(OUT_DIR/"task1_model_comparison.csv", index=False)
    with open(OUT_DIR/"task1_best_model.txt","w") as f:
        f.write(str(best_name))

    return res, fitted, idx_to_class


def shap_task2(
    best_model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    id_series: Optional[pd.Series],
    patient_id: str,
    idx_to_class: Optional[Dict[int, str]] = None,
):
    if not HAVE_SHAP:
        print("SHAP not available — install `shap` to run Task 2.")
        return
    estimator = best_model.named_steps[list(best_model.named_steps.keys())[-1]]
    explainer = shap.TreeExplainer(estimator)

    # (a) Top-10 per cancer
    df_full = X.copy()
    df_full["__y__"] = y.values
    idxs = []
    for cls in sorted(CANCER_SET):
        cls_idx = df_full.index[df_full["__y__"] == cls].tolist()
        if len(cls_idx) > SHAP_SAMPLES_PER_CLASS:
            rng = np.random.RandomState(RANDOM_STATE)
            cls_idx = list(rng.choice(cls_idx, SHAP_SAMPLES_PER_CLASS, replace=False))
        idxs.extend(cls_idx)
    idxs = sorted(set(idxs))
    X_shap = X.loc[idxs]
    y_shap = y.loc[idxs]

    shap_values = explainer.shap_values(X_shap)
    records = []
    if isinstance(shap_values, list):
        classes = list(estimator.classes_)
        for i, raw_cls in enumerate(classes):
            cls_name = idx_to_class.get(int(raw_cls), str(raw_cls)) if idx_to_class else str(raw_cls)
            vals = shap_values[i]
            mean_abs = np.abs(vals).mean(axis=0)
            top_idx = np.argsort(mean_abs)[::-1][:10]
            for rank, j in enumerate(top_idx, start=1):
                records.append({"CancerType": str(cls_name), "Rank": rank, "Feature": X_shap.columns[j], "Mean|SHAP|": float(mean_abs[j])})
    else:
        mean_abs = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:10]
        for rank, j in enumerate(top_idx, start=1):
            records.append({"CancerType": "ALL", "Rank": rank, "Feature": X_shap.columns[j], "Mean|SHAP|": float(mean_abs[j])})
    pd.DataFrame(records).to_csv(OUT_DIR/"task2a_top10_features_per_cancer.csv", index=False)

    # (b) Force plots for patient across 5 cancer types
    shap.initjs()
    # Try to find the patient row; fallback to first row
    if id_series is not None and id_series.notna().any():
        try:
            idx = id_series[id_series.astype(str) == patient_id].index[0]
        except Exception:
            idx = X.index[0]
    else:
        idx = X.index[0]
    x_row = X.loc[[idx]]
    shap_row = explainer.shap_values(x_row)

    def save_force_html(path, force_obj):
        try:
            shap.save_html(str(path), force_obj)
            return True
        except Exception:
            return False

    if isinstance(shap_row, list):
        classes = list(estimator.classes_)
        for i, raw_cls in enumerate(classes):
            cls_name = idx_to_class.get(int(raw_cls), str(raw_cls)) if idx_to_class else str(raw_cls)
            expected_values = explainer.expected_value
            if isinstance(expected_values, (list, np.ndarray)):
                expected = expected_values[i]
            else:
                expected = expected_values
            force = shap.force_plot(expected, shap_row[i][0, :], x_row, feature_names=x_row.columns, matplotlib=False)
            save_force_html(OUT_DIR/f"task2b_forceplot_{cls_name}_patient_{patient_id.replace(':','-')}.html", force)
    else:
        expected = explainer.expected_value
        force = shap.force_plot(expected, shap_row[0, :], x_row, feature_names=x_row.columns, matplotlib=False)
        save_force_html(OUT_DIR/f"task2b_forceplot_patient_{patient_id.replace(':','-')}.html", force)


# -------------------- REGRESSION (Tasks 3–4) --------------------

def memory_savvy_read_gdsc2(csv_path: str, max_features: int) -> Tuple[pd.DataFrame, pd.Series, pd.Series, Dict[str, int]]:
    """Load GDSC2 data assuming columns: ['cell_line','drug_name','LN_IC50', <gene features...>].
       Returns X (float32), y (float32), keys (cell_line|drug), and some metadata.
    """
    # Read once to find columns (no huge memory)
    header_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    # Heuristics for key/target columns:
    target_col = "LN_IC50"
    id_cols = []
    for cand in ["cell_line","CELL_LINE","CellLine","cellLine","cell_line_name"]:
        if cand in header_cols:
            id_cols.append(cand)
            break
    for cand in ["drug_name","Drug","DRUG","drug"]:
        if cand in header_cols:
            id_cols.append(cand)
            break
    if target_col not in header_cols:
        raise RuntimeError("Expected LN_IC50 column missing in GDSC2 CSV.")
    feature_cols = [c for c in header_cols if c not in id_cols+[target_col]]
    selected_feature_cols = feature_cols[:max_features]
    usecols = id_cols + [target_col] + selected_feature_cols

    df = pd.read_csv(csv_path, usecols=usecols)
    y = pd.to_numeric(df[target_col], errors="coerce").astype(np.float32)
    keys = df[id_cols[0]].astype(str) + "|" + df[id_cols[1]].astype(str) if len(id_cols)>=2 else df[id_cols[0]].astype(str)

    X = df.drop(columns=[target_col]+id_cols)
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.astype(np.float32)
    X = X.loc[:, X.notna().any(axis=0)]
    meta = {"n_rows": int(len(df)), "n_features": int(X.shape[1]), "id_cols": id_cols, "target": target_col}
    return X, y, keys, meta


def train_compare_regressors(X: pd.DataFrame, y: pd.Series, random_state=RANDOM_STATE) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    preprocess = ColumnTransformer([("num", SimpleImputer(strategy="median"), X.columns)], remainder="drop")

    models = {}
    models["DecisionTreeReg"] = Pipeline([("prep", preprocess), ("reg", DecisionTreeRegressor(random_state=random_state, min_samples_leaf=2))])
    models["RandomForestReg"] = Pipeline([("prep", preprocess), ("reg", RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1))])
    models["GBMReg"] = Pipeline([("prep", preprocess), ("reg", GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=random_state))])

    if HAVE_XGB:
        models["XGBReg"] = Pipeline([("prep", preprocess), ("reg", XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=random_state))])
    if HAVE_LGBM:
        models["LGBMReg"] = Pipeline([("prep", preprocess), ("reg", LGBMRegressor(n_estimators=800, learning_rate=0.05, num_leaves=63, subsample=0.8, colsample_bytree=0.8, random_state=random_state))])
    if HAVE_CAT:
        models["CatBoostReg"] = Pipeline([("prep", preprocess), ("reg", CatBoostRegressor(iterations=700, learning_rate=0.05, depth=6, loss_function="RMSE", random_seed=random_state, verbose=False))])

    rows = []
    fitted = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        mse = mean_squared_error(y_test, pred)
        rmse = math.sqrt(mse)
        r2 = r2_score(y_test, pred)
        rows.append({"Model": name, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2})
        fitted[name] = pipe

    res = pd.DataFrame(rows).sort_values(by=["RMSE","MAE"], ascending=[True, True]).reset_index(drop=True)
    res.to_csv(OUT_DIR/"task3_regressor_comparison.csv", index=False)

    best_name = res.iloc[0]["Model"]
    with open(OUT_DIR/"task3_best_model.txt","w") as f:
        f.write(str(best_name))

    return res, fitted


def shap_task4(best_reg_model: Pipeline, X: pd.DataFrame, y: pd.Series, keys: pd.Series):
    if not HAVE_SHAP:
        print("SHAP not available — install `shap` to run Task 4.")
        return

    est = best_reg_model.named_steps[list(best_reg_model.named_steps.keys())[-1]]
    explainer = shap.TreeExplainer(est)

    # (a) per-drug top-10 features (requires a 'drug' component in key: cell|drug)
    # Parse drug names if keys are "cell|drug"
    if keys.str.contains(r"\|").any():
        drug_names = keys.str.split("|").str[1]
    else:
        # If single ID, we can't group by drug
        drug_names = pd.Series(["ALL"]*len(keys), index=keys.index)

    # Subsample for speed
    rng = np.random.RandomState(RANDOM_STATE)
    sample_idx = rng.choice(X.index, size=min(SHAP_SAMPLES_REG, len(X)), replace=False)
    X_shap = X.loc[sample_idx]
    drugs_shap = drug_names.loc[sample_idx]

    shap_vals = explainer.shap_values(X_shap)
    # shap_vals shape: (n_samples, n_features) for regression
    assert isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 2

    # Compute per-drug mean|SHAP|
    recs = []
    for drug in sorted(drugs_shap.unique()):
        mask = (drugs_shap == drug).values
        if mask.sum() == 0:
            continue
        mean_abs = np.abs(shap_vals[mask]).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:10]
        for rank, j in enumerate(top_idx, start=1):
            recs.append({"Drug": drug, "Rank": rank, "Feature": X_shap.columns[j], "Mean|SHAP|": float(mean_abs[j])})
    pd.DataFrame(recs).to_csv(OUT_DIR/"task4a_top10_features_per_drug.csv", index=False)

    # (b) Least-error pair: compute prediction errors and take min absolute error
    preds = best_reg_model.predict(X)
    errors = np.abs(preds - y.values)
    idx_min = int(np.argmin(errors))
    least_key = keys.iloc[idx_min]
    # SHAP for that row
    x_row = X.iloc[[idx_min]]
    row_shap = explainer.shap_values(x_row)[0, :]
    mean_abs = np.abs(row_shap)
    top_idx = np.argsort(mean_abs)[::-1][:10]
    pd.DataFrame({
        "Rank": np.arange(1, 11),
        "Feature": X.columns[top_idx],
        "Absolute_SHAP": mean_abs[top_idx].astype(float)
    }).to_csv(OUT_DIR/f"task4b_top10_features_least_error_{least_key.replace('|','_')}.csv", index=False)


# -------------------- MAIN --------------------
if __name__ == "__main__":
    print("=== HW3 Runner ===")

    # ---- Tasks 1–2 (Classification) ----
    if os.path.exists(CANCER_CSV):
        print("[Task 1] Loading cancer CSV (memory-optimized)...")
        Xc, yc, ids, target_col, id_col = memory_savvy_read_cancers(CANCER_CSV, MAX_FEATURES_CLASSIF)
        print(f"[Task 1] Loaded: rows={len(Xc)}, features={Xc.shape[1]}, target={target_col}, id={id_col}")
        res_cls, fitted_cls, idx_to_class = train_compare_classifiers(Xc, yc, RANDOM_STATE)
        best_cls_name = res_cls.iloc[0]["Model"]
        best_cls = fitted_cls[best_cls_name]
        print(f"[Task 1] Best classifier: {best_cls_name}")
        if HAVE_SHAP:
            print("[Task 2] Running SHAP for best classifier...")
            shap_task2(best_cls, Xc, yc, ids, PATIENT_ID_TO_PLOT, idx_to_class)
        else:
            print("[Task 2] shap not installed — skipping.")
    else:
        print(f"[Task 1-2] Missing {CANCER_CSV}. Put it next to this script.")

    # ---- Tasks 3–4 (Regression) ----
    if os.path.exists(GDSC2_CSV):
        print("[Task 3] Loading GDSC2 CSV (memory-optimized)...")
        Xr, yr, keys, meta = memory_savvy_read_gdsc2(GDSC2_CSV, MAX_FEATURES_REGRESS)
        print(f"[Task 3] Loaded: rows={meta['n_rows']}, features={meta['n_features']} (target={meta['target']}, ids={meta['id_cols']})")
        res_reg, fitted_reg = train_compare_regressors(Xr, yr, RANDOM_STATE)
        best_reg_name = res_reg.iloc[0]["Model"]
        best_reg = fitted_reg[best_reg_name]
        print(f"[Task 3] Best regressor: {best_reg_name}")
        if HAVE_SHAP:
            print("[Task 4] Running SHAP for best regressor...")
            shap_task4(best_reg, Xr, yr, keys)
        else:
            print("[Task 4] shap not installed — skipping.")
    else:
        print(f"[Task 3-4] Missing {GDSC2_CSV}. Provide it locally to run.")
