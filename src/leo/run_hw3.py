"""Automations for CAP5610 HW3 tree-based modeling and SHAP analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

from . import project_root

RANDOM_STATE = 42
BACKGROUND_SIZE = 200


@dataclass
class ClassificationResults:
    metrics_df: pd.DataFrame
    test_metrics: Dict[str, float]
    best_model_name: str
    report: str
    shap_top_features: pd.DataFrame
    shap_force_paths: Dict[str, Path]


@dataclass
class RegressionResults:
    metrics_df: pd.DataFrame
    test_metrics: Dict[str, float]
    best_model_name: str
    shap_top_features: pd.DataFrame
    min_error_features: pd.DataFrame


def _prepare_output_dirs() -> Dict[str, Path]:
    root = project_root()
    paths = {
        "root": root,
        "reports": root / "reports",
        "figures": root / "reports" / "figures",
        "processed": root / "data" / "processed",
        "models": root / "models",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _load_classification_data() -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    data_path = project_root() / "data" / "raw" / "lncRNA_5_Cancers.csv"
    df = pd.read_csv(data_path)
    sample_ids = df["Ensembl_ID"].copy()
    X = df.drop(columns=["Class", "Ensembl_ID"])
    y = df["Class"].copy()
    return X, y, sample_ids


def _load_regression_data() -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    data_path = project_root() / "data" / "raw" / "hw3-drug-screening-data.csv"
    df = pd.read_csv(data_path)
    meta_cols = ["CELL_LINE_NAME", "DRUG_NAME"]
    y = df["LN_IC50"].copy()
    X = df.drop(columns=["LN_IC50"])
    cell_ids = df["CELL_LINE_NAME"].copy()
    drug_names = df["DRUG_NAME"].copy()
    return X, y, cell_ids, drug_names


def _tree_background(sample: np.ndarray) -> np.ndarray:
    if sample.shape[0] <= BACKGROUND_SIZE:
        return sample
    idx = np.random.default_rng(RANDOM_STATE).choice(
        sample.shape[0], size=BACKGROUND_SIZE, replace=False
    )
    return sample[idx]


def run_classification(paths: Dict[str, Path]) -> ClassificationResults:
    X, y, sample_ids = _load_classification_data()
    feature_names = X.columns.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    base_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeClassifier()),  # placeholder, replaced per iteration
    ])

    classifiers = {
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "xgboost": XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=RANDOM_STATE,
            eval_metric="mlogloss",
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=600,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "catboost": CatBoostClassifier(
            iterations=700,
            depth=6,
            learning_rate=0.05,
            random_seed=RANDOM_STATE,
            task_type="CPU",
            verbose=False,
        ),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows: List[Dict[str, float]] = []
    best_name = None
    best_score = -np.inf
    best_pipeline = None

    for name, estimator in classifiers.items():
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", clone(estimator)),
        ])
        fold_acc: List[float] = []
        fold_f1: List[float] = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr = X_train.iloc[train_idx]
            X_val = X_train.iloc[val_idx]
            y_tr = y_train.iloc[train_idx]
            y_val = y_train.iloc[val_idx]

            pipeline.fit(X_tr, y_tr)
            preds = pipeline.predict(X_val)
            fold_acc.append(accuracy_score(y_val, preds))
            fold_f1.append(f1_score(y_val, preds, average="macro"))

        mean_acc = float(np.mean(fold_acc))
        std_acc = float(np.std(fold_acc))
        mean_f1 = float(np.mean(fold_f1))
        std_f1 = float(np.std(fold_f1))
        rows.append(
            {
                "model": name,
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc,
                "mean_f1_macro": mean_f1,
                "std_f1_macro": std_f1,
            }
        )

        if mean_f1 > best_score:
            best_score = mean_f1
            best_name = name
            best_pipeline = pipeline

    metrics_df = pd.DataFrame(rows).sort_values("mean_f1_macro", ascending=False)
    assert best_pipeline is not None and best_name is not None

    best_pipeline.fit(X_train, y_train)
    test_preds = best_pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds, average="macro")
    report = classification_report(y_test, test_preds)

    test_metrics = {
        "accuracy": float(test_acc),
        "f1_macro": float(test_f1),
    }

    # Fit on the full dataset for SHAP analysis
    best_pipeline_full = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", clone(classifiers[best_name])),
    ])
    best_pipeline_full.fit(X, y)

    imputer = best_pipeline_full.named_steps["imputer"]
    model = best_pipeline_full.named_steps["model"]
    X_imputed = imputer.transform(X)
    background = _tree_background(X_imputed)

    explainer = shap.TreeExplainer(
        model,
        data=background,
        feature_perturbation="tree_path_dependent",
    )
    shap_values_raw = explainer.shap_values(X_imputed)

    if isinstance(shap_values_raw, list):
        shap_by_class = shap_values_raw
    elif isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3:
        shap_by_class = [shap_values_raw[:, :, i] for i in range(shap_values_raw.shape[2])]
    else:
        shap_by_class = [shap_values_raw]

    classes = model.classes_ if hasattr(model, "classes_") else np.unique(y)

    top_rows: List[Dict[str, object]] = []
    for class_index, class_name in enumerate(classes):
        class_mask = (y == class_name).to_numpy()
        class_shap = shap_by_class[class_index][class_mask]
        mean_abs = np.abs(class_shap).mean(axis=0)
        top_indices = np.argsort(mean_abs)[::-1][:10]
        for rank, feat_idx in enumerate(top_indices, start=1):
            top_rows.append(
                {
                    "class": class_name,
                    "rank": rank,
                    "feature": feature_names[feat_idx],
                    "mean_abs_shap": float(mean_abs[feat_idx]),
                }
            )

    shap_top_features = pd.DataFrame(top_rows)

    figures_dir = paths["figures"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    patient_id = "TCGA-39-5011-01A"
    if patient_id in sample_ids.values:
        sample_loc = sample_ids[sample_ids == patient_id].index[0]
        shap_force_paths: Dict[str, Path] = {}
        for class_index, class_name in enumerate(classes):
            per_sample_shap = shap_by_class[class_index][sample_loc]
            top_indices = np.argsort(np.abs(per_sample_shap))[::-1][:20]
            reduced_shap = per_sample_shap[top_indices]
            reduced_values = X_imputed[sample_loc, top_indices]
            reduced_features = feature_names[top_indices]
            force = shap.force_plot(
                np.asarray(explainer.expected_value)[class_index],
                reduced_shap,
                reduced_values,
                feature_names=reduced_features,
            )
            output_path = figures_dir / f"force_plot_{patient_id}_{class_name}.html"
            shap.save_html(str(output_path), force)
            shap_force_paths[class_name] = output_path
    else:
        shap_force_paths = {}

    metrics_df.to_csv(paths["reports"] / "classification_metrics.csv", index=False)
    pd.DataFrame([test_metrics]).to_csv(
        paths["reports"] / "classification_test_metrics.csv", index=False
    )
    shap_top_features.to_csv(
        paths["reports"] / "classification_top_features.csv", index=False
    )

    return ClassificationResults(
        metrics_df=metrics_df,
        test_metrics=test_metrics,
        best_model_name=best_name,
        report=report,
        shap_top_features=shap_top_features,
        shap_force_paths=shap_force_paths,
    )


def run_regression(paths: Dict[str, Path]) -> RegressionResults:
    X, y, cell_ids, drug_names = _load_regression_data()

    numeric_features = [
        col for col in X.columns if col not in {"CELL_LINE_NAME", "DRUG_NAME"}
    ]
    categorical_features = ["CELL_LINE_NAME", "DRUG_NAME"]

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", numeric_transformer, numeric_features),
        ]
    )

    regressors = {
        "decision_tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "random_forest": RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "gradient_boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "xgboost": XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=RANDOM_STATE,
        ),
        "lightgbm": LGBMRegressor(
            n_estimators=600,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "catboost": CatBoostRegressor(
            iterations=700,
            depth=6,
            learning_rate=0.05,
            random_seed=RANDOM_STATE,
            task_type="CPU",
            verbose=False,
        ),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows: List[Dict[str, float]] = []
    best_name = None
    best_score = -np.inf
    best_pipeline = None

    for name, estimator in regressors.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", clone(estimator)),
        ])
        fold_mae: List[float] = []
        fold_rmse: List[float] = []
        fold_r2: List[float] = []
        for train_idx, val_idx in kf.split(X_train):
            X_tr = X_train.iloc[train_idx]
            X_val = X_train.iloc[val_idx]
            y_tr = y_train.iloc[train_idx]
            y_val = y_train.iloc[val_idx]

            pipeline.fit(X_tr, y_tr)
            preds = pipeline.predict(X_val)
            fold_mae.append(mean_absolute_error(y_val, preds))
            mse = mean_squared_error(y_val, preds)
            fold_rmse.append(np.sqrt(mse))
            fold_r2.append(r2_score(y_val, preds))

        mean_mae = float(np.mean(fold_mae))
        std_mae = float(np.std(fold_mae))
        mean_rmse = float(np.mean(fold_rmse))
        std_rmse = float(np.std(fold_rmse))
        mean_r2 = float(np.mean(fold_r2))
        std_r2 = float(np.std(fold_r2))
        rows.append(
            {
                "model": name,
                "mean_mae": mean_mae,
                "std_mae": std_mae,
                "mean_rmse": mean_rmse,
                "std_rmse": std_rmse,
                "mean_r2": mean_r2,
                "std_r2": std_r2,
            }
        )

        if mean_r2 > best_score:
            best_score = mean_r2
            best_name = name
            best_pipeline = pipeline

    metrics_df = pd.DataFrame(rows).sort_values("mean_r2", ascending=False)
    assert best_pipeline is not None and best_name is not None

    best_pipeline.fit(X_train, y_train)
    test_preds = best_pipeline.predict(X_test)
    test_metrics = {
        "mae": float(mean_absolute_error(y_test, test_preds)),
        "mse": float(mean_squared_error(y_test, test_preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, test_preds))),
        "r2": float(r2_score(y_test, test_preds)),
    }

    best_pipeline_full = Pipeline([
        ("preprocessor", preprocessor),
        ("model", clone(regressors[best_name])),
    ])
    best_pipeline_full.fit(X, y)

    pre = best_pipeline_full.named_steps["preprocessor"]
    model = best_pipeline_full.named_steps["model"]
    X_transformed = pre.transform(X)
    feature_names = pre.get_feature_names_out()
    background = _tree_background(X_transformed)

    explainer = shap.TreeExplainer(
        model,
        data=background,
        feature_perturbation="tree_path_dependent",
    )
    shap_values = explainer.shap_values(X_transformed)

    top_rows: List[Dict[str, object]] = []
    unique_drugs = sorted(drug_names.unique())
    for drug in unique_drugs:
        mask = (drug_names == drug).to_numpy()
        if not mask.any():
            continue
        drug_shap = shap_values[mask]
        mean_abs = np.abs(drug_shap).mean(axis=0)
        top_indices = np.argsort(mean_abs)[::-1][:10]
        for rank, feat_idx in enumerate(top_indices, start=1):
            top_rows.append(
                {
                    "drug": drug,
                    "rank": rank,
                    "feature": feature_names[feat_idx],
                    "mean_abs_shap": float(mean_abs[feat_idx]),
                }
            )

    shap_top_features = pd.DataFrame(top_rows)

    preds_full = best_pipeline_full.predict(X)
    errors = np.abs(preds_full - y.to_numpy())
    min_idx = int(np.argmin(errors))
    min_pair_features = shap_values[min_idx]
    top_indices = np.argsort(np.abs(min_pair_features))[::-1][:10]
    min_rows = []
    for rank, feat_idx in enumerate(top_indices, start=1):
        min_rows.append(
            {
                "rank": rank,
                "feature": feature_names[feat_idx],
                "abs_shap": float(np.abs(min_pair_features[feat_idx])),
                "shap_value": float(min_pair_features[feat_idx]),
                "value": float(X_transformed[min_idx, feat_idx]),
            }
        )
    min_error_features = pd.DataFrame(min_rows)
    min_error_features.insert(0, "cell_line", cell_ids.iloc[min_idx])
    min_error_features.insert(1, "drug", drug_names.iloc[min_idx])
    min_error_features.insert(2, "true_ln_ic50", float(y.iloc[min_idx]))
    min_error_features.insert(3, "pred_ln_ic50", float(preds_full[min_idx]))
    min_error_features.insert(4, "abs_error", float(errors[min_idx]))

    metrics_df.to_csv(paths["reports"] / "regression_metrics.csv", index=False)
    pd.DataFrame([test_metrics]).to_csv(
        paths["reports"] / "regression_test_metrics.csv", index=False
    )
    shap_top_features.to_csv(
        paths["reports"] / "regression_top_features.csv", index=False
    )
    min_error_features.to_csv(
        paths["reports"] / "regression_min_error_features.csv", index=False
    )

    return RegressionResults(
        metrics_df=metrics_df,
        test_metrics=test_metrics,
        best_model_name=best_name,
        shap_top_features=shap_top_features,
        min_error_features=min_error_features,
    )


def main() -> None:
    paths = _prepare_output_dirs()
    classification_results = run_classification(paths)
    regression_results = run_regression(paths)

    summary = {
        "classification": {
            "best_model": classification_results.best_model_name,
            "test_metrics": classification_results.test_metrics,
        },
        "regression": {
            "best_model": regression_results.best_model_name,
            "test_metrics": regression_results.test_metrics,
        },
    }

    with open(paths["reports"] / "summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)


if __name__ == "__main__":
    main()
