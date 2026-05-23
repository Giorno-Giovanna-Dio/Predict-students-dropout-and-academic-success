"""
XGBoost binary classification: Graduate vs Not_Graduate.

Not_Graduate = Enrolled ∪ Dropout (combined).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path(__file__).parent / "dataset"
TARGET_COL = "Target"
# LabelEncoder order: Graduate=0, Not_Graduate=1
CLASS_NAMES = ["Graduate", "Not_Graduate"]
NOT_GRADUATE_SOURCES = {"Dropout", "Enrolled"}


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"{name}.csv")


def binarize_target(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    unknown = set(out[TARGET_COL].unique()) - {"Graduate"} - NOT_GRADUATE_SOURCES
    if unknown:
        raise ValueError(f"Unexpected Target values: {unknown}")

    out[TARGET_COL] = out[TARGET_COL].replace(
        {"Graduate": "Graduate", "Dropout": "Not_Graduate", "Enrolled": "Not_Graduate"}
    )
    return out


def prepare_xy(df: pd.DataFrame, le: LabelEncoder) -> tuple[pd.DataFrame, np.ndarray]:
    X = df.drop(columns=[TARGET_COL])
    y = le.transform(df[TARGET_COL])
    return X, y


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_proba_pos: np.ndarray) -> dict:
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    try:
        auc = roc_auc_score(y_true, y_proba_pos)
    except ValueError:
        auc = None

    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "not_graduate_recall": float(report["Not_Graduate"]["recall"]),
        "not_graduate_f1": float(report["Not_Graduate"]["f1-score"]),
        "graduate_recall": float(report["Graduate"]["recall"]),
        "roc_auc": auc,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def main() -> None:
    train_df = binarize_target(load_split("train"))
    val_df = binarize_target(load_split("validation"))
    test_df = binarize_target(load_split("test"))

    le = LabelEncoder()
    le.fit(CLASS_NAMES)

    X_train, y_train = prepare_xy(train_df, le)
    X_val, y_val = prepare_xy(val_df, le)
    X_test, y_test = prepare_xy(test_df, le)

    print("Class counts (train):", train_df[TARGET_COL].value_counts().to_dict())

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
    }

    evals = [(dtrain, "train"), (dval, "validation")]
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=evals,
        early_stopping_rounds=30,
        verbose_eval=50,
    )

    def predict_split(dm: xgb.DMatrix, y: np.ndarray, split_name: str) -> dict:
        proba_pos = booster.predict(dm)
        pred = (proba_pos >= 0.5).astype(int)
        metrics = evaluate(y, pred, proba_pos)
        metrics["split"] = split_name
        metrics["best_iteration"] = int(booster.best_iteration)
        return metrics

    results = {
        "target_mapping": {
            "Graduate": "Graduate",
            "Dropout": "Not_Graduate",
            "Enrolled": "Not_Graduate",
        },
        "validation": predict_split(dval, y_val, "validation"),
        "test": predict_split(dtest, y_test, "test"),
        "note": (
            "Binary: Graduate vs (Enrolled + Dropout). "
            "Primary retention metric: not_graduate_recall."
        ),
    }

    out_path = Path(__file__).parent / "metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    model_path = Path(__file__).parent / "xgboost_model.json"
    booster.save_model(model_path)

    for split in ("validation", "test"):
        m = results[split]
        print(f"\n=== {split} ===")
        print(f"  balanced_accuracy:   {m['balanced_accuracy']:.4f}")
        print(f"  macro_f1:            {m['macro_f1']:.4f}")
        print(f"  not_graduate_recall: {m['not_graduate_recall']:.4f}  (Enrolled+Dropout)")
        print(f"  graduate_recall:     {m['graduate_recall']:.4f}")
        if m["roc_auc"] is not None:
            print(f"  ROC-AUC:             {m['roc_auc']:.4f}")
        print("  confusion_matrix [Graduate, Not_Graduate] (rows=true, cols=pred):")
        print(np.array(m["confusion_matrix"]))

    print(f"\nSaved metrics -> {out_path}")
    print(f"Saved model   -> {model_path}")


if __name__ == "__main__":
    main()
