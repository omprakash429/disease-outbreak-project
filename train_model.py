"""
train_model.py  -  trains the Disease Outbreak transmission-mode model.

Run:   python train_model.py
Needs: data/NORS_20261027.csv

What it does, step by step:
  1. Loads the CSV and prints a data report (columns, missing values, duplicates)
  2. Cleans the data (see src/data_utils.py)
  3. Splits into 80% training / 20% testing
  4. Trains a "dummy" baseline, Logistic Regression, and Random Forest
  5. Evaluates on the TEST set (accuracy, F1, confusion matrix)
  6. Saves the model + charts + metrics into models/ and reports/
"""

import argparse
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")  # draw charts to files, no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             ConfusionMatrixDisplay, confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from src.data_utils import (CATEGORY_FEATURES, FEATURES, KEEP_CLASSES,
                            NUMBER_FEATURES, TARGET, load_raw,
                            prepare_training_data)

RANDOM_STATE = 42
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)


def print_data_report(df, path):
    """Step 1: describe the raw dataset and save it as a text file."""
    lines = []
    lines.append(f"Rows: {len(df)}   Columns: {df.shape[1]}")
    lines.append(f"Exact duplicate rows: {df.duplicated().sum()}")
    lines.append("")
    lines.append(f"{'Column':32} {'Type':10} {'Missing %':>10} {'Unique':>8}")
    for col in df.columns:
        lines.append(f"{col:32} {str(df[col].dtype):10} "
                     f"{df[col].isna().mean() * 100:10.1f} {df[col].nunique():8}")
    lines.append("")
    lines.append(f"Target options in '{TARGET}':")
    lines.append(df[TARGET].value_counts().to_string())
    text = "\n".join(lines)
    print(text)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def build_pipeline(model):
    """Preprocessing + model in one object, so the same steps run for training and prediction."""
    preprocess = ColumnTransformer([
        # text columns -> 0/1 columns
        ("categories", OneHotEncoder(handle_unknown="ignore"), CATEGORY_FEATURES),
        # illnesses can range from 2 to thousands -> log scale keeps it manageable
        ("numbers", FunctionTransformer(np.log1p), NUMBER_FEATURES),
    ])
    return Pipeline([("prep", preprocess), ("model", model)])


def evaluate(name, pipe, X_test, y_test):
    pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")
    print(f"{name:22} accuracy = {acc:.3f}   macro-F1 = {f1:.3f}")
    return pred, acc, f1


def main(csv_path):
    # ---- 1. Load + describe ------------------------------------------------
    if not os.path.exists(csv_path):
        raise SystemExit(f"Cannot find {csv_path}\n"
                         "Put NORS_20261027.csv inside the 'data' folder and run again.")
    raw = load_raw(csv_path)
    print("=" * 60, "\nDATA REPORT (raw file)\n" + "=" * 60)
    print_data_report(raw, "reports/data_report.txt")

    # ---- 2. Clean ----------------------------------------------------------
    X, y, options = prepare_training_data(raw)
    print("\n" + "=" * 60)
    print(f"After cleaning: {len(X)} rows, {len(KEEP_CLASSES)} classes")
    print(y.value_counts().to_string())

    # ---- 3. Train/test split (stratify keeps class proportions equal) -----
    X_train, X_test, y_train, y_test = train_test_split(
        X[FEATURES], y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    print(f"\nTrain rows: {len(X_train)}   Test rows: {len(X_test)}")

    # ---- 4. Train models ---------------------------------------------------
    models = {
        "Baseline (always guess most common)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=-1, random_state=RANDOM_STATE),
    }
    print("\n" + "=" * 60 + "\nRESULTS ON THE TEST SET\n" + "=" * 60)
    results, fitted = {}, {}
    for name, model in models.items():
        pipe = build_pipeline(model)
        pipe.fit(X_train, y_train)
        pred, acc, f1 = evaluate(name, pipe, X_test, y_test)
        results[name] = {"accuracy": acc, "macro_f1": f1}
        fitted[name] = (pipe, pred)

    # ---- 5. Detailed evaluation of the main model (Random Forest) --------
    main_name = "Random Forest"
    pipe, pred = fitted[main_name]
    classes = list(pipe.classes_)
    report_text = classification_report(y_test, pred, digits=3)
    print(f"\nClassification report - {main_name}\n{report_text}")
    report_dict = classification_report(y_test, pred, output_dict=True)

    # Confusion matrix chart
    cm = confusion_matrix(y_test, pred, labels=classes)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(
        ax=ax, cmap="Blues", xticks_rotation=30, colorbar=False)
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig("reports/confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Which inputs matter most? (permutation importance on the test set)
    imp = permutation_importance(pipe, X_test, y_test, n_repeats=3,
                                 random_state=RANDOM_STATE, scoring="f1_macro")
    importance = pd.Series(imp.importances_mean, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    importance.plot.barh(ax=ax, color="#2b7de9")
    ax.set_xlabel("Drop in macro-F1 when the column is shuffled")
    ax.set_title("Which inputs matter most?")
    fig.tight_layout()
    fig.savefig("reports/feature_importance.png", dpi=150)
    plt.close(fig)

    # Class distribution chart
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    y.value_counts().plot.bar(ax=ax, color="#2b7de9")
    ax.set_title("Outbreaks per transmission mode (cleaned data)")
    ax.set_xlabel("")
    ax.set_ylabel("Number of outbreaks")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig("reports/class_distribution.png", dpi=150)
    plt.close(fig)

    # ---- 6. Save everything ------------------------------------------------
    metrics = {
        "rows_after_cleaning": int(len(X)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "class_counts": {k: int(v) for k, v in y.value_counts().items()},
        "model_comparison": results,
        "main_model": main_name,
        "per_class": {c: report_dict[c] for c in classes},
        "feature_importance": {k: float(v) for k, v in importance.items()},
    }
    with open("reports/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open("reports/classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    joblib.dump({"pipeline": pipe, "options": options, "classes": classes},
                "models/outbreak_model.joblib")
    print("\nSaved: models/outbreak_model.joblib and files in reports/")
    print("Next step:  streamlit run app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join("data", "NORS_20261027.csv"))
    main(parser.parse_args().data)
