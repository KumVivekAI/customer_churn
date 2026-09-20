"""Train and save the final churn prediction pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.preprocessing import build_model_pipeline, load_and_clean_data

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "TelcoCustomerChurn.csv"
MODEL_PATH = PROJECT_ROOT / "model" / "churn_model.pkl"
METRICS_PATH = PROJECT_ROOT / "model" / "model_metrics.json"


def evaluate_model(name: str, model, X_test, y_test) -> dict:
    """Compute evaluation metrics for a fitted model."""
    predictions = model.predict(X_test)
    metrics = {
        "model_name": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, pos_label="Yes"),
        "recall": recall_score(y_test, predictions, pos_label="Yes"),
        "f1": f1_score(y_test, predictions, pos_label="Yes"),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(y_test, predictions),
    }
    return metrics


def main() -> None:
    df = load_and_clean_data(DATA_PATH)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model_configs = {
        "baseline_tree": build_model_pipeline(),
        "tuned_tree": build_model_pipeline(
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced",
        ),
    }

    comparison = []
    fitted_models = {}
    for name, pipeline in model_configs.items():
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(name, pipeline, X_test, y_test)
        comparison.append(metrics)
        fitted_models[name] = pipeline
        print(f"\n{name}")
        print(metrics["classification_report"])

    final_name = max(comparison, key=lambda item: item["f1"])["model_name"]
    final_model = fitted_models[final_name]
    final_metrics = next(item for item in comparison if item["model_name"] == final_name)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)

    output = {
        "selected_model": final_name,
        "comparison": comparison,
        "final_model_metrics": final_metrics,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }
    METRICS_PATH.write_text(json.dumps(output, indent=2))

    print(f"\nSelected model: {final_name}")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
