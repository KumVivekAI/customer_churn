"""Preprocessing utilities for the Telco churn prediction pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


SERVICE_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

CATEGORICAL_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

NUMERIC_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "AvgMonthlyCharge",
    "NumAddOnServices",
    "IsMonthToMonth",
]

REQUIRED_INPUT_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create engineered features used during training and inference."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
        data["MonthlyCharges"] = pd.to_numeric(data["MonthlyCharges"], errors="coerce")

        tenure = data["tenure"].replace(0, np.nan)
        data["AvgMonthlyCharge"] = data["TotalCharges"] / tenure
        data["AvgMonthlyCharge"] = data["AvgMonthlyCharge"].fillna(data["MonthlyCharges"])

        data["NumAddOnServices"] = data[SERVICE_COLUMNS].eq("Yes").sum(axis=1)
        data["IsMonthToMonth"] = (data["Contract"] == "Month-to-month").astype(int)

        return data[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]


def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    """Load the Telco dataset and apply basic cleaning."""
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["tenure"] == 0, "TotalCharges"] = df.loc[df["tenure"] == 0, "MonthlyCharges"]
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])
    return df


def build_preprocessor() -> ColumnTransformer:
    """Build preprocessing steps that can be reused on unseen data."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_COLUMNS),
            ("cat", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )


def build_model_pipeline(
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    class_weight: str | dict | None = None,
) -> Pipeline:
    """Create the full sklearn pipeline for churn prediction."""
    classifier = DecisionTreeClassifier(
        random_state=42,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
    )

    return Pipeline(
        steps=[
            ("features", FeatureEngineer()),
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )
