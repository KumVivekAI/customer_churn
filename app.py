"""FastAPI service for Telco customer churn prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.preprocessing import REQUIRED_INPUT_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "model" / "churn_model.pkl"

app = FastAPI(
    title="Telco Customer Churn API",
    description="Predict customer churn using a trained decision tree pipeline.",
    version="1.0.0",
)

model = None


class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(gt=0)
    TotalCharges: float = Field(ge=0)

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, value: str) -> str:
        allowed = {"Male", "Female"}
        if value not in allowed:
            raise ValueError(f"gender must be one of {sorted(allowed)}")
        return value

    @field_validator("Partner", "Dependents", "PhoneService", "PaperlessBilling")
    @classmethod
    def validate_yes_no(cls, value: str) -> str:
        allowed = {"Yes", "No"}
        if value not in allowed:
            raise ValueError(f"value must be one of {sorted(allowed)}")
        return value

    @field_validator("MultipleLines")
    @classmethod
    def validate_multiple_lines(cls, value: str) -> str:
        allowed = {"Yes", "No", "No phone service"}
        if value not in allowed:
            raise ValueError(f"MultipleLines must be one of {sorted(allowed)}")
        return value

    @field_validator("InternetService")
    @classmethod
    def validate_internet_service(cls, value: str) -> str:
        allowed = {"DSL", "Fiber optic", "No"}
        if value not in allowed:
            raise ValueError(f"InternetService must be one of {sorted(allowed)}")
        return value

    @field_validator(
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    )
    @classmethod
    def validate_service_fields(cls, value: str) -> str:
        allowed = {"Yes", "No", "No internet service"}
        if value not in allowed:
            raise ValueError("service field must be Yes, No, or No internet service")
        return value

    @field_validator("Contract")
    @classmethod
    def validate_contract(cls, value: str) -> str:
        allowed = {"Month-to-month", "One year", "Two year"}
        if value not in allowed:
            raise ValueError(f"Contract must be one of {sorted(allowed)}")
        return value

    @field_validator("PaymentMethod")
    @classmethod
    def validate_payment_method(cls, value: str) -> str:
        allowed = {
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        }
        if value not in allowed:
            raise ValueError(f"PaymentMethod must be one of {sorted(allowed)}")
        return value


class PredictionResponse(BaseModel):
    prediction: str
    churn_probability: float


@app.on_event("startup")
def load_model() -> None:
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. Run train_model.py first."
        )
    model = joblib.load(MODEL_PATH)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput) -> PredictionResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    payload = customer.model_dump()
    missing_columns = [col for col in REQUIRED_INPUT_COLUMNS if col not in payload]
    if missing_columns:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required fields: {missing_columns}",
        )

    input_df = pd.DataFrame([payload])
    try:
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        classes = list(model.named_steps["classifier"].classes_)
        churn_index = classes.index("Yes")
        churn_probability = float(probabilities[churn_index])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")

    return PredictionResponse(
        prediction=str(prediction),
        churn_probability=round(churn_probability, 4),
    )
