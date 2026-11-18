"""app.py : expose la fonction de prédiction du modèle churn via FastAPI."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model_pipeline import FEATURE_COLUMNS, load_model, predict_model

MODEL_PATH = "classifier.joblib"
state = {"model": None}


@asynccontextmanager
async def lifespan(_app):
    """Charge le modèle depuis le disque au démarrage de l'API."""
    try:
        state["model"] = load_model(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    except FileNotFoundError as err:
        print(f"Erreur au chargement du modèle : {err}")
    yield


app = FastAPI(title="Churn Prediction API", version="1.0", lifespan=lifespan)


class CustomerFeatures(BaseModel):
    """Caractéristiques d'un client (mêmes colonnes que l'entraînement)."""

    CreditScore: int = Field(..., ge=0)
    Gender: int = Field(..., ge=0, le=1, description="0 = Female, 1 = Male")
    Age: int = Field(..., ge=18)
    Tenure: int = Field(..., ge=0)
    Balance: float = Field(..., ge=0)
    NumOfProducts: int = Field(..., ge=1)
    HasCrCard: int = Field(..., ge=0, le=1)
    IsActiveMember: int = Field(..., ge=0, le=1)
    EstimatedSalary: float = Field(..., ge=0)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "CreditScore": 600,
                    "Gender": 1,
                    "Age": 40,
                    "Tenure": 3,
                    "Balance": 60000.0,
                    "NumOfProducts": 2,
                    "HasCrCard": 1,
                    "IsActiveMember": 1,
                    "EstimatedSalary": 50000.0,
                }
            ]
        }
    }


@app.get("/")
def health():
    """Vérifie que l'API tourne et que le modèle est chargé."""
    return {"status": "ok", "model_loaded": state["model"] is not None}


@app.post("/predict")
def predict(features: CustomerFeatures):
    """Prédit si un client risque de partir (1) ou de rester (0)."""
    model = state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")
    try:
        values = [getattr(features, col) for col in FEATURE_COLUMNS]
        result = predict_model(model, values)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    except Exception as err:  # erreur inattendue du modèle
        raise HTTPException(status_code=500, detail=f"Erreur : {err}") from err
    return {"prediction": result, "label": "Part" if result else "Reste"}
