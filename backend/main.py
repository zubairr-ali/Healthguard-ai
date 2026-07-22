from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import json

from model import load_heart, load_diabetes, predict_patient, train_models
from shap_explainer import get_global_shap, get_individual_shap
from llm_advisor import generate_advisory, generate_fallback_advisory
from database import init_db, save_prediction, get_all_predictions, get_stats

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="HealthGuard AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Initialise database on startup ───────────────────────────────────────────
@app.on_event("startup")
async def startup():
    init_db()
    print("✅ HealthGuard AI API started.")


# ── Request models ───────────────────────────────────────────────────────────
class HeartPatient(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


class DiabetesPatient(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float


# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "HealthGuard AI API is running", "version": "1.0.0"}


# ── Predict endpoint ─────────────────────────────────────────────────────────
@app.post("/api/predict/heart")
def predict_heart(patient: HeartPatient):
    try:
        patient_dict = patient.model_dump()

        # Prediction
        result = predict_patient("heart", patient_dict)

        # Individual SHAP
        shap_vals = get_individual_shap("heart", patient_dict)

        # Advisory
        advisory = generate_fallback_advisory(
            "heart disease",
            result["risk_score"],
            result["risk_level"],
            list(shap_vals.items())[:3]
        )

        # Save to database
        save_prediction(
            condition="heart",
            patient_data=patient_dict,
            risk_score=result["risk_score"],
            risk_level=result["risk_level"],
            model_used=result["model_used"],
            top_factors=dict(list(shap_vals.items())[:5]),
            advisory=advisory
        )

        return {
            "condition": "heart",
            "risk_score": result["risk_score"],
            "risk_level": result["risk_level"],
            "model_used": result["model_used"],
            "shap_values": shap_vals,
            "advisory": advisory
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict/diabetes")
def predict_diabetes(patient: DiabetesPatient):
    try:
        patient_dict = patient.model_dump()

        result = predict_patient("diabetes", patient_dict)
        shap_vals = get_individual_shap("diabetes", patient_dict)

        advisory = generate_fallback_advisory(
            "diabetes",
            result["risk_score"],
            result["risk_level"],
            list(shap_vals.items())[:3]
        )

        save_prediction(
            condition="diabetes",
            patient_data=patient_dict,
            risk_score=result["risk_score"],
            risk_level=result["risk_level"],
            model_used=result["model_used"],
            top_factors=dict(list(shap_vals.items())[:5]),
            advisory=advisory
        )

        return {
            "condition": "diabetes",
            "risk_score": result["risk_score"],
            "risk_level": result["risk_level"],
            "model_used": result["model_used"],
            "shap_values": shap_vals,
            "advisory": advisory
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Global SHAP endpoint ─────────────────────────────────────────────────────
@app.get("/api/shap/global/{condition}")
def global_shap(condition: str):
    try:
        if condition == "heart":
            X, y, _ = load_heart()
        elif condition == "diabetes":
            X, y, _ = load_diabetes()
        else:
            raise HTTPException(status_code=400, detail="Invalid condition")

        shap_vals = get_global_shap(condition, X)
        return {"condition": condition, "global_shap": shap_vals}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Model comparison endpoint ────────────────────────────────────────────────
@app.get("/api/models/{condition}")
def model_comparison(condition: str):
    try:
        if condition == "heart":
            X, y, _ = load_heart()
            results, best = train_models(X, y, condition)
        elif condition == "diabetes":
            X, y, _ = load_diabetes()
            results, best = train_models(X, y, condition)
        else:
            raise HTTPException(status_code=400, detail="Invalid condition")

        return {
            "condition": condition,
            "best_model": best,
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── History endpoint ─────────────────────────────────────────────────────────
@app.get("/api/history")
def history():
    try:
        records = get_all_predictions()
        return {"records": records, "total": len(records)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Stats endpoint ───────────────────────────────────────────────────────────
@app.get("/api/stats")
def stats():
    try:
        return get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    