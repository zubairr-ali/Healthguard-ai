from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import json
import os

from model import load_heart, load_diabetes, predict_patient, encode_heart_patient, RESULTS_DIR

def clean_json(obj):
    """Recursively convert numpy types to plain Python types for JSON serialization."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json(v) for v in obj]
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return round(float(obj), 3)
    elif isinstance(obj, float):
        return round(obj, 3)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    else:
        return obj
    
from shap_explainer import get_global_shap, get_individual_shap
from llm_advisor import generate_advisory, generate_fallback_advisory
from database import init_db, save_prediction, get_all_predictions, get_stats

app = FastAPI(title="HealthGuard AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    init_db()
    print("[OK] HealthGuard AI API started.")


class HeartPatient(BaseModel):
    Age: float
    Sex: str
    ChestPainType: str
    RestingBP: float
    Cholesterol: float
    FastingBS: int
    RestingECG: str
    MaxHR: float
    ExerciseAngina: str
    Oldpeak: float
    ST_Slope: str


class DiabetesPatient(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float


@app.get("/")
def root():
    return {"message": "HealthGuard AI API is running", "version": "1.0.0"}


@app.post("/api/predict/heart")
def predict_heart(patient: HeartPatient):
    try:
        raw_dict = patient.model_dump()
        patient_dict = encode_heart_patient(raw_dict)

        result = predict_patient("heart", patient_dict)

        shap_vals = get_individual_shap("heart", patient_dict)

        advisory = generate_fallback_advisory(
            "heart disease",
            result["risk_score"],
            result["risk_level"],
            list(shap_vals.items())[:3]
        )

        save_prediction(
            condition="heart",
            patient_data=clean_json(patient_dict),
            risk_score=float(result["risk_score"]),
            risk_level=result["risk_level"],
            model_used=result["model_used"],
            top_factors=clean_json(dict(list(shap_vals.items())[:5])),
            advisory=advisory
        )

        return clean_json({
            "condition": "heart",
            "risk_score": result["risk_score"],
            "risk_level": result["risk_level"],
            "model_used": result["model_used"],
            "shap_values": shap_vals,
            "advisory": advisory
        })

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
            patient_data=clean_json(patient_dict),
            risk_score=float(result["risk_score"]),
            risk_level=result["risk_level"],
            model_used=result["model_used"],
            top_factors=clean_json(dict(list(shap_vals.items())[:5])),
            advisory=advisory
        )

        return clean_json({
            "condition": "diabetes",
            "risk_score": result["risk_score"],
            "risk_level": result["risk_level"],
            "model_used": result["model_used"],
            "shap_values": shap_vals,
            "advisory": advisory
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


@app.get("/api/models/{condition}")
def model_comparison(condition: str):
    if condition not in ("heart", "diabetes"):
        raise HTTPException(status_code=400, detail="Invalid condition")

    results_path = os.path.join(RESULTS_DIR, f"{condition}_results.json")
    try:
        with open(results_path) as f:
            saved = json.load(f)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail=f"No saved results for '{condition}' yet — run model.py's training pipeline first.",
        )

    return {
        "condition": condition,
        "best_model": saved["best_model"],
        "results": saved["results"],
    }


@app.get("/api/history")
def history():
    try:
        records = get_all_predictions()
        return {"records": records, "total": len(records)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
def stats():
    try:
        return get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    