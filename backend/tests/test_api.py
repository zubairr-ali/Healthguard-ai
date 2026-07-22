import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from main import app

from database import init_db
init_db()  # ensure tables exist before any test runs

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_predict_heart_endpoint_valid_input():
    payload = {
        "Age": 54, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 145,
        "Cholesterol": 233, "FastingBS": 1, "RestingECG": "Normal",
        "MaxHR": 150, "ExerciseAngina": "N", "Oldpeak": 2.3, "ST_Slope": "Flat"
    }
    response = client.post("/api/predict/heart", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "risk_score" in data
    assert "shap_values" in data
    assert "advisory" in data


def test_predict_diabetes_endpoint_valid_input():
    payload = {
        "Pregnancies": 2, "Glucose": 148, "BloodPressure": 72,
        "SkinThickness": 35, "Insulin": 0, "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627, "Age": 50
    }
    response = client.post("/api/predict/diabetes", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "risk_score" in data


def test_predict_heart_invalid_input_rejected():
    payload = {"Age": "not a number"}  # malformed input
    response = client.post("/api/predict/heart", json=payload)
    assert response.status_code == 422  # FastAPI validation error


def test_stats_endpoint():
    response = client.get("/api/stats")
    assert response.status_code == 200
    assert "total" in response.json()


def test_history_endpoint():
    response = client.get("/api/history")
    assert response.status_code == 200
    assert "records" in response.json()