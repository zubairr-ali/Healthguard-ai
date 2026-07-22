import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from model import load_heart, load_diabetes, predict_patient, encode_heart_patient


def test_load_heart_shape():
    X, y, features = load_heart()
    assert X.shape[0] == 918
    assert len(features) == X.shape[1]
    assert set(y.unique()) <= {0, 1}


def test_load_heart_no_missing_values():
    X, y, _ = load_heart()
    assert X.isnull().sum().sum() == 0


def test_load_diabetes_shape():
    X, y, features = load_diabetes()
    assert X.shape[0] == 768
    assert set(y.unique()) <= {0, 1}


def test_encode_heart_patient_produces_correct_keys():
    raw = {
        "Age": 54, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 145,
        "Cholesterol": 233, "FastingBS": 1, "RestingECG": "Normal",
        "MaxHR": 150, "ExerciseAngina": "N", "Oldpeak": 2.3, "ST_Slope": "Flat"
    }
    encoded = encode_heart_patient(raw)
    assert encoded["Sex_M"] == 1
    assert encoded["ChestPainType_ASY"] == 1
    assert encoded["ST_Slope_Flat"] == 1
    assert encoded["Age"] == 54


def test_predict_patient_heart_returns_valid_output():
    raw = {
        "Age": 54, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 145,
        "Cholesterol": 233, "FastingBS": 1, "RestingECG": "Normal",
        "MaxHR": 150, "ExerciseAngina": "N", "Oldpeak": 2.3, "ST_Slope": "Flat"
    }
    encoded = encode_heart_patient(raw)
    result = predict_patient("heart", encoded)
    assert 0 <= result["risk_score"] <= 100
    assert result["risk_level"] in ["Low", "Medium", "High"]
    assert isinstance(result["risk_score"], float)


def test_predict_patient_diabetes_returns_valid_output():
    patient = {
        "Pregnancies": 2, "Glucose": 148, "BloodPressure": 72,
        "SkinThickness": 35, "Insulin": 0, "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627, "Age": 50
    }
    result = predict_patient("diabetes", patient)
    assert 0 <= result["risk_score"] <= 100
    assert result["risk_level"] in ["Low", "Medium", "High"]


def test_risk_level_thresholds_are_consistent():
    """A very low-risk-looking patient should not be classified High."""
    healthy_patient = {
        "Age": 25, "Sex": "F", "ChestPainType": "ATA", "RestingBP": 110,
        "Cholesterol": 180, "FastingBS": 0, "RestingECG": "Normal",
        "MaxHR": 190, "ExerciseAngina": "N", "Oldpeak": 0, "ST_Slope": "Up"
    }
    encoded = encode_heart_patient(healthy_patient)
    result = predict_patient("heart", encoded)
    assert result["risk_level"] in ["Low", "Medium"]