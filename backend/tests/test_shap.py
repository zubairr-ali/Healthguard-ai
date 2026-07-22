import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shap_explainer import get_individual_shap, aggregate_categorical_shap
from model import load_heart, encode_heart_patient


def test_aggregate_categorical_shap_combines_dummies():
    raw_shap = {"ST_Slope_Up": 1.5, "ST_Slope_Flat": 0.3, "Age": 0.1}
    result = aggregate_categorical_shap(raw_shap)
    assert "ST_Slope" in result
    assert "ST_Slope_Up" not in result
    assert round(result["ST_Slope"], 2) == 1.8


def test_individual_shap_returns_clean_clinical_features():
    raw = {
        "Age": 54, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 145,
        "Cholesterol": 233, "FastingBS": 1, "RestingECG": "Normal",
        "MaxHR": 150, "ExerciseAngina": "N", "Oldpeak": 2.3, "ST_Slope": "Flat"
    }
    encoded = encode_heart_patient(raw)
    result = get_individual_shap("heart", encoded)
    # Should be collapsed to original clinical variable names, not one-hot dummies
    for key in result.keys():
        assert "_" not in key or key in ["ExerciseAngina", "ChestPainType", "ST_Slope", "RestingECG"]