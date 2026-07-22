import shap
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving files
import matplotlib.pyplot as plt

# ── paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# ── load model + data ────────────────────────────────────────────────────────
def load_model_and_features(condition):
    model = pickle.load(open(os.path.join(MODELS_DIR, f"{condition}_model.pkl"), "rb"))
    features = pickle.load(open(os.path.join(MODELS_DIR, f"{condition}_features.pkl"), "rb"))
    return model, features


# ── GLOBAL SHAP — top features across whole dataset ─────────────────────────
def get_global_shap(condition, X: pd.DataFrame):
    model, features = load_model_and_features(condition)
    X = X[features]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # For binary classification shap_values may be a list
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    # Mean absolute SHAP value per feature
    # Mean absolute SHAP value per feature
    sv = np.array(sv)
    if sv.ndim == 3:
        sv = sv[1]
    mean_shap = np.abs(sv).mean(axis=0)
    feature_importance = dict(zip(features, mean_shap.tolist()))

    # Sort descending
    sorted_importance = dict(
        sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    )

    # Save global SHAP bar chart
    plt.figure(figsize=(10, 6))
    feat_names = list(sorted_importance.keys())[:10]
    feat_vals = list(sorted_importance.values())[:10]
    colors = ["#E24B4A" if float(v) > 0 else "#1D9E75" for v in feat_vals]
    plt.barh(feat_names[::-1], feat_vals[::-1], color=colors[::-1])
    plt.xlabel("Mean |SHAP Value|")
    plt.title(f"Global Feature Importance — {condition.capitalize()}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{condition}_global_shap.png"), dpi=150)
    plt.close()

    return sorted_importance


# ── INDIVIDUAL SHAP — why THIS patient got this prediction ───────────────────
def get_individual_shap(condition, patient_data: dict):
    model, features = load_model_and_features(condition)
    df = pd.DataFrame([patient_data]).reindex(columns=features, fill_value=0)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)

    sv = np.array(shap_values)
    # Flatten to 1D array of feature contributions
    while sv.ndim > 1:
        sv = sv[0]

    # Build per-feature dict
    individual = {}
    for feat, val in zip(features, sv.tolist()):
        individual[feat] = round(val, 4)

    # Sort by absolute impact
    sorted_individual = dict(
        sorted(individual.items(), key=lambda x: abs(x[1]), reverse=True)
    )

    # Save individual waterfall-style chart
    plt.figure(figsize=(10, 6))
    feat_names = list(sorted_individual.keys())[:10]
    feat_vals = list(sorted_individual.values())[:10]
    colors = ["#E24B4A" if float(v) < 0 else "#1D9E75" for v in feat_vals]
    plt.barh(feat_names[::-1], feat_vals[::-1], color=colors[::-1])
    plt.axvline(x=0, color="black", linewidth=0.8)
    plt.xlabel("SHAP Value (impact on prediction)")
    plt.title(f"Individual Explanation — {condition.capitalize()}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{condition}_individual_shap.png"), dpi=150)
    plt.close()

    return sorted_individual


# ── TEST when run directly ───────────────────────────────────────────────────
if __name__ == "__main__":
    from model import load_heart, load_diabetes

    print("Testing Global SHAP — Heart Disease...")
    X_h, y_h, _ = load_heart()
    global_heart = get_global_shap("heart", X_h)
    print("Top 5 features:", list(global_heart.items())[:5])

    print("\nTesting Individual SHAP — Heart Disease...")
    from model import encode_heart_patient
    raw_patient = {
        "Age": 54, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 145,
        "Cholesterol": 233, "FastingBS": 1, "RestingECG": "Normal",
        "MaxHR": 150, "ExerciseAngina": "N", "Oldpeak": 2.3, "ST_Slope": "Flat"
    }
    sample_patient = encode_heart_patient(raw_patient)
    individual_heart = get_individual_shap("heart", sample_patient)
    print("Top 5 individual factors:", list(individual_heart.items())[:5])

    print("\nTesting Global SHAP — Diabetes...")
    X_d, y_d, _ = load_diabetes()
    global_diabetes = get_global_shap("diabetes", X_d)
    print("Top 5 features:", list(global_diabetes.items())[:5])

    print("\n✅ SHAP module working perfectly.")
