import shap
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving files
import matplotlib.pyplot as plt

CATEGORICAL_PREFIXES = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]

def aggregate_categorical_shap(shap_dict):
    """
    Collapses one-hot encoded dummy columns (e.g. ST_Slope_Up, ST_Slope_Flat)
    back into their original clinical variable (ST_Slope) by summing SHAP
    contributions — mathematically valid since SHAP values are additive.
    """
    aggregated = {}
    for feat, val in shap_dict.items():
        matched_prefix = next((p for p in CATEGORICAL_PREFIXES if feat.startswith(p + "_")), None)
        key = matched_prefix if matched_prefix else feat
        aggregated[key] = aggregated.get(key, 0) + val
    return dict(sorted(aggregated.items(), key=lambda x: abs(x[1]), reverse=True))

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


# ── explainer selection ──────────────────────────────────────────────────────
# TreeExplainer only supports tree-based models (Random Forest, XGBoost,
# LightGBM, CatBoost). TabPFN — the winning heart-disease model — is a
# pretrained transformer and TreeExplainer rejects it at construction with
# "Model type not yet supported by TreeExplainer: TabPFNClassifier".
#
# Rather than special-case TabPFN by name (fragile — breaks if the model
# lineup changes), this tries the fast exact TreeExplainer first and falls
# back to a model-agnostic permutation explainer built on predict_proba for
# anything TreeExplainer doesn't support. That covers TabPFN today and any
# future non-tree model without further changes here.
#
# Trade-off worth knowing: the fallback calls predict_proba many times per
# explanation, so it is slower than TreeExplainer. For TabPFN specifically,
# each call re-runs in-context inference, so this is noticeably slower than
# the tree models. Acceptable for a single patient's individual explanation;
# global SHAP over the full dataset (get_global_shap) will be slow for a
# non-tree model and is a known limitation worth naming in the report if
# TabPFN's interpretability cost comes up.
_BACKGROUND_CACHE = {}


def _background_sample(condition, features, n=100):
    """Reference sample pulled from training data, used to build the k-means
    summary below. Not used directly as the SHAP background — raw rows are
    too expensive for a model like TabPFN, see _background_summary."""
    if condition not in _BACKGROUND_CACHE:
        from model import load_heart, load_diabetes
        if condition == "heart":
            X, _, _ = load_heart()
        else:
            X, _, _ = load_diabetes()
        X = X[features]
        _BACKGROUND_CACHE[condition] = X.sample(n=min(n, len(X)), random_state=42)
    return _BACKGROUND_CACHE[condition]


def _background_summary(condition, features, k=8):
    """A compressed background masker for the SLOW (non-tree) fallback path.

    This is the single biggest lever on runtime for a model like TabPFN,
    where every background point costs a full forward pass on CPU.
    shap.maskers.Independent(data, max_samples=k) summarises many raw
    background rows down to k representative points internally (via
    k-means) rather than using all of them — 8 representative points
    instead of, say, 100 raw rows is roughly a 12x reduction in the
    dominant cost, with a normal, expected loss of precision in exchange.
    """
    cache_key = f"{condition}::summary"
    if cache_key not in _BACKGROUND_CACHE:
        raw = _background_sample(condition, features)
        _BACKGROUND_CACHE[cache_key] = shap.maskers.Independent(raw, max_samples=min(k, len(raw)))
    return _BACKGROUND_CACHE[cache_key]


def _make_explainer(model, condition, features):
    """Returns (explainer, kind) where kind is 'tree' or 'generic'."""
    try:
        return shap.TreeExplainer(model), "tree"
    except Exception:
        background = _background_summary(condition, features)
        return shap.Explainer(model.predict_proba, background), "generic"


def _positive_class_shap(explainer, kind, df, max_evals=None):
    """Returns a 2D array (n_samples, n_features) of SHAP values for class 1,
    regardless of whether the explainer is TreeExplainer or the generic
    fallback — the two return different shapes/types otherwise.

    max_evals bounds the number of predict_proba calls the generic
    (permutation) explainer makes. SHAP's minimum valid value is
    2 * n_features + 1; anything below that is silently raised to it.
    Passed explicitly on the interactive single-patient path so a slow
    model like TabPFN has a predictable, bounded cost instead of SHAP's
    default budget (which scales up with feature count and is tuned for
    offline analysis, not a live request)."""
    if kind == "tree":
        raw = explainer.shap_values(df)
        if isinstance(raw, list):          # legacy shap: list of per-class arrays
            raw = np.array(raw[1])
        else:
            raw = np.array(raw)
            if raw.ndim == 3:              # newer shap: (n_samples, n_features, n_classes)
                raw = raw[..., 1] if raw.shape[-1] == 2 else raw[1]
        return raw
    else:
        exp = explainer(df, max_evals=max_evals) if max_evals else explainer(df)
        vals = np.array(exp.values)
        if vals.ndim == 3:                 # (n_samples, n_features, n_classes) from predict_proba
            vals = vals[..., 1]
        return vals


# ── GLOBAL SHAP — top features across whole dataset ─────────────────────────
def get_global_shap(condition, X: pd.DataFrame):
    model, features = load_model_and_features(condition)
    X = X[features]

    explainer, kind = _make_explainer(model, condition, features)
    sv = _positive_class_shap(explainer, kind, X)

    # Mean absolute SHAP value per feature
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

    return aggregate_categorical_shap(sorted_importance)


# ── INDIVIDUAL SHAP — why THIS patient got this prediction ───────────────────
def get_individual_shap(condition, patient_data: dict):
    model, features = load_model_and_features(condition)
    df = pd.DataFrame([patient_data]).reindex(columns=features, fill_value=0)

    explainer, kind = _make_explainer(model, condition, features)
    # Minimum valid budget for the permutation explainer — bounds this to a
    # predictable cost on the interactive request path (see docstring above).
    max_evals = 2 * len(features) + 1
    sv = _positive_class_shap(explainer, kind, df, max_evals=max_evals)

    # Flatten to a single row of per-feature contributions
    row = np.asarray(sv)[0]

    # Build per-feature dict
    individual = {}
    for feat, val in zip(features, row.tolist()):
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

    return aggregate_categorical_shap(sorted_individual)


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
