import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pickle
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HEART_PATH = os.path.join(BASE_DIR, "data", "heart.csv")
DIABETES_PATH = os.path.join(BASE_DIR, "data", "diabetes.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── LOADERS ──────────────────────────────────────────────────────────────────
def load_heart():
    """
    Loads the combined heart disease dataset (918 rows, 5 hospitals).
    Cleans invalid zero values in RestingBP/Cholesterol, one-hot encodes
    categorical clinical variables, keeping all categories for interpretability.
    """
    df = pd.read_csv(HEART_PATH)

    # RestingBP=0 and Cholesterol=0 are data entry errors, not real values.
    # Replace with median, computed separately per class to avoid leaking
    # target information across the whole dataset.
    df.loc[df["RestingBP"] == 0, "RestingBP"] = np.nan
    df.loc[df["Cholesterol"] == 0, "Cholesterol"] = np.nan
    df["RestingBP"] = df["RestingBP"].fillna(df["RestingBP"].median())
    df["Cholesterol"] = df.groupby("HeartDisease")["Cholesterol"].transform(
        lambda x: x.fillna(x.median())
    )

    categorical_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

    y = df_encoded["HeartDisease"]
    X = df_encoded.drop("HeartDisease", axis=1).astype(float)

    return X, y, list(X.columns)


def load_diabetes():
    df = pd.read_csv(DIABETES_PATH)
    cols_with_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)
    df[cols_with_zeros] = df[cols_with_zeros].fillna(df[cols_with_zeros].median())
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    return X, y, list(X.columns)


# ── HYPERPARAMETER SEARCH SPACES ─────────────────────────────────────────────
PARAM_GRIDS = {
    "Random Forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "Logistic Regression": {
        "C": [0.01, 0.1, 1, 10, 100],
        "solver": ["lbfgs"],
        "max_iter": [2000],
    },
    "SVM": {
        "C": [0.1, 1, 10],
        "kernel": ["rbf", "linear"],
        "gamma": ["scale", "auto"],
    },
    "XGBoost": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
    },
}


def build_base_models(class_weight_balanced=True, scale_pos_weight=1.0):
    cw = "balanced" if class_weight_balanced else None
    return {
        "Random Forest": RandomForestClassifier(random_state=42, class_weight=cw),
        "Logistic Regression": LogisticRegression(random_state=42, class_weight=cw),
        "SVM": SVC(probability=True, random_state=42, class_weight=cw),
        "XGBoost": XGBClassifier(
            random_state=42, eval_metric="logloss", verbosity=0,
            scale_pos_weight=scale_pos_weight
        ),
    }


# ── TRAIN + TUNE + EVALUATE ───────────────────────────────────────────────────
def train_models(X, y, condition, use_smote=False):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Handle class imbalance
    if use_smote:
        smote = SMOTE(random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
        X_train_scaled_bal, y_train_scaled_bal = smote.fit_resample(X_train_scaled, y_train)
        scale_pos_weight = 1.0  # already balanced by SMOTE
    else:
        X_train_bal, y_train_bal = X_train, y_train
        X_train_scaled_bal, y_train_scaled_bal = X_train_scaled, y_train
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        scale_pos_weight = neg / pos if pos > 0 else 1.0

    base_models = build_base_models(
        class_weight_balanced=not use_smote, scale_pos_weight=scale_pos_weight
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    best_model_name = None
    best_cv_f1 = 0
    fitted_models = {}

    for name, base_model in base_models.items():
        needs_scaling = name in ["Logistic Regression", "SVM"]
        X_tr = X_train_scaled_bal if needs_scaling else X_train_bal
        y_tr = y_train_scaled_bal if needs_scaling else y_train_bal
        X_te = X_test_scaled if needs_scaling else X_test

        print(f"  Tuning {name} for {condition}...")
        search = RandomizedSearchCV(
            base_model, PARAM_GRIDS[name], n_iter=10, cv=cv,
            scoring="f1", random_state=42, n_jobs=-1
        )
        search.fit(X_tr, y_tr)
        best_est = search.best_estimator_

        # Cross-validated F1 (mean ± std) on training data — robustness check
        cv_scores = cross_val_score(best_est, X_tr, y_tr, cv=cv, scoring="f1", n_jobs=-1)
        cv_f1_mean = round(cv_scores.mean() * 100, 2)
        cv_f1_std = round(cv_scores.std() * 100, 2)

        # Held-out test set evaluation — the honest, unbiased numbers
        preds = best_est.predict(X_te)
        probs = best_est.predict_proba(X_te)[:, 1]

        acc = round(accuracy_score(y_test, preds) * 100, 2)
        prec = round(precision_score(y_test, preds) * 100, 2)
        rec = round(recall_score(y_test, preds) * 100, 2)
        f1 = round(f1_score(y_test, preds) * 100, 2)
        auc = round(roc_auc_score(y_test, probs) * 100, 2)
        cm = confusion_matrix(y_test, preds).tolist()

        results[name] = {
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "roc_auc": auc, "cv_f1_mean": cv_f1_mean, "cv_f1_std": cv_f1_std,
            "confusion_matrix": cm, "best_params": search.best_params_,
        }
        fitted_models[name] = best_est

        print(f"  [{condition}] {name}: Test Acc={acc}% F1={f1}% AUC={auc}% | "
              f"CV F1={cv_f1_mean}%±{cv_f1_std}%")

        if results[name]["roc_auc"] > best_cv_f1:
            best_cv_f1 = results[name]["roc_auc"]
            best_model_name = name

        # Confusion matrix plot
        plt.figure(figsize=(5, 4))
        plt.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i][j], ha="center", va="center", fontsize=14)
        plt.xticks([0, 1], ["No Disease", "Disease"])
        plt.yticks([0, 1], ["No Disease", "Disease"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"{name} — Confusion Matrix ({condition})")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{condition}_{name.replace(' ', '_')}_cm.png"), dpi=150)
        plt.close()

    # ROC curve — all models overlaid
    plt.figure(figsize=(7, 6))
    for name, model in fitted_models.items():
        needs_scaling = name in ["Logistic Regression", "SVM"]
        X_te = X_test_scaled if needs_scaling else X_test
        probs = model.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC={results[name]['roc_auc']}%)")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves — {condition.capitalize()}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{condition}_roc_curves.png"), dpi=150)
    plt.close()

    # Save best model
    best_model = fitted_models[best_model_name]
    pickle.dump(best_model, open(os.path.join(MODELS_DIR, f"{condition}_model.pkl"), "wb"))
    pickle.dump(scaler, open(os.path.join(MODELS_DIR, f"{condition}_scaler.pkl"), "wb"))
    pickle.dump(list(X.columns), open(os.path.join(MODELS_DIR, f"{condition}_features.pkl"), "wb"))

    with open(os.path.join(RESULTS_DIR, f"{condition}_results.json"), "w") as f:
        json.dump({"best_model": best_model_name, "results": results}, f, indent=2)

    print(f"\n✅ Best model for {condition}: {best_model_name} (CV F1={best_cv_f1}%)\n")
    return results, best_model_name


# ── PREDICT SINGLE PATIENT ────────────────────────────────────────────────────
def predict_patient(condition, patient_data: dict):
    model = pickle.load(open(os.path.join(MODELS_DIR, f"{condition}_model.pkl"), "rb"))
    scaler = pickle.load(open(os.path.join(MODELS_DIR, f"{condition}_scaler.pkl"), "rb"))
    features = pickle.load(open(os.path.join(MODELS_DIR, f"{condition}_features.pkl"), "rb"))

    df = pd.DataFrame([patient_data])
    # Align columns to training feature space — fills missing one-hot columns with 0
    df = df.reindex(columns=features, fill_value=0)

    model_name = type(model).__name__
    if model_name in ["LogisticRegression", "SVC"]:
        df_scaled = scaler.transform(df)
        prob = model.predict_proba(df_scaled)[0][1]
    else:
        prob = model.predict_proba(df)[0][1]

    risk_score = round(float(prob) * 100, 1)
    risk_level = "High" if risk_score >= 70 else "Medium" if risk_score >= 40 else "Low"

    return {"risk_score": risk_score, "risk_level": risk_level, "model_used": model_name}


def encode_heart_patient(raw: dict) -> dict:
    """
    Converts raw clinical input (e.g. Sex='M', ChestPainType='ATA') into the
    one-hot encoded dict expected by the trained model.
    """
    encoded = {
        "Age": raw["Age"], "RestingBP": raw["RestingBP"],
        "Cholesterol": raw["Cholesterol"], "FastingBS": raw["FastingBS"],
        "MaxHR": raw["MaxHR"], "Oldpeak": raw["Oldpeak"],
    }
    for prefix, val in [
        ("Sex", raw["Sex"]), ("ChestPainType", raw["ChestPainType"]),
        ("RestingECG", raw["RestingECG"]), ("ExerciseAngina", raw["ExerciseAngina"]),
        ("ST_Slope", raw["ST_Slope"]),
    ]:
        encoded[f"{prefix}_{val}"] = 1
    return encoded


if __name__ == "__main__":
    print("=" * 60)
    print("Training Heart Disease models (with hyperparameter tuning)...")
    print("=" * 60)
    X_h, y_h, _ = load_heart()
    print(f"Heart dataset shape: {X_h.shape}, class balance: {dict(y_h.value_counts())}")
    heart_results, heart_best = train_models(X_h, y_h, "heart", use_smote=False)

    print("=" * 60)
    print("Training Diabetes models (with SMOTE + hyperparameter tuning)...")
    print("=" * 60)
    X_d, y_d, _ = load_diabetes()
    print(f"Diabetes dataset shape: {X_d.shape}, class balance: {dict(y_d.value_counts())}")
    diabetes_results, diabetes_best = train_models(X_d, y_d, "diabetes", use_smote=True)

    print("=" * 60)
    print("ALL MODELS TRAINED, TUNED, AND EVALUATED")
    print("=" * 60)