import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import pickle
import os

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HEART_PATH = os.path.join(BASE_DIR, "data", "heart.csv")
DIABETES_PATH = os.path.join(BASE_DIR, "data", "diabetes.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# ── loaders ────────────────────────────────────────────────────────────────
def load_heart():
    df = pd.read_csv(HEART_PATH)
    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y, list(X.columns)


def load_diabetes():
    df = pd.read_csv(DIABETES_PATH)
    # Replace biological zeros with NaN then fill with median
    cols_with_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)
    df[cols_with_zeros] = df[cols_with_zeros].fillna(df[cols_with_zeros].median())
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    return X, y, list(X.columns)


# ── train all four models ───────────────────────────────────────────────────
def train_models(X, y, condition):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=100, random_state=42,
            eval_metric="logloss", verbosity=0
        ),
    }

    results = {}
    best_model_name = None
    best_f1 = 0

    for name, model in models.items():
        # SVM and LR need scaled data; RF and XGBoost don't
        if name in ["Logistic Regression", "SVM"]:
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        acc = round(accuracy_score(y_test, preds) * 100, 2)
        prec = round(precision_score(y_test, preds) * 100, 2)
        rec = round(recall_score(y_test, preds) * 100, 2)
        f1 = round(f1_score(y_test, preds) * 100, 2)

        results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

        print(f"[{condition}] {name}: Acc={acc}% | Prec={prec}% | Rec={rec}% | F1={f1}%")

    # Save best model + scaler
    best_model = models[best_model_name]
    pickle.dump(best_model, open(os.path.join(MODELS_DIR, f"{condition}_model.pkl"), "wb"))
    pickle.dump(scaler, open(os.path.join(MODELS_DIR, f"{condition}_scaler.pkl"), "wb"))
    pickle.dump(X.columns.tolist(), open(os.path.join(MODELS_DIR, f"{condition}_features.pkl"), "wb"))

    print(f"\n✅ Best model for {condition}: {best_model_name} (F1={best_f1}%)\n")
    return results, best_model_name


# ── predict single patient ──────────────────────────────────────────────────
def predict_patient(condition, patient_data: dict):
    model = pickle.load(open(os.path.join(MODELS_DIR, f"{condition}_model.pkl"), "rb"))
    scaler = pickle.load(open(os.path.join(MODELS_DIR, f"{condition}_scaler.pkl"), "rb"))
    features = pickle.load(open(os.path.join(MODELS_DIR, f"{condition}_features.pkl"), "rb"))

    df = pd.DataFrame([patient_data])[features]

    model_name = type(model).__name__
    if model_name in ["LogisticRegression", "SVC"]:
        df_scaled = scaler.transform(df)
        prob = model.predict_proba(df_scaled)[0][1]
    else:
        prob = model.predict_proba(df)[0][1]

    risk_score = round(prob * 100, 1)

    if risk_score >= 70:
        risk_level = "High"
    elif risk_score >= 40:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "model_used": model_name,
    }


# ── run training when file is executed directly ─────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Training Heart Disease models...")
    print("=" * 50)
    X_h, y_h, _ = load_heart()
    heart_results, heart_best = train_models(X_h, y_h, "heart")

    print("=" * 50)
    print("Training Diabetes models...")
    print("=" * 50)
    X_d, y_d, _ = load_diabetes()
    diabetes_results, diabetes_best = train_models(X_d, y_d, "diabetes")

    print("=" * 50)
    print("ALL MODELS TRAINED AND SAVED")
    print("=" * 50)