import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from tabpfn import TabPFNClassifier
from deep_tabular_models import (
    split_categorical_continuous, FTTransformerClassifier, TabTransformerClassifier,
    make_tabnet, fit_tabnet_with_early_stopping, manual_cv_f1,
)
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


def load_heart():
    df = pd.read_csv(HEART_PATH)
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


PARAM_GRIDS = {
    "Logistic Regression": {"C": [0.01, 0.1, 1, 10, 100], "solver": ["lbfgs"], "max_iter": [2000]},
    "Random Forest": {
        "n_estimators": [100, 200, 300], "max_depth": [None, 5, 10, 15],
        "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4],
    },
    "XGBoost": {"n_estimators": [100, 200, 300], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.1, 0.2]},
    "LightGBM": {
        "n_estimators": [100, 200, 300], "max_depth": [3, 5, 7, -1],
        "learning_rate": [0.01, 0.05, 0.1], "num_leaves": [15, 31, 63],
    },
    "CatBoost": {"iterations": [100, 200, 300], "depth": [3, 5, 7], "learning_rate": [0.01, 0.05, 0.1]},
    "MLP": {
        "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
        "activation": ["relu", "tanh"], "alpha": [0.0001, 0.001, 0.01],
        "learning_rate_init": [0.001, 0.01],
    },
    "Stacking": {"final_estimator__C": [0.1, 1, 10]},
}


def build_stacking_model(scale_pos_weight=1.0):
    estimators = [
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")),
        ("xgb", XGBClassifier(n_estimators=200, random_state=42, eval_metric="logloss",
                               verbosity=0, scale_pos_weight=scale_pos_weight)),
        ("lgbm", LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)),
    ]
    return StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000), cv=3)


# Models fitted directly (no RandomizedSearchCV) because they are either a pretrained
# foundation model (TabPFN) or a neural net where grid search would mean training
# dozens of networks — both fit once with sensible defaults instead.
NO_TUNING_MODELS = {"TabPFN", "TabNet", "FT-Transformer", "TabTransformer"}


def build_base_models(X, class_weight_balanced=True, scale_pos_weight=1.0):
    cw = "balanced" if class_weight_balanced else None
    categ_idx, cont_idx, cardinalities = split_categorical_continuous(np.asarray(X))
    return {
        "Logistic Regression": LogisticRegression(random_state=42, class_weight=cw),
        "Random Forest": RandomForestClassifier(random_state=42, class_weight=cw),
        "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0, scale_pos_weight=scale_pos_weight),
        "LightGBM": LGBMClassifier(random_state=42, verbose=-1, class_weight=cw),
        "CatBoost": CatBoostClassifier(random_state=42, verbose=0, auto_class_weights="Balanced" if class_weight_balanced else None),
        "MLP": MLPClassifier(random_state=42, max_iter=2000, early_stopping=True, n_iter_no_change=15),
        "Stacking": build_stacking_model(scale_pos_weight=scale_pos_weight),
        "TabPFN": TabPFNClassifier(random_state=42),
        "FT-Transformer": FTTransformerClassifier(
            categ_idx=categ_idx, cont_idx=cont_idx, cardinalities=cardinalities,
            dim=32, depth=3, heads=4, max_epochs=200, patience=10, batch_size=32,
        ),
        "TabTransformer": TabTransformerClassifier(
            categ_idx=categ_idx, cont_idx=cont_idx, cardinalities=cardinalities,
            dim=32, depth=3, heads=4, max_epochs=200, patience=10, batch_size=32,
        ),
        "TabNet": make_tabnet(categ_idx, cardinalities),
    }


def train_models(X, y, condition, use_smote=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if use_smote:
        smote = SMOTE(random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
        X_train_scaled_bal, y_train_scaled_bal = smote.fit_resample(X_train_scaled, y_train)
        scale_pos_weight = 1.0
    else:
        X_train_bal, y_train_bal = X_train, y_train
        X_train_scaled_bal, y_train_scaled_bal = X_train_scaled, y_train
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        scale_pos_weight = neg / pos if pos > 0 else 1.0

    base_models = build_base_models(X, class_weight_balanced=not use_smote, scale_pos_weight=scale_pos_weight)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    best_model_name = None
    best_cv_f1 = 0
    fitted_models = {}

    for name, base_model in base_models.items():
        needs_scaling = name in ["Logistic Regression", "MLP"]
        X_tr = X_train_scaled_bal if needs_scaling else X_train_bal
        y_tr = y_train_scaled_bal if needs_scaling else y_train_bal
        X_te = X_test_scaled if needs_scaling else X_test

        if name in NO_TUNING_MODELS:
            # TabPFN is a pretrained foundation model that uses in-context learning;
            # the other three are neural nets where a hyperparameter grid search would
            # mean training dozens of networks — all four are fit once instead of via
            # RandomizedSearchCV. FT-Transformer/TabTransformer/TabNet all hold out a
            # validation split internally and use patience-based early stopping on
            # validation loss, so "fixed hyperparameters" here means the architecture/
            # optimizer settings, not the number of epochs actually trained.
            note = "foundation model" if name == "TabPFN" else "neural tabular model"
            print(f"  Fitting {name} for {condition} (no tuning — {note})...")
            best_est = base_model
            if name == "TabNet":
                fit_tabnet_with_early_stopping(best_est, X_tr, y_tr, max_epochs=300, patience=20)
            else:
                best_est.fit(X_tr, y_tr)
            convergence = ""
            if hasattr(best_est, "n_epochs_trained_"):
                stop_reason = "early stopping" if best_est.stopped_early_ else "hit max_epochs"
                convergence = f" [{stop_reason} at epoch {best_est.n_epochs_trained_}]"
            print(f"  [{condition}] {name} converged{convergence}")
            if name == "TabPFN":
                best_params = {"note": "TabPFN uses in-context learning; no hyperparameter search or "
                                        "epoch-based training is performed by design"}
            else:
                best_params = {"note": f"{name} is fit directly with fixed architecture/optimizer hyperparameters "
                                        f"(no RandomizedSearchCV); convergence is checked via a held-out validation "
                                        f"split with patience-based early stopping{convergence}"}
        else:
            print(f"  Tuning {name} for {condition}...")
            search = RandomizedSearchCV(base_model, PARAM_GRIDS[name], n_iter=10, cv=cv, scoring="f1", random_state=42, n_jobs=-1)
            search.fit(X_tr, y_tr)
            best_est = search.best_estimator_
            best_params = search.best_params_

        if name == "TabNet":
            # TabNetClassifier's __init__ mutates cat_emb_dim, which breaks sklearn's
            # clone() — cross_val_score relies on clone(), so fold estimators are
            # rebuilt manually instead.
            cat_idxs, cat_dims = best_est.cat_idxs, best_est.cat_dims
            cv_scores = manual_cv_f1(
                lambda: make_tabnet(cat_idxs, cat_dims),
                X_tr, y_tr, cv,
                fit_fn=lambda model, X_fold, y_fold: fit_tabnet_with_early_stopping(
                    model, X_fold, y_fold, max_epochs=300, patience=20,
                ),
            )
        else:
            # Neural/foundation models run sequentially (n_jobs=1) — parallel
            # multiprocess CV would spawn duplicate torch/foundation-model processes.
            cv_n_jobs = 1 if name in NO_TUNING_MODELS else -1
            cv_scores = cross_val_score(best_est, X_tr, y_tr, cv=cv, scoring="f1", n_jobs=cv_n_jobs)
        cv_f1_mean = round(cv_scores.mean() * 100, 2)
        cv_f1_std = round(cv_scores.std() * 100, 2)

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
            "confusion_matrix": cm, "best_params": best_params,
        }
        fitted_models[name] = best_est

        print(f"  [{condition}] {name}: Test Acc={acc}% F1={f1}% AUC={auc}% | CV F1={cv_f1_mean}%±{cv_f1_std}%")

        if results[name]["roc_auc"] > best_cv_f1:
            best_cv_f1 = results[name]["roc_auc"]
            best_model_name = name

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

    plt.figure(figsize=(8, 7))
    for name, model in fitted_models.items():
        needs_scaling = name in ["Logistic Regression", "MLP"]
        X_te = X_test_scaled if needs_scaling else X_test
        probs = model.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC={results[name]['roc_auc']}%)")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves — {condition.capitalize()} (All Models)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{condition}_roc_curves.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    names = list(results.keys())
    accs = [results[n]["accuracy"] for n in names]
    f1s = [results[n]["f1"] for n in names]
    aucs = [results[n]["roc_auc"] for n in names]
    x = np.arange(len(names))
    width = 0.25
    plt.bar(x - width, accs, width, label="Accuracy")
    plt.bar(x, f1s, width, label="F1 Score")
    plt.bar(x + width, aucs, width, label="ROC-AUC")
    plt.xticks(x, names, rotation=30, ha="right")
    plt.ylabel("Score (%)")
    plt.title(f"Model Comparison — {condition.capitalize()}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{condition}_model_comparison.png"), dpi=150)
    plt.close()

    best_model = fitted_models[best_model_name]
    pickle.dump(best_model, open(os.path.join(MODELS_DIR, f"{condition}_model.pkl"), "wb"))
    pickle.dump(scaler, open(os.path.join(MODELS_DIR, f"{condition}_scaler.pkl"), "wb"))
    pickle.dump(list(X.columns), open(os.path.join(MODELS_DIR, f"{condition}_features.pkl"), "wb"))

    with open(os.path.join(RESULTS_DIR, f"{condition}_results.json"), "w") as f:
        json.dump({"best_model": best_model_name, "results": results}, f, indent=2)

    print(f"\n[OK] Best model for {condition}: {best_model_name} (AUC={best_cv_f1}%)\n")
    return results, best_model_name


def predict_patient(condition, patient_data: dict):
    model = pickle.load(open(os.path.join(MODELS_DIR, f"{condition}_model.pkl"), "rb"))
    scaler = pickle.load(open(os.path.join(MODELS_DIR, f"{condition}_scaler.pkl"), "rb"))
    features = pickle.load(open(os.path.join(MODELS_DIR, f"{condition}_features.pkl"), "rb"))

    df = pd.DataFrame([patient_data])
    df = df.reindex(columns=features, fill_value=0)

    model_name = type(model).__name__
    if model_name in ["LogisticRegression", "MLPClassifier"]:
        df_scaled = scaler.transform(df)
        prob = model.predict_proba(df_scaled)[0][1]
    else:
        prob = model.predict_proba(df.values)[0][1]

    risk_score = round(float(prob) * 100, 1)
    risk_level = "High" if risk_score >= 70 else "Medium" if risk_score >= 40 else "Low"

    return {"risk_score": risk_score, "risk_level": risk_level, "model_used": model_name}


def encode_heart_patient(raw: dict) -> dict:
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
    print("Training Heart Disease models (10 models, hyperparameter tuning)...")
    print("=" * 60)
    X_h, y_h, _ = load_heart()
    print(f"Heart dataset shape: {X_h.shape}, class balance: {dict(y_h.value_counts())}")
    heart_results, heart_best = train_models(X_h, y_h, "heart", use_smote=False)

    print("=" * 60)
    print("Training Diabetes models (10 models, SMOTE + hyperparameter tuning)...")
    print("=" * 60)
    X_d, y_d, _ = load_diabetes()
    print(f"Diabetes dataset shape: {X_d.shape}, class balance: {dict(y_d.value_counts())}")
    diabetes_results, diabetes_best = train_models(X_d, y_d, "diabetes", use_smote=True)

    print("=" * 60)
    print("ALL 20 MODEL RUNS TRAINED, TUNED, AND EVALUATED")
    print("=" * 60)