"""
HealthGuard AI - Cross-Dataset External Validation
===================================================
Module 6WCM0029 | Muhammad Zubair | University of Hertfordshire

Addresses supervisor feedback points (4), (7), (13) and (14): compare models
across multiple datasets with overlapping feature subsets, so that any claim
about "the most accurate model" is justified rather than assumed.

TWO EXPERIMENTS
---------------
Experiment A - DIABETES (clean external validation)
    Pima Indians (768 rows) <-> Frankfurt Hospital (2,000 rows)
    Identical 8-feature schema. No harmonisation needed. This is a true
    external validation: same features, same label definition, different
    population and different data collection site.

Experiment B - CARDIOVASCULAR (harmonised external validation)
    Heart Disease combined (918 rows) <-> Cardiovascular Disease (70,000 rows)
    Different schemas. A 5-feature clinically-common subset is derived.
    NOTE: the label definitions genuinely differ (angiographic coronary
    disease vs. broader cardiovascular disease presence). Degradation here
    therefore reflects BOTH distribution shift and construct mismatch, and
    must be reported as such. This is a limitation, not a bug.

WHAT IS MEASURED
----------------
For each direction (source -> target):
    internal  = stratified 5-fold CV ROC-AUC on the source dataset
    external  = train on all of source, evaluate on all of target
    degradation = internal - external
Degradation is the headline number. It answers the supervisor's question
about whether benchmark performance implies real-world reliability.

USAGE
-----
    python cross_dataset_validation.py
Expects CSVs in ../data/ (see DATA_DIR below). Writes results to
../results/cross_dataset_results.json and prints summary tables.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             confusion_matrix, matthews_corrcoef,
                             brier_score_loss)

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

HERE = Path(__file__).resolve().parent

# Expected filenames (wherever the data folder turns out to be)
F_PIMA = "diabetes.csv"
F_FRANKFURT = "diabetes_frankfurt.csv"
F_HEART = "heart.csv"
F_CARDIO = "cardio_train.csv"
EXPECTED = [F_PIMA, F_FRANKFURT, F_HEART, F_CARDIO]


def find_data_dir() -> Path:
    """Locate the data folder without the user having to care where it is.

    Checks the common layouts (data/ beside the script, data/ at repo root,
    the script's own folder, the current working directory) and picks
    whichever contains the most of the expected CSVs.
    """
    candidates = [
        HERE / "data",           # backend/data/
        HERE.parent / "data",    # repo-root/data/
        HERE,                    # beside the script
        Path.cwd() / "data",
        Path.cwd(),
    ]
    seen, ranked = set(), []
    for c in candidates:
        c = c.resolve()
        if c in seen or not c.is_dir():
            continue
        seen.add(c)
        ranked.append((sum((c / f).exists() for f in EXPECTED), c))

    ranked.sort(key=lambda t: -t[0])
    if ranked and ranked[0][0] > 0:
        return ranked[0][1]
    return (HERE / "data").resolve()


def report_data_dir(data_dir: Path) -> None:
    print(f"\nLooking for CSVs in: {data_dir}")
    for f in EXPECTED:
        path = data_dir / f
        if path.exists():
            print(f"   FOUND    {f}")
        else:
            print(f"   MISSING  {f}")
    if not any((data_dir / f).exists() for f in EXPECTED):
        print("\n   No CSVs found in any of the usual places.")
        print("   Put your four CSVs in a folder named 'data' next to this")
        print("   script, or pass the folder explicitly:")
        print("       python cross_dataset_validation.py /path/to/data")


DATA_DIR = find_data_dir()
RESULTS_DIR = HERE / "results"


# ══════════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════════

def build_models():
    """Return {name: estimator}. Optional libraries are skipped if absent."""
    models = {
        "Logistic Regression": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=SEED)),
        ]),
        "Random Forest": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=300, random_state=SEED,
                                           n_jobs=-1)),
        ]),
    }

    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=5,
            eval_metric="logloss", random_state=SEED, n_jobs=-1)
    except ImportError:
        print("  [skip] xgboost not installed")

    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300, learning_rate=0.1, random_state=SEED,
            n_jobs=-1, verbose=-1)
    except ImportError:
        print("  [skip] lightgbm not installed")

    try:
        from catboost import CatBoostClassifier
        models["CatBoost"] = CatBoostClassifier(
            iterations=300, learning_rate=0.1, depth=5,
            random_seed=SEED, verbose=0, allow_writing_files=False)
    except ImportError:
        print("  [skip] catboost not installed")

    try:
        from tabpfn import TabPFNClassifier  # noqa: F401
        models["TabPFN"] = "TABPFN"  # special-cased in fit_predict
    except ImportError:
        print("  [skip] tabpfn not installed")

    return models


# TabPFN refuses >5000 training samples on CPU. It is designed for the
# small-sample regime; this cap is a property of the model, not a workaround.
TABPFN_MAX_TRAIN = 5_000


def _instantiate(model):
    if model == "TABPFN":
        from tabpfn import TabPFNClassifier
        try:
            return TabPFNClassifier(ignore_pretraining_limits=True)
        except TypeError:            # older tabpfn without the kwarg
            return TabPFNClassifier()
    from sklearn.base import clone
    return clone(model)


def fit_predict_proba(model, X_tr, y_tr, X_te):
    """Fit a fresh copy and return P(class=1) on X_te."""
    est = _instantiate(model)

    if model == "TABPFN":
        # TabPFN cannot ingest NaNs reliably and is capped on training size
        X_tr = pd.DataFrame(X_tr).fillna(pd.DataFrame(X_tr).median()).values
        X_te = pd.DataFrame(X_te).fillna(pd.DataFrame(X_te).median()).values
        if len(X_tr) > TABPFN_MAX_TRAIN:
            idx = np.random.RandomState(SEED).choice(
                len(X_tr), TABPFN_MAX_TRAIN, replace=False)
            X_tr, y_tr = X_tr[idx], np.asarray(y_tr)[idx]

    est.fit(X_tr, y_tr)
    return est.predict_proba(X_te)[:, 1]


def score_all(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "sensitivity": round(float(tp / (tp + fn)) if (tp + fn) else 0.0, 4),
        "specificity": round(float(tn / (tn + fp)) if (tn + fp) else 0.0, 4),
        "mcc": round(float(matthews_corrcoef(y_true, pred)), 4),
        "brier": round(float(brier_score_loss(y_true, proba)), 4),
    }


# ══════════════════════════════════════════════════════════════════════
# LOADERS + HARMONISATION
# ══════════════════════════════════════════════════════════════════════

DIABETES_FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                     "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

# Columns where a literal 0 is physiologically impossible => missing data
DIABETES_ZERO_AS_NAN = ["Glucose", "BloodPressure", "SkinThickness",
                        "Insulin", "BMI"]

CARDIO_COMMON = ["age", "sex", "systolic_bp", "chol_cat", "gluc_elevated"]


def load_diabetes(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in DIABETES_FEATURES + ["Outcome"] if c not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing columns {missing}. Found: {list(df.columns)}")
    df = df[DIABETES_FEATURES + ["Outcome"]].copy()
    # Recode impossible zeros as NaN; imputation happens inside each pipeline
    for c in DIABETES_ZERO_AS_NAN:
        df.loc[df[c] == 0, c] = np.nan
    return df


def _chol_mgdl_to_cat(series: pd.Series) -> pd.Series:
    """Map mg/dL cholesterol onto the cardio dataset's 1/2/3 ordinal scale.

    Clinical cut-points (NCEP ATP III):
        <200 = normal (1), 200-239 = above normal (2), >=240 = well above (3)
    """
    out = pd.Series(np.nan, index=series.index, dtype=float)
    out[series < 200] = 1
    out[(series >= 200) & (series < 240)] = 2
    out[series >= 240] = 3
    return out


def load_heart_common(path: Path) -> pd.DataFrame:
    """fedesoriano combined heart dataset -> harmonised common subset."""
    df = pd.read_csv(path)
    need = ["Age", "Sex", "RestingBP", "Cholesterol", "FastingBS", "HeartDisease"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"heart.csv: missing columns {missing}. Found: {list(df.columns)}")

    out = pd.DataFrame(index=df.index)
    out["age"] = df["Age"].astype(float)
    out["sex"] = (df["Sex"].astype(str).str.upper() == "M").astype(int)

    # RestingBP == 0 is a known data-quality artefact in this release
    bp = df["RestingBP"].astype(float).replace(0, np.nan)
    out["systolic_bp"] = bp

    # Cholesterol == 0 encodes "not measured" for ~172 rows
    chol = df["Cholesterol"].astype(float).replace(0, np.nan)
    out["chol_cat"] = _chol_mgdl_to_cat(chol)

    # FastingBS is already binary: 1 if fasting blood sugar > 120 mg/dL
    out["gluc_elevated"] = df["FastingBS"].astype(int)

    out["target"] = df["HeartDisease"].astype(int)
    return out.dropna(subset=["target"])


def load_cardio_common(path: Path) -> pd.DataFrame:
    """sulianova cardiovascular dataset -> harmonised common subset."""
    # This file ships semicolon-delimited; fall back to comma if needed.
    df = pd.read_csv(path, sep=";")
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=",")

    need = ["age", "gender", "ap_hi", "cholesterol", "gluc", "cardio"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"cardio_train.csv: missing columns {missing}. Found: {list(df.columns)}")

    n_raw = len(df)
    out = pd.DataFrame(index=df.index)
    out["age"] = df["age"].astype(float) / 365.25          # stored in days
    out["sex"] = (df["gender"].astype(int) == 2).astype(int)  # 2 = male, 1 = female
    out["systolic_bp"] = df["ap_hi"].astype(float)
    out["chol_cat"] = df["cholesterol"].astype(float)       # already 1/2/3
    out["gluc_elevated"] = (df["gluc"].astype(int) > 1).astype(int)
    out["target"] = df["cardio"].astype(int)

    # This release contains extreme BP artefacts (negatives, values >10000).
    # Restrict to a physiologically plausible range.
    ok = out["systolic_bp"].between(70, 250) & out["age"].between(18, 100)
    out = out[ok].copy()
    print(f"    cardio: kept {len(out):,} of {n_raw:,} rows after BP/age sanity filter")
    return out


# ══════════════════════════════════════════════════════════════════════
# CONTAMINATION GATE
# ══════════════════════════════════════════════════════════════════════

CONTAMINATION_THRESHOLD_PCT = 5.0


def overlap_pct(a: pd.DataFrame, b: pd.DataFrame) -> tuple:
    """Return (% of a found in b, % of b found in a) on shared columns."""
    shared = [c for c in a.columns if c in b.columns]
    if not shared:
        return 0.0, 0.0

    def fp(df):
        sub = df[shared].copy()
        for c in sub.columns:
            if pd.api.types.is_numeric_dtype(sub[c]):
                sub[c] = sub[c].round(4)
        arr = sub.to_numpy(dtype=object)
        return pd.Series(["|".join(map(str, r)) for r in arr], index=sub.index)

    fa, fb = fp(a), fp(b)
    return (100 * fa.isin(set(fb)).sum() / len(a),
            100 * fb.isin(set(fa)).sum() / len(b))


def independence_gate(a_name, a, b_name, b, results) -> bool:
    """Refuse to run an external validation on non-independent datasets."""
    pa, pb = overlap_pct(a, b)
    print(f"\n    Independence check: {pa:.1f}% of {a_name} appears in "
          f"{b_name}; {pb:.1f}% of {b_name} appears in {a_name}")

    if max(pa, pb) < CONTAMINATION_THRESHOLD_PCT:
        print("    -> independent; proceeding")
        return True

    print("\n    *** ABORTED: DATASETS ARE NOT INDEPENDENT ***")
    print("    An 'external' validation between these datasets would measure")
    print("    memorisation, not generalisation. No AUC figures are reported")
    print("    for this experiment. See dataset_integrity_check.py output.")
    results[f"{a_name} <-> {b_name}"] = {
        "status": "ABORTED_CONTAMINATED",
        "pct_of_source_in_target": round(float(pa), 2),
        "pct_of_target_in_source": round(float(pb), 2),
        "threshold_pct": CONTAMINATION_THRESHOLD_PCT,
        "reason": ("Datasets share rows and cannot be treated as independent "
                   "cohorts. Any external-validation metric computed between "
                   "them reflects memorisation of duplicated patients."),
    }
    return False


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT RUNNER
# ══════════════════════════════════════════════════════════════════════

def run_direction(models, src_name, src_df, tgt_name, tgt_df, features, results):
    """Internal CV on source, then external evaluation on target."""
    Xs = src_df[features].values.astype(float)
    ys = src_df["target"].values.astype(int)
    Xt = tgt_df[features].values.astype(float)
    yt = tgt_df["target"].values.astype(int)

    key = f"{src_name} -> {tgt_name}"
    print(f"\n  {key}   (train n={len(Xs):,}  test n={len(Xt):,})")
    print(f"  {'model':<22} {'internal':>9} {'external':>9} {'drop':>8}")
    print("  " + "-" * 52)

    results[key] = {
        "source": src_name, "target": tgt_name,
        "n_source": int(len(Xs)), "n_target": int(len(Xt)),
        "features": features, "models": {},
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for name, model in models.items():
        try:
            # --- internal: 5-fold CV on the source dataset ---
            if model == "TABPFN":
                # cross_val_score cannot clone the sentinel; do folds manually
                aucs = []
                for tr, te in cv.split(Xs, ys):
                    p = fit_predict_proba(model, Xs[tr], ys[tr], Xs[te])
                    aucs.append(roc_auc_score(ys[te], p))
                internal = float(np.mean(aucs))
            else:
                internal = float(np.mean(cross_val_score(
                    _instantiate(model), Xs, ys, cv=cv, scoring="roc_auc",
                    n_jobs=1)))

            # --- external: fit on all of source, score on all of target ---
            proba = fit_predict_proba(model, Xs, ys, Xt)
            ext = score_all(yt, proba)
            drop = internal - ext["roc_auc"]

            results[key]["models"][name] = {
                "internal_cv_auc": round(internal, 4),
                "external": ext,
                "auc_degradation": round(drop, 4),
            }
            print(f"  {name:<22} {internal:>9.4f} {ext['roc_auc']:>9.4f} "
                  f"{drop:>+8.4f}")

        except Exception as e:
            print(f"  {name:<22}  FAILED: {type(e).__name__}: {e}")
            results[key]["models"][name] = {"error": f"{type(e).__name__}: {e}"}


def summarise(results):
    rows = []
    for key, block in results.items():
        if block.get("status", "").startswith("ABORTED"):
            continue
        for model, m in block["models"].items():
            if "error" in m:
                continue
            rows.append({
                "direction": key,
                "model": model,
                "internal_auc": m["internal_cv_auc"],
                "external_auc": m["external"]["roc_auc"],
                "degradation": m["auc_degradation"],
                "ext_sensitivity": m["external"]["sensitivity"],
                "ext_specificity": m["external"]["specificity"],
                "ext_brier": m["external"]["brier"],
            })
    if not rows:
        return None
    df = pd.DataFrame(rows)

    print("\n" + "=" * 78)
    print("PER-DIRECTION RESULTS (report these; the averages below are secondary)")
    print("=" * 78)
    for key in df["direction"].unique():
        sub = df[df.direction == key].sort_values("degradation")
        print(f"\n  {key}")
        print(sub[["model", "internal_auc", "external_auc",
                   "degradation"]].to_string(index=False))

    # Only average over directions that EVERY model completed. Averaging a
    # model that failed a hard direction against models that attempted it
    # would rank the failure as robustness.
    n_dir = df["direction"].nunique()
    coverage = df.groupby("model")["direction"].nunique()
    complete = coverage[coverage == n_dir].index.tolist()
    dropped = coverage[coverage < n_dir]

    print("\n" + "=" * 78)
    print("MEAN AUC DEGRADATION (models that completed all directions only)")
    print("=" * 78)
    if len(dropped):
        print("EXCLUDED from the ranking - did not complete every direction:")
        for m, c in dropped.items():
            print(f"   {m}: {c}/{n_dir} directions")
        print("   Averaging these against complete models would reward failure.")
        print()

    if complete:
        rank = (df[df.model.isin(complete)].groupby("model")
                  .agg(mean_internal=("internal_auc", "mean"),
                       mean_external=("external_auc", "mean"),
                       mean_degradation=("degradation", "mean"))
                  .sort_values("mean_degradation").round(4))
        print(rank.to_string())
    else:
        print("No model completed every direction; use the per-direction table.")

    print("\nNOTE: the model with the highest internal AUC is not necessarily")
    print("the one with the smallest degradation. That distinction is the point")
    print("of this experiment - report both.")
    print("\nWARNING: a NEGATIVE degradation (external better than internal) is")
    print("not good news. It usually means the two datasets are not independent.")
    print("Run dataset_integrity_check.py before interpreting any such result.")
    return df


def main():
    global DATA_DIR
    print("=" * 78)
    print("HealthGuard AI - Cross-Dataset External Validation")
    print("=" * 78)

    if len(sys.argv) > 1:                       # explicit override wins
        DATA_DIR = Path(sys.argv[1]).resolve()
    report_data_dir(DATA_DIR)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nBuilding model registry...")
    models = build_models()
    print(f"  {len(models)} models ready: {', '.join(models)}")

    results = {}

    # ---------- Experiment A : diabetes ----------
    pima_p, frank_p = DATA_DIR / F_PIMA, DATA_DIR / F_FRANKFURT
    if pima_p.exists() and frank_p.exists():
        print("\n" + "=" * 78)
        print("EXPERIMENT A - DIABETES (identical schema, true external validation)")
        print("=" * 78)
        pima = load_diabetes(pima_p, "Pima")
        frank = load_diabetes(frank_p, "Frankfurt")
        print(f"    Pima: {len(pima):,} rows, {pima['Outcome'].mean():.1%} positive")
        print(f"    Frankfurt: {len(frank):,} rows, {frank['Outcome'].mean():.1%} positive")
        pima = pima.rename(columns={"Outcome": "target"})
        frank = frank.rename(columns={"Outcome": "target"})
        if independence_gate("Pima", pima, "Frankfurt", frank, results):
            run_direction(models, "Pima", pima, "Frankfurt", frank,
                          DIABETES_FEATURES, results)
            run_direction(models, "Frankfurt", frank, "Pima", pima,
                          DIABETES_FEATURES, results)
    else:
        print(f"\n[skip] Experiment A - need {F_PIMA} and {F_FRANKFURT} in {DATA_DIR}")

    # ---------- Experiment B : cardiovascular ----------
    heart_p, cardio_p = DATA_DIR / F_HEART, DATA_DIR / F_CARDIO
    if heart_p.exists() and cardio_p.exists():
        print("\n" + "=" * 78)
        print("EXPERIMENT B - CARDIOVASCULAR (harmonised 5-feature subset)")
        print("=" * 78)
        print("    WARNING: label constructs differ between these datasets.")
        print("    Heart = angiographic coronary disease; Cardio = CVD presence.")
        print("    Degradation reflects construct mismatch as well as shift.")
        heart = load_heart_common(heart_p)
        cardio = load_cardio_common(cardio_p)
        print(f"    Heart: {len(heart):,} rows, {heart['target'].mean():.1%} positive")
        print(f"    Cardio: {len(cardio):,} rows, {cardio['target'].mean():.1%} positive")
        run_direction(models, "Heart", heart, "Cardio", cardio,
                      CARDIO_COMMON, results)
        run_direction(models, "Cardio", cardio, "Heart", heart,
                      CARDIO_COMMON, results)
    else:
        print(f"\n[skip] Experiment B - need {F_HEART} and {F_CARDIO} in {DATA_DIR}")

    if not results:
        print("\nNo experiments ran. Check your CSV filenames in ../data/.")
        sys.exit(1)

    df = summarise(results)

    out_path = RESULTS_DIR / "cross_dataset_results.json"
    with open(out_path, "w") as f:
        json.dump({"seed": SEED,
                   "tabpfn_max_train": TABPFN_MAX_TRAIN,
                   "results": results}, f, indent=2)
    print(f"\nSaved: {out_path}")

    if df is not None:
        csv_path = RESULTS_DIR / "cross_dataset_summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
