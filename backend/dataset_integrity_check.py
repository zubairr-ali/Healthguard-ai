"""
HealthGuard AI - Dataset Integrity Check
========================================
Module 6WCM0029 | Muhammad Zubair | University of Hertfordshire

Run this BEFORE trusting any cross-dataset validation result.

It answers one question: are the datasets actually independent?

If a "target" dataset contains rows from the "source" dataset, then an
external validation using them is not external at all - the model is being
tested on data it memorised. Tree ensembles will report near-perfect scores
while a linear model will not, because only the trees can memorise.

Checks performed:
  1. Exact duplicate rows WITHIN each dataset  (inflates cross-validation)
  2. Exact row OVERLAP BETWEEN paired datasets (invalidates external tests)
  3. Feature-only overlap, ignoring the label
  4. Class balance and basic distribution comparison

Usage:
    python dataset_integrity_check.py
    python dataset_integrity_check.py /path/to/data
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXPECTED = ["diabetes.csv", "diabetes_frankfurt.csv", "heart.csv", "cardio_train.csv"]


def find_data_dir() -> Path:
    candidates = [HERE / "data", HERE.parent / "data", HERE,
                  Path.cwd() / "data", Path.cwd()]
    seen, ranked = set(), []
    for c in candidates:
        c = c.resolve()
        if c in seen or not c.is_dir():
            continue
        seen.add(c)
        ranked.append((sum((c / f).exists() for f in EXPECTED), c))
    ranked.sort(key=lambda t: -t[0])
    return ranked[0][1] if ranked and ranked[0][0] else (HERE / "data").resolve()


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    if df.shape[1] == 1:
        df = pd.read_csv(path)
    return df


def _fingerprint(df: pd.DataFrame, cols) -> pd.Series:
    """Round floats then join to a string so near-identical rows match."""
    sub = df[cols].copy()
    for c in sub.columns:
        if pd.api.types.is_numeric_dtype(sub[c]):
            sub[c] = sub[c].round(4)
    arr = sub.to_numpy(dtype=object)
    return pd.Series(["|".join(map(str, r)) for r in arr], index=sub.index)


def within(name: str, df: pd.DataFrame) -> None:
    n = len(df)
    dup = int(df.duplicated().sum())
    pct = 100 * dup / n if n else 0
    flag = "  <-- inflates cross-validation" if pct > 1 else ""
    print(f"  {name:<26} {n:>7,} rows   {dup:>6,} exact duplicates "
          f"({pct:5.2f}%){flag}")


def between(a_name: str, a: pd.DataFrame, b_name: str, b: pd.DataFrame,
            label: str) -> dict:
    shared = [c for c in a.columns if c in b.columns]
    feats = [c for c in shared if c != label]
    if not feats:
        print(f"  {a_name} / {b_name}: no shared columns, cannot compare")
        return {}

    fa_full, fb_full = _fingerprint(a, shared), _fingerprint(b, shared)
    fa_feat, fb_feat = _fingerprint(a, feats), _fingerprint(b, feats)

    set_b_full, set_b_feat = set(fb_full), set(fb_feat)
    full_hits = int(fa_full.isin(set_b_full).sum())
    feat_hits = int(fa_feat.isin(set_b_feat).sum())

    pct_a = 100 * full_hits / len(a)
    pct_b = 100 * int(fb_full.isin(set(fa_full)).sum()) / len(b)

    print(f"\n  {a_name}  vs  {b_name}")
    print(f"    shared columns              : {len(shared)}")
    print(f"    rows of {a_name} found in {b_name} : {full_hits:,} "
          f"({pct_a:.1f}% of {a_name})")
    print(f"    rows of {b_name} found in {a_name} : "
          f"{int(fb_full.isin(set(fa_full)).sum()):,} ({pct_b:.1f}% of {b_name})")
    print(f"    feature-only matches (label ignored): {feat_hits:,}")

    if pct_a > 5 or pct_b > 5:
        print(f"\n    *** CONTAMINATION DETECTED ***")
        print(f"    These datasets are NOT independent. Any 'external'")
        print(f"    validation between them measures memorisation, not")
        print(f"    generalisation. Do not report those AUC figures as")
        print(f"    external validation.")
    else:
        print(f"    -> no meaningful overlap; safe to treat as independent")

    return {"shared_cols": len(shared), "a_in_b": full_hits,
            "pct_a": round(pct_a, 2), "pct_b": round(pct_b, 2),
            "feature_only": feat_hits}


def compare_distributions(a_name, a, b_name, b, label):
    shared = [c for c in a.columns if c in b.columns and c != label]
    num = [c for c in shared if pd.api.types.is_numeric_dtype(a[c])
           and pd.api.types.is_numeric_dtype(b[c])]
    if not num:
        return
    rows = []
    for c in num:
        rows.append({
            "feature": c,
            f"{a_name}_mean": round(float(a[c].mean()), 2),
            f"{b_name}_mean": round(float(b[c].mean()), 2),
            f"{a_name}_std": round(float(a[c].std()), 2),
            f"{b_name}_std": round(float(b[c].std()), 2),
        })
    print(f"\n    Feature distributions:")
    print(pd.DataFrame(rows).to_string(index=False))
    if label in a.columns and label in b.columns:
        print(f"\n    positive rate: {a_name} {a[label].mean():.3f}   "
              f"{b_name} {b[label].mean():.3f}")


def main():
    data_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else find_data_dir()
    print("=" * 74)
    print("DATASET INTEGRITY CHECK")
    print("=" * 74)
    print(f"Data directory: {data_dir}\n")

    loaded = {}
    for f in EXPECTED:
        p = data_dir / f
        if p.exists():
            loaded[f] = _read(p)
        else:
            print(f"  MISSING {f}")

    print("1. DUPLICATE ROWS WITHIN EACH DATASET")
    print("-" * 74)
    for f, df in loaded.items():
        within(f, df)

    print("\n\n2. OVERLAP BETWEEN PAIRED DATASETS")
    print("-" * 74)

    if "diabetes.csv" in loaded and "diabetes_frankfurt.csv" in loaded:
        pima, frank = loaded["diabetes.csv"], loaded["diabetes_frankfurt.csv"]
        between("Pima", pima, "Frankfurt", frank, "Outcome")
        compare_distributions("Pima", pima, "Frankfurt", frank, "Outcome")

    if "heart.csv" in loaded and "cardio_train.csv" in loaded:
        heart, cardio = loaded["heart.csv"], loaded["cardio_train.csv"]
        common = [c for c in heart.columns if c in cardio.columns]
        print(f"\n  Heart  vs  Cardio")
        print(f"    raw shared column names     : {len(common)} {common}")
        print(f"    -> schemas differ by design; harmonisation is handled")
        print(f"       inside cross_dataset_validation.py, so a raw row")
        print(f"       overlap check is not meaningful here.")

    print("\n" + "=" * 74)
    print("HOW TO READ THIS")
    print("=" * 74)
    print("""
If overlap between Pima and Frankfurt is large, Experiment A in your
cross-dataset validation is invalid as written. That is a FINDING, not a
failure - dataset provenance problems of exactly this kind are a known
and under-reported issue in clinical ML benchmarking, and documenting one
in your own pipeline is stronger evidence of methodological rigour than
a clean result would have been.

The fix is to deduplicate: remove from the target dataset every row that
also appears in the source, then re-run. Report both the contaminated and
the cleaned figures, and explain the difference.
""")


if __name__ == "__main__":
    main()
