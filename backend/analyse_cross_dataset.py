"""
HealthGuard AI - Cross-Dataset Results: Figures and Rank Analysis
=================================================================
Module 6WCM0029 | Muhammad Zubair | University of Hertfordshire

Reads results/cross_dataset_results.json (written by cross_dataset_validation.py)
and produces the figures and statistics for the results chapter.

Outputs to results/figures/:
    fig_rank_inversion.png   - slopegraph: internal -> external, per direction
    fig_internal_vs_external.png - scatter with parity line and Spearman rho
    fig_robustness_trap.png  - mean external AUC vs mean degradation

Also prints Spearman rank correlations, which quantify the central claim:
whether in-sample model ranking predicts out-of-sample model ranking.

Usage:
    python analyse_cross_dataset.py
    python analyse_cross_dataset.py path/to/cross_dataset_results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import spearmanr
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
    print("[warn] scipy not installed - rank correlations will be omitted")

HERE = Path(__file__).resolve().parent


def find_results() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).resolve()
    for c in [HERE / "results" / "cross_dataset_results.json",
              HERE.parent / "results" / "cross_dataset_results.json",
              Path.cwd() / "results" / "cross_dataset_results.json"]:
        if c.exists():
            return c.resolve()
    sys.exit("Could not find cross_dataset_results.json - pass its path as an argument.")


def load(path: Path):
    blob = json.loads(path.read_text())
    results = blob.get("results", blob)

    rows, aborted = [], []
    for key, block in results.items():
        if str(block.get("status", "")).startswith("ABORTED"):
            aborted.append((key, block))
            continue
        for model, m in block.get("models", {}).items():
            if "error" in m:
                continue
            rows.append({
                "direction": key,
                "model": model,
                "internal": m["internal_cv_auc"],
                "external": m["external"]["roc_auc"],
                "degradation": m["auc_degradation"],
                "sensitivity": m["external"]["sensitivity"],
                "specificity": m["external"]["specificity"],
                "brier": m["external"]["brier"],
            })
    return pd.DataFrame(rows), aborted


def rank_stats(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for d, sub in df.groupby("direction", sort=False):
        rho, p = (spearmanr(sub.internal, sub.external) if HAVE_SCIPY
                  else (np.nan, np.nan))
        ri = sub.internal.rank(ascending=False)
        re = sub.external.rank(ascending=False)
        out.append({
            "direction": d, "n_models": len(sub),
            "spearman_rho": round(float(rho), 3),
            "p_value": round(float(p), 3),
            "max_rank_shift": int((ri - re).abs().max()),
            "mean_degradation": round(float(sub.degradation.mean()), 4),
        })
    return pd.DataFrame(out)


def fig_rank_inversion(df, outdir):
    dirs = list(df.direction.unique())
    fig, axes = plt.subplots(1, len(dirs), figsize=(7.4 * len(dirs), 6.6))
    axes = np.atleast_1d(axes)
    cmap = plt.get_cmap("tab10")
    models = sorted(df.model.unique())
    colours = {m: cmap(i % 10) for i, m in enumerate(models)}

    def spread(values, min_gap):
        """Nudge near-identical label positions apart so text stays readable."""
        order = np.argsort(-np.asarray(values))
        placed = np.array(values, dtype=float)
        for i in range(1, len(order)):
            hi, lo = order[i - 1], order[i]
            if placed[hi] - placed[lo] < min_gap:
                placed[lo] = placed[hi] - min_gap
        return placed

    for ax, d in zip(axes, dirs):
        sub = df[df.direction == d].sort_values("internal", ascending=False)
        span = max(sub.internal.max(), sub.external.max()) - min(
            sub.internal.min(), sub.external.min(), 0.5)
        gap = span * 0.055
        li = spread(sub.internal.tolist(), gap)
        le = spread(sub.external.tolist(), gap)

        for (_, r), yi, ye in zip(sub.iterrows(), li, le):
            ax.plot([0, 1], [r.internal, r.external], "-o", lw=2.2,
                    ms=7, color=colours[r.model], alpha=0.85)
            # labels sit at the de-collided heights, joined back by leader lines
            ax.text(-0.10, yi, f"{r.model}  {r.internal:.3f}", ha="right",
                    va="center", fontsize=8.5, color=colours[r.model])
            ax.text(1.10, ye, f"{r.external:.3f}", ha="left", va="center",
                    fontsize=8.5, color=colours[r.model])
            ax.plot([-0.09, 0], [yi, r.internal], lw=0.7,
                    color=colours[r.model], alpha=0.45)
            ax.plot([1, 1.09], [r.external, ye], lw=0.7,
                    color=colours[r.model], alpha=0.45)
        ax.set_xlim(-1.05, 1.42)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["internal\n(5-fold CV on source)",
                            "external\n(held-out target dataset)"], fontsize=9)
        ax.axhline(0.5, ls=":", lw=1.2, color="grey")
        ax.text(1.40, 0.505, "chance", fontsize=8, color="grey", ha="right")
        title = d
        if HAVE_SCIPY:
            sd = rank_stats(df)
            rho = sd.loc[sd.direction == d, "spearman_rho"].iloc[0]
            pv = sd.loc[sd.direction == d, "p_value"].iloc[0]
            title += f"\nSpearman $\\rho$ = {rho:.3f} (p = {pv:.3f})"
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("ROC-AUC")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Does in-sample ranking survive external validation?",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    p = outdir / "fig_rank_inversion.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_scatter(df, outdir):
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    markers = {d: m for d, m in zip(df.direction.unique(), ["o", "s", "^", "D"])}
    cmap = plt.get_cmap("tab10")
    colours = {m: cmap(i % 10) for i, m in enumerate(sorted(df.model.unique()))}

    for _, r in df.iterrows():
        ax.scatter(r.internal, r.external, s=110, alpha=0.85,
                   color=colours[r.model], marker=markers[r.direction],
                   edgecolor="white", linewidth=1.2)

    lo = min(df.internal.min(), df.external.min()) - 0.03
    hi = max(df.internal.max(), df.external.max()) + 0.03
    ax.plot([lo, hi], [lo, hi], "--", color="grey", lw=1.4)
    ax.text(hi, hi, " perfect transfer", fontsize=9, color="grey", va="center")
    ax.axhline(0.5, ls=":", lw=1.2, color="crimson")
    ax.text(lo, 0.505, " chance (AUC 0.5)", fontsize=9, color="crimson")

    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=colours[m], label=m, ms=9)
         for m in sorted(df.model.unique())]
    h += [Line2D([], [], marker=markers[d], ls="", color="grey", label=d, ms=9)
          for d in df.direction.unique()]
    ax.legend(handles=h, fontsize=8, loc="upper left", framealpha=0.9)

    ax.set_xlabel("internal ROC-AUC (5-fold CV on the source dataset)")
    ax.set_ylabel("external ROC-AUC (held-out target dataset)")
    ax.set_title("Every point sits below the parity line:\n"
                 "no model transferred without loss", fontsize=12)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = outdir / "fig_internal_vs_external.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_trap(df, outdir):
    agg = (df.groupby("model")
             .agg(mean_internal=("internal", "mean"),
                  mean_external=("external", "mean"),
                  mean_drop=("degradation", "mean"))
             .sort_values("mean_external", ascending=False))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.4, 5.4))
    y = np.arange(len(agg))

    a1.barh(y, agg.mean_external, color="steelblue", alpha=0.85)
    a1.set_yticks(y); a1.set_yticklabels(agg.index, fontsize=9)
    a1.invert_yaxis()
    a1.axvline(0.5, ls=":", color="crimson", lw=1.5)
    a1.set_xlim(0.45, max(0.70, agg.mean_external.max() + 0.03))
    a1.set_xlabel("mean external ROC-AUC")
    a1.set_title("Ranked by external performance\n(what matters clinically)",
                 fontsize=11)
    for i, v in enumerate(agg.mean_external):
        a1.text(v + 0.003, i, f"{v:.4f}", va="center", fontsize=9)

    agg2 = agg.sort_values("mean_drop")
    y2 = np.arange(len(agg2))
    a2.barh(y2, agg2.mean_drop, color="indianred", alpha=0.85)
    a2.set_yticks(y2); a2.set_yticklabels(agg2.index, fontsize=9)
    a2.invert_yaxis()
    a2.set_xlabel("mean AUC degradation (smaller = 'more robust')")
    a2.set_title("Ranked by degradation\n(a misleading metric on its own)",
                 fontsize=11)
    for i, v in enumerate(agg2.mean_drop):
        a2.text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=9)

    fig.suptitle("Smallest degradation does not mean best external model",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    p = outdir / "fig_robustness_trap.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p, agg


def main():
    path = find_results()
    print("=" * 74)
    print("CROSS-DATASET RESULTS ANALYSIS")
    print("=" * 74)
    print(f"Reading: {path}")

    df, aborted = load(path)
    if df.empty:
        sys.exit("No usable results in that JSON.")

    outdir = path.parent / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    if aborted:
        print("\nEXPERIMENTS ABORTED BY THE INDEPENDENCE GATE")
        print("-" * 74)
        for key, blk in aborted:
            print(f"  {key}: {blk.get('status')}")
            print(f"    source rows found in target: {blk.get('pct_of_source_in_target')}%")
            print(f"    target rows found in source: {blk.get('pct_of_target_in_source')}%")

    print("\nRANK CORRELATION: does internal ranking predict external ranking?")
    print("-" * 74)
    rs = rank_stats(df)
    print(rs.to_string(index=False))
    if HAVE_SCIPY:
        print("\n  rho near 1.0  -> in-sample ranking transfers")
        print("  rho near 0.0  -> in-sample ranking is uninformative out-of-sample")

    print("\n\nPER-DIRECTION DETAIL")
    print("-" * 74)
    for d, sub in df.groupby("direction", sort=False):
        s = sub.sort_values("external", ascending=False)
        s = s.assign(rank_internal=sub.internal.rank(ascending=False).astype(int),
                     rank_external=sub.external.rank(ascending=False).astype(int))
        print(f"\n  {d}")
        print(s[["model", "internal", "external", "degradation",
                 "rank_internal", "rank_external"]].to_string(index=False))

    f1 = fig_rank_inversion(df, outdir)
    f2 = fig_scatter(df, outdir)
    f3, agg = fig_trap(df, outdir)

    print("\n\nAGGREGATE (mean across all directions)")
    print("-" * 74)
    print(agg.round(4).to_string())

    best_ext = agg.index[0]
    best_drop = agg.sort_values("mean_drop").index[0]
    if best_ext != best_drop:
        print(f"\n  Best mean EXTERNAL AUC : {best_ext}")
        print(f"  Smallest DEGRADATION   : {best_drop}")
        print("  These disagree. Degradation alone is not a robustness metric -")
        print("  a model that starts weak has less room to fall. Rank by")
        print("  external performance and report degradation alongside it.")

    print("\n\nFigures written:")
    for f in (f1, f2, f3):
        print(f"  {f}")

    csv = path.parent / "cross_dataset_analysis.csv"
    df.to_csv(csv, index=False)
    print(f"  {csv}")


if __name__ == "__main__":
    main()
