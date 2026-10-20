"""Paired comparisons of the GP variants against every reference model.

Regenerates the numbers of the baseline tables in the results chapter from the
pipeline artifacts, so they are reproducible instead of resting on an ad-hoc
calculation. Reads the two prediction files the pipeline writes -- the baselines
stage publishes most models, the evaluate stage the one-hot random forest -- and
prints both tables as LaTeX rows.

    uv run --no-sync python experiments/08_baseline_paired_tests.py
    uv run --no-sync python experiments/08_baseline_paired_tests.py --experiment reference_cpu

The bootstrap is seeded, so repeated runs give identical intervals. Holm
correction is applied within the comparisons of each GP variant, so adding or
removing a reference model changes every corrected p-value in that block.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

GP_VARIANTS = {
    "gp_hardened_physical_epistatic": "Epistatic GP, fixed configuration",
    "gp_nested_source_selection": "Epistatic GP, nested selection",
}

# Printed label and representation for the results tables.
REFERENCES = {
    "random_forest_onehot_fixed": ("Random forest (one-hot)", "one-hot"),
    "tabicl_v2_physical": ("TabICL", "physical"),
    "random_forest_physical_fixed": ("Random forest (physical)", "physical"),
    "ridge_physical": ("Ridge regression", "physical"),
    "bayesian_ridge_physical": ("Bayesian ridge regression", "physical"),
    "knn_physical": ("$k$-nearest neighbors", "physical"),
    "train_mean": ("Training mean", "---"),
}

BOOTSTRAP = 10_000
SEED = 20261011


def load(root, experiment):
    base = Path(root) / "results" / experiment
    columns = ["model", "variant_id", "observed", "predicted", "abs_error"]
    frames = [
        pd.read_csv(base / "baselines/all_loocv_predictions.csv")[columns],
        pd.read_csv(base / "evaluate/baseline_predictions.csv")[columns],
    ]
    data = pd.concat(frames, ignore_index=True)
    # train_mean is published by both stages with identical values.
    data = data.drop_duplicates(["model", "variant_id"], keep="first")
    return data


def paired(data, gp, reference, rng):
    """Per-variant error difference, aligned on variant_id."""
    a = data[data.model == gp].set_index("variant_id")
    b = data[data.model == reference].set_index("variant_id")
    ids = a.index.intersection(b.index)
    if len(ids) != len(a) or len(ids) != len(b):
        raise ValueError(f"{gp} and {reference} do not cover the same variants")
    gp_err = a.loc[ids, "abs_error"].to_numpy(float)
    ref_err = b.loc[ids, "abs_error"].to_numpy(float)
    if not (np.isfinite(gp_err).all() and np.isfinite(ref_err).all()):
        raise ValueError(f"Non-finite absolute errors for {gp} vs {reference}")
    delta = ref_err - gp_err  # positive = the GP is more accurate
    n = len(delta)
    draws = rng.integers(0, n, size=(BOOTSTRAP, n))
    means = delta[draws].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    # Wilcoxon is undefined when every paired difference is zero.
    p = float(wilcoxon(delta)[1]) if np.any(delta != 0) else float("nan")
    observed = a.loc[ids, "observed"].to_numpy(float)
    rho_gp = spearmanr(observed, a.loc[ids, "predicted"]).statistic
    rho_ref = spearmanr(observed, b.loc[ids, "predicted"]).statistic
    return {
        "reference": reference,
        "delta_mae": delta.mean(),
        "low": low,
        "high": high,
        "median": float(np.median(delta)),
        "n_plus": int((delta > 0).sum()),
        "p": p,
        "delta_rho": rho_gp - rho_ref,
    }


def holm(p_values):
    """Holm-Bonferroni within one family of comparisons."""
    order = np.argsort(p_values)
    corrected = np.empty(len(p_values))
    running = 0.0
    for rank, index in enumerate(order):
        value = (len(p_values) - rank) * p_values[index]
        running = max(running, value)
        corrected[index] = min(running, 1.0)
    return corrected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=".")
    parser.add_argument("--experiment", default="baselines_tabicl")
    args = parser.parse_args(argv)

    data = load(args.root, args.experiment)
    present = set(data.model)
    references = [name for name in REFERENCES if name in present]
    missing = [name for name in REFERENCES if name not in present]
    if missing:
        print(f"note: not in this run, omitted: {', '.join(missing)}\n")

    metrics = pd.read_csv(
        Path(args.root) / "results" / args.experiment / "baselines/all_loocv_metrics.csv"
    )
    print(f"% n = {data.variant_id.nunique()} variants; bootstrap {BOOTSTRAP}, seed {SEED}")
    print("\n% ---- tab:baselines ----")
    merged = pd.concat(
        [
            metrics,
            pd.read_csv(
                Path(args.root) / "results" / args.experiment / "evaluate/baseline_metrics.csv"
            ),
        ],
        ignore_index=True,
    ).drop_duplicates("model", keep="first")
    for _, row in merged.sort_values("mae").iterrows():
        if row.model in GP_VARIANTS or row.model in REFERENCES:
            label = GP_VARIANTS.get(row.model) or REFERENCES[row.model][0]
            rep = "selected" if row.model == "gp_nested_source_selection" else (
                "physical" if row.model in GP_VARIANTS else REFERENCES[row.model][1]
            )
            rho = "---" if row.model == "train_mean" else f"{row.spearman:.3f}"
            r2 = f"{row.r2:.3f}" if row.r2 >= 0 else f"$-{abs(row.r2):.3f}$"
            print(f"    {label:25s} & {rep:8s} & {row.mae:5.2f} & {r2} & {rho} \\\\")

    print("\n% ---- tab:baseline_tests ----")
    for gp, title in GP_VARIANTS.items():
        rng = np.random.default_rng(SEED)
        rows = [paired(data, gp, name, rng) for name in references]
        corrected = holm(np.array([r["p"] for r in rows]))
        print(f"    \\multicolumn{{7}}{{l}}{{\\textit{{{title}}}}} \\\\")
        for row, p_h in zip(rows, corrected):
            label = REFERENCES[row["reference"]][0]
            ci = f"[{row['low']:.2f},\\,{row['high']:.2f}]".replace("[-", "[$-$").replace(",\\,-", ",\\,$-$")
            median = f"{row['median']:.2f}" if row["median"] >= 0 else f"$-{abs(row['median']):.2f}$"
            fmt = lambda v: f"{v:.4f}" if v < 0.01 else f"{v:.3f}"
            print(
                f"    {label:25s} & {row['delta_mae']:5.2f} & {ci:22s} & {median:>8s} & "
                f"{row['n_plus']:2d} & {fmt(row['p']):6s} & {fmt(p_h):6s} \\\\"
            )
        print(f"    % delta rho: " + ", ".join(
            f"{REFERENCES[r['reference']][0]} {r['delta_rho']:+.3f}" for r in rows
            if r["reference"] != "train_mean"
        ))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
