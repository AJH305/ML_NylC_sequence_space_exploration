"""Prediction error grouped by mutation order, with rank correlations.

The nested outer predictions answer how well the model generalizes, but not
whether the error depends on how many substitutions a variant carries. The
absolute error and the mutation order are confounded by the activity level:
higher-order variants are more active, and an absolute error naturally grows
with the magnitude of the quantity it refers to. This module therefore reports
three Spearman correlations side by side -- order against absolute error, order
against observed activity, and order against the error normalized to the
observed activity -- so that the confounding is visible rather than implied.

Spearman's rho is used because the mutation order is an ordinal variable with
five levels and the error distribution is right-skewed; a Pearson correlation
would be driven by the few largest deviations.

The wild type forms a group of its own with a single observation. It is
included by default, because the group sizes must add up to the number of
evaluated variants, and the statistics are additionally reported without it so
that the sensitivity to that single point can be checked.

Run it on a finished experiment::

    python -m nylc.validation.error_by_order --config configs/default.yaml

It reads ``evaluate/nested_predictions.csv`` and writes
``evaluate/error_by_order_groups.csv`` and
``evaluate/error_by_order_correlations.csv`` next to it. Nothing is refitted.
"""

import argparse
from pathlib import Path

import pandas as pd
from scipy import stats

GROUP_FILE = "error_by_order_groups.csv"
CORRELATION_FILE = "error_by_order_correlations.csv"


def prepare(predictions):
    """Add the activity-normalized error used for the confounding check."""
    frame = predictions.copy()
    missing = {"mutation_order", "observed", "predicted"} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in predictions: {sorted(missing)}")
    if "abs_error" not in frame:
        frame["abs_error"] = (frame["observed"] - frame["predicted"]).abs()
    if (frame["observed"] <= 0).any():
        raise ValueError("Relative error is undefined for non-positive observed activities")
    frame["relative_error"] = frame["abs_error"] / frame["observed"]
    return frame


def group_statistics(frame):
    """Per-order group sizes, error location and mean observed activity."""
    grouped = (
        frame.groupby("mutation_order")
        .agg(
            n=("abs_error", "size"),
            mean_absolute_error=("abs_error", "mean"),
            median_absolute_error=("abs_error", "median"),
            mean_observed_activity=("observed", "mean"),
            median_relative_error=("relative_error", "median"),
        )
        .reset_index()
    )
    return grouped


def correlations(frame, scope):
    """Spearman correlations of mutation order against the three quantities."""
    rows = []
    for name, column in (
        ("absolute_error", "abs_error"),
        ("observed_activity", "observed"),
        ("relative_error", "relative_error"),
    ):
        result = stats.spearmanr(frame["mutation_order"], frame[column])
        rows.append(
            {
                "scope": scope,
                "n": len(frame),
                "against": name,
                "spearman_rho": float(result.statistic),
                "p_value": float(result.pvalue),
            }
        )
    return pd.DataFrame(rows)


def analyze(predictions):
    """Group statistics and correlations, with and without the wild type."""
    frame = prepare(predictions)
    without_wild_type = frame[frame["mutation_order"] > 0]
    tables = [correlations(frame, "all_variants")]
    if len(without_wild_type) != len(frame):
        tables.append(correlations(without_wild_type, "excluding_wild_type"))
    return group_statistics(frame), pd.concat(tables, ignore_index=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=".", help="ML_NylC project directory")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--experiment",
        help="overrides the experiment named in the configuration",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    if args.experiment:
        experiment = args.experiment
    else:
        from nylc.config import load_config

        experiment = load_config(root / args.config)["experiment"]

    base = root / "results" / experiment / "evaluate"
    source = base / "nested_predictions.csv"
    if not source.is_file():
        parser.error(f"No nested predictions at {source}; run the evaluate stage first")

    groups, rho = analyze(pd.read_csv(source))
    groups.to_csv(base / GROUP_FILE, index=False)
    rho.to_csv(base / CORRELATION_FILE, index=False)

    with pd.option_context("display.width", 120):
        print(f"Source: {source}\n")
        print(groups.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print()
        print(rho.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nWritten: {base / GROUP_FILE}\n         {base / CORRELATION_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
