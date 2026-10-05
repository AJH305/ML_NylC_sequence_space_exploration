#!/usr/bin/env python3
"""Recompute every derived number of the thesis results chapter.

Most figures quoted in the results chapter are read straight out of a published
stage artifact. A handful are not: they are simple derivations over the
per-variant tables -- a median, a share, a reference error, a correlation
between two published columns -- and until this script existed they lived only
in the chat log in which they were computed. That is a gap in the provenance
chain, because a reader with the repository could not reproduce them.

This script closes that gap. It reads only published artifacts, derives every
such number, and prints it under the label it carries in the chapter. It fits
no model and writes nothing into the results tree.

Two further uses follow from that. ``--save`` records the current values as a
reference, and ``--check`` compares a later run against that reference, which
turns the script into a regression test: after the pipeline is rerun, any
number in the chapter that moved is reported here instead of being discovered
during a defense.

Deliberately placed in ``tools/`` rather than ``src/``. ``provenance.code_identity``
hashes every ``.py`` under ``src/``, so a module added there would invalidate the
signature of every stage and force a full rerun. A script here does not.

Standard library only, so it runs without the project environment::

    python tools/thesis_numbers.py
    python tools/thesis_numbers.py --save tools/thesis_numbers_reference.json
    python tools/thesis_numbers.py --check tools/thesis_numbers_reference.json
"""

import argparse
import csv
import json
import math
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

Z95 = 1.95996398  # two-sided normal quantile, as used by the coverage table

REPORTED_REFERENCES = (
    "random_forest_onehot_fixed",
    "random_forest_physical_fixed",
    "ridge_physical",
    "bayesian_ridge_physical",
    "knn_physical",
    "train_mean",
)


# ------------------------------------------------------------------ helpers ---


class Artifacts:
    """Read-only accessor; a missing file yields None rather than raising."""

    def __init__(self, base):
        self.base = Path(base)
        self.missing = []

    def rows(self, relative):
        path = self.base / relative
        if not path.is_file():
            self.missing.append(relative)
            return None
        with path.open(encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def json(self, relative):
        path = self.base / relative
        if not path.is_file():
            self.missing.append(relative)
            return None
        return json.loads(path.read_text(encoding="utf-8"))


def column(rows, name):
    return [float(row[name]) for row in rows]


def ranks(values):
    """Average ranks, so the rank correlation matches scipy's convention."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def pearson(a, b):
    mean_a, mean_b = st.mean(a), st.mean(b)
    numerator = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    denominator = math.sqrt(
        sum((x - mean_a) ** 2 for x in a) * sum((y - mean_b) ** 2 for y in b)
    )
    return numerator / denominator if denominator else float("nan")


def spearman(a, b):
    return pearson(ranks(a), ranks(b))


def spread(values):
    return {"min": min(values), "median": st.median(values), "max": max(values)}


# ------------------------------------------------------------- computations ---


def nested_performance(art, out):
    """5.1.2 Nested cross-validation performance."""
    metrics = art.rows("evaluate/nested_metrics.csv")
    predictions = art.rows("evaluate/nested_predictions.csv")
    if metrics is None or predictions is None:
        return
    by_mode = {row["uncertainty"]: row for row in metrics}
    for mode in ("uncalibrated", "sigma_scaled"):
        if mode in by_mode:
            out[f"nested.nlpd.{mode}"] = float(by_mode[mode]["nlpd_observed"])
    headline = by_mode["uncalibrated"]
    for key in ("mae", "rmse", "r2", "pearson", "spearman"):
        out[f"nested.{key}"] = float(headline[key])

    observed = column(predictions, "observed")
    errors = column(predictions, "abs_error")
    out["dataset.n_variants"] = len(predictions)
    out["dataset.activity_min"] = min(observed)
    out["dataset.activity_max"] = max(observed)
    out["dataset.activity_median"] = st.median(observed)
    out["dataset.activity_sd"] = st.stdev(observed)
    out["dataset.activity_range"] = max(observed) - min(observed)
    out["nested.mae_pct_of_sd"] = 100 * st.mean(errors) / st.stdev(observed)
    out["nested.mae_pct_of_range"] = 100 * st.mean(errors) / (max(observed) - min(observed))
    out["nested.median_abs_error"] = st.median(errors)

    ordered = sorted(predictions, key=lambda row: float(row["abs_error"]))
    out["nested.smallest_error"] = float(ordered[0]["abs_error"])
    out["nested.smallest_error_variant"] = ordered[0]["variant_id"]
    for rank, row in enumerate(reversed(ordered[-5:]), start=1):
        out[f"nested.largest_error_{rank}"] = float(row["abs_error"])
        out[f"nested.largest_error_{rank}_variant"] = row["variant_id"]

    counts = Counter(int(row["mutation_order"]) for row in predictions)
    for order in sorted(counts):
        out[f"dataset.n_order_{order}"] = counts[order]

    correlations = art.rows("evaluate/error_by_order_correlations.csv")
    if correlations:
        for row in correlations:
            if row["scope"] != "all_variants":
                continue
            out[f"order.rho_{row['against']}"] = float(row["spearman_rho"])
            out[f"order.p_{row['against']}"] = float(row["p_value"])


def robustness(art, out):
    """5.1.3 Robustness and selected configuration."""
    predictions = art.rows("evaluate/nested_predictions.csv")
    if predictions:
        chosen = Counter(
            (row["selected_descriptor_set"], row["selected_model"]) for row in predictions
        )
        for (source, model), count in sorted(chosen.items()):
            out[f"selection.folds.{source}.{model}"] = count
        deviating = [
            row["variant_id"]
            for row in predictions
            if row["selected_descriptor_set"] != "physical"
        ]
        out["selection.non_physical_folds"] = ";".join(sorted(deviating))
        for name in (
            "fitted_lengthscale_multiplier",
            "fitted_sigma_main",
            "fitted_sigma_epi",
            "fitted_sigma_noise",
            "resolved_lengthscale",
        ):
            for stat, value in spread(column(predictions, name)).items():
                out[f"folds.{name}.{stat}"] = value
        out["folds.optimizer_success"] = sum(
            row["optimizer_success"] == "True" for row in predictions
        )
        out["folds.boundary_hits"] = sum(
            row["optimizer_any_boundary_hit"] == "True" for row in predictions
        )

    holdout = art.rows("evaluate/position_holdout_metrics.csv")
    settings = art.rows("evaluate/position_holdout_settings.csv")
    holdout_predictions = art.rows("evaluate/position_holdout_predictions.csv")
    core = art.rows("prepare/core.csv")
    if holdout and holdout_predictions and core:
        activity = {row["variant_id"]: float(row["activity_pa6"]) for row in core}
        grouped = defaultdict(list)
        for row in holdout_predictions:
            grouped[row["heldout_position"]].append(row)
        chosen = {row["heldout_position"]: row for row in (settings or [])}
        for row in holdout:
            position = row["heldout_position"]
            tested = grouped[position]
            held = {entry["variant_id"] for entry in tested}
            # The reference a prediction has to beat: the mean activity of the
            # variants the model was actually trained on for this holdout.
            train = [value for key, value in activity.items() if key not in held]
            reference = st.mean(train)
            out[f"holdout.{position}.mae"] = float(row["mae"])
            out[f"holdout.{position}.n_test"] = len(tested)
            out[f"holdout.{position}.n_train"] = len(train)
            out[f"holdout.{position}.train_mean_mae"] = st.mean(
                abs(float(entry["observed"]) - reference) for entry in tested
            )
            if position in chosen:
                out[f"holdout.{position}.selected_model"] = chosen[position]["selected_model"]
                out[f"holdout.{position}.sigma_epi"] = float(chosen[position]["fitted_sigma_epi"])

    # Restart insensitivity: the diagnostics stage fits with the inner restart
    # count, the family sweep with the final one. Agreement bounds how much the
    # reported optima depend on where the optimizer was started.
    exploratory = art.rows("diagnostics/exploratory_source_sensitivity.csv")
    sweep = art.rows("evaluate/mll_family_sweep_metrics.csv")
    if exploratory and sweep:
        left = {(row["descriptor_set"], row["model"]): float(row["mae"]) for row in exploratory}
        right = {(row["descriptor_set"], row["model"]): float(row["mae"]) for row in sweep}
        shared = sorted(set(left) & set(right))
        if shared:
            out["restarts.n_settings_compared"] = len(shared)
            out["restarts.max_abs_mae_difference"] = max(
                abs(left[key] - right[key]) for key in shared
            )

    model = art.json("fit/model.json")
    if model:
        for name, value in model["hyperparameters"].items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out[f"deployment.{name}"] = float(value)
        out["deployment.sigma_scale"] = float(model["sigma_scale"])


def epistatic_contribution(art, out):
    """5.1.4 Contribution of epistatic terms."""
    metrics = art.rows("evaluate/mll_family_sweep_metrics.csv")
    tests = art.rows("evaluate/mll_family_sweep_tests.csv")
    paired = art.rows("evaluate/mll_family_sweep_paired.csv")
    predictions = art.rows("evaluate/mll_family_sweep_predictions.csv")
    if metrics:
        families = {"additive": {}, "epistatic": {}}
        for row in metrics:
            families[row["model"]][row["descriptor_set"]] = row
        for family, byset in families.items():
            if not byset:
                continue
            for name in ("mae", "r2", "nlpd_observed", "mean_predicted_std_observed",
                         "median_fitted_sigma_main", "median_fitted_sigma_epi",
                         "median_fitted_sigma_noise", "median_fitted_lengthscale_multiplier"):
                values = [float(row[name]) for row in byset.values()]
                out[f"sweep.{family}.{name}.min"] = min(values)
                out[f"sweep.{family}.{name}.max"] = max(values)
                out[f"sweep.{family}.{name}.median"] = st.median(values)
            for source, row in sorted(byset.items()):
                out[f"sweep.{family}.{source}.mae"] = float(row["mae"])
        if families["additive"] and families["epistatic"]:
            out["sweep.nlpd_better_for_epistatic"] = sum(
                float(families["epistatic"][key]["nlpd_observed"])
                < float(families["additive"][key]["nlpd_observed"])
                for key in families["epistatic"]
            )
    if predictions:
        out["sweep.n_optimizations"] = len(predictions)
        out["sweep.n_converged"] = sum(row["optimizer_success"] == "True" for row in predictions)
        out["sweep.n_boundary_hits"] = sum(row["any_boundary_hit"] == "True" for row in predictions)
    if tests:
        deltas = [float(row["delta_mae"]) for row in tests]
        out["sweep.delta_mae.min"] = min(deltas)
        out["sweep.delta_mae.max"] = max(deltas)
        better = [int(row["n_focus_better"]) for row in tests]
        out["sweep.n_focus_better.min"] = min(better)
        out["sweep.n_focus_better.max"] = max(better)
        out["sweep.n_ci_excluding_zero"] = sum(float(row["delta_mae_ci_low"]) > 0 for row in tests)
        for row in tests:
            source = row["descriptor_set"]
            out[f"sweep.{source}.wilcoxon_p"] = float(row["wilcoxon_p"])
            out[f"sweep.{source}.wilcoxon_p_holm"] = float(row["wilcoxon_p_holm"])
        if tests[0].get("delta_spearman"):
            rho = [float(row["delta_spearman"]) for row in tests]
            out["sweep.delta_spearman.min"] = min(rho)
            out["sweep.delta_spearman.max"] = max(rho)
            out["sweep.n_rho_ci_including_zero"] = sum(
                float(row["delta_spearman_ci_low"]) < 0 < float(row["delta_spearman_ci_high"])
                for row in tests
            )
    if paired:
        # How much of the result is a property of the representation rather than
        # of the variants: the per-variant reductions are correlated across
        # descriptor sets, so the sets are not independent replications.
        sources = sorted({row["descriptor_set"] for row in paired})
        table = defaultdict(dict)
        for row in paired:
            table[row["variant_id"]][row["descriptor_set"]] = float(
                row["error_reduction_epistatic_vs_additive"]
            )
        variants = sorted(table)
        offdiagonal, physical_pairs = [], []
        for i, left in enumerate(sources):
            for right in sources[i + 1:]:
                value = pearson(
                    [table[v][left] for v in variants], [table[v][right] for v in variants]
                )
                offdiagonal.append(value)
                if "physical" in (left, right):
                    physical_pairs.append(value)
        if offdiagonal:
            out["sweep.pairwise_r.min"] = min(offdiagonal)
            out["sweep.pairwise_r.max"] = max(offdiagonal)
            out["sweep.pairwise_r.median"] = st.median(offdiagonal)
            out["sweep.pairwise_r.physical_max"] = max(physical_pairs)
        improved = [v for v in variants if all(table[v][s] > 0 for s in sources)]
        worsened = [v for v in variants if all(table[v][s] < 0 for s in sources)]
        out["sweep.improved_in_all_sets"] = len(improved)
        out["sweep.worsened_in_all_sets"] = len(worsened)
        out["sweep.sign_depends_on_set"] = len(variants) - len(improved) - len(worsened)
        mean_delta = lambda v: st.mean([table[v][s] for s in sources])
        for rank, variant in enumerate(sorted(improved, key=mean_delta, reverse=True)[:3], 1):
            out[f"sweep.top_improvement_{rank}"] = variant
            out[f"sweep.top_improvement_{rank}_mean"] = mean_delta(variant)
        for rank, variant in enumerate(sorted(worsened, key=mean_delta)[:2], 1):
            out[f"sweep.top_deterioration_{rank}"] = variant
            out[f"sweep.top_deterioration_{rank}_mean"] = -mean_delta(variant)


def baselines(art, out):
    """5.1.5 Comparison with baseline models."""
    stage = art.rows("baselines/all_loocv_metrics.csv")
    evaluate = art.rows("evaluate/baseline_metrics.csv")
    if stage and evaluate:
        merged = {}
        for row in list(stage) + list(evaluate):
            merged.setdefault(row["model"], row)
        wanted = ("gp_hardened_physical_epistatic", "gp_nested_source_selection")
        for name in wanted + REPORTED_REFERENCES:
            if name not in merged:
                continue
            for key in ("mae", "r2", "spearman"):
                out[f"baseline.{name}.{key}"] = float(merged[name][key])
    tests = art.rows("evaluate/paired_comparison_tests.csv")
    if tests:
        for row in tests:
            if not row["family"].startswith("reported::"):
                continue
            focus = row["focus_model"]
            against = row["against_model"]
            for key in ("delta_mae", "delta_mae_median", "wilcoxon_p", "wilcoxon_p_holm"):
                out[f"paired.{focus}.{against}.{key}"] = float(row[key])
            out[f"paired.{focus}.{against}.n_focus_better"] = int(row["n_focus_better"])
            if row.get("delta_spearman"):
                out[f"paired.{focus}.{against}.delta_spearman"] = float(row["delta_spearman"])
        for focus in sorted({row["focus_model"] for row in tests if row["family"].startswith("reported::")}):
            family = [
                row for row in tests
                if row["family"].startswith("reported::") and row["focus_model"] == focus
            ]
            out[f"paired.{focus}.n_raw_below_005"] = sum(
                float(row["wilcoxon_p"]) < 0.05 for row in family
            )
            out[f"paired.{focus}.n_holm_below_005"] = sum(
                float(row["wilcoxon_p_holm"]) < 0.05 for row in family
            )


def uncertainty(art, out):
    """5.1.6 Predictive uncertainty and calibration."""
    coverage = art.rows("evaluate/coverage.csv")
    if coverage:
        for row in coverage:
            label = row["interval"].replace("%", "")
            out[f"coverage.{label}.uncalibrated"] = float(row["uncalibrated_coverage"])
            out[f"coverage.{label}.calibrated"] = float(row["calibrated_coverage"])
    predictions = art.rows("evaluate/nested_predictions.csv")
    if not predictions:
        return
    errors = column(predictions, "abs_error")
    uncal = column(predictions, "predicted_std_observed")
    cal = column(predictions, "predicted_std_observed_calibrated")
    out["uncertainty.mean_sd_uncalibrated"] = st.mean(uncal)
    out["uncertainty.mean_sd_calibrated"] = st.mean(cal)
    rms = lambda sd: math.sqrt(sum((e / s) ** 2 for e, s in zip(errors, sd)) / len(errors))
    out["uncertainty.rms_standardized_uncalibrated"] = rms(uncal)
    out["uncertainty.rms_standardized_calibrated"] = rms(cal)
    out["uncertainty.mean_width95_uncalibrated"] = st.mean(2 * Z95 * s for s in uncal)
    out["uncertainty.mean_width95_calibrated"] = st.mean(2 * Z95 * s for s in cal)
    for label, sd in (("uncalibrated", uncal), ("calibrated", cal)):
        outside = [
            row["variant_id"]
            for row, e, s in zip(predictions, errors, sd)
            if e > Z95 * s
        ]
        out[f"uncertainty.outside95_{label}"] = len(outside)
        out[f"uncertainty.outside95_{label}_variants"] = ";".join(sorted(outside))
    if "sigma_scale_inner" in predictions[0]:
        for stat, value in spread(column(predictions, "sigma_scale_inner")).items():
            out[f"uncertainty.sigma_scale_inner.{stat}"] = value
    # The decisive question for how the variance may be used: does a larger
    # predicted standard deviation mark a prediction that is actually worse?
    out["uncertainty.rho_sd_vs_error"] = spearman(uncal, errors)


SECTIONS = (
    ("5.1.2  Nested cross-validation performance", nested_performance),
    ("5.1.3  Robustness and selected configuration", robustness),
    ("5.1.4  Contribution of epistatic terms", epistatic_contribution),
    ("5.1.5  Comparison with baseline models", baselines),
    ("5.1.6  Predictive uncertainty and calibration", uncertainty),
)


# -------------------------------------------------------------------- main ---


def compute(base):
    art = Artifacts(base)
    values, grouped = {}, []
    for title, function in SECTIONS:
        before = set(values)
        function(art, values)
        grouped.append((title, [key for key in values if key not in before]))
    return values, grouped, art.missing


def render(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def compare(values, reference, tolerance):
    drift = []
    for key, expected in reference.items():
        if key not in values:
            drift.append((key, expected, "<fehlt>"))
            continue
        found = values[key]
        if isinstance(expected, (int, float)) and isinstance(found, (int, float)):
            scale = max(1.0, abs(float(expected)))
            if abs(float(found) - float(expected)) > tolerance * scale:
                drift.append((key, expected, found))
        elif found != expected:
            drift.append((key, expected, found))
    for key in values:
        if key not in reference:
            drift.append((key, "<neu>", values[key]))
    return drift


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=".", help="ML_NylC project directory")
    parser.add_argument("--experiment", default="reference_cpu")
    parser.add_argument("--save", metavar="PATH", help="write the values as a reference")
    parser.add_argument("--check", metavar="PATH", help="compare against a saved reference")
    parser.add_argument("--tolerance", type=float, default=1e-6,
                        help="relative tolerance for --check (default 1e-6)")
    args = parser.parse_args(argv)

    base = Path(args.root).resolve() / "results" / args.experiment
    if not base.is_dir():
        parser.error(f"No results tree at {base}")
    values, grouped, missing = compute(base)

    for title, keys in grouped:
        if not keys:
            continue
        print(f"\n{title}")
        print("-" * len(title))
        width = max(len(key) for key in keys)
        for key in keys:
            print(f"  {key:<{width}}  {render(values[key])}")

    if missing:
        print("\nFehlende Artefakte (zugehoerige Zahlen wurden uebersprungen):")
        for name in sorted(set(missing)):
            print(f"  {name}")

    if args.save:
        Path(args.save).write_text(
            json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"\n{len(values)} Werte gespeichert: {args.save}")

    if args.check:
        reference = json.loads(Path(args.check).read_text(encoding="utf-8"))
        drift = compare(values, reference, args.tolerance)
        print(f"\n=== Abgleich gegen {args.check} (Toleranz {args.tolerance:g}) ===")
        if not drift:
            print(f"  {len(reference)} Werte unveraendert.")
            return 0
        for key, expected, found in drift:
            print(f"  {key}\n      Referenz {render(expected)}\n      jetzt    {render(found)}")
        print(f"\n  {len(drift)} Abweichung(en) von {len(reference)} Werten.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
