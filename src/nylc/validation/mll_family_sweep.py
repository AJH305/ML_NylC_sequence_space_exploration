"""Additive against epistatic kernel, fitted the way the nested LOOCV fits.

Why this module exists
----------------------
``validation.comparison`` compares the two kernel families with the grid search
migrated from notebook 03. That grid hands its value straight to
``build_position_kernels`` as an *absolute* length scale, while the nested
pipeline in ``models.epistatic_gp`` multiplies ``lengthscale_multiplier`` by the
median non-zero pairwise descriptor distance. The two parameters share a name
and differ in unit, so the grid comparison runs at length scales the deployment
model never visits. On the Kidera factors the three grid values 0.2, 0.5 and 1.0
correspond to multipliers 0.042, 0.106 and 0.211 -- all below the optimizer's
lower bound of 0.25 -- while the nested folds fit 0.370 to 0.513. At such length
scales ``exp(-d^2 / (2 l^2))`` underflows for every pair that is not identical at
a position, the descriptor kernel degenerates into a one-hot kernel, and the
descriptor choice stops mattering. The grid also selects its smallest length
scale and its largest epistasis amplitude in almost every fold, so the optimum
lies outside the grid in both directions.

This module therefore repeats the family comparison with the *same* fitting
procedure as the nested evaluation: in every fold the continuous
hyperparameters are optimized against the regularized negative log marginal
likelihood, using the bounds, log priors, noise prior and restart count from
``config['gp']``. Only the discrete setting is held fixed -- descriptor set,
kernel family and PCA components -- so a difference between the families is
attributable to the kernel structure rather than to a mis-centered grid.

What the numbers mean
---------------------
Each reported error is a leave-one-variant-out error for a *fixed* discrete
setting. It is therefore comparable between the two families and between
descriptor sets, but it is NOT comparable to ``nested_metrics.csv``, where the
descriptor set and the kernel family are themselves selected inside each fold.
Holding the setting fixed across all folds uses information the nested estimate
withholds, so these errors are optimistic as estimates of deployment error and
are reported only as a family contrast.

This module writes its own files and touches neither
``model_family_metrics.csv`` nor ``kernel_family_sweep_metrics.csv``.

Reporting obligation: a sweep is evidence only if every descriptor set that was
run is reported, including any for which the additive kernel wins. Choosing the
favourable ones afterwards is what the nested design exists to prevent.

Runtime grows with the number of descriptor sets; each one runs two families
times one leave-one-variant-out loop times ``--restarts`` optimizer starts. Run
a single set first and read the reported duration before queueing the rest::

    python -m nylc.validation.mll_family_sweep --descriptors physical
    python -m nylc.validation.mll_family_sweep

Results accumulate: descriptor sets from earlier runs are kept, a descriptor set
that is run again replaces its own rows, and every file is rewritten after each
descriptor set so a later failure costs no completed runtime.
"""

import argparse
import time
from pathlib import Path

import pandas as pd

METRIC_FILE = "mll_family_sweep_metrics.csv"
PREDICTION_FILE = "mll_family_sweep_predictions.csv"
PAIRED_FILE = "mll_family_sweep_paired.csv"
TEST_FILE = "mll_family_sweep_tests.csv"

FAMILIES = ("additive", "epistatic")
FOCUS, AGAINST = ("epistatic", "additive")

HYPERPARAMETER_COLUMNS = (
    "lengthscale_multiplier",
    "sigma_main",
    "sigma_epi",
    "sigma_noise",
)


def loocv_for_setting(model, core, setting, n_starts):
    """Leave-one-variant-out predictions for one fixed discrete setting.

    Mirrors ``EpistaticGP.loocv_predictions_for_fixed_setting`` but also keeps
    the per-fold hyperparameters, the resolved length scale and the optimizer
    status, because those are what distinguish this comparison from the
    grid-based one.
    """
    from nylc.models.kernels import regression_metrics

    source = model.get_descriptor_source(setting["descriptor_set"])
    core = core.reset_index(drop=True)
    rows = []
    for test_idx in range(len(core)):
        df_test = core.iloc[[test_idx]]
        df_train = core.drop(index=test_idx)
        prediction = model.fit_and_predict_one_fold_hardened(
            df_train, df_test, source, setting, n_starts=n_starts
        )
        hyperparameters = prediction["fitted_hyperparameters"]
        observed = float(df_test["activity_pa6"].iloc[0])
        predicted = float(prediction["mean"][0])
        row = {
            "descriptor_set": setting["descriptor_set"],
            "model": setting["model"],
            "variant_id": df_test["variant_id"].iloc[0],
            "mutation_signature": df_test["mutation_signature"].iloc[0],
            "mutation_order": int(df_test["mutation_order"].iloc[0]),
            "observed": observed,
            "predicted": predicted,
            "predicted_std_latent": float(prediction["std_latent"][0]),
            "predicted_std_observed": float(prediction["std_observed"][0]),
            "abs_error": abs(observed - predicted),
            "training_distance_scale": prediction["training_distance_scale"],
            "resolved_lengthscale": prediction["resolved_lengthscale"],
            "optimizer_success": bool(hyperparameters["optimizer_success"]),
            "any_boundary_hit": bool(hyperparameters["any_boundary_hit"]),
        }
        row.update(
            {f"fitted_{name}": hyperparameters[name] for name in HYPERPARAMETER_COLUMNS}
        )
        rows.append(row)

    predictions = pd.DataFrame(rows)
    metrics = regression_metrics(
        predictions["observed"],
        predictions["predicted"],
        predictions["predicted_std_observed"],
    )
    metrics.update(
        {
            "descriptor_set": setting["descriptor_set"],
            "model": setting["model"],
            "mean_predicted_std_observed": float(
                predictions["predicted_std_observed"].mean()
            ),
            "median_resolved_lengthscale": float(predictions["resolved_lengthscale"].median()),
            "optimizer_success_fraction": float(predictions["optimizer_success"].mean()),
            "boundary_hit_fraction": float(predictions["any_boundary_hit"].mean()),
        }
    )
    metrics.update(
        {
            f"median_fitted_{name}": float(predictions[f"fitted_{name}"].median())
            for name in HYPERPARAMETER_COLUMNS
        }
    )
    return predictions, metrics


def paired_table(predictions, descriptor_set):
    """Per-variant absolute errors of both families, side by side."""
    paired = (
        predictions.pivot(index="variant_id", columns="model", values="abs_error")
        .reset_index()
        .assign(descriptor_set=descriptor_set)
    )
    paired["error_reduction_epistatic_vs_additive"] = paired[AGAINST] - paired[FOCUS]
    return paired


def paired_test(paired, predictions, descriptor_set):
    """Bootstrap interval, Wilcoxon test and Spearman contrast for one set."""
    from nylc.validation.paired_model_comparison import (
        bootstrap_indices,
        compare_errors,
        compare_spearman,
    )

    errors = paired[["variant_id", FOCUS, AGAINST]]
    indices = bootstrap_indices(len(errors))
    row = compare_errors(
        errors, FOCUS, AGAINST, f"mll_family_sweep::{descriptor_set}", indices
    )
    row.update(compare_spearman(predictions, FOCUS, AGAINST, indices))
    row["descriptor_set"] = descriptor_set
    return row


def compare_one(model, core, descriptor_set, n_starts):
    """Both kernel families for a single descriptor set."""
    started = time.perf_counter()
    prediction_frames, metric_rows = ([], [])
    for family in FAMILIES:
        setting = {
            "descriptor_set": descriptor_set,
            "model": family,
            "pca_components": None,
        }
        predictions, metrics = loocv_for_setting(model, core, setting, n_starts)
        prediction_frames.append(predictions)
        metric_rows.append(metrics)
    elapsed = time.perf_counter() - started

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows).assign(
        n_starts=n_starts, seconds=round(elapsed, 1)
    )
    paired = paired_table(predictions, descriptor_set)
    test = pd.DataFrame([paired_test(paired, predictions, descriptor_set)])
    return predictions, metrics, paired, test, elapsed


def merge(fresh, path):
    """Keep descriptor sets from earlier runs, replace the ones just computed."""
    if path.is_file():
        previous = pd.read_csv(path)
        if "descriptor_set" in previous.columns:
            kept = previous[~previous["descriptor_set"].isin(fresh["descriptor_set"].unique())]
            fresh = pd.concat([kept, fresh], ignore_index=True)
    keys = [c for c in ("descriptor_set", "model", "variant_id") if c in fresh.columns]
    return fresh.sort_values(keys, ignore_index=True) if keys else fresh


def publish(collected, output):
    """Write everything computed so far, Holm-correcting across descriptor sets."""
    from nylc.validation.paired_model_comparison import holm

    written = {}
    for name, frames in collected.items():
        written[name] = merge(pd.concat(frames, ignore_index=True), output / name)
    tests = written[TEST_FILE]
    if "wilcoxon_p" in tests.columns and len(tests):
        tests = tests.copy()
        tests["wilcoxon_p_holm"] = holm(tests["wilcoxon_p"].to_numpy())
        written[TEST_FILE] = tests
    for name, frame in written.items():
        frame.to_csv(output / name, index=False)
    return written[METRIC_FILE], written[TEST_FILE]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=".", help="ML_NylC project directory")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--descriptors",
        nargs="+",
        help="descriptor sets to compare; defaults to every set in features.sources",
    )
    parser.add_argument(
        "--restarts",
        type=int,
        default=None,
        help="optimizer restarts per fold; defaults to gp.final_restarts",
    )
    args = parser.parse_args(argv)

    from nylc.config import load_config
    from nylc.data.lab import load_prepared
    from nylc.features.descriptors import DESCRIPTOR_SOURCES, residue_descriptor_table
    from nylc.models.epistatic_gp import EpistaticGP

    root = Path(args.root).resolve()
    config = load_config(root / args.config)
    base = root / "results" / config["experiment"]
    core_path = base / "prepare" / "core.csv"
    if not core_path.is_file():
        parser.error(f"No prepared variants at {core_path}; run the prepare stage first")

    requested = args.descriptors or [
        name for name in config["features"]["sources"] if name in DESCRIPTOR_SOURCES
    ]
    unknown = [name for name in requested if name not in DESCRIPTOR_SOURCES]
    if unknown:
        parser.error(f"Unknown descriptor sets: {unknown}")

    n_starts = args.restarts or config["gp"]["final_restarts"]
    core = load_prepared(core_path)
    output = base / "evaluate"
    output.mkdir(parents=True, exist_ok=True)
    model = EpistaticGP(config["gp"], residue_descriptor_table, requested)

    print(
        f"Variants: {len(core)} | descriptor sets: {', '.join(requested)} "
        f"| restarts per fold: {n_starts}\n"
    )

    collected = {METRIC_FILE: [], PREDICTION_FILE: [], PAIRED_FILE: [], TEST_FILE: []}
    for index, name in enumerate(requested, start=1):
        print(f"[{index}/{len(requested)}] {name} ...", flush=True)
        predictions, metrics, paired, test, elapsed = compare_one(model, core, name, n_starts)
        collected[PREDICTION_FILE].append(predictions)
        collected[METRIC_FILE].append(metrics)
        collected[PAIRED_FILE].append(paired)
        collected[TEST_FILE].append(test)
        publish(collected, output)

        mae = metrics.set_index("model")["mae"]
        row = test.iloc[0]
        print(
            f"    MAE epistatic {mae[FOCUS]:.2f} | additive {mae[AGAINST]:.2f} "
            f"| delta {row['delta_mae']:.2f} "
            f"[{row['delta_mae_ci_low']:.2f}, {row['delta_mae_ci_high']:.2f}] "
            f"| p {row['wilcoxon_p']:.4f} | {elapsed / 60:.1f} min",
            flush=True,
        )
        if index == 1 and len(requested) > 1:
            print(
                f"    estimated total for {len(requested)} sets: "
                f"{elapsed * len(requested) / 60:.0f} min",
                flush=True,
            )

    metrics, tests = publish(collected, output)

    print("\n=== all descriptor sets computed so far ===")
    view = metrics.pivot(index="descriptor_set", columns="model", values="mae")
    view["difference"] = view[AGAINST] - view[FOCUS]
    print(view.sort_values(FOCUS).to_string(float_format=lambda v: f"{v:.2f}"))
    print("\n=== paired tests ===")
    columns = [
        "descriptor_set",
        "delta_mae",
        "delta_mae_ci_low",
        "delta_mae_ci_high",
        "n_focus_better",
        "wilcoxon_p",
        "wilcoxon_p_holm",
    ]
    print(
        tests[[c for c in columns if c in tests.columns]]
        .sort_values("descriptor_set")
        .to_string(index=False, float_format=lambda v: f"{v:.4f}")
    )
    print("\nWritten:")
    for name in (METRIC_FILE, PREDICTION_FILE, PAIRED_FILE, TEST_FILE):
        print(f"  {output / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
