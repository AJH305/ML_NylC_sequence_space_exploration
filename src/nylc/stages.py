"""File-to-file scientific stages shared by the CLI and Snakemake."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from nylc.data.lab import (
    load_lab,
    read_reference,
    load_prepared,
    serializable_frame,
    write_sequence_features,
    training_sem,
)
from nylc.data.candidates import (
    add_boltzgen_candidates_from_full_sequences,
    check_designed_sequence_consistency,
)
from nylc.features.store import create_source, write_feature_store, store_factory
from nylc.features.descriptors import residue_descriptor_table
from nylc.models.epistatic_gp import EpistaticGP
from nylc.models.kernels import feature_source_diagnostics
from nylc.provenance import sha256_file, write_json


def prepared(config, root):
    base = root / "results" / config["experiment"]
    return base, load_prepared(base / "prepare/lab.csv"), load_prepared(base / "prepare/core.csv")


def engine(config, root, sources=None):
    base = root / "results" / config["experiment"]
    return EpistaticGP(
        config["gp"], store_factory(base / "features"), sources or config["features"]["sources"]
    )


def prepare(config, root, output):
    expected = json.loads((root / config["inputs"]["checksums"]).read_text(encoding="utf-8"))
    for role in ("lab_data", "wt_fasta", "candidates"):
        path = root / config["inputs"][role]
        if sha256_file(path) != expected[role]:
            raise ValueError(
                f"Input checksum mismatch: {role}; create a new documented data snapshot"
            )
    wt = read_reference(root / config["inputs"]["wt_fasta"])
    lab = load_lab(root / config["inputs"]["lab_data"], wt, config["data"]["subset_variant_ids"])
    core = lab.loc[lab["only_target_positions"]].reset_index(drop=True)
    if len(core) < 5:
        raise ValueError("Nested model selection requires at least five eligible variants")
    if not core["aa_tuple"].is_unique:
        raise ValueError(
            "Duplicate core sequences: combine replicates or define grouped validation"
        )
    serializable_frame(lab).to_csv(output / "lab.csv", index=False)
    serializable_frame(core).to_csv(output / "core.csv", index=False)
    serializable_frame(lab.loc[~lab["only_target_positions"]]).to_csv(
        output / "excluded.csv", index=False
    )
    write_sequence_features(lab, wt, output)
    raw = pd.read_csv(root / config["inputs"]["candidates"]).copy()
    raw["source_file"] = Path(config["inputs"]["candidates"]).name
    raw["source_path"] = config["inputs"]["candidates"]
    # Preserve the recorded BoltzGen run identity from the original snapshot.
    raw["boltzgen_run"] = config["data"]["candidate_run"]
    all_candidates, valid = add_boltzgen_candidates_from_full_sequences(raw)
    valid = check_designed_sequence_consistency(valid, strict=True)
    if valid.empty or not valid["candidate_id"].is_unique:
        raise ValueError("No valid candidates or ambiguous candidate IDs")
    valid["already_tested"] = valid["mutation_set"].isin(set(lab["mutation_set"]))
    serializable_frame(all_candidates).to_csv(output / "candidate_audit.csv", index=False)
    serializable_frame(valid).to_csv(output / "candidates.csv", index=False)
    write_json(
        output / "summary.json",
        {
            "n_lab": len(lab),
            "n_core": len(core),
            "n_excluded": len(lab) - len(core),
            "n_raw_candidates": len(raw),
            "n_valid_unique_candidates": len(valid),
            "units": {
                "activity_pa6": "as recorded in source lab table; see data documentation",
                "tm_celsius": "degree Celsius",
            },
            "scope": "four-position GP; mutations outside 99,134,304,330 excluded from fitting",
        },
    )


def features(config, root, output):
    base, lab, core = prepared(config, root)
    candidates = load_prepared(base / "prepare/candidates.csv")
    wt = read_reference(root / config["inputs"]["wt_fasta"])
    pockets = core["pocket"].tolist() + candidates["pocket"].tolist()
    descriptions = []
    for name in config["features"]["sources"]:
        source = create_source(name, config, root, wt)
        descriptions.append(write_feature_store(name, source, pockets, output))
    write_json(output / "sources.json", descriptions)


def diagnostics(config, root, output):
    _, _, core = prepared(config, root)
    model = engine(config, root)
    tables = [
        feature_source_diagnostics(core, model.get_descriptor_source(name))
        for name in config["features"]["sources"]
    ]
    pd.concat(tables, ignore_index=True).to_csv(output / "feature_diagnostics.csv", index=False)
    from nylc.validation.comparison import nonidentical_kernel_diagnostics

    summaries, distributions = [], []
    for name in config["features"]["sources"]:
        summary, distribution = nonidentical_kernel_diagnostics(
            core, model.get_descriptor_source(name)
        )
        summaries.append(summary)
        distributions.append(distribution)
    pd.concat(summaries, ignore_index=True).to_csv(
        output / "nonidentical_pair_diagnostics.csv", index=False
    )
    pd.concat(distributions, ignore_index=True).to_csv(
        output / "nonidentical_pair_distributions.csv", index=False
    )
    # Fixed-source LOOCV is exploratory, distinct from nested performance estimates.
    grid = model.make_hardened_joint_grid()
    rows = [{**setting, **model.evaluate_hardened_setting_loocv(core, setting)} for setting in grid]
    pd.DataFrame(rows).to_csv(output / "exploratory_source_sensitivity.csv", index=False)


def evaluate(config, root, output):
    _, _, core = prepared(config, root)
    model = engine(config, root)
    grid = model.make_hardened_joint_grid()
    pred, metrics, selections, inner = model.nested_loocv_hardened_feature_selection_calibrated(
        core, grid
    )
    pred.to_csv(output / "nested_predictions.csv", index=False)
    metrics.to_csv(output / "nested_metrics.csv", index=False)
    selections.to_csv(output / "selected_settings.csv", index=False)
    pd.concat(
        [table.assign(outer_test_variant=variant) for variant, table in inner.items()],
        ignore_index=True,
    ).to_csv(output / "inner_grids.csv", index=False)
    model.uncertainty_coverage_table(pred).to_csv(output / "coverage.csv", index=False)
    if config["validation"]["compare_model_families"]:
        from nylc.validation.comparison import nested_loocv_model_comparison

        p, m, _ = nested_loocv_model_comparison(
            core, residue_descriptor_table(config["validation"]["baseline_descriptor"])
        )
        p.to_csv(output / "model_family_predictions.csv", index=False)
        m.to_csv(output / "model_family_metrics.csv", index=False)
        paired = p.pivot(index="variant_id", columns="model", values="abs_error")
        paired["error_reduction_epistatic_vs_additive"] = paired["additive"] - paired["epistatic"]
        paired.to_csv(output / "paired_model_errors.csv")
    if config["validation"]["baselines"]:
        from nylc.validation.baselines import loocv_baselines

        p, m = loocv_baselines(
            core, residue_descriptor_table(config["validation"]["baseline_descriptor"])
        )
        p.to_csv(output / "baseline_predictions.csv", index=False)
        m.to_csv(output / "baseline_metrics.csv", index=False)
    if config["validation"]["family_holdout"]:
        p, m, s = model.strict_position_family_holdout_hardened(core, grid)
        p.to_csv(output / "position_holdout_predictions.csv", index=False)
        m.to_csv(output / "position_holdout_metrics.csv", index=False)
        s.to_csv(output / "position_holdout_settings.csv", index=False)


def fit(config, root, output):
    _, _, core = prepared(config, root)
    model = engine(config, root, config["selection"]["sources"])
    setting, grid = model.select_hardened_setting_inner_loocv(
        core, model.make_hardened_joint_grid()
    )
    source = model.get_descriptor_source(setting["descriptor_set"])
    prepared_features = model.prepare_fold_features(
        core["pocket"].tolist(),
        core["pocket"].tolist(),
        source,
        pca_components=setting.get("pca_components"),
    )
    hp = model.optimize_gp_hyperparameters(
        prepared_features, core["activity_pa6"].to_numpy(), training_sem(core), setting["model"]
    )
    calibration = model.loocv_predictions_for_fixed_setting(
        core, setting, n_starts=config["gp"]["final_restarts"]
    )
    sigma = model.fit_sigma_scale(
        calibration.observed, calibration.predicted, calibration.predicted_std_observed
    )
    calibration["predicted_std_observed_calibrated"] = sigma * calibration.predicted_std_observed
    grid.to_csv(output / "selection_grid.csv", index=False)
    calibration.to_csv(output / "deployment_calibration.csv", index=False)
    write_json(
        output / "model.json",
        {
            "model_name": "Epistatic GP",
            "setting": setting,
            "hyperparameters": hp,
            "sigma_scale": sigma,
            "n_training_variants": len(core),
            "calibration_scope": "deployment scale after full-data model selection; unbiased coverage is evaluated in the separate nested outer loop",
        },
    )


def predict(config, root, output):
    base, _, core = prepared(config, root)
    candidates = load_prepared(base / "prepare/candidates.csv")
    final = json.loads((base / "fit/model.json").read_text())
    model = engine(config, root, config["selection"]["sources"])
    setting, hp = final["setting"], final["hyperparameters"]
    source = model.get_descriptor_source(setting["descriptor_set"])
    prediction = model.fit_and_predict_one_fold_hardened(
        core, candidates, source, setting, fitted_hyperparameters=hp
    )
    candidates["predicted_activity_gp_mean"] = prediction["mean"]
    candidates["predicted_activity_gp_std_latent"] = prediction["std_latent"]
    candidates["predicted_activity_gp_std_observed"] = prediction["std_observed"]
    candidates["predicted_activity_gp_std_observed_calibrated"] = (
        final["sigma_scale"] * prediction["std_observed"]
    )
    candidates["gp_sigma_scale"] = final["sigma_scale"]
    candidates["gp_descriptor_set"] = setting["descriptor_set"]
    candidates["gp_model_family"] = setting["model"]
    candidates["gp_resolved_lengthscale"] = prediction["resolved_lengthscale"]
    if not np.isfinite(prediction["mean"]).all():
        raise ValueError("Non-finite candidate predictions")
    prepared_features = model.prepare_fold_features(
        core["pocket"].tolist(),
        candidates["pocket"].tolist(),
        source,
        setting.get("pca_components"),
    )
    from nylc.models.kernels import (
        robust_distance_scale,
        rbf_kernel_from_descriptors,
        build_total_epistatic_kernel,
    )

    lengthscale = hp["lengthscale_multiplier"] * robust_distance_scale(prepared_features["train"])
    if prepared_features["kind"] == "global":
        kernel = hp["sigma_main"] ** 2 * rbf_kernel_from_descriptors(
            prepared_features["test"]["global"], lengthscale=lengthscale
        )
    else:
        position_kernels = {
            p: rbf_kernel_from_descriptors(x, lengthscale=lengthscale)
            for p, x in prepared_features["test"].items()
        }
        kernel, _, _ = build_total_epistatic_kernel(
            position_kernels, sigma_main=hp["sigma_main"], sigma_epi=hp["sigma_epi"]
        )
    kernel = (kernel + kernel.T) / 2
    if np.linalg.eigvalsh(kernel).min() < -1e-7:
        raise ValueError("Candidate kernel is not positive semidefinite")
    serializable_frame(candidates).to_csv(output / "candidates.csv", index=False)
    np.savez_compressed(
        output / "candidate_kernel.npz",
        kernel=kernel,
        candidate_ids=candidates.candidate_id.to_numpy(dtype=str),
    )


def select(config, root, output):
    from nylc.selection.panel import select_panel

    base, lab, _ = prepared(config, root)
    candidates = pd.read_csv(base / "predict/candidates.csv")
    with np.load(base / "predict/candidate_kernel.npz", allow_pickle=False) as arrays:
        if not np.array_equal(arrays["candidate_ids"], candidates.candidate_id.to_numpy(dtype=str)):
            raise ValueError("Candidate/kernel identifier mismatch")
        select_panel(candidates, arrays["kernel"], lab, config["selection"], output)


def baselines(config, root, output):
    base, _, core = prepared(config, root)
    if not config["baseline_comparison"]["enabled"]:
        write_json(output / "summary.json", {"status": "disabled"})
        return
    from nylc.validation.notebook_baselines import run_comparison

    run_comparison(
        config, root, output, core, pd.read_csv(base / "evaluate/nested_predictions.csv")
    )


def report(config, root, output):
    from nylc.reporting.figures import make_report

    make_report(config, root, output)


def figures(config, root, output):
    """Publication figures for the thesis; reads published artifacts only."""
    from nylc.reporting.thesis_figures import make_thesis_figures

    make_thesis_figures(config, root, output)


STAGES = {
    "prepare": prepare,
    "features": features,
    "diagnostics": diagnostics,
    "evaluate": evaluate,
    "fit": fit,
    "predict": predict,
    "select": select,
    "baselines": baselines,
    "report": report,
    "figures": figures,
}
DEPENDENCIES = {
    "prepare": [],
    "features": ["prepare"],
    "diagnostics": ["features"],
    "evaluate": ["features"],
    "fit": ["features"],
    "predict": ["prepare", "features", "fit"],
    "select": ["prepare", "predict"],
    "baselines": ["evaluate"],
    "report": ["diagnostics", "evaluate", "select", "baselines"],
    "figures": ["diagnostics", "evaluate", "fit", "select", "baselines"],
}
CONFIG_KEYS = {
    "prepare": ["inputs", "data"],
    "features": ["features", "runtime"],
    "diagnostics": ["gp", "features"],
    "evaluate": ["gp", "features", "validation"],
    "fit": ["gp"],
    "predict": ["gp"],
    "select": ["selection"],
    "baselines": ["baseline_comparison", "gp", "runtime"],
    "report": ["baseline_comparison"],
    # Figure styling lives in the module, not in the configuration; the stage
    # signature already covers the source checksum, so a pure styling change
    # re-runs this stage on its own.
    "figures": ["features", "gp", "validation", "selection", "baseline_comparison"],
}
