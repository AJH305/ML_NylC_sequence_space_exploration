"""Baseline comparison methods migrated from the separate Baslines notebook."""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from nylc.models.kernels import regression_metrics
from nylc.validation.baselines import flattened_descriptor_features
from nylc.features.descriptors import residue_descriptor_table


def inner_loocv_mae(estimator, X, y):
    predictions = np.empty(len(y), dtype=float)
    for valid_idx in range(len(y)):
        train_idx = np.delete(np.arange(len(y)), valid_idx)
        fitted = clone(estimator).fit(X.iloc[train_idx], y[train_idx])
        predictions[valid_idx] = fitted.predict(X.iloc[[valid_idx]])[0]
    return mean_absolute_error(y, predictions)


def nested_loocv_candidates(frame, X, model_name, candidates):
    y = frame["activity_pa6"].to_numpy(float)
    rows = []
    for test_idx in range(len(frame)):
        train_idx = np.delete(np.arange(len(frame)), test_idx)
        scores = [inner_loocv_mae(est, X.iloc[train_idx], y[train_idx]) for _, est in candidates]
        best_idx = int(np.argmin(scores))
        setting, estimator = candidates[best_idx]
        fitted = clone(estimator).fit(X.iloc[train_idx], y[train_idx])
        prediction = float(fitted.predict(X.iloc[[test_idx]])[0])
        rows.append(
            {
                "model": model_name,
                "variant_id": frame.loc[test_idx, "variant_id"],
                "mutation_signature": frame.loc[test_idx, "mutation_signature"],
                "mutation_order": int(frame.loc[test_idx, "mutation_order"]),
                "observed": y[test_idx],
                "predicted": prediction,
                "abs_error": abs(y[test_idx] - prediction),
                "selected_setting": setting,
                "inner_mae": scores[best_idx],
            }
        )
    return pd.DataFrame(rows)


def fixed_outer_loocv(frame, X, model_name, estimator):
    y = frame["activity_pa6"].to_numpy(float)
    rows = []
    for test_idx in range(len(frame)):
        train_idx = np.delete(np.arange(len(frame)), test_idx)
        fitted = clone(estimator).fit(X.iloc[train_idx], y[train_idx])
        prediction = float(fitted.predict(X.iloc[[test_idx]])[0])
        rows.append(
            {
                "model": model_name,
                "variant_id": frame.loc[test_idx, "variant_id"],
                "mutation_signature": frame.loc[test_idx, "mutation_signature"],
                "mutation_order": int(frame.loc[test_idx, "mutation_order"]),
                "observed": y[test_idx],
                "predicted": prediction,
                "abs_error": abs(y[test_idx] - prediction),
                "selected_setting": "pre-specified",
            }
        )
    return pd.DataFrame(rows)


def summarize_predictions(predictions):
    rows = []
    for name, group in predictions.groupby("model"):
        if name == "esm2_zero_shot_direct":
            rows.append(
                {
                    "model": name,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "r2": np.nan,
                    "pearson": pearsonr(group.observed, group.predicted).statistic,
                    "spearman": spearmanr(group.observed, group.predicted).statistic,
                }
            )
        else:
            rows.append({"model": name, **regression_metrics(group.observed, group.predicted)})
    return pd.DataFrame(rows).sort_values("mae", na_position="last").reset_index(drop=True)


def run_classical(frame, settings, threads=1):
    """Nested LOOCV with scaling fitted inside every inner/outer training split."""
    frame = frame.reset_index(drop=True)
    physical = pd.DataFrame(
        flattened_descriptor_features(frame, residue_descriptor_table("physical"))
    )
    physical.columns = [f"physical_{i}" for i in range(physical.shape[1])]
    count = frame[["mutation_order"]].astype(float)
    y = frame.activity_pa6.to_numpy(float)
    mean = frame[["variant_id", "mutation_signature", "mutation_order"]].copy()
    mean["model"] = "train_mean"
    mean["observed"] = y
    mean["predicted"] = (y.sum() - y) / (len(y) - 1)
    mean["abs_error"] = np.abs(y - mean.predicted)
    frames = [mean]
    ridge = [
        (f"alpha={a}", make_pipeline(StandardScaler(), Ridge(alpha=a))) for a in settings["alphas"]
    ]
    for name, x in [("ridge_mutation_count", count), ("ridge_physical", physical)]:
        frames.append(nested_loocv_candidates(frame, x, name, ridge))
    # Inner training sets contain n-2 observations in nested LOOCV.
    neighbors = [
        (
            f"k={k}",
            make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=k, weights="distance")),
        )
        for k in settings["knn_neighbors"]
        if k <= len(frame) - 2
    ]
    if not neighbors:
        raise ValueError("No kNN setting fits the inner training folds")
    frames.append(nested_loocv_candidates(frame, physical, "knn_physical", neighbors))
    frames.append(
        fixed_outer_loocv(
            frame,
            physical,
            "bayesian_ridge_physical",
            make_pipeline(StandardScaler(), BayesianRidge()),
        )
    )
    forest = RandomForestRegressor(
        n_estimators=settings["forest_estimators"],
        min_samples_leaf=2,
        random_state=settings["seed"],
        n_jobs=threads,
    )
    frames.append(fixed_outer_loocv(frame, physical, "random_forest_physical_fixed", forest))
    return frames, physical, ridge


def tabicl_predictions(frame, physical, settings, runtime, root):
    """Use a verified local checkpoint without implicit model downloads."""
    from nylc.provenance import sha256_file
    from nylc.runtime import setup_torch

    checkpoint = root / settings["checkpoint"]
    if sha256_file(checkpoint) != settings["sha256"]:
        raise ValueError("TabICL checkpoint checksum mismatch")
    from tabicl import TabICLRegressor

    device = setup_torch(runtime)
    estimator = TabICLRegressor(
        n_estimators=settings["n_estimators"],
        random_state=settings["seed"],
        model_path=str(checkpoint),
        allow_auto_download=False,
        checkpoint_version="tabicl-regressor-v2-20260212.ckpt",
        device=device,
        n_jobs=runtime["threads"],
        verbose=False,
    )
    return fixed_outer_loocv(frame, physical, "tabicl_v2_physical", estimator)


def run_comparison(config, root, output, core, current_gp):
    from nylc.models.epistatic_gp import EpistaticGP
    from nylc.provenance import write_json

    settings = config["baseline_comparison"]
    core = core.reset_index(drop=True).copy()
    frames, physical, ridge = run_classical(core, settings, config["runtime"]["threads"])
    status = {"esm2_zero_shot": "disabled", "tabicl": "disabled"}
    if settings["zero_shot"]["enabled"]:
        from nylc.features.esm2_zero_shot import ESM2ZeroShotScorer
        from nylc.data.lab import read_reference
        from nylc.features.descriptors import WT_POCKET
        from nylc.runtime import setup_torch
        from nylc.models.zero_shot_gp import loocv_hardened_zero_shot_prior

        zs = settings["zero_shot"]
        device = setup_torch(config["runtime"])
        scorer = ESM2ZeroShotScorer(
            wt_sequence=read_reference(root / config["inputs"]["wt_fasta"]),
            wt_pocket=WT_POCKET,
            model_name=zs["model"],
            revision=zs["revision"],
            device=device,
            batch_size=config["runtime"]["batch_size"],
            local_files_only=config["runtime"]["offline"],
            cache_dir=root / "cache/esm2_zero_shot",
            model_cache_dir=root / "cache/huggingface",
        )
        core["esm2_zero_shot"] = scorer.score_pockets(core.pocket.tolist())
        if not np.isfinite(core.esm2_zero_shot).all():
            raise ValueError("Zero-shot scores must be finite for every variant")
        core[["variant_id", "esm2_zero_shot"]].to_csv(
            output / "esm2_zero_shot_scores.csv", index=False
        )
        direct = core[
            ["variant_id", "mutation_signature", "mutation_order", "activity_pa6", "esm2_zero_shot"]
        ].rename(columns={"activity_pa6": "observed", "esm2_zero_shot": "predicted"})
        direct["model"] = "esm2_zero_shot_direct"
        direct["abs_error"] = np.nan  # Raw scores are not expressed in activity units.
        frames.append(direct)
        augmented = physical.assign(esm2_zero_shot=core.esm2_zero_shot.to_numpy())
        frames.append(
            nested_loocv_candidates(core, augmented, "ridge_physical_plus_esm2_zero_shot", ridge)
        )
        prior = loocv_hardened_zero_shot_prior(
            core, physical, n_starts=config["gp"]["final_restarts"], settings=config["gp"]
        )
        prior.to_csv(output / "gp_zero_shot_prior_predictions.csv", index=False)
        frames.append(prior)
        status["esm2_zero_shot"] = {
            "model": zs["model"],
            "revision": zs["revision"],
            "device": device,
        }
    if settings["tabicl"]["enabled"]:
        frames.append(
            tabicl_predictions(core, physical, settings["tabicl"], config["runtime"], root)
        )
        status["tabicl"] = {
            "checkpoint": settings["tabicl"]["checkpoint"],
            "sha256": settings["tabicl"]["sha256"],
        }
    # Compute the fixed physical Epistatic GP comparator directly; do not relabel
    # nested source selection as if every fold had selected the same model.
    reference_name = "gp_hardened_physical_epistatic"
    model = EpistaticGP(config["gp"], residue_descriptor_table, ["physical"])
    reference = model.loocv_predictions_for_fixed_setting(
        core,
        {"descriptor_set": "physical", "model": "epistatic", "pca_components": None},
        n_starts=config["gp"]["final_restarts"],
    )
    reference["model"] = reference_name
    reference["abs_error"] = abs(reference.observed - reference.predicted)
    frames.append(reference)
    current_gp = current_gp.copy()
    current_gp["model"] = "gp_nested_source_selection"
    current_gp["abs_error"] = abs(current_gp.observed - current_gp.predicted)
    frames.append(current_gp)
    predictions = pd.concat(frames, ignore_index=True, sort=False)
    expected = set(core.variant_id)
    for name, group in predictions.groupby("model"):
        if (
            len(group) != len(core)
            or set(group.variant_id) != expected
            or not group.variant_id.is_unique
        ):
            raise ValueError(f"Comparator {name} has mismatched held-out variants")
        observed = group.set_index("variant_id").observed.reindex(core.variant_id)
        np.testing.assert_allclose(observed, core.activity_pa6, rtol=0, atol=1e-10)
    metrics = summarize_predictions(predictions)
    predictions.to_csv(output / "all_loocv_predictions.csv", index=False)
    metrics.to_csv(output / "all_loocv_metrics.csv", index=False)
    best = metrics.loc[metrics.mae.notna()].iloc[0]["model"]
    paired = predictions.pivot(index="variant_id", columns="model", values="abs_error")
    paired[f"gain_{best}_vs_{reference_name}"] = paired[reference_name] - paired[best]
    paired.to_csv(output / "paired_model_errors.csv")
    write_json(
        output / "summary.json",
        {
            "models": metrics.model.tolist(),
            "optional_models": status,
            "n_variants": len(core),
            "best_model_by_observed_mae": best,
            "interpretation": "Best-model ranking is descriptive, not an independently validated selection procedure.",
            "gp_reference": "Physical epistatic setting fixed in advance; parameters refitted in each outer fold.",
            "zero_shot_uncertainty": "Prior-GP intervals retain the original plug-in calculation; uncertainty in fitted mean coefficients is not propagated.",
        },
    )
