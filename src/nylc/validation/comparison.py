"""Original fixed-descriptor model comparison and nonidentical-pair diagnostics.

Retains the earlier grid-based comparison in notebook 03 separately from
feature-hardened marginal-likelihood model selection.
"""

from itertools import product
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

KERNEL_DIAGNOSTIC_MULTIPLIERS = (0.5, 1.0, 2.0)
from nylc.models.kernels import (
    pockets_to_position_descriptor_arrays,
    rbf_kernel_from_descriptors,
    build_position_kernels,
    build_total_epistatic_kernel,
    build_total_cross_kernel,
    fit_gp_from_kernel,
    gp_predict_from_fit,
    regression_metrics,
    source_training_arrays,
    robust_distance_scale,
)


def nonidentical_kernel_diagnostics(df_eval, source, multipliers=KERNEL_DIAGNOSTIC_MULTIPLIERS):
    arrays = source_training_arrays(df_eval["pocket"].tolist(), source)
    reference_scale = robust_distance_scale(arrays)
    rows = []
    distributions = []
    for component, X in arrays.items():
        distances_matrix = pairwise_distances(np.asarray(X, dtype=float), metric="euclidean")
        distances = distances_matrix[np.triu_indices_from(distances_matrix, k=1)]
        finite = np.isfinite(distances)
        identical = finite & (distances <= 1e-12)
        nonidentical_distances = distances[finite & ~identical]
        n_finite = int(finite.sum())
        if nonidentical_distances.size == 0:
            continue
        for multiplier in multipliers:
            lengthscale = float(multiplier) * reference_scale
            similarities = np.exp(-(nonidentical_distances**2) / (2.0 * lengthscale**2))
            rows.append(
                {
                    "source": source.name,
                    "component": component,
                    "lengthscale_multiplier": float(multiplier),
                    "reference_distance_scale": reference_scale,
                    "resolved_lengthscale": lengthscale,
                    "n_finite_offdiagonal_pairs": n_finite,
                    "n_identical_pairs": int(identical.sum()),
                    "fraction_identical_pairs": float(identical.sum() / n_finite),
                    "n_nonidentical_pairs": int(nonidentical_distances.size),
                    "kernel_q05_nonidentical": float(np.quantile(similarities, 0.05)),
                    "kernel_q25_nonidentical": float(np.quantile(similarities, 0.25)),
                    "kernel_median_nonidentical": float(np.median(similarities)),
                    "kernel_q75_nonidentical": float(np.quantile(similarities, 0.75)),
                    "kernel_q95_nonidentical": float(np.quantile(similarities, 0.95)),
                    "fraction_nonidentical_kernel_lt_1e-3": float(np.mean(similarities < 0.001)),
                    "fraction_nonidentical_kernel_lt_0p01": float(np.mean(similarities < 0.01)),
                    "fraction_nonidentical_kernel_gt_0p9": float(np.mean(similarities > 0.9)),
                }
            )
            distributions.extend(
                (
                    {
                        "source": source.name,
                        "component": component,
                        "lengthscale_multiplier": float(multiplier),
                        "kernel_similarity": float(value),
                    }
                    for value in similarities
                )
            )
    return (pd.DataFrame(rows), pd.DataFrame(distributions))


def fit_and_predict_one_fold(
    df_train, df_test, source, lengthscale, sigma_main, sigma_epi, sigma_noise
):
    from nylc.data.lab import training_sem

    df_train = df_train.copy()
    df_train["activity_sem_for_gp"] = training_sem(df_train)
    train_pockets = df_train["pocket"].tolist()
    test_pockets = df_test["pocket"].tolist()
    if getattr(source, "representation", None) == "global_delta":
        X_train = source.feature_matrix(train_pockets)
        X_test = source.feature_matrix(test_pockets)
        K_train = float(sigma_main) ** 2 * rbf_kernel_from_descriptors(
            X_train, lengthscale=lengthscale
        )
        K_test_train = float(sigma_main) ** 2 * rbf_kernel_from_descriptors(
            X_test, X2=X_train, lengthscale=lengthscale
        )
        K_test_diag = np.full(len(df_test), float(sigma_main) ** 2)
    else:
        train_arrays = pockets_to_position_descriptor_arrays(train_pockets, source)
        train_kernels = build_position_kernels(train_arrays, lengthscales=lengthscale)
        K_train, _, _ = build_total_epistatic_kernel(
            train_kernels, sigma_main=sigma_main, sigma_epi=sigma_epi
        )
        K_test_train = build_total_cross_kernel(
            train_pockets=train_pockets,
            test_pockets=test_pockets,
            source=source,
            lengthscales=lengthscale,
            sigma_main=sigma_main,
            sigma_epi=sigma_epi,
        )
        K_test_diag = np.full(len(df_test), float(sigma_main) ** 2 + float(sigma_epi) ** 2)
    gp_fit = fit_gp_from_kernel(
        K_train,
        y=df_train["activity_pa6"].to_numpy(dtype=float),
        sem=df_train["activity_sem_for_gp"].to_numpy(dtype=float),
        sigma_noise=sigma_noise,
    )
    return gp_predict_from_fit(gp_fit, K_test_train, K_test_diag)


def loocv_epistatic_gp(
    df_eval, source, lengthscale, sigma_main, sigma_epi, sigma_noise, label="model"
):
    rows = []
    df_eval = df_eval.reset_index(drop=True)
    for test_idx in range(len(df_eval)):
        df_test = df_eval.iloc[[test_idx]].copy()
        df_train = df_eval.drop(index=test_idx).copy()
        pred = fit_and_predict_one_fold(
            df_train, df_test, source, lengthscale, sigma_main, sigma_epi, sigma_noise
        )
        observed = float(df_test["activity_pa6"].iloc[0])
        predicted = float(pred["mean"][0])
        rows.append(
            {
                "model": label,
                "variant_id": df_test["variant_id"].iloc[0],
                "mutations": df_test["mutations"].iloc[0],
                "mutation_signature": df_test["mutation_signature"].iloc[0],
                "mutation_order": int(df_test["mutation_order"].iloc[0]),
                "observed": observed,
                "predicted": predicted,
                "predicted_std_latent": float(pred["std_latent"][0]),
                "predicted_std_observed": float(pred["std_observed"][0]),
                "abs_error": abs(observed - predicted),
                "lengthscale": lengthscale,
                "sigma_main": sigma_main,
                "sigma_epi": sigma_epi,
                "sigma_noise": sigma_noise,
            }
        )
    result = pd.DataFrame(rows)
    metrics = regression_metrics(
        result["observed"], result["predicted"], result["predicted_std_observed"]
    )
    return (result, metrics)


def evaluate_grid_loocv(df_eval, source, grid, label="model"):
    rows = []
    for params in grid:
        _, metrics = loocv_epistatic_gp(df_eval, source=source, label=label, **params)
        rows.append({**params, **metrics})
    return pd.DataFrame(rows).sort_values(["mae", "rmse"]).reset_index(drop=True)


def make_param_grid(model_type):
    lengthscale_values = (0.2, 0.5, 1.0)
    sigma_main_values = (0.5, 1.0)
    sigma_noise_values = (0.3, 0.5, 1.0)
    if model_type == "additive":
        sigma_epi_values = (0.0,)
    elif model_type == "epistatic":
        sigma_epi_values = (1.0, 2.0, 4.0)
    else:
        raise ValueError("model_type must be additive or epistatic")
    return [
        {
            "lengthscale": lengthscale,
            "sigma_main": sigma_main,
            "sigma_epi": sigma_epi,
            "sigma_noise": sigma_noise,
        }
        for lengthscale, sigma_main, sigma_epi, sigma_noise in product(
            lengthscale_values, sigma_main_values, sigma_epi_values, sigma_noise_values
        )
    ]


def select_hyperparameters_inner_loocv(df_train, source, model_type):
    grid_result = evaluate_grid_loocv(
        df_train, source=source, grid=make_param_grid(model_type), label=model_type
    )
    best = grid_result.iloc[0].to_dict()
    best_params = {
        key: best[key] for key in ["lengthscale", "sigma_main", "sigma_epi", "sigma_noise"]
    }
    return (best_params, grid_result)


def nested_loocv_model_comparison(df_eval, source, model_types=("additive", "epistatic")):
    df_eval = df_eval.reset_index(drop=True)
    rows = []
    inner_grids = {}
    for outer_idx in range(len(df_eval)):
        df_test = df_eval.iloc[[outer_idx]].copy()
        df_train = df_eval.drop(index=outer_idx).copy()
        variant_id = df_test["variant_id"].iloc[0]
        for model_type in model_types:
            best_params, inner_grid = select_hyperparameters_inner_loocv(
                df_train, source=source, model_type=model_type
            )
            inner_grids[variant_id, model_type] = inner_grid
            pred = fit_and_predict_one_fold(df_train, df_test, source=source, **best_params)
            observed = float(df_test["activity_pa6"].iloc[0])
            predicted = float(pred["mean"][0])
            rows.append(
                {
                    "model": model_type,
                    "variant_id": variant_id,
                    "mutations": df_test["mutations"].iloc[0],
                    "mutation_signature": df_test["mutation_signature"].iloc[0],
                    "mutation_order": int(df_test["mutation_order"].iloc[0]),
                    "observed": observed,
                    "predicted": predicted,
                    "predicted_std_latent": float(pred["std_latent"][0]),
                    "predicted_std_observed": float(pred["std_observed"][0]),
                    "abs_error": abs(observed - predicted),
                    **{f"selected_{key}": value for key, value in best_params.items()},
                    "inner_mae": float(inner_grid.iloc[0]["mae"]),
                }
            )
        print(f"{outer_idx + 1:02d}/{len(df_eval)} {variant_id}")
    result = pd.DataFrame(rows)
    metrics = []
    for model, group in result.groupby("model"):
        metrics.append(
            {
                "model": model,
                **regression_metrics(
                    group["observed"], group["predicted"], group["predicted_std_observed"]
                ),
            }
        )
    return (result, pd.DataFrame(metrics).sort_values("mae"), inner_grids)
