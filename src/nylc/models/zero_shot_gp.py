"""Hardened Epistatic GP with a linear zero-shot prior mean.

The covariance construction and regularized marginal-likelihood optimization
match the hardened Epistatic GP workflow.  The only model extension is a linear
transformation of a scalar zero-shot score in the GP mean function.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from nylc.data.lab import training_sem


MLL_N_STARTS = 8
MLL_RANDOM_SEED = 20250308
MLL_MAXITER = 300
MLL_BOUNDS = {
    "lengthscale_multiplier": (0.25, 4.0),
    "sigma_main": (0.05, 4.0),
    "sigma_epi": (1e-3, 4.0),
    "sigma_noise": (0.03, 2.0),
}
MLL_LOG_PRIOR_SD = {
    "lengthscale_multiplier": 0.70,
    "sigma_main": 1.00,
    "sigma_noise": 1.00,
}
MLL_NOISE_PRIOR_CENTER = 0.30
MLL_EPI_SHRINKAGE_SCALE = 1.00


def _positive_pairwise_distances(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2:
        return np.array([], dtype=float)
    differences = x[:, None, :] - x[None, :, :]
    distances = np.sqrt(np.sum(differences * differences, axis=2))
    values = distances[np.triu_indices_from(distances, k=1)]
    return values[np.isfinite(values) & (values > 1e-12)]


def robust_distance_scale(x: np.ndarray, n_positions: int = 4) -> float:
    """Median non-zero distance across position-specific training arrays."""
    x = np.asarray(x, dtype=float)
    if x.shape[1] % n_positions:
        raise ValueError("Feature count must be divisible by the number of positions.")
    width = x.shape[1] // n_positions
    values = []
    for position_idx in range(n_positions):
        block = x[:, position_idx * width : (position_idx + 1) * width]
        values.extend(_positive_pairwise_distances(block).tolist())
    if not values:
        raise ValueError("No non-zero pairwise training distances are available.")
    return float(np.median(np.asarray(values, dtype=float)))


def _rbf_kernel(x_a: np.ndarray, x_b: np.ndarray, lengthscale: float) -> np.ndarray:
    differences = x_a[:, None, :] - x_b[None, :, :]
    squared_distances = np.sum(differences * differences, axis=2)
    return np.exp(-0.5 * squared_distances / float(lengthscale) ** 2)


def epistatic_kernel(
    x_a: np.ndarray,
    x_b: np.ndarray,
    lengthscale: float,
    sigma_main: float,
    sigma_epi: float,
    n_positions: int = 4,
) -> np.ndarray:
    """Mean main-effect kernel plus mean of all pairwise kernel products."""
    x_a, x_b = np.asarray(x_a, float), np.asarray(x_b, float)
    if x_a.shape[1] != x_b.shape[1] or x_a.shape[1] % n_positions:
        raise ValueError("Incompatible position-wise feature matrices.")
    width = x_a.shape[1] // n_positions
    position_kernels = []
    for position_idx in range(n_positions):
        start, stop = position_idx * width, (position_idx + 1) * width
        position_kernels.append(_rbf_kernel(x_a[:, start:stop], x_b[:, start:stop], lengthscale))
    main = np.mean(np.stack(position_kernels, axis=0), axis=0)
    interactions = [
        position_kernels[i] * position_kernels[j]
        for i in range(n_positions)
        for j in range(i + 1, n_positions)
    ]
    epistatic = np.mean(np.stack(interactions, axis=0), axis=0)
    return float(sigma_main) ** 2 * main + float(sigma_epi) ** 2 * epistatic


def _standardize_target(y: np.ndarray) -> tuple[np.ndarray, float, float]:
    y = np.asarray(y, dtype=float)
    mean = float(np.mean(y))
    std = float(np.std(y, ddof=0))
    if not np.isfinite(std) or std <= 0:
        std = 1.0
    return (y - mean) / std, mean, std


def _zero_shot_design(z_train: np.ndarray, z_test: np.ndarray | None = None):
    z_train = np.asarray(z_train, dtype=float).reshape(-1)
    mean = float(np.mean(z_train))
    std = float(np.std(z_train, ddof=0))
    if not np.isfinite(std) or std <= 0:
        std = 1.0
    train = np.column_stack([np.ones(len(z_train)), (z_train - mean) / std])
    if z_test is None:
        return train, None
    z_test = np.asarray(z_test, dtype=float).reshape(-1)
    test = np.column_stack([np.ones(len(z_test)), (z_test - mean) / std])
    return train, test


def _gls_coefficients(chol: np.ndarray, design: np.ndarray, target: np.ndarray) -> np.ndarray:
    precision_design = np.linalg.solve(chol.T, np.linalg.solve(chol, design))
    precision_target = np.linalg.solve(chol.T, np.linalg.solve(chol, target))
    normal_matrix = design.T @ precision_design + 1e-8 * np.eye(design.shape[1])
    return np.linalg.solve(normal_matrix, design.T @ precision_target)


def _unpack(log_values: Sequence[float]) -> dict[str, float]:
    names = ("lengthscale_multiplier", "sigma_main", "sigma_epi", "sigma_noise")
    return dict(zip(names, np.exp(np.asarray(log_values, dtype=float))))


def regularized_gp_objective(
    log_values: Sequence[float],
    x_train: np.ndarray,
    y: np.ndarray,
    sem: np.ndarray,
    zero_shot: np.ndarray,
    settings=None,
) -> float:
    settings = settings or {}
    MLL_LOG_PRIOR_SD = settings.get("log_prior_sd", globals()["MLL_LOG_PRIOR_SD"])
    MLL_NOISE_PRIOR_CENTER = settings.get("noise_prior_center", globals()["MLL_NOISE_PRIOR_CENTER"])
    MLL_EPI_SHRINKAGE_SCALE = settings.get(
        "epistasis_shrinkage_scale", globals()["MLL_EPI_SHRINKAGE_SCALE"]
    )
    params = _unpack(log_values)
    try:
        distance_scale = robust_distance_scale(x_train)
        lengthscale = params["lengthscale_multiplier"] * distance_scale
        kernel = epistatic_kernel(
            x_train, x_train, lengthscale, params["sigma_main"], params["sigma_epi"]
        )
        y_scaled, _, y_std = _standardize_target(y)
        sem_scaled = np.asarray(sem, dtype=float) / y_std
        covariance = (kernel + kernel.T) / 2
        covariance += np.diag(sem_scaled**2 + params["sigma_noise"] ** 2 + 1e-8)
        chol = np.linalg.cholesky(covariance)
        design, _ = _zero_shot_design(zero_shot)
        beta = _gls_coefficients(chol, design, y_scaled)
        residual = y_scaled - design @ beta
        alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, residual))
        nll = (
            0.5 * float(residual @ alpha)
            + float(np.log(np.diag(chol)).sum())
            + 0.5 * len(y_scaled) * np.log(2.0 * np.pi)
        )
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return 1e30

    penalty = 0.0
    penalty += (
        0.5
        * (np.log(params["lengthscale_multiplier"]) / MLL_LOG_PRIOR_SD["lengthscale_multiplier"])
        ** 2
    )
    penalty += 0.5 * (np.log(params["sigma_main"]) / MLL_LOG_PRIOR_SD["sigma_main"]) ** 2
    penalty += (
        0.5
        * (np.log(params["sigma_noise"] / MLL_NOISE_PRIOR_CENTER) / MLL_LOG_PRIOR_SD["sigma_noise"])
        ** 2
    )
    penalty += 0.5 * (params["sigma_epi"] / MLL_EPI_SHRINKAGE_SCALE) ** 2
    return float(nll + penalty)


def optimize_hyperparameters(
    x_train: np.ndarray,
    y: np.ndarray,
    sem: np.ndarray,
    zero_shot: np.ndarray,
    n_starts: int = MLL_N_STARTS,
    settings=None,
) -> dict[str, object]:
    names = ("lengthscale_multiplier", "sigma_main", "sigma_epi", "sigma_noise")
    settings = settings or {}
    bounds = settings.get("bounds", MLL_BOUNDS)
    log_bounds = [tuple(np.log(bounds[name])) for name in names]
    initial = np.log([1.0, 1.0, 0.5, MLL_NOISE_PRIOR_CENTER])
    starts = [initial]
    rng = np.random.default_rng(settings.get("seed", MLL_RANDOM_SEED) + len(y) + len(names))
    for _ in range(max(int(n_starts) - 1, 0)):
        starts.append(np.asarray([rng.uniform(low, high) for low, high in log_bounds]))
    results = [
        minimize(
            regularized_gp_objective,
            start,
            args=(x_train, y, sem, zero_shot, settings),
            method="L-BFGS-B",
            bounds=log_bounds,
            options={"maxiter": settings.get("max_iterations", MLL_MAXITER), "ftol": 1e-10},
        )
        for start in starts
    ]
    finite = [result for result in results if np.isfinite(result.fun) and result.fun < 1e29]
    if not finite:
        raise RuntimeError("All hardened GP optimization starts failed.")
    best = min(finite, key=lambda result: result.fun)
    fitted: dict[str, object] = _unpack(best.x)
    fitted.update(
        regularized_neg_log_marginal_likelihood=float(best.fun),
        optimizer_success=bool(best.success),
        optimizer_message=str(best.message),
        optimizer_iterations=int(best.nit),
        successful_restarts=int(sum(result.success for result in finite)),
        n_restarts=len(results),
    )
    return fitted


def fit_predict_one_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    sem_train: np.ndarray,
    z_train: np.ndarray,
    x_test: np.ndarray,
    z_test: np.ndarray,
    n_starts: int = MLL_N_STARTS,
    settings=None,
) -> dict[str, object]:
    fitted = optimize_hyperparameters(
        x_train, y_train, sem_train, z_train, n_starts=n_starts, settings=settings
    )
    distance_scale = robust_distance_scale(x_train)
    lengthscale = float(fitted["lengthscale_multiplier"]) * distance_scale
    kernel_train = epistatic_kernel(
        x_train,
        x_train,
        lengthscale,
        float(fitted["sigma_main"]),
        float(fitted["sigma_epi"]),
    )
    kernel_cross = epistatic_kernel(
        x_test,
        x_train,
        lengthscale,
        float(fitted["sigma_main"]),
        float(fitted["sigma_epi"]),
    )
    y_scaled, y_mean, y_std = _standardize_target(y_train)
    sem_scaled = np.asarray(sem_train, dtype=float) / y_std
    covariance = (kernel_train + kernel_train.T) / 2
    covariance += np.diag(sem_scaled**2 + float(fitted["sigma_noise"]) ** 2 + 1e-8)
    chol = np.linalg.cholesky(covariance)
    design_train, design_test = _zero_shot_design(z_train, z_test)
    beta = _gls_coefficients(chol, design_train, y_scaled)
    residual = y_scaled - design_train @ beta
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, residual))
    mean_scaled = design_test @ beta + kernel_cross @ alpha
    mean = mean_scaled * y_std + y_mean

    test_diagonal = float(fitted["sigma_main"]) ** 2 + float(fitted["sigma_epi"]) ** 2
    projection = np.linalg.solve(chol, kernel_cross.T)
    latent_variance_scaled = np.maximum(test_diagonal - np.sum(projection**2, axis=0), 0.0)
    latent_std = np.sqrt(latent_variance_scaled) * y_std
    observed_std = np.sqrt(latent_variance_scaled + float(fitted["sigma_noise"]) ** 2) * y_std
    return {
        "mean": np.asarray(mean, dtype=float),
        "std_latent": np.asarray(latent_std, dtype=float),
        "std_observed": np.asarray(observed_std, dtype=float),
        "fitted_hyperparameters": fitted,
        "zero_shot_intercept_scaled": float(beta[0]),
        "zero_shot_slope_scaled": float(beta[1]),
        "training_distance_scale": distance_scale,
        "resolved_lengthscale": lengthscale,
        "min_train_kernel_eigenvalue": float(
            np.linalg.eigvalsh((kernel_train + kernel_train.T) / 2).min()
        ),
    }


def loocv_hardened_zero_shot_prior(
    frame: pd.DataFrame,
    physical_features: pd.DataFrame | np.ndarray,
    n_starts: int = MLL_N_STARTS,
    settings=None,
) -> pd.DataFrame:
    """Outer LOOCV; every fold refits the prior and all GP parameters."""
    frame = frame.reset_index(drop=True)
    x = np.asarray(physical_features, dtype=float)
    y = frame["activity_pa6"].to_numpy(dtype=float)
    zero_shot = frame["esm2_zero_shot"].to_numpy(dtype=float)
    rows = []
    for test_idx in range(len(frame)):
        train_idx = np.delete(np.arange(len(frame)), test_idx)
        prediction = fit_predict_one_fold(
            x[train_idx],
            y[train_idx],
            training_sem(frame.iloc[train_idx]),
            zero_shot[train_idx],
            x[[test_idx]],
            zero_shot[[test_idx]],
            n_starts=n_starts,
            settings=settings,
        )
        fitted = prediction["fitted_hyperparameters"]
        predicted = float(prediction["mean"][0])
        rows.append(
            {
                "model": "gp_hardened_physical_epistatic_with_esm2_zero_shot_prior",
                "variant_id": frame.loc[test_idx, "variant_id"],
                "mutation_signature": frame.loc[test_idx, "mutation_signature"],
                "mutation_order": int(frame.loc[test_idx, "mutation_order"]),
                "observed": y[test_idx],
                "predicted": predicted,
                "predicted_std_latent": float(prediction["std_latent"][0]),
                "predicted_std_observed": float(prediction["std_observed"][0]),
                "abs_error": abs(y[test_idx] - predicted),
                "descriptor_set": "physical",
                "model_family": "epistatic",
                "training_distance_scale": prediction["training_distance_scale"],
                "resolved_lengthscale": prediction["resolved_lengthscale"],
                "min_train_kernel_eigenvalue": prediction["min_train_kernel_eigenvalue"],
                "zero_shot_intercept_scaled": prediction["zero_shot_intercept_scaled"],
                "zero_shot_slope_scaled": prediction["zero_shot_slope_scaled"],
                **{f"fitted_{name}": fitted[name] for name in MLL_BOUNDS},
                "regularized_neg_log_marginal_likelihood": fitted[
                    "regularized_neg_log_marginal_likelihood"
                ],
                "optimizer_success": fitted["optimizer_success"],
            }
        )
        print(f"Hardened Physical GP + ESM-2 prior: {test_idx + 1:02d}/{len(frame)}")
    return pd.DataFrame(rows)
