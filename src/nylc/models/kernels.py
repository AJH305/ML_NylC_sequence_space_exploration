"""Epistatic GP kernels and Gaussian predictions, extracted from the reference methods."""

import math
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, pairwise_distances
from nylc.features.descriptors import POSITIONS
import pandas as pd

LEARNED_SOURCE_PREFIXES = ("esm2_", "metl_")


def aa_descriptor_vector(aa, source):
    aa = str(aa).upper()
    if aa not in source.scaled_table.index:
        raise ValueError(f"Unknown or unsupported amino acid: {aa}")
    return source.scaled_table.loc[aa].to_numpy(dtype=float)


def pockets_to_position_descriptor_arrays(pockets, source, positions=POSITIONS):
    if hasattr(source, "position_arrays"):
        return source.position_arrays(pockets, positions=positions)
    arrays = {}
    for pos in positions:
        arrays[pos] = np.vstack([aa_descriptor_vector(pocket[pos], source) for pocket in pockets])
    return arrays


def rbf_kernel_from_descriptors(X1, X2=None, lengthscale=1.0):
    if X2 is None:
        X2 = X1
    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)
    squared_distances = ((X1[:, None, :] - X2[None, :, :]) ** 2).sum(axis=2)
    return np.exp(-squared_distances / (2.0 * float(lengthscale) ** 2))


def normalize_lengthscales(lengthscales, positions=POSITIONS):
    if isinstance(lengthscales, dict):
        return {pos: float(lengthscales[pos]) for pos in positions}
    return {pos: float(lengthscales) for pos in positions}


def build_position_kernels(position_arrays, lengthscales=1.0):
    ls = normalize_lengthscales(lengthscales, positions=position_arrays.keys())
    return {
        pos: rbf_kernel_from_descriptors(X_pos, lengthscale=ls[pos])
        for pos, X_pos in position_arrays.items()
    }


def build_position_cross_kernels(
    train_pockets, test_pockets, source, lengthscales=1.0, positions=POSITIONS
):
    ls = normalize_lengthscales(lengthscales, positions=positions)
    train_arrays = pockets_to_position_descriptor_arrays(train_pockets, source, positions=positions)
    test_arrays = pockets_to_position_descriptor_arrays(test_pockets, source, positions=positions)
    return {
        pos: rbf_kernel_from_descriptors(
            test_arrays[pos], X2=train_arrays[pos], lengthscale=ls[pos]
        )
        for pos in positions
    }


def build_main_kernel(position_kernels):
    return np.mean(np.stack(list(position_kernels.values()), axis=0), axis=0)


def build_epistasis_kernel(position_kernels):
    positions = list(position_kernels)
    products = []
    for i, pos_a in enumerate(positions):
        for pos_b in positions[i + 1 :]:
            products.append(position_kernels[pos_a] * position_kernels[pos_b])
    if not products:
        raise ValueError("At least two positions are required for pairwise epistasis.")
    return np.mean(np.stack(products, axis=0), axis=0)


def build_total_epistatic_kernel(position_kernels, sigma_main=1.0, sigma_epi=0.0):
    k_main = build_main_kernel(position_kernels)
    k_epi = build_epistasis_kernel(position_kernels)
    k_total = float(sigma_main) ** 2 * k_main + float(sigma_epi) ** 2 * k_epi
    return (k_total, k_main, k_epi)


def build_total_cross_kernel(
    train_pockets, test_pockets, source, lengthscales=1.0, sigma_main=1.0, sigma_epi=0.0
):
    cross = build_position_cross_kernels(
        train_pockets, test_pockets, source, lengthscales=lengthscales
    )
    k_main = build_main_kernel(cross)
    k_epi = build_epistasis_kernel(cross)
    return float(sigma_main) ** 2 * k_main + float(sigma_epi) ** 2 * k_epi


def standardize_target(y):
    y = np.asarray(y, dtype=float)
    y_mean = float(np.mean(y))
    y_std = float(np.std(y, ddof=1))
    if not np.isfinite(y_std) or y_std <= 0:
        raise ValueError("Target has zero or invalid standard deviation.")
    return ((y - y_mean) / y_std, y_mean, y_std)


def fit_gp_from_kernel(K_total, y, sem=None, sigma_noise=0.5, jitter=1e-08):
    y_scaled, y_mean, y_std = standardize_target(y)
    n = len(y_scaled)
    if sem is None:
        sem_scaled = np.zeros(n)
    else:
        sem_scaled = np.asarray(sem, dtype=float) / y_std
        sem_scaled = np.nan_to_num(sem_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    K_y = (
        np.asarray(K_total, dtype=float)
        + np.diag(sem_scaled**2)
        + float(sigma_noise) ** 2 * np.eye(n)
        + float(jitter) * np.eye(n)
    )
    L = np.linalg.cholesky(K_y)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_scaled))
    return {
        "K_total": K_total,
        "K_y": K_y,
        "L": L,
        "alpha": alpha,
        "y_mean": y_mean,
        "y_std": y_std,
        "sigma_noise": float(sigma_noise),
        "sem_scaled": sem_scaled,
    }


def gp_predict_from_fit(gp_fit, K_test_train, K_test_diag):
    K_test_train = np.asarray(K_test_train, dtype=float)
    K_test_diag = np.asarray(K_test_diag, dtype=float)
    mean_scaled = K_test_train @ gp_fit["alpha"]
    v = np.linalg.solve(gp_fit["L"], K_test_train.T)
    var_latent_scaled = np.maximum(K_test_diag - np.sum(v**2, axis=0), 0.0)
    var_observed_scaled = var_latent_scaled + gp_fit["sigma_noise"] ** 2
    return {
        "mean": gp_fit["y_mean"] + gp_fit["y_std"] * mean_scaled,
        "std_latent": gp_fit["y_std"] * np.sqrt(var_latent_scaled),
        "std_observed": gp_fit["y_std"] * np.sqrt(var_observed_scaled),
    }


def gaussian_nlpd(y_true, y_pred, std):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    std = np.maximum(np.asarray(std, dtype=float), 1e-09)
    return float(np.mean(0.5 * np.log(2 * np.pi * std**2) + 0.5 * ((y_true - y_pred) / std) ** 2))


def regression_metrics(y_true, y_pred, pred_std_observed=None):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": math.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "pearson": pearsonr(y_true, y_pred).statistic if len(y_true) > 2 else np.nan,
        "spearman": spearmanr(y_true, y_pred).statistic if len(y_true) > 2 else np.nan,
    }
    if pred_std_observed is not None:
        out["nlpd_observed"] = gaussian_nlpd(y_true, y_pred, pred_std_observed)
    return out


def is_learned_source_name(source_name):
    return str(source_name).startswith(LEARNED_SOURCE_PREFIXES)


def positive_pairwise_distances(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2:
        return np.array([], dtype=float)
    distances = pairwise_distances(X, metric="euclidean")
    values = distances[np.triu_indices_from(distances, k=1)]
    return values[np.isfinite(values) & (values > 1e-12)]


def source_training_arrays(pockets, source, positions=POSITIONS):
    if getattr(source, "representation", None) == "global_delta":
        return {"global": np.asarray(source.feature_matrix(pockets), dtype=float)}
    return {
        pos: np.asarray(values, dtype=float)
        for pos, values in pockets_to_position_descriptor_arrays(
            pockets, source, positions=positions
        ).items()
    }


def robust_distance_scale(feature_arrays):
    values = []
    for X in feature_arrays.values():
        values.extend(positive_pairwise_distances(X).tolist())
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("No non-zero pairwise feature distances are available.")
    return float(np.median(values))


def feature_source_diagnostics(df_eval, source):
    arrays = source_training_arrays(df_eval["pocket"].tolist(), source)
    rows = []
    for component, X in arrays.items():
        if not np.isfinite(X).all():
            raise AssertionError(f"Non-finite features for {source.name}:{component}")
        distances = positive_pairwise_distances(X)
        if distances.size == 0:
            rows.append(
                {
                    "source": source.name,
                    "component": component,
                    "n": X.shape[0],
                    "dimension": X.shape[1],
                    "n_nonzero_distances": 0,
                    "median_distance": np.nan,
                    "q10_distance": np.nan,
                    "q90_distance": np.nan,
                    "fraction_kernel_lt_1e-3": np.nan,
                    "fraction_kernel_gt_0p9": np.nan,
                    "min_kernel_eigenvalue": np.nan,
                }
            )
            continue
        reference_lengthscale = float(np.median(distances))
        K = rbf_kernel_from_descriptors(X, lengthscale=reference_lengthscale)
        offdiag = K[np.triu_indices_from(K, k=1)]
        rows.append(
            {
                "source": source.name,
                "component": component,
                "n": X.shape[0],
                "dimension": X.shape[1],
                "n_nonzero_distances": distances.size,
                "median_distance": reference_lengthscale,
                "q10_distance": float(np.quantile(distances, 0.1)),
                "q90_distance": float(np.quantile(distances, 0.9)),
                "fraction_kernel_lt_1e-3": float(np.mean(offdiag < 0.001)),
                "fraction_kernel_gt_0p9": float(np.mean(offdiag > 0.9)),
                "min_kernel_eigenvalue": float(np.linalg.eigvalsh((K + K.T) / 2).min()),
            }
        )
    return pd.DataFrame(rows)
