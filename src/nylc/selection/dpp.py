"""Exact fixed-size k-DPP sampling and diversity diagnostics."""

import numpy as np


def normalize_to_correlation(kernel):
    kernel = np.asarray(kernel, dtype=float)
    diagonal = np.maximum(np.diag(kernel), 1e-15)
    correlation = kernel / np.sqrt(np.outer(diagonal, diagonal))
    correlation = (correlation + correlation.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    correlation = eigenvectors * eigenvalues @ eigenvectors.T
    diagonal = np.maximum(np.diag(correlation), 1e-15)
    correlation = correlation / np.sqrt(np.outer(diagonal, diagonal))
    return (correlation + correlation.T) / 2.0


def elementary_symmetric_polynomials(eigenvalues, k):
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    E = np.zeros((k + 1, len(eigenvalues) + 1), dtype=float)
    E[0, :] = 1.0
    for n in range(1, len(eigenvalues) + 1):
        for ell in range(1, min(k, n) + 1):
            E[ell, n] = E[ell, n - 1] + eigenvalues[n - 1] * E[ell - 1, n - 1]
    return E


def sample_projection_dpp(eigenvectors, rng):
    V = np.asarray(eigenvectors, dtype=float).copy()
    selected = []
    while V.shape[1] > 0:
        probabilities = np.sum(V**2, axis=1) / V.shape[1]
        probabilities = np.maximum(probabilities, 0.0)
        probabilities /= probabilities.sum()
        item = int(rng.choice(len(probabilities), p=probabilities))
        selected.append(item)
        column_probabilities = V[item, :] ** 2
        column_probabilities /= column_probabilities.sum()
        column = int(rng.choice(V.shape[1], p=column_probabilities))
        pivot = V[item, column]
        if abs(pivot) < 1e-14:
            raise np.linalg.LinAlgError("Projection-DPP pivot is numerically zero.")
        V = V - np.outer(V[:, column], V[item, :] / pivot)
        V = np.delete(V, column, axis=1)
        if V.shape[1] > 0:
            V, _ = np.linalg.qr(V)
    return selected


def sample_k_dpp(L, k, seed, jitter=1e-09, eigenvalue_tolerance=1e-10):
    L = np.asarray(L, dtype=float)
    if k < 0 or k > len(L):
        raise ValueError("k must be between zero and the DPP ground-set size.")
    if k == 0:
        return []
    L = (L + L.T) / 2.0 + jitter * np.eye(len(L))
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    eigenvalues = np.where(eigenvalues > eigenvalue_tolerance, eigenvalues, 0.0)
    if np.count_nonzero(eigenvalues) < k:
        raise np.linalg.LinAlgError(
            f"DPP kernel rank {np.count_nonzero(eigenvalues)} is smaller than k={k}."
        )
    E = elementary_symmetric_polynomials(eigenvalues, k)
    rng = np.random.default_rng(seed)
    chosen_eigenvectors = []
    remaining = k
    for n in range(len(eigenvalues), 0, -1):
        if remaining == 0:
            break
        denominator = E[remaining, n]
        probability = (
            eigenvalues[n - 1] * E[remaining - 1, n - 1] / denominator if denominator > 0 else 0.0
        )
        if rng.random() < min(max(probability, 0.0), 1.0):
            chosen_eigenvectors.append(n - 1)
            remaining -= 1
    if remaining != 0:
        raise RuntimeError("Failed to sample the requested number of DPP eigenvectors.")
    return sample_projection_dpp(eigenvectors[:, chosen_eigenvectors], rng)


def kernel_batch_diagnostics(kernel):
    correlation = normalize_to_correlation(kernel)
    n = len(correlation)
    off_diagonal = correlation[np.triu_indices(n, k=1)]
    sign, logdet = np.linalg.slogdet(correlation + 1e-12 * np.eye(n))
    geometric_eigenvalue = np.exp(logdet / n) if sign > 0 else 0.0
    arithmetic_eigenvalue = float(np.trace(correlation) / n)
    return {
        "mean_kernel_similarity": float(off_diagonal.mean()) if off_diagonal.size else 0.0,
        "log_determinant": float(logdet) if sign > 0 else -np.inf,
        "isometry_score": float(geometric_eigenvalue / arithmetic_eigenvalue),
    }
