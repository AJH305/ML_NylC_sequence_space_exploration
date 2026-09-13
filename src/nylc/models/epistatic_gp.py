"""Fold-local Epistatic GP fitting, nested model selection and sigma calibration.

Methods migrated from notebook 03; sigma calibration from notebook 04.
All optimizer settings and feature registries belong to this instance.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from nylc.features.descriptors import POSITIONS
from nylc.models.kernels import (
    is_learned_source_name,
    pockets_to_position_descriptor_arrays,
    robust_distance_scale,
    rbf_kernel_from_descriptors,
    build_total_epistatic_kernel,
    build_main_kernel,
    build_epistasis_kernel,
    standardize_target,
    fit_gp_from_kernel,
    gp_predict_from_fit,
    regression_metrics,
)

CALIBRATION_LEVELS = {"50%": 0.67448975, "68%": 1.0, "90%": 1.64485363, "95%": 1.95996398}


class EpistaticGP:
    def __init__(self, settings, source_factory, available_sources):
        self.make_feature_source = source_factory
        self.AVAILABLE_FEATURE_SOURCES = tuple(available_sources)
        self.DESCRIPTOR_CACHE = {}
        self.HARDENED_PCA_COMPONENTS = tuple(settings["pca_components"])
        self.MLL_N_STARTS = settings["final_restarts"]
        self.MLL_INNER_N_STARTS = settings["inner_restarts"]
        self.MLL_RANDOM_SEED = settings["seed"]
        self.MLL_MAXITER = settings["max_iterations"]
        self.MLL_BOUNDS = settings["bounds"]
        self.MLL_LOG_PRIOR_SD = settings["log_prior_sd"]
        self.MLL_NOISE_PRIOR_CENTER = settings["noise_prior_center"]
        self.MLL_EPI_SHRINKAGE_SCALE = settings["epistasis_shrinkage_scale"]

    def get_descriptor_source(self, source_name):
        """Create each frozen feature source once and reuse its cached representation."""
        if source_name not in self.AVAILABLE_FEATURE_SOURCES:
            raise KeyError(f"Feature source is unavailable or disabled: {source_name}")
        if source_name not in self.DESCRIPTOR_CACHE:
            self.DESCRIPTOR_CACHE[source_name] = self.make_feature_source(source_name)
        return self.DESCRIPTOR_CACHE[source_name]

    def fit_pca_train_only(self, X_train, X_test, requested_components):
        X_train = np.asarray(X_train, dtype=float)
        X_test = np.asarray(X_test, dtype=float)
        if requested_components is None:
            return (X_train, X_test, None)
        n_components = min(int(requested_components), X_train.shape[0] - 1, X_train.shape[1])
        if n_components < 1:
            raise ValueError("PCA requires at least one valid component.")
        reducer = PCA(n_components=n_components, svd_solver="full")
        X_train_reduced = reducer.fit_transform(X_train)
        X_test_reduced = reducer.transform(X_test)
        return (X_train_reduced, X_test_reduced, reducer)

    def prepare_fold_features(self, train_pockets, test_pockets, source, pca_components=None):
        representation = getattr(source, "representation", None)
        if representation == "global_delta":
            train_raw = np.asarray(source.feature_matrix(train_pockets), dtype=float)
            test_raw = np.asarray(source.feature_matrix(test_pockets), dtype=float)
            train_reduced, test_reduced, reducer = self.fit_pca_train_only(
                train_raw, test_raw, pca_components
            )
            return {
                "kind": "global",
                "train": {"global": train_reduced},
                "test": {"global": test_reduced},
                "reducers": {"global": reducer},
            }
        train_raw = pockets_to_position_descriptor_arrays(train_pockets, source)
        test_raw = pockets_to_position_descriptor_arrays(test_pockets, source)
        train_out, test_out, reducers = ({}, {}, {})
        for pos in POSITIONS:
            apply_pca = pca_components if is_learned_source_name(source.name) else None
            train_out[pos], test_out[pos], reducers[pos] = self.fit_pca_train_only(
                train_raw[pos], test_raw[pos], apply_pca
            )
        return {"kind": "position", "train": train_out, "test": test_out, "reducers": reducers}

    def kernels_from_prepared_features(
        self, prepared, lengthscale_multiplier, sigma_main, sigma_epi
    ):
        distance_scale = robust_distance_scale(prepared["train"])
        lengthscale = float(lengthscale_multiplier) * distance_scale
        if not np.isfinite(lengthscale) or lengthscale <= 0:
            raise ValueError("Resolved length scale must be positive and finite.")
        if prepared["kind"] == "global":
            X_train = prepared["train"]["global"]
            X_test = prepared["test"]["global"]
            K_train = float(sigma_main) ** 2 * rbf_kernel_from_descriptors(
                X_train, lengthscale=lengthscale
            )
            K_test_train = float(sigma_main) ** 2 * rbf_kernel_from_descriptors(
                X_test, X2=X_train, lengthscale=lengthscale
            )
            K_test_diag = np.full(X_test.shape[0], float(sigma_main) ** 2)
        else:
            train_position_kernels = {
                pos: rbf_kernel_from_descriptors(X, lengthscale=lengthscale)
                for pos, X in prepared["train"].items()
            }
            cross_position_kernels = {
                pos: rbf_kernel_from_descriptors(
                    prepared["test"][pos], X2=prepared["train"][pos], lengthscale=lengthscale
                )
                for pos in POSITIONS
            }
            K_train, _, _ = build_total_epistatic_kernel(
                train_position_kernels, sigma_main=sigma_main, sigma_epi=sigma_epi
            )
            K_test_train = float(sigma_main) ** 2 * build_main_kernel(
                cross_position_kernels
            ) + float(sigma_epi) ** 2 * build_epistasis_kernel(cross_position_kernels)
            K_test_diag = np.full(
                len(next(iter(prepared["test"].values()))),
                float(sigma_main) ** 2 + float(sigma_epi) ** 2,
            )
        return (K_train, K_test_train, K_test_diag, distance_scale, lengthscale)

    def training_kernel_from_prepared(
        self, prepared, lengthscale_multiplier, sigma_main, sigma_epi
    ):
        """Build a training kernel without touching held-out features."""
        distance_scale = robust_distance_scale(prepared["train"])
        lengthscale = float(lengthscale_multiplier) * distance_scale
        if prepared["kind"] == "global":
            X = prepared["train"]["global"]
            base_main = rbf_kernel_from_descriptors(X, lengthscale=lengthscale)
            base_epi = np.zeros_like(base_main)
        else:
            position_kernels = {
                pos: rbf_kernel_from_descriptors(X, lengthscale=lengthscale)
                for pos, X in prepared["train"].items()
            }
            base_main = build_main_kernel(position_kernels)
            base_epi = build_epistasis_kernel(position_kernels)
        K = float(sigma_main) ** 2 * base_main + float(sigma_epi) ** 2 * base_epi
        return (K, distance_scale, lengthscale)

    def regularized_gp_objective(self, log_values, parameter_names, prepared, y, sem):
        """Negative log marginal likelihood plus weak scientific regularization."""
        params = dict(zip(parameter_names, np.exp(np.asarray(log_values, dtype=float))))
        sigma_epi = params.get("sigma_epi", 0.0)
        try:
            K, _, _ = self.training_kernel_from_prepared(
                prepared, params["lengthscale_multiplier"], params["sigma_main"], sigma_epi
            )
            y_scaled, _, y_std = standardize_target(y)
            sem_scaled = np.asarray(sem, dtype=float) / y_std
            diagonal = sem_scaled**2 + params["sigma_noise"] ** 2 + 1e-08
            K_y = (K + K.T) / 2 + np.diag(diagonal)
            L = np.linalg.cholesky(K_y)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_scaled))
            nll = (
                0.5 * float(y_scaled @ alpha)
                + float(np.log(np.diag(L)).sum())
                + 0.5 * len(y_scaled) * np.log(2.0 * np.pi)
            )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            return 1e30
        penalty = 0.0
        penalty += (
            0.5
            * (
                np.log(params["lengthscale_multiplier"])
                / self.MLL_LOG_PRIOR_SD["lengthscale_multiplier"]
            )
            ** 2
        )
        penalty += 0.5 * (np.log(params["sigma_main"]) / self.MLL_LOG_PRIOR_SD["sigma_main"]) ** 2
        penalty += (
            0.5
            * (
                np.log(params["sigma_noise"] / self.MLL_NOISE_PRIOR_CENTER)
                / self.MLL_LOG_PRIOR_SD["sigma_noise"]
            )
            ** 2
        )
        if "sigma_epi" in params:
            penalty += 0.5 * (params["sigma_epi"] / self.MLL_EPI_SHRINKAGE_SCALE) ** 2
        return float(nll + penalty)

    def optimize_gp_hyperparameters(self, prepared, y, sem, model, n_starts=None):
        if n_starts is None:
            n_starts = self.MLL_N_STARTS
        "Fit continuous GP parameters using bounded multi-start MAP/MLL."
        parameter_names = ["lengthscale_multiplier", "sigma_main"]
        if model == "epistatic":
            parameter_names.append("sigma_epi")
        parameter_names.append("sigma_noise")
        if model not in {"global_rbf", "additive", "epistatic"}:
            raise ValueError(f"Unknown model family: {model}")
        if model == "global_rbf" and prepared["kind"] != "global":
            raise ValueError("global_rbf requires a global representation")
        log_bounds = [tuple(np.log(self.MLL_BOUNDS[name])) for name in parameter_names]
        initial = {
            "lengthscale_multiplier": 1.0,
            "sigma_main": 1.0,
            "sigma_epi": 0.5,
            "sigma_noise": self.MLL_NOISE_PRIOR_CENTER,
        }
        starts = [np.log([initial[name] for name in parameter_names])]
        rng = np.random.default_rng(self.MLL_RANDOM_SEED + len(y) + len(parameter_names))
        for _ in range(max(int(n_starts) - 1, 0)):
            starts.append(np.asarray([rng.uniform(low, high) for low, high in log_bounds]))
        results = [
            minimize(
                self.regularized_gp_objective,
                start,
                args=(
                    parameter_names,
                    prepared,
                    np.asarray(y, dtype=float),
                    np.asarray(sem, dtype=float),
                ),
                method="L-BFGS-B",
                bounds=log_bounds,
                options={"maxiter": self.MLL_MAXITER, "ftol": 1e-10},
            )
            for start in starts
        ]
        finite_results = [
            result for result in results if np.isfinite(result.fun) and result.fun < 1e29
        ]
        if not finite_results:
            raise RuntimeError("All marginal-likelihood optimization starts failed.")
        best = min(finite_results, key=lambda result: result.fun)
        fitted = dict(zip(parameter_names, np.exp(best.x)))
        fitted.setdefault("sigma_epi", 0.0)
        boundary_hits = {}
        for name, value in fitted.items():
            if name not in self.MLL_BOUNDS or (name == "sigma_epi" and model != "epistatic"):
                continue
            low, high = self.MLL_BOUNDS[name]
            log_span = np.log(high) - np.log(low)
            boundary_hits[name] = bool(
                min(abs(np.log(value) - np.log(low)), abs(np.log(high) - np.log(value)))
                <= 0.02 * log_span
            )
        fitted.update(
            {
                "regularized_neg_log_marginal_likelihood": float(best.fun),
                "optimizer_success": bool(best.success),
                "optimizer_message": str(best.message),
                "optimizer_iterations": int(best.nit),
                "successful_restarts": int(sum((result.success for result in finite_results))),
                "n_restarts": int(len(results)),
                "boundary_hits": boundary_hits,
                "any_boundary_hit": bool(any(boundary_hits.values())),
            }
        )
        return fitted

    def fit_and_predict_one_fold_hardened(
        self, df_train, df_test, source, setting, fitted_hyperparameters=None, n_starts=None
    ):
        # Missing replicate SEM is estimated within this training fold only.
        from nylc.data.lab import training_sem

        df_train = df_train.copy()
        df_train["activity_sem_for_gp"] = training_sem(df_train)
        if n_starts is None:
            n_starts = self.MLL_N_STARTS
        train_pockets = df_train["pocket"].tolist()
        test_pockets = df_test["pocket"].tolist()
        prepared = self.prepare_fold_features(
            train_pockets, test_pockets, source, pca_components=setting.get("pca_components")
        )
        if fitted_hyperparameters is None:
            fitted_hyperparameters = self.optimize_gp_hyperparameters(
                prepared,
                y=df_train["activity_pa6"].to_numpy(dtype=float),
                sem=df_train["activity_sem_for_gp"].to_numpy(dtype=float),
                model=setting["model"],
                n_starts=n_starts,
            )
        K_train, K_test_train, K_test_diag, distance_scale, lengthscale = (
            self.kernels_from_prepared_features(
                prepared,
                lengthscale_multiplier=fitted_hyperparameters["lengthscale_multiplier"],
                sigma_main=fitted_hyperparameters["sigma_main"],
                sigma_epi=fitted_hyperparameters["sigma_epi"],
            )
        )
        gp_fit = fit_gp_from_kernel(
            K_train,
            y=df_train["activity_pa6"].to_numpy(dtype=float),
            sem=df_train["activity_sem_for_gp"].to_numpy(dtype=float),
            sigma_noise=fitted_hyperparameters["sigma_noise"],
        )
        prediction = gp_predict_from_fit(gp_fit, K_test_train, K_test_diag)
        prediction["training_distance_scale"] = distance_scale
        prediction["resolved_lengthscale"] = lengthscale
        prediction["min_train_kernel_eigenvalue"] = float(
            np.linalg.eigvalsh((K_train + K_train.T) / 2).min()
        )
        prediction["fitted_hyperparameters"] = dict(fitted_hyperparameters)
        return prediction

    def source_pca_options(self, source_name):
        if not is_learned_source_name(source_name):
            return (None,)
        return (None,) + tuple(self.HARDENED_PCA_COMPONENTS)

    def make_hardened_joint_grid(self):
        """Enumerate only discrete choices; continuous values are optimized per fold."""
        rows = []
        for source_name in self.AVAILABLE_FEATURE_SOURCES:
            source = self.get_descriptor_source(source_name)
            is_global = getattr(source, "representation", None) == "global_delta"
            model_specs = ("global_rbf",) if is_global else ("additive", "epistatic")
            for pca_components in self.source_pca_options(source_name):
                for model in model_specs:
                    rows.append(
                        {
                            "descriptor_set": source_name,
                            "model": model,
                            "pca_components": pca_components,
                        }
                    )
        return rows

    def evaluate_hardened_setting_loocv(self, df_subset, setting):
        df_subset = df_subset.reset_index(drop=True)
        source = self.get_descriptor_source(setting["descriptor_set"])
        observed, predicted, predicted_std, optimization_rows = ([], [], [], [])
        for test_idx in range(len(df_subset)):
            df_test = df_subset.iloc[[test_idx]]
            df_train = df_subset.drop(index=test_idx)
            pred = self.fit_and_predict_one_fold_hardened(
                df_train, df_test, source, setting, n_starts=self.MLL_INNER_N_STARTS
            )
            observed.append(float(df_test["activity_pa6"].iloc[0]))
            predicted.append(float(pred["mean"][0]))
            predicted_std.append(float(pred["std_observed"][0]))
            hp = pred["fitted_hyperparameters"]
            optimization_rows.append(
                {
                    "lengthscale_multiplier": hp["lengthscale_multiplier"],
                    "sigma_main": hp["sigma_main"],
                    "sigma_epi": hp["sigma_epi"],
                    "sigma_noise": hp["sigma_noise"],
                    "regularized_neg_log_marginal_likelihood": hp[
                        "regularized_neg_log_marginal_likelihood"
                    ],
                    "optimizer_success": hp["optimizer_success"],
                    "any_boundary_hit": hp["any_boundary_hit"],
                }
            )
        metrics = regression_metrics(observed, predicted, predicted_std)
        optimization_df = pd.DataFrame(optimization_rows)
        for name in ["lengthscale_multiplier", "sigma_main", "sigma_epi", "sigma_noise"]:
            metrics[f"median_fitted_{name}"] = float(optimization_df[name].median())
        metrics["optimizer_success_fraction"] = float(optimization_df["optimizer_success"].mean())
        metrics["boundary_hit_fraction"] = float(optimization_df["any_boundary_hit"].mean())
        return metrics

    def select_hardened_setting_inner_loocv(self, df_train, joint_grid):
        rows = []
        for setting in joint_grid:
            metrics = self.evaluate_hardened_setting_loocv(df_train, setting)
            rows.append({**setting, **metrics})
        grid_df = (
            pd.DataFrame(rows).sort_values(["mae", "nlpd_observed", "rmse"]).reset_index(drop=True)
        )
        setting_keys = ["descriptor_set", "model", "pca_components"]
        best_setting = {key: grid_df.iloc[0][key] for key in setting_keys}
        if pd.isna(best_setting["pca_components"]):
            best_setting["pca_components"] = None
        return (best_setting, grid_df)

    def nested_loocv_hardened_feature_selection(self, df_eval, joint_grid):
        df_eval = df_eval.reset_index(drop=True)
        rows, inner_grids = ([], {})
        for outer_idx in range(len(df_eval)):
            df_test = df_eval.iloc[[outer_idx]].copy()
            df_train = df_eval.drop(index=outer_idx).copy()
            variant_id = df_test["variant_id"].iloc[0]
            best_setting, inner_grid = self.select_hardened_setting_inner_loocv(
                df_train, joint_grid
            )
            inner_grids[variant_id] = inner_grid
            source = self.get_descriptor_source(best_setting["descriptor_set"])
            pred = self.fit_and_predict_one_fold_hardened(df_train, df_test, source, best_setting)
            hp = pred["fitted_hyperparameters"]
            observed = float(df_test["activity_pa6"].iloc[0])
            predicted = float(pred["mean"][0])
            rows.append(
                {
                    "variant_id": variant_id,
                    "mutations": df_test["mutations"].iloc[0],
                    "mutation_signature": df_test["mutation_signature"].iloc[0],
                    "mutation_order": int(df_test["mutation_order"].iloc[0]),
                    "observed": observed,
                    "predicted": predicted,
                    "predicted_std_latent": float(pred["std_latent"][0]),
                    "predicted_std_observed": float(pred["std_observed"][0]),
                    "abs_error": abs(observed - predicted),
                    "training_distance_scale": pred["training_distance_scale"],
                    "resolved_lengthscale": pred["resolved_lengthscale"],
                    "min_train_kernel_eigenvalue": pred["min_train_kernel_eigenvalue"],
                    **{f"selected_{key}": value for key, value in best_setting.items()},
                    "fitted_lengthscale_multiplier": hp["lengthscale_multiplier"],
                    "fitted_sigma_main": hp["sigma_main"],
                    "fitted_sigma_epi": hp["sigma_epi"],
                    "fitted_sigma_noise": hp["sigma_noise"],
                    "regularized_neg_log_marginal_likelihood": hp[
                        "regularized_neg_log_marginal_likelihood"
                    ],
                    "optimizer_success": hp["optimizer_success"],
                    "optimizer_iterations": hp["optimizer_iterations"],
                    "optimizer_successful_restarts": hp["successful_restarts"],
                    "optimizer_any_boundary_hit": hp["any_boundary_hit"],
                    "optimizer_boundary_hits": str(hp["boundary_hits"]),
                    "inner_mae": float(inner_grid.iloc[0]["mae"]),
                    "inner_nlpd_observed": float(inner_grid.iloc[0]["nlpd_observed"]),
                }
            )
            pass
        predictions = pd.DataFrame(rows)
        metrics = pd.DataFrame(
            [
                {
                    "analysis": "nested_feature_model_selection_with_regularized_mll",
                    **regression_metrics(
                        predictions["observed"],
                        predictions["predicted"],
                        predictions["predicted_std_observed"],
                    ),
                }
            ]
        )
        selection_columns = [column for column in predictions if column.startswith("selected_")]
        selection_summary = (
            predictions[selection_columns].value_counts(dropna=False).reset_index(name="count")
        )
        return (predictions, metrics, selection_summary, inner_grids)

    def run_mll_smoke_test(self, frame, source_name):
        """Fail early before the expensive nested run if the optimizer is invalid."""
        smoke_df = frame.iloc[: min(12, len(frame))].copy()
        source = self.get_descriptor_source(source_name)
        prepared = self.prepare_fold_features(
            smoke_df["pocket"].tolist(), smoke_df.iloc[:1]["pocket"].tolist(), source
        )
        hp = self.optimize_gp_hyperparameters(
            prepared,
            smoke_df["activity_pa6"].to_numpy(dtype=float),
            smoke_df["activity_sem_for_gp"].to_numpy(dtype=float),
            model="epistatic",
            n_starts=2,
        )
        for name in ["lengthscale_multiplier", "sigma_main", "sigma_epi", "sigma_noise"]:
            low, high = self.MLL_BOUNDS[name]
            assert low <= hp[name] <= high
        K, _, _ = self.training_kernel_from_prepared(
            prepared, hp["lengthscale_multiplier"], hp["sigma_main"], hp["sigma_epi"]
        )
        assert np.linalg.eigvalsh((K + K.T) / 2).min() > -1e-08
        assert np.isfinite(hp["regularized_neg_log_marginal_likelihood"])
        return hp

    def strict_position_family_holdout_hardened(self, df_eval, joint_grid):
        prediction_rows, selection_rows = ([], [])
        for pos in POSITIONS:
            is_test = df_eval["mutation_positions"].apply(lambda positions: pos in positions)
            df_test = df_eval[is_test].copy()
            df_train = df_eval[~is_test].copy()
            if df_test.empty or len(df_train) < 5:
                continue
            best_setting, inner_grid = self.select_hardened_setting_inner_loocv(
                df_train, joint_grid
            )
            source = self.get_descriptor_source(best_setting["descriptor_set"])
            pred = self.fit_and_predict_one_fold_hardened(df_train, df_test, source, best_setting)
            hp = pred["fitted_hyperparameters"]
            tmp = df_test[
                ["variant_id", "mutations", "mutation_signature", "mutation_order", "activity_pa6"]
            ].copy()
            tmp["heldout_position"] = pos
            tmp["observed"] = tmp.pop("activity_pa6")
            tmp["predicted"] = pred["mean"]
            tmp["predicted_std_observed"] = pred["std_observed"]
            tmp["abs_error"] = np.abs(tmp["observed"] - tmp["predicted"])
            tmp["training_distance_scale"] = pred["training_distance_scale"]
            tmp["resolved_lengthscale"] = pred["resolved_lengthscale"]
            for key, value in best_setting.items():
                tmp[f"selected_{key}"] = value
            for key in ["lengthscale_multiplier", "sigma_main", "sigma_epi", "sigma_noise"]:
                tmp[f"fitted_{key}"] = hp[key]
            prediction_rows.append(tmp)
            selection_rows.append(
                {
                    "heldout_position": pos,
                    "n_train": len(df_train),
                    "n_test": len(df_test),
                    "inner_mae": float(inner_grid.iloc[0]["mae"]),
                    **{f"selected_{key}": value for key, value in best_setting.items()},
                    **{
                        f"fitted_{key}": hp[key]
                        for key in [
                            "lengthscale_multiplier",
                            "sigma_main",
                            "sigma_epi",
                            "sigma_noise",
                        ]
                    },
                }
            )
            pass
        if not prediction_rows:
            raise ValueError("No eligible position-family holdout splits")
        predictions = pd.concat(prediction_rows, ignore_index=True)
        summary = (
            predictions.groupby("heldout_position")
            .agg(
                n=("abs_error", "size"),
                mae=("abs_error", "mean"),
                rmse=("abs_error", lambda values: float(np.sqrt(np.mean(np.asarray(values) ** 2)))),
                median_abs_error=("abs_error", "median"),
                mean_predicted_std=("predicted_std_observed", "mean"),
            )
            .reset_index()
        )
        return (predictions, summary, pd.DataFrame(selection_rows))

    def fit_sigma_scale(self, y_true, y_pred, pred_std):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        pred_std = np.asarray(pred_std, dtype=float)
        valid = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(pred_std) & (pred_std > 0)
        if not np.any(valid):
            raise ValueError(
                "Sigma scaling requires finite predictions and positive standard deviations."
            )
        standardized_residual = (y_true[valid] - y_pred[valid]) / pred_std[valid]
        scale = float(np.sqrt(np.mean(standardized_residual**2)))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Estimated sigma scale must be positive and finite.")
        return scale

    def uncertainty_coverage_table(self, predictions):
        error = np.abs(predictions["observed"] - predictions["predicted"])
        rows = []
        for label, z_value in CALIBRATION_LEVELS.items():
            rows.append(
                {
                    "interval": label,
                    "nominal_coverage": float(label.rstrip("%")) / 100.0,
                    "nominal_z": z_value,
                    "uncalibrated_coverage": float(
                        (error <= z_value * predictions["predicted_std_observed"]).mean()
                    ),
                    "calibrated_coverage": float(
                        (error <= z_value * predictions["predicted_std_observed_calibrated"]).mean()
                    ),
                }
            )
        return pd.DataFrame(rows)

    def loocv_predictions_for_fixed_setting(self, df_eval, setting, n_starts=None):
        if n_starts is None:
            n_starts = self.MLL_INNER_N_STARTS
        df_eval = df_eval.reset_index(drop=True)
        source = self.get_descriptor_source(setting["descriptor_set"])
        rows = []
        for test_idx in range(len(df_eval)):
            df_test = df_eval.iloc[[test_idx]]
            df_train = df_eval.drop(index=test_idx)
            pred = self.fit_and_predict_one_fold_hardened(
                df_train, df_test, source, setting, n_starts=n_starts
            )
            rows.append(
                {
                    "variant_id": df_test["variant_id"].iloc[0],
                    "observed": float(df_test["activity_pa6"].iloc[0]),
                    "predicted": float(pred["mean"][0]),
                    "predicted_std_observed": float(pred["std_observed"][0]),
                }
            )
        return pd.DataFrame(rows)

    def nested_loocv_hardened_feature_selection_calibrated(self, df_eval, joint_grid):
        df_eval = df_eval.reset_index(drop=True)
        rows, inner_grids = ([], {})
        for outer_idx in range(len(df_eval)):
            df_test = df_eval.iloc[[outer_idx]].copy()
            df_train = df_eval.drop(index=outer_idx).copy()
            variant_id = df_test["variant_id"].iloc[0]
            best_setting, inner_grid = self.select_hardened_setting_inner_loocv(
                df_train, joint_grid
            )
            inner_grids[variant_id] = inner_grid
            inner_predictions = self.loocv_predictions_for_fixed_setting(
                df_train, best_setting, n_starts=self.MLL_INNER_N_STARTS
            )
            sigma_scale = self.fit_sigma_scale(
                inner_predictions["observed"],
                inner_predictions["predicted"],
                inner_predictions["predicted_std_observed"],
            )
            source = self.get_descriptor_source(best_setting["descriptor_set"])
            pred = self.fit_and_predict_one_fold_hardened(df_train, df_test, source, best_setting)
            hp = pred["fitted_hyperparameters"]
            observed = float(df_test["activity_pa6"].iloc[0])
            predicted = float(pred["mean"][0])
            std_observed = float(pred["std_observed"][0])
            rows.append(
                {
                    "variant_id": variant_id,
                    "mutations": df_test["mutations"].iloc[0],
                    "mutation_signature": df_test["mutation_signature"].iloc[0],
                    "mutation_order": int(df_test["mutation_order"].iloc[0]),
                    "observed": observed,
                    "predicted": predicted,
                    "predicted_std_latent": float(pred["std_latent"][0]),
                    "predicted_std_observed": std_observed,
                    "sigma_scale_inner": sigma_scale,
                    "predicted_std_observed_calibrated": sigma_scale * std_observed,
                    "abs_error": abs(observed - predicted),
                    "training_distance_scale": pred["training_distance_scale"],
                    "resolved_lengthscale": pred["resolved_lengthscale"],
                    "min_train_kernel_eigenvalue": pred["min_train_kernel_eigenvalue"],
                    **{f"selected_{key}": value for key, value in best_setting.items()},
                    "fitted_lengthscale_multiplier": hp["lengthscale_multiplier"],
                    "fitted_sigma_main": hp["sigma_main"],
                    "fitted_sigma_epi": hp["sigma_epi"],
                    "fitted_sigma_noise": hp["sigma_noise"],
                    "regularized_neg_log_marginal_likelihood": hp[
                        "regularized_neg_log_marginal_likelihood"
                    ],
                    "optimizer_success": hp["optimizer_success"],
                    "optimizer_any_boundary_hit": hp["any_boundary_hit"],
                }
            )
            pass
        predictions = pd.DataFrame(rows)
        raw_metrics = regression_metrics(
            predictions["observed"], predictions["predicted"], predictions["predicted_std_observed"]
        )
        calibrated_metrics = regression_metrics(
            predictions["observed"],
            predictions["predicted"],
            predictions["predicted_std_observed_calibrated"],
        )
        metrics = pd.DataFrame(
            [
                {"uncertainty": "uncalibrated", **raw_metrics},
                {"uncertainty": "sigma_scaled", **calibrated_metrics},
            ]
        )
        selection_columns = [column for column in predictions if column.startswith("selected_")]
        selection_summary = (
            predictions[selection_columns].value_counts(dropna=False).reset_index(name="count")
        )
        return (predictions, metrics, selection_summary, inner_grids)
