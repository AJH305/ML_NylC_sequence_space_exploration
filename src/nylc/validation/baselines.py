"""Prespecified comparison models from notebook 03."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from nylc.features.descriptors import POSITIONS, CANONICAL_AA
from nylc.models.kernels import aa_descriptor_vector, regression_metrics


def onehot_pocket_features(df_eval):
    rows = []
    for pocket in df_eval["pocket"]:
        row = {}
        for pos in POSITIONS:
            for aa in CANONICAL_AA:
                row[f"pos{pos}_{aa}"] = int(pocket[pos] == aa)
        rows.append(row)
    return pd.DataFrame(rows, index=df_eval.index).to_numpy(dtype=float)


def flattened_descriptor_features(df_eval, source):
    rows = []
    for pocket in df_eval["pocket"]:
        rows.append(
            np.concatenate([aa_descriptor_vector(pocket[pos], source) for pos in POSITIONS])
        )
    return np.vstack(rows)


def inner_loocv_regressor_mae(X_train, y_train, fit_predict_factory):
    preds = []
    obs = []
    for i in range(len(y_train)):
        train_idx = [j for j in range(len(y_train)) if j != i]
        pred = fit_predict_factory(X_train[train_idx], y_train[train_idx], X_train[[i]])
        preds.append(float(pred[0]))
        obs.append(float(y_train[i]))
    return mean_absolute_error(obs, preds)


def select_ridge_alpha(X_train, y_train, alphas=(0.01, 0.1, 1.0, 10.0, 100.0)):
    rows = []
    for alpha in alphas:
        mae = inner_loocv_regressor_mae(
            X_train, y_train, lambda Xtr, ytr, Xte: Ridge(alpha=alpha).fit(Xtr, ytr).predict(Xte)
        )
        rows.append({"alpha": alpha, "inner_mae": mae})
    return pd.DataFrame(rows).sort_values("inner_mae").iloc[0].to_dict()


def select_knn_k(X_train, y_train, k_values=(1, 3, 5, 7)):
    rows = []
    for k in k_values:
        if k >= len(y_train):
            continue
        mae = inner_loocv_regressor_mae(
            X_train,
            y_train,
            lambda Xtr, ytr, Xte: (
                KNeighborsRegressor(n_neighbors=k, weights="distance").fit(Xtr, ytr).predict(Xte)
            ),
        )
        rows.append({"k": k, "inner_mae": mae})
    return pd.DataFrame(rows).sort_values("inner_mae").iloc[0].to_dict()


def loocv_baselines(df_eval, descriptor_source_for_knn):
    df_eval = df_eval.reset_index(drop=True)
    y = df_eval["activity_pa6"].to_numpy(dtype=float)
    X_onehot = onehot_pocket_features(df_eval)
    X_desc = flattened_descriptor_features(df_eval, descriptor_source_for_knn)
    rows = []
    for test_idx in range(len(df_eval)):
        train_idx = [i for i in range(len(df_eval)) if i != test_idx]
        variant_id = df_eval.loc[test_idx, "variant_id"]
        observed = float(y[test_idx])
        y_train = y[train_idx]
        rows.append(
            {
                "model": "train_mean",
                "variant_id": variant_id,
                "observed": observed,
                "predicted": float(np.mean(y_train)),
                "abs_error": abs(observed - float(np.mean(y_train))),
            }
        )
        ridge_best = select_ridge_alpha(X_onehot[train_idx], y_train)
        ridge = Ridge(alpha=ridge_best["alpha"]).fit(X_onehot[train_idx], y_train)
        ridge_pred = float(ridge.predict(X_onehot[[test_idx]])[0])
        rows.append(
            {
                "model": "ridge_onehot",
                "variant_id": variant_id,
                "observed": observed,
                "predicted": ridge_pred,
                "abs_error": abs(observed - ridge_pred),
                "selected_alpha": ridge_best["alpha"],
            }
        )
        knn_best = select_knn_k(X_desc[train_idx], y_train)
        knn = KNeighborsRegressor(n_neighbors=int(knn_best["k"]), weights="distance").fit(
            X_desc[train_idx], y_train
        )
        knn_pred = float(knn.predict(X_desc[[test_idx]])[0])
        rows.append(
            {
                "model": "knn_descriptor",
                "variant_id": variant_id,
                "observed": observed,
                "predicted": knn_pred,
                "abs_error": abs(observed - knn_pred),
                "selected_k": int(knn_best["k"]),
            }
        )
        rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=42).fit(
            X_onehot[train_idx], y_train
        )
        rf_pred = float(rf.predict(X_onehot[[test_idx]])[0])
        rows.append(
            {
                "model": "random_forest_onehot_fixed",
                "variant_id": variant_id,
                "observed": observed,
                "predicted": rf_pred,
                "abs_error": abs(observed - rf_pred),
            }
        )
    result = pd.DataFrame(rows)
    metrics = []
    for model, group in result.groupby("model"):
        metrics.append(
            {"model": model, **regression_metrics(group["observed"], group["predicted"])}
        )
    return (result, pd.DataFrame(metrics).sort_values("mae"))
