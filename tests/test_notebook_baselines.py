import json
from copy import deepcopy
from types import SimpleNamespace
import sys
import numpy as np
import pandas as pd
import pytest
from nylc.validation.notebook_baselines import (
    run_classical,
    summarize_predictions,
    tabicl_predictions,
)
from nylc.features.esm2_zero_shot import ESM2ZeroShotScorer
from nylc.features.descriptors import WT_POCKET
from nylc.models.zero_shot_gp import fit_predict_one_fold
from nylc.data.lab import read_reference
from nylc.provenance import sha256_file
from conftest import ROOT


def test_classical_baselines_match_original_notebook(lab, config):
    ref = pd.read_csv(ROOT / "references/baselines/classical_nine_variant_predictions.csv")
    frame = lab.set_index("variant_id").loc[ref.variant_id.drop_duplicates()].reset_index()
    frames, _, _ = run_classical(frame, config["baseline_comparison"])
    actual = pd.concat(frames)
    compared = ref.merge(
        actual, on=["model", "variant_id"], suffixes=("_ref", "_new"), validate="one_to_one"
    )
    assert len(compared) == len(ref) == 45
    np.testing.assert_allclose(compared.predicted_ref, compared.predicted_new, rtol=1e-9, atol=1e-8)
    np.testing.assert_allclose(
        compared.inner_mae_ref, compared.inner_mae_new, rtol=1e-9, atol=1e-8, equal_nan=True
    )


def test_zero_shot_gp_matches_original_support_script(monkeypatch):
    ref = json.loads((ROOT / "references/baselines/zero_shot_gp_reference.json").read_text())
    x = np.random.default_rng(ref["seed"]).normal(size=(7, 8))
    z = np.array([0.0, -2.0, 1.0, 3.0, -1.0, 2.0, 4.0])
    result = fit_predict_one_fold(
        x[:6],
        np.array([1.0, 3.0, 2.0, 8.0, 4.0, 5.0]),
        np.ones(6) * 0.2,
        z[:6],
        x[6:],
        z[6:],
        n_starts=1,
    )
    # Optimisation can vary slightly with platform-specific linear algebra.
    for name in ("mean", "std_latent", "std_observed"):
        np.testing.assert_allclose(result[name], ref[name], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        result["fitted_hyperparameters"]["regularized_neg_log_marginal_likelihood"],
        ref["hyperparameters"]["regularized_neg_log_marginal_likelihood"],
        rtol=1e-8,
        atol=1e-8,
    )
    # Separately require tight agreement for the migrated prediction formula.
    monkeypatch.setattr(
        "nylc.models.zero_shot_gp.optimize_hyperparameters",
        lambda *args, **kwargs: ref["hyperparameters"],
    )
    fixed = fit_predict_one_fold(
        x[:6],
        np.array([1.0, 3.0, 2.0, 8.0, 4.0, 5.0]),
        np.ones(6) * 0.2,
        z[:6],
        x[6:],
        z[6:],
        n_starts=1,
    )
    for name in ("mean", "std_latent", "std_observed"):
        np.testing.assert_allclose(fixed[name], ref[name], rtol=1e-10, atol=1e-10)


def test_zero_shot_cache_and_raw_score_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr("importlib.metadata.version", lambda name: "test-version")
    scorer = ESM2ZeroShotScorer(
        read_reference(ROOT / "inputs/WT.fasta"),
        WT_POCKET,
        revision="a" * 40,
        device="cpu",
        cache_dir=tmp_path,
    )
    calls = []

    def fake_batch(sequences, positions):
        calls.append(positions)
        return [-float(len(p)) for p in positions]

    monkeypatch.setattr(scorer, "_score_batch", fake_batch)
    pockets = [WT_POCKET, dict(WT_POCKET, **{}) | {99: "G"}, dict(WT_POCKET) | {99: "G", 134: "W"}]
    np.testing.assert_equal(scorer.score_pockets(pockets), [0, -1, -2])
    scorer.score_pockets(pockets)
    assert len(calls) == 1 and calls[0] == [(99,), (99, 134)]
    original = scorer._cache_path("ACD")
    scorer.revision = "b" * 40
    assert scorer._cache_path("ACD") != original
    scores = pd.DataFrame(
        {
            "model": ["esm2_zero_shot_direct"] * 3,
            "observed": [2.0, 3.0, 4.0],
            "predicted": [0.0, 1.0, 2.0],
        }
    )
    metrics = summarize_predictions(scores).iloc[0]
    assert np.isnan(metrics.mae) and np.isnan(metrics.rmse) and np.isnan(metrics.r2)
    assert metrics.spearman == pytest.approx(1)


def test_tabicl_adapter_requires_local_verified_weights(tmp_path, monkeypatch, config):
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_bytes(b"test-only")
    settings = deepcopy(config["baseline_comparison"]["tabicl"])
    settings.update(checkpoint=str(checkpoint), sha256=sha256_file(checkpoint))
    captured = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setitem(sys.modules, "tabicl", SimpleNamespace(TabICLRegressor=factory))
    monkeypatch.setattr("nylc.runtime.setup_torch", lambda runtime: "cpu")
    monkeypatch.setattr("nylc.validation.notebook_baselines.fixed_outer_loocv", lambda *args: "ok")
    assert tabicl_predictions(None, None, settings, config["runtime"], tmp_path) == "ok"
    assert captured["allow_auto_download"] is False
    assert captured["device"] == "cpu" and captured["n_estimators"] == 8
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        tabicl_predictions(None, None, settings, config["runtime"], tmp_path)


def test_optional_zero_shot_comparison_exports(config, lab, tmp_path, monkeypatch):
    from nylc.validation.notebook_baselines import run_comparison

    configured = deepcopy(config)
    configured["baseline_comparison"]["zero_shot"]["enabled"] = True
    configured["baseline_comparison"]["forest_estimators"] = 10
    core = lab.set_index("variant_id").loc[config["data"]["subset_variant_ids"]].reset_index()
    monkeypatch.setattr("nylc.runtime.setup_torch", lambda runtime: "cpu")
    monkeypatch.setattr(
        ESM2ZeroShotScorer,
        "score_pockets",
        lambda self, pockets: np.arange(len(pockets), dtype=float),
    )
    current = pd.DataFrame(
        {
            "variant_id": core.variant_id,
            "observed": core.activity_pa6,
            "predicted": np.arange(len(core), dtype=float),
        }
    )
    run_comparison(configured, ROOT, tmp_path, core, current)
    predictions = pd.read_csv(tmp_path / "all_loocv_predictions.csv")
    metrics = pd.read_csv(tmp_path / "all_loocv_metrics.csv")
    assert len(metrics) == 11 and len(predictions) == 11 * len(core)
    assert len(pd.read_csv(tmp_path / "gp_zero_shot_prior_predictions.csv")) == len(core)
    assert (
        metrics.set_index("model").loc["esm2_zero_shot_direct", ["mae", "rmse", "r2"]].isna().all()
    )
