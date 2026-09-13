import json
import numpy as np
import pandas as pd
import pytest
from nylc.data.lab import parse_mutations, read_reference, training_sem
from nylc.data.candidates import (
    add_boltzgen_candidates_from_full_sequences,
    check_designed_sequence_consistency,
)
from nylc.features.descriptors import residue_descriptor_table, POSITIONS
from nylc.models.kernels import (
    rbf_kernel_from_descriptors,
    build_total_epistatic_kernel,
    fit_gp_from_kernel,
    gp_predict_from_fit,
)
from nylc.selection.dpp import sample_k_dpp, normalize_to_correlation
from nylc.selection.panel import select_panel
from conftest import ROOT


def test_mutation_mapping_and_strict_validation():
    wt = read_reference(ROOT / "inputs/WT.fasta")
    pocket, mutations = parse_mutations("D99G;F134W", wt)
    assert pocket[99] == "G" and pocket[134] == "W" and len(mutations) == 2
    for text in ("D99G garbage", "D99G;D99V", "A99G", "D9999G", "D99D"):
        with pytest.raises(ValueError):
            parse_mutations(text, wt)


def test_lab_replicates_and_exclusions(lab):
    row = lab.set_index("variant_id").loc["WT"]
    np.testing.assert_allclose(row.activity_pa6, np.mean([79, 75, 79]))
    np.testing.assert_allclose(row.activity_sem, np.std([79, 75, 79], ddof=1) / np.sqrt(3))
    assert len(lab) == 36
    assert lab.loc[~lab.only_target_positions, "variant_id"].tolist() == ["F301L"]
    assert row.mutation_order == 0


def test_sem_imputation_uses_only_training_fold():
    train = pd.DataFrame({"activity_sem": [1.0, 3.0, np.nan]})
    np.testing.assert_array_equal(training_sem(train), [1, 3, 2])


def test_epistatic_kernel_and_prediction_match_linear_algebra():
    arrays = {
        p: np.arange(12, dtype=float).reshape(6, 2) / (i + 1) for i, p in enumerate(POSITIONS)
    }
    kernels = {p: rbf_kernel_from_descriptors(x, lengthscale=2) for p, x in arrays.items()}
    kernel, _, _ = build_total_epistatic_kernel(kernels, sigma_main=1.2, sigma_epi=0.4)
    np.testing.assert_allclose(np.diag(kernel), 1.2**2 + 0.4**2)
    assert np.linalg.eigvalsh(kernel).min() > -1e-10
    y = np.array([1.0, 2.0, 4.0, 3.0, 5.0, 6.0])
    sem = np.full(6, 0.1)
    fit = fit_gp_from_kernel(kernel, y, sem=sem, sigma_noise=0.3)
    prediction = gp_predict_from_fit(fit, kernel, np.diag(kernel))
    # The reference method uses the sample standard deviation for the target.
    ystd = y.std(ddof=1)
    noisy = kernel + np.diag((sem / ystd) ** 2 + 0.3**2 + 1e-8)
    expected = y.mean() + ystd * kernel @ np.linalg.solve(noisy, (y - y.mean()) / ystd)
    np.testing.assert_allclose(prediction["mean"], expected, atol=1e-8)


def test_heldout_features_do_not_change_training_pca(model):
    train = np.arange(24, dtype=float).reshape(6, 4)
    a, _, reducer_a = model.fit_pca_train_only(train, np.ones((1, 4)), 2)
    b, _, reducer_b = model.fit_pca_train_only(train, np.ones((1, 4)) * 1e9, 2)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(reducer_a.mean_, reducer_b.mean_)


def test_sigma_scaling_known_residuals(model):
    scale = model.fit_sigma_scale([2.0, 4.0], [0.0, 0.0], [1.0, 2.0])
    assert scale == 2.0


def test_dpp_size_seed_and_invalid_rank():
    kernel = np.eye(12)
    a = sample_k_dpp(kernel, 5, 42)
    assert a == sample_k_dpp(kernel, 5, 42)
    assert len(a) == len(set(a)) == 5
    with pytest.raises(np.linalg.LinAlgError):
        sample_k_dpp(np.ones((5, 5)), 3, 42, jitter=0)


def test_panel_kernel_alignment_after_activity_sort(config, tmp_path):
    ids = ["a", "b", "c", "d", "e", "f", "g", "h"]
    frame = pd.DataFrame(
        {
            "candidate_id": ids,
            "predicted_activity_gp_mean": [3.0, 8.0, 2.0, 7.0, 1.0, 6.0, 5.0, 4.0],
            "already_tested": False,
            "mutations": ["D99G"] * 8,
        }
    )
    rng = np.random.default_rng(8)
    features = rng.normal(size=(8, 5))
    kernel = features @ features.T + np.eye(8)
    lab = pd.DataFrame({"variant_id": ["WT", "mut"], "activity_pa6": [1.0, 2.0]})
    select_panel(frame, kernel, lab, config["selection"], tmp_path)
    with np.load(tmp_path / "primary_kernel.npz", allow_pickle=False) as saved:
        indices = [ids.index(name) for name in saved["candidate_ids"]]
        expected = normalize_to_correlation(kernel[np.ix_(indices, indices)])
        np.testing.assert_allclose(saved["kernel"], expected, atol=1e-12)
        assert len(saved["selected_ids"]) == 3


@pytest.mark.reference
def test_fixed_reference_parameters_reproduce_candidate_predictions(lab, model):
    raw = pd.read_csv(ROOT / "inputs/boltzgen_candidates.csv").copy()
    raw["boltzgen_run"] = "160"
    _, candidates = add_boltzgen_candidates_from_full_sequences(raw)
    candidates = check_designed_sequence_consistency(candidates)
    ref = ROOT / "references/dpp_calibrated"
    selection = pd.read_csv(ref / "final_hardened_model_selection.csv").iloc[0]
    hyper = pd.read_csv(ref / "final_hardened_hyperparameters.csv").iloc[0]
    hp = {
        key: float(hyper[key])
        for key in ("lengthscale_multiplier", "sigma_main", "sigma_epi", "sigma_noise")
    }
    setting = {
        "descriptor_set": selection.descriptor_set,
        "model": selection.model,
        "pca_components": None,
    }
    source = residue_descriptor_table(setting["descriptor_set"])
    core = lab.loc[lab.only_target_positions].reset_index(drop=True)
    prediction = model.fit_and_predict_one_fold_hardened(
        core, candidates, source, setting, fitted_hyperparameters=hp
    )
    actual = pd.DataFrame(
        {
            "candidate_id": candidates.candidate_id,
            "mean": prediction["mean"],
            "std": prediction["std_observed"],
        }
    )
    expected = pd.read_csv(ref / "all_candidates_activity_ranked.csv")
    joined = actual.merge(expected, on="candidate_id", validate="one_to_one")
    assert len(joined) == len(actual) == len(expected)
    np.testing.assert_allclose(
        joined["mean"], joined.predicted_activity_gp_mean, rtol=1e-7, atol=1e-6
    )
    np.testing.assert_allclose(
        joined["std"], joined.predicted_activity_gp_std_observed, rtol=1e-7, atol=1e-6
    )


def test_reference_input_hashes():
    from nylc.provenance import sha256_file

    provenance = json.loads((ROOT / "references/dpp_calibrated/run_provenance.json").read_text())
    assert sha256_file(ROOT / "inputs/lab_data.csv") == provenance["inputs"]["lab_data"]["sha256"]
    assert (
        sha256_file(ROOT / "inputs/boltzgen_candidates.csv")
        == provenance["inputs"]["boltzgen_candidates"]["sha256"]
    )
