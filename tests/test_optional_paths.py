import numpy as np
import pytest
from nylc.features.descriptors import residue_descriptor_table, WT_POCKET, POSITIONS
from nylc.features.protein_llm_embeddings import ProteinLLMEmbeddingSource
from nylc.features.store import write_feature_store, StoredSource
from nylc.data.lab import read_reference
from nylc.validation import comparison
from conftest import ROOT


class FakeESM(ProteinLLMEmbeddingSource):
    def _embed_uncached(self, sequences):
        return [
            np.asarray([[ord(aa), i + 1, 1.0] for i, aa in enumerate(seq)], dtype=np.float32)
            for seq in sequences
        ]


def test_pinned_embedding_cache_and_feature_artifact(tmp_path):
    wt = read_reference(ROOT / "inputs/WT.fasta")
    a = FakeESM(wt, WT_POCKET, revision="a" * 40, cache_dir=tmp_path / "cache", device="cpu")
    b = FakeESM(wt, WT_POCKET, revision="b" * 40, cache_dir=tmp_path / "cache", device="cpu")
    assert a._cache_path(wt) != b._cache_path(wt)
    pockets = [dict(WT_POCKET), {**WT_POCKET, 99: "G"}]
    write_feature_store("esm2_target", a, pockets, tmp_path)
    restored = StoredSource(tmp_path / "esm2_target.npz")
    expected = a.position_arrays(pockets, positions=POSITIONS)
    for pos, values in restored.position_arrays(pockets).items():
        np.testing.assert_array_equal(values, expected[pos])
    with pytest.raises(ValueError, match="immutable"):
        FakeESM(wt, WT_POCKET, revision="main")


def test_original_nested_model_comparison_small_grid(lab, monkeypatch):
    frame = lab.loc[lab.only_target_positions].iloc[:6].copy()

    def grid(family):
        return [
            {
                "lengthscale": 0.5,
                "sigma_main": 1.0,
                "sigma_epi": 0.5 if family == "epistatic" else 0.0,
                "sigma_noise": 0.3,
            }
        ]

    monkeypatch.setattr(comparison, "make_param_grid", grid)
    predictions, metrics, grids = comparison.nested_loocv_model_comparison(
        frame, residue_descriptor_table("physical")
    )
    assert len(predictions) == 12 and len(metrics) == 2 and len(grids) == 12
    assert np.isfinite(predictions.predicted).all()


def test_family_holdout_refits_on_training_subset(lab, model):
    ids = ["WT", "D99G", "D99V", "D99R", "F134W", "D304M", "R330A", "R330Q"]
    frame = lab.set_index("variant_id").loc[ids].reset_index()
    grid = [{"descriptor_set": "physical", "model": "epistatic", "pca_components": None}]
    predictions, summary, settings = model.strict_position_family_holdout_hardened(frame, grid)
    assert set(summary.heldout_position) == {99, 134, 304, 330}
    assert np.isfinite(predictions.predicted).all()
    assert (settings.n_train + settings.n_test == len(frame)).all()
