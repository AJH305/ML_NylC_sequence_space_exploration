"""Feature artifacts decouple pretrained model inference from CPU statistics."""

from pathlib import Path
import numpy as np
from nylc.features.descriptors import (
    POSITIONS,
    WT_POCKET,
    DESCRIPTOR_SOURCES,
    residue_descriptor_table,
)
from nylc.provenance import sha256_file


class StoredSource:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as arrays:
            self.name = str(arrays["name"])
            self.representation = str(arrays["representation"])
            self.keys = [tuple(x) for x in arrays["keys"].tolist()]
            self.values = {k: arrays[k].copy() for k in arrays.files if k.startswith("component_")}
        self.index = {k: i for i, k in enumerate(self.keys)}

    def _indices(self, pockets):
        return [self.index[tuple(p[pos] for pos in POSITIONS)] for p in pockets]

    def position_arrays(self, pockets, positions=POSITIONS):
        idx = self._indices(pockets)
        return {p: self.values[f"component_{p}"][idx] for p in positions}

    def feature_matrix(self, pockets):
        return self.values["component_global"][self._indices(pockets)]


def create_source(name, config, root, wt_sequence):
    if name in DESCRIPTOR_SOURCES:
        return residue_descriptor_table(name)
    settings = config["features"]["learned"][name]
    runtime = config["runtime"]
    from nylc.runtime import setup_torch

    device = setup_torch(runtime)
    common = dict(
        wt_sequence=wt_sequence,
        wt_pocket=WT_POCKET,
        name=name,
        device=device,
        batch_size=runtime["batch_size"],
        cache_dir=root / "cache" / name,
    )
    if settings["kind"] == "esm2":
        from nylc.features.protein_llm_embeddings import ProteinLLMEmbeddingSource

        return ProteinLLMEmbeddingSource(
            **common,
            model_name=settings["model"],
            revision=settings["revision"],
            model_cache_dir=root / "cache/huggingface",
            local_files_only=runtime["offline"],
            representation=settings["representation"],
            window_radius=settings.get("window_radius", 16),
        )
    if settings["kind"] == "metl":
        from nylc.features.metl_embeddings import METLEmbeddingSource

        path = (root / settings["checkpoint"]).resolve()
        if sha256_file(path) != settings["sha256"]:
            raise ValueError("METL checkpoint checksum mismatch")
        return METLEmbeddingSource(
            **common,
            model_id=settings["model"],
            checkpoint_path=path,
            checkpoint_sha256=settings["sha256"],
        )
    raise ValueError(f"Unsupported feature backend: {settings['kind']}")


def write_feature_store(name, source, pockets, output):
    keys = list(dict.fromkeys(tuple(p[pos] for pos in POSITIONS) for p in pockets))
    pockets = [dict(zip(POSITIONS, k)) for k in keys]
    representation = getattr(source, "representation", "position")
    if name in DESCRIPTOR_SOURCES:
        arrays = {
            p: np.vstack([source.scaled_table.loc[pocket[p]].to_numpy() for pocket in pockets])
            for p in POSITIONS
        }
    elif representation == "global_delta":
        arrays = {"global": source.feature_matrix(pockets)}
    else:
        arrays = source.position_arrays(pockets, positions=POSITIONS)
    for values in arrays.values():
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite features: {name}")
    np.savez_compressed(
        output / f"{name}.npz",
        name=np.asarray(name),
        representation=np.asarray(representation),
        keys=np.asarray(keys),
        **{f"component_{p}": x for p, x in arrays.items()},
    )
    return {
        "name": name,
        "representation": representation,
        "resolved_device": getattr(source, "device", "cpu"),
        "n_pockets": len(keys),
        "dimensions": {str(p): v.shape[1] for p, v in arrays.items()},
    }


def store_factory(directory):
    directory = Path(directory)
    return lambda name: StoredSource(directory / f"{name}.npz")
