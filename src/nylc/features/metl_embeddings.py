"""Frozen METL representations for the NylC Epistatic GP notebooks.

The optional ``metl-pretrained`` dependency and model weights are loaded only
when embeddings that are not already cached are requested.  The class exposes
the per-residue transformer output immediately before METL's global average
pooling layer, so it can be used by the same position-wise epistatic kernel as the
ESM-2 target representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import numpy as np

from nylc.features.protein_llm_embeddings import build_variant_sequence


@dataclass
class METLEmbeddingSource:
    """Position-specific embeddings from a frozen METL-Global source model."""

    wt_sequence: str
    wt_pocket: Mapping[int, str]
    model_id: str = "metl-g-20m-1d"
    checkpoint_path: Optional[Union[str, Path]] = None
    checkpoint_sha256: str = ""
    pdb_path: Optional[Union[str, Path]] = None
    cache_dir: Union[str, Path] = Path("results") / "metl_embedding_cache"
    batch_size: int = 4
    device: str = "auto"
    name: str = "metl_g_20m_1d_target"
    method: str = "metl_biophysics_pretrained_residue_embeddings"
    representation: str = field(default="target", init=False)
    description: str = field(init=False)
    _model: object = field(default=None, init=False, repr=False)
    _data_encoder: object = field(default=None, init=False, repr=False)
    _torch: object = field(default=None, init=False, repr=False)
    _resolved_device: Optional[str] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.checkpoint_path is None:
            raise ValueError("METL requires a local, checksum-pinned checkpoint")
        self.checkpoint_path = Path(self.checkpoint_path)
        if sha256(self.checkpoint_path.read_bytes()).hexdigest() != self.checkpoint_sha256:
            raise ValueError("METL checkpoint checksum mismatch")
        self.wt_sequence = self.wt_sequence.upper()
        self.model_id = self.model_id.lower()
        self.cache_dir = Path(self.cache_dir)
        self.pdb_path = None if self.pdb_path is None else Path(self.pdb_path)
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least one")
        if not self.model_id.startswith("metl-g-"):
            raise ValueError(
                "NylC requires a METL-Global source model; published METL-Local "
                "checkpoints are specific to other proteins."
            )
        if self.requires_pdb and self.pdb_path is None:
            raise ValueError(f"{self.model_id} requires the NylC wild-type PDB file")
        if self.pdb_path is not None and not self.pdb_path.is_file():
            raise FileNotFoundError(f"METL PDB file not found: {self.pdb_path}")
        build_variant_sequence(self.wt_sequence, self.wt_pocket, self.wt_pocket)
        rpe = "3D structure-relative" if self.requires_pdb else "1D sequence-relative"
        self.description = (
            f"Target-position embeddings from frozen {self.model_id} with {rpe} attention."
        )

    @property
    def requires_pdb(self) -> bool:
        return self.model_id.endswith("-3d")

    def _load_backend(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            import metl
        except ImportError as exc:
            raise ImportError(
                "METL embeddings require torch and the official metl-pretrained "
                "package. Install it with: pip install "
                "git+https://github.com/gitter-lab/metl-pretrained.git"
            ) from exc

        resolved_device = self.device
        if resolved_device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

        model, data_encoder = metl.get_from_checkpoint(str(self.checkpoint_path))
        model = model.to(resolved_device)
        model.eval()

        self._model = model
        self._data_encoder = data_encoder
        self._torch = torch
        self._resolved_device = resolved_device

    def _pdb_digest(self) -> str:
        if self.pdb_path is None:
            return "no-pdb"
        return sha256(self.pdb_path.read_bytes()).hexdigest()

    def _cache_path(self, sequence: str) -> Path:
        from importlib.metadata import version, PackageNotFoundError

        versions = []
        for package in ("torch", "metl-pretrained"):
            try:
                versions.append(version(package))
            except PackageNotFoundError:
                versions.append("not-installed")
        implementation = sha256(Path(__file__).read_bytes()).hexdigest()
        identity = f"ml-nylc-v1\0{self.model_id}\0{self.checkpoint_sha256}\0{self.device}\0{versions}\0{implementation}\0{self._pdb_digest()}\0{sequence}"
        digest = sha256(identity.encode("utf-8")).hexdigest()
        return self.cache_dir / self.model_id / f"{digest}.npz"

    def _read_cache(self, sequence: str) -> Optional[np.ndarray]:
        path = self._cache_path(sequence)
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as cached:
            vectors = np.asarray(cached["embeddings"], dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(sequence):
            return None
        return vectors

    def _write_cache(self, sequence: str, vectors: np.ndarray) -> None:
        path = self._cache_path(sequence)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, embeddings=np.asarray(vectors, dtype=np.float32))
        temporary.replace(path)

    def _pooling_layer(self):
        """Locate the layer whose input is METL's residue-wise representation."""
        network = getattr(self._model, "model", None)
        if network is None:
            raise RuntimeError("Loaded METL model does not expose its internal network")
        for name, layer in network.named_children():
            if name == "avg_pooling":
                return layer
        raise RuntimeError("Loaded METL model has no global average pooling layer")

    def _embed_uncached(self, sequences: Sequence[str]) -> list[np.ndarray]:
        self._load_backend()
        torch = self._torch
        output_vectors: list[np.ndarray] = []

        for start in range(0, len(sequences), self.batch_size):
            batch = list(sequences[start : start + self.batch_size])
            if len({len(sequence) for sequence in batch}) != 1:
                raise ValueError("All METL sequences in one batch must have equal length")
            encoded = self._data_encoder.encode_sequences(batch)
            tokens = torch.as_tensor(encoded, dtype=torch.long, device=self._resolved_device)
            captured: list[object] = []

            def capture_pooling_input(_module, inputs):
                captured.append(inputs[0].detach())

            handle = self._pooling_layer().register_forward_pre_hook(capture_pooling_input)
            try:
                with torch.inference_mode():
                    if self.requires_pdb:
                        self._model(tokens, pdb_fn=str(self.pdb_path))
                    else:
                        self._model(tokens)
            finally:
                handle.remove()

            if len(captured) != 1:
                raise RuntimeError("Could not capture METL representation before pooling")
            hidden = captured[0].cpu()
            if hidden.ndim != 3 or hidden.shape[0] != len(batch):
                raise RuntimeError(f"Unexpected METL hidden-state shape: {tuple(hidden.shape)}")

            for row, sequence in enumerate(batch):
                vectors = hidden[row].numpy().astype(np.float32)
                if vectors.shape[0] != len(sequence):
                    raise ValueError(
                        "METL residue alignment failed: "
                        f"expected {len(sequence)} residues, got {vectors.shape[0]}"
                    )
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                vectors = vectors / np.maximum(norms, np.finfo(np.float32).eps)
                output_vectors.append(vectors)
        return output_vectors

    def _sequence_embeddings(
        self,
        pockets: Sequence[Mapping[int, str]],
    ) -> tuple[list[str], dict[str, np.ndarray]]:
        sequences = [
            build_variant_sequence(self.wt_sequence, pocket, self.wt_pocket) for pocket in pockets
        ]
        unique_sequences = list(dict.fromkeys(sequences))
        by_sequence: dict[str, np.ndarray] = {}
        missing: list[str] = []

        for sequence in unique_sequences:
            cached = self._read_cache(sequence)
            if cached is None:
                missing.append(sequence)
            else:
                by_sequence[sequence] = cached

        if missing:
            for sequence, vectors in zip(missing, self._embed_uncached(missing)):
                by_sequence[sequence] = vectors
                self._write_cache(sequence, vectors)
        return sequences, by_sequence

    def position_arrays(
        self,
        pockets: Sequence[Mapping[int, str]],
        positions: Sequence[int],
    ) -> dict[int, np.ndarray]:
        """Return one contextual METL embedding matrix per target position."""
        positions = tuple(int(position) for position in positions)
        if not positions:
            raise ValueError("At least one target position is required")
        if min(positions) < 1 or max(positions) > len(self.wt_sequence):
            raise ValueError("A requested METL position is outside the wild-type sequence")
        sequences, by_sequence = self._sequence_embeddings(pockets)
        return {
            position: np.vstack([by_sequence[sequence][position - 1] for sequence in sequences])
            for position in positions
        }
