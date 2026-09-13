"""Contextual residue embeddings for the NylC Epistatic GP notebooks.

The model and its optional dependencies are loaded lazily.  Importing this
module therefore does not require torch or transformers; they are only needed
when ``position_arrays`` is called for the first time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import numpy as np


def read_fasta_sequence(path: Union[str, Path]) -> str:
    """Read exactly one protein sequence from a FASTA file."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    sequence = "".join(line.strip() for line in lines if line.strip() and not line.startswith(">"))
    sequence = sequence.upper()
    if not sequence:
        raise ValueError(f"No sequence found in FASTA file: {path}")
    if not sequence.isalpha():
        raise ValueError(f"FASTA sequence contains unsupported characters: {path}")
    return sequence


def build_variant_sequence(
    wt_sequence: str,
    pocket: Mapping[int, str],
    wt_pocket: Mapping[int, str],
) -> str:
    """Apply a 1-based pocket definition to the wild-type sequence."""
    residues = list(wt_sequence.upper())
    for position, wt_aa in wt_pocket.items():
        if position < 1 or position > len(residues):
            raise ValueError(f"Position {position} is outside sequence length {len(residues)}")
        observed_wt = residues[position - 1]
        if observed_wt != str(wt_aa).upper():
            raise ValueError(
                f"WT mismatch at position {position}: FASTA has {observed_wt}, "
                f"configuration expects {wt_aa}"
            )
        mutant_aa = str(pocket[position]).upper()
        if len(mutant_aa) != 1 or not mutant_aa.isalpha():
            raise ValueError(f"Unsupported residue at position {position}: {mutant_aa!r}")
        residues[position - 1] = mutant_aa
    return "".join(residues)


@dataclass
class ProteinLLMEmbeddingSource:
    """ESM-2 representations exposed like a notebook feature source.

    ``representation`` controls the downstream representation:

    - ``target``: contextual vectors at the configured mutation positions;
    - ``global_delta``: the flattened full-protein difference to WT;
    - ``window_delta``: flattened WT differences in one local window per
      configured mutation position.

    Residue embeddings are L2-normalized before differences are calculated.
    Flattened delta representations are divided by the square root of their
    number of residues, making squared Euclidean distance an average over
    positions and keeping the existing RBF length-scale grid interpretable.
    """

    wt_sequence: str
    wt_pocket: Mapping[int, str]
    model_name: str = "facebook/esm2_t6_8M_UR50D"
    revision: str = ""
    model_cache_dir: Optional[Union[str, Path]] = None
    cache_dir: Union[str, Path] = Path("results") / "protein_llm_embedding_cache"
    batch_size: int = 8
    device: str = "auto"
    local_files_only: bool = False
    representation: str = "target"
    window_radius: int = 16
    name: str = "esm2_target"
    method: str = "contextual_residue_embeddings"
    description: str = field(init=False)
    _tokenizer: object = field(default=None, init=False, repr=False)
    _model: object = field(default=None, init=False, repr=False)
    _torch: object = field(default=None, init=False, repr=False)
    _resolved_device: Optional[str] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        import re

        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("ESM-2 requires an immutable 40-character model revision")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.wt_sequence = self.wt_sequence.upper()
        self.cache_dir = Path(self.cache_dir)
        valid_representations = {"target", "global_delta", "window_delta"}
        if self.representation not in valid_representations:
            raise ValueError(
                f"Unknown representation {self.representation!r}; "
                f"expected one of {sorted(valid_representations)}"
            )
        if self.window_radius < 0:
            raise ValueError("window_radius must be non-negative")
        descriptions = {
            "target": "contextual residue embeddings at the target positions",
            "global_delta": "flattened full-protein embedding difference to WT",
            "window_delta": (
                f"WT-difference embeddings in +/-{self.window_radius}-residue target windows"
            ),
        }
        self.description = f"ESM-2 {descriptions[self.representation]} from {self.model_name}."
        # Fail early on numbering mistakes before a model download is attempted.
        build_variant_sequence(self.wt_sequence, self.wt_pocket, self.wt_pocket)

    def _load_backend(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoTokenizer, EsmModel
        except ImportError as exc:
            raise ImportError(
                "Protein-LLM embeddings require torch and transformers. "
                "Install them in the active notebook environment first."
            ) from exc

        resolved_device = self.device
        if resolved_device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            revision=self.revision,
            cache_dir=self.model_cache_dir,
            local_files_only=self.local_files_only,
        )
        self._model = EsmModel.from_pretrained(
            self.model_name,
            revision=self.revision,
            cache_dir=self.model_cache_dir,
            local_files_only=self.local_files_only,
            add_pooling_layer=False,
        ).to(resolved_device)
        self._model.eval()
        self._torch = torch
        self._resolved_device = resolved_device

    def _cache_path(self, sequence: str) -> Path:
        from importlib.metadata import version, PackageNotFoundError

        versions = []
        for package in ("torch", "transformers"):
            try:
                versions.append(version(package))
            except PackageNotFoundError:
                versions.append("not-installed")
        implementation = sha256(Path(__file__).read_bytes()).hexdigest()
        identity = f"ml-nylc-v1\0{self.model_name}\0{self.revision}\0{self.device}\0{versions}\0{implementation}\0{sequence}"
        digest = sha256(identity.encode("utf-8")).hexdigest()
        model_slug = self.model_name.replace("/", "--")
        return self.cache_dir / model_slug / f"{digest}.npz"

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
        np.savez_compressed(
            temporary,
            embeddings=np.asarray(vectors, dtype=np.float32),
        )
        temporary.replace(path)

    def _embed_uncached(self, sequences: Sequence[str]) -> list[np.ndarray]:
        self._load_backend()
        output_vectors: list[np.ndarray] = []
        torch = self._torch

        for start in range(0, len(sequences), self.batch_size):
            batch = list(sequences[start : start + self.batch_size])
            encoded = self._tokenizer(
                batch,
                add_special_tokens=True,
                padding=True,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            special_tokens_mask = encoded.pop("special_tokens_mask")
            attention_mask = encoded["attention_mask"]
            model_inputs = {key: value.to(self._resolved_device) for key, value in encoded.items()}
            with torch.inference_mode():
                hidden = self._model(**model_inputs).last_hidden_state.detach().cpu()

            for row, sequence in enumerate(batch):
                residue_mask = (attention_mask[row] == 1) & (special_tokens_mask[row] == 0)
                residue_hidden = hidden[row][residue_mask]
                if residue_hidden.shape[0] != len(sequence):
                    raise ValueError(
                        "Tokenizer-to-residue alignment failed: "
                        f"expected {len(sequence)} residues, got {residue_hidden.shape[0]}"
                    )
                vectors = residue_hidden.numpy().astype(np.float32)
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
        # Delta representations always require the WT reference, even when the
        # WT is absent from a particular cross-validation training fold.
        requested_sequences = list(sequences)
        if self.representation.endswith("_delta"):
            requested_sequences.append(self.wt_sequence)
        unique_sequences = list(dict.fromkeys(requested_sequences))
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
        """Return one contextual embedding matrix per mutational position."""
        positions = tuple(int(position) for position in positions)
        if self.representation == "global_delta":
            raise TypeError("global_delta uses feature_matrix(), not position_arrays()")
        if max(positions) > len(self.wt_sequence):
            raise ValueError(
                f"Requested position {max(positions)} exceeds sequence length {len(self.wt_sequence)}"
            )
        sequences, by_sequence = self._sequence_embeddings(pockets)

        if self.representation == "target":
            return {
                position: np.vstack([by_sequence[sequence][position - 1] for sequence in sequences])
                for position in positions
            }

        wt_embeddings = by_sequence[self.wt_sequence]
        arrays = {}
        for position in positions:
            start = max(0, position - 1 - self.window_radius)
            stop = min(len(self.wt_sequence), position + self.window_radius)
            n_residues = stop - start
            arrays[position] = np.vstack(
                [
                    (
                        (by_sequence[sequence][start:stop] - wt_embeddings[start:stop])
                        / np.sqrt(n_residues)
                    ).reshape(-1)
                    for sequence in sequences
                ]
            )
        return arrays

    def feature_matrix(
        self,
        pockets: Sequence[Mapping[int, str]],
    ) -> np.ndarray:
        """Return the full-protein WT-delta matrix for the global RBF-GP."""
        if self.representation != "global_delta":
            raise TypeError("feature_matrix() is only available for global_delta")
        sequences, by_sequence = self._sequence_embeddings(pockets)
        wt_embeddings = by_sequence[self.wt_sequence]
        scale = np.sqrt(len(self.wt_sequence))
        return np.vstack(
            [
                ((by_sequence[sequence] - wt_embeddings) / scale).reshape(-1)
                for sequence in sequences
            ]
        )
