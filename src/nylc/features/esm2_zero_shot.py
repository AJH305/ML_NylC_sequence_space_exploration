"""True zero-shot ESM-2 likelihood-ratio scores for NylC variants.

The scorer follows the masked-marginal setup used in protein-fitness
benchmarks: all mutated positions are masked simultaneously and the score is
the sum of mutant-versus-wild-type log-probability ratios at those positions.
No assay labels are used. Scores are cached per model, wild type and variant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import numpy as np

from nylc.features.protein_llm_embeddings import build_variant_sequence


@dataclass
class ESM2ZeroShotScorer:
    wt_sequence: str
    wt_pocket: Mapping[int, str]
    model_name: str = "facebook/esm2_t6_8M_UR50D"
    cache_dir: Union[str, Path] = Path("results") / "esm2_zero_shot_cache"
    batch_size: int = 8
    device: str = "auto"
    local_files_only: bool = True
    revision: str = ""
    model_cache_dir: Union[str, Path] = Path("cache/huggingface")
    _tokenizer: object = field(default=None, init=False, repr=False)
    _model: object = field(default=None, init=False, repr=False)
    _torch: object = field(default=None, init=False, repr=False)
    _resolved_device: Optional[str] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        import re

        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("ESM revision must be an immutable commit")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.wt_sequence = self.wt_sequence.upper()
        self.wt_pocket = {int(pos): str(aa).upper() for pos, aa in self.wt_pocket.items()}
        self.cache_dir = Path(self.cache_dir)
        build_variant_sequence(self.wt_sequence, self.wt_pocket, self.wt_pocket)

    def _load_backend(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoTokenizer, EsmForMaskedLM
        except ImportError as exc:
            raise ImportError("ESM-2 zero-shot scoring requires torch and transformers.") from exc

        resolved_device = self.device
        if resolved_device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=self.local_files_only,
            revision=self.revision,
            cache_dir=str(self.model_cache_dir),
        )
        self._model = EsmForMaskedLM.from_pretrained(
            self.model_name,
            local_files_only=self.local_files_only,
            revision=self.revision,
            cache_dir=str(self.model_cache_dir),
        ).to(resolved_device)
        self._model.eval()
        self._torch = torch
        self._resolved_device = resolved_device

    def _cache_path(self, variant_sequence: str) -> Path:
        from importlib.metadata import version

        implementation = sha256(Path(__file__).read_bytes()).hexdigest()
        identity = (
            f"{self.revision}\0{self.device}\0{version('torch')}\0{version('transformers')}\0{implementation}\0"
            f"{self.model_name}\0masked_mutation_set_log_odds\0"
            f"{self.wt_sequence}\0{variant_sequence}"
        )
        digest = sha256(identity.encode("utf-8")).hexdigest()
        model_slug = self.model_name.replace("/", "--")
        return self.cache_dir / model_slug / f"{digest}.npz"

    def _read_cache(self, variant_sequence: str) -> Optional[float]:
        path = self._cache_path(variant_sequence)
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as cached:
            return float(cached["score"])

    def _write_cache(self, variant_sequence: str, score: float) -> None:
        path = self._cache_path(variant_sequence)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, score=np.asarray(score, dtype=np.float64))
        temporary.replace(path)

    def _score_batch(
        self,
        variant_sequences: Sequence[str],
        mutation_positions: Sequence[Sequence[int]],
    ) -> list[float]:
        self._load_backend()
        torch = self._torch
        encoded = self._tokenizer(
            list(variant_sequences),
            add_special_tokens=True,
            padding=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        special_tokens_mask = encoded.pop("special_tokens_mask")
        attention_mask = encoded["attention_mask"]
        input_ids = encoded["input_ids"].clone()
        residue_token_indices = []

        for row, (sequence, positions) in enumerate(zip(variant_sequences, mutation_positions)):
            residue_mask = (attention_mask[row] == 1) & (special_tokens_mask[row] == 0)
            token_indices = torch.where(residue_mask)[0]
            if len(token_indices) != len(sequence):
                raise ValueError(
                    f"Tokenizer alignment failed: expected {len(sequence)} residues, "
                    f"got {len(token_indices)}."
                )
            selected = [int(token_indices[int(pos) - 1]) for pos in positions]
            residue_token_indices.append(selected)
            input_ids[row, selected] = self._tokenizer.mask_token_id

        model_inputs = {
            "input_ids": input_ids.to(self._resolved_device),
            "attention_mask": attention_mask.to(self._resolved_device),
        }
        with torch.inference_mode():
            log_probs = torch.log_softmax(self._model(**model_inputs).logits, dim=-1).cpu()

        scores = []
        for row, (sequence, positions, token_indices) in enumerate(
            zip(variant_sequences, mutation_positions, residue_token_indices)
        ):
            score = 0.0
            for pos, token_idx in zip(positions, token_indices):
                mutant_aa = sequence[int(pos) - 1]
                wt_aa = self.wt_sequence[int(pos) - 1]
                mutant_id = self._tokenizer.convert_tokens_to_ids(mutant_aa)
                wt_id = self._tokenizer.convert_tokens_to_ids(wt_aa)
                score += float(
                    log_probs[row, token_idx, mutant_id] - log_probs[row, token_idx, wt_id]
                )
            scores.append(score)
        return scores

    def score_pockets(self, pockets: Sequence[Mapping[int, str]]) -> np.ndarray:
        sequences = [
            build_variant_sequence(self.wt_sequence, pocket, self.wt_pocket) for pocket in pockets
        ]
        positions_by_variant = [
            tuple(pos for pos in self.wt_pocket if str(pocket[pos]).upper() != self.wt_pocket[pos])
            for pocket in pockets
        ]
        scores: list[Optional[float]] = [None] * len(sequences)
        missing_indices = []

        for idx, (sequence, positions) in enumerate(zip(sequences, positions_by_variant)):
            if not positions:
                scores[idx] = 0.0
                continue
            cached = self._read_cache(sequence)
            if cached is None:
                missing_indices.append(idx)
            else:
                scores[idx] = cached

        for start in range(0, len(missing_indices), self.batch_size):
            batch_indices = missing_indices[start : start + self.batch_size]
            batch_scores = self._score_batch(
                [sequences[idx] for idx in batch_indices],
                [positions_by_variant[idx] for idx in batch_indices],
            )
            for idx, score in zip(batch_indices, batch_scores):
                scores[idx] = score
                self._write_cache(sequences[idx], score)

        return np.asarray(scores, dtype=float)
