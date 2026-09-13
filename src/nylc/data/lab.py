"""Validated lab data and 1-based reference sequence mapping."""

from itertools import combinations
import re
import numpy as np
import pandas as pd
from Bio import SeqIO
from nylc.features.descriptors import WT_POCKET, POSITIONS, TARGET_POSITIONS, CANONICAL_AA

MUTATION_PATTERN = re.compile(r"([A-Z])(\d+)([A-Z])")


def read_reference(path):
    records = list(SeqIO.parse(str(path), "fasta"))
    if len(records) != 1:
        raise ValueError("Exactly one WT FASTA record is required")
    sequence = str(records[0].seq).upper()
    if not sequence or set(sequence) - set(CANONICAL_AA):
        raise ValueError("WT FASTA contains unsupported amino acids")
    for position, expected in WT_POCKET.items():
        if position > len(sequence) or sequence[position - 1] != expected:
            raise ValueError(f"WT sequence does not match {expected}{position}")
    return sequence


def parse_mutations(value, wt_sequence=None):
    pocket = dict(WT_POCKET)
    if pd.isna(value) or str(value).strip().lower() in {"", "wt", "wildtype", "wild type", "nan"}:
        return pocket, []
    text = str(value).strip().upper()
    matches = list(MUTATION_PATTERN.finditer(text))
    remainder = MUTATION_PATTERN.sub("", text)
    if not matches or re.sub(r"[\s;,/_+]+", "", remainder):
        raise ValueError(f"Malformed mutation specification: {value}")
    mutations, seen = [], set()
    for match in matches:
        wt, pos, mutant = match.groups()
        pos = int(pos)
        if pos < 1 or pos in seen or wt not in CANONICAL_AA or mutant not in CANONICAL_AA:
            raise ValueError(f"Invalid or duplicate mutation: {match.group()}")
        expected = WT_POCKET.get(pos)
        if wt_sequence is not None:
            if pos > len(wt_sequence):
                raise ValueError(f"Mutation outside WT sequence: {match.group()}")
            expected = wt_sequence[pos - 1]
        if expected is not None and wt != expected:
            raise ValueError(f"WT mismatch: {match.group()}, expected {expected}{pos}")
        if wt == mutant:
            raise ValueError(f"No-op mutation: {match.group()}")
        seen.add(pos)
        mutations.append((wt, pos, mutant))
        if pos in pocket:
            pocket[pos] = mutant
    return pocket, mutations


def mutation_positions(mutations):
    return {p for _, p, _ in mutations}


def mutation_signature(positions):
    return "+".join(str(p) for p in sorted(positions)) if positions else "WT"


def mutation_list_to_string(mutations):
    return ";".join(f"{wt}{pos}{mut}" for wt, pos, mut in sorted(mutations, key=lambda x: x[1]))


def annotate(frame, wt_sequence=None):
    frame = frame.copy()
    parsed = frame["mutations"].apply(lambda s: parse_mutations(s, wt_sequence))
    frame["pocket"] = parsed.map(lambda p: p[0])
    frame["parsed_mutations"] = parsed.map(lambda p: p[1])
    frame["all_mutations"] = frame["parsed_mutations"]
    frame["mutation_positions"] = frame["parsed_mutations"].map(mutation_positions)
    frame["mutation_signature"] = frame["mutation_positions"].map(mutation_signature)
    frame["mutation_order"] = frame["parsed_mutations"].map(len)
    frame["aa_tuple"] = frame["pocket"].map(lambda p: tuple(p[pos] for pos in POSITIONS))
    frame["only_target_positions"] = frame["mutation_positions"].map(
        lambda p: p.issubset(TARGET_POSITIONS)
    )
    frame["mutation_set"] = frame["parsed_mutations"].map(
        lambda ms: frozenset(f"{w}{p}{m}" for w, p, m in ms)
    )
    return frame


def load_lab(path, wt_sequence, subset_ids=()):
    frame = pd.read_csv(path, sep=";", decimal=",", keep_default_na=True)
    if not {"variant_id", "mutations"}.issubset(frame):
        raise ValueError("Lab data requires variant_id and mutations")
    if frame["variant_id"].isna().any() or not frame["variant_id"].is_unique:
        raise ValueError("Lab variant IDs must be present and unique")
    if not frame["variant_id"].astype(str).str.fullmatch(r"[A-Za-z0-9_-]+").all():
        raise ValueError("Unsafe variant identifier")
    activity = [c for c in frame if c.startswith("activity_pa6_")]
    tm = [c for c in frame if c.startswith("tm_celsius_")]
    if not activity:
        raise ValueError("No activity replicate columns found")
    for col in activity + tm:
        original = frame[col]
        converted = pd.to_numeric(original, errors="coerce")
        if (original.notna() & converted.isna()).any() or np.isinf(converted).any():
            raise ValueError(f"Invalid measurement in {col}")
        frame[col] = converted
    frame["activity_n"] = frame[activity].count(axis=1)
    frame["activity_pa6"] = frame[activity].mean(axis=1)
    frame["activity_sd"] = frame[activity].std(axis=1, ddof=1)
    frame["activity_sem"] = frame["activity_sd"] / np.sqrt(frame["activity_n"])
    frame["activity_sem_for_gp"] = frame["activity_sem"]
    frame["tm_celsius"] = frame[tm].mean(axis=1) if tm else np.nan
    if frame["activity_pa6"].isna().any():
        raise ValueError("At least one variant has no usable activity")
    frame = annotate(frame, wt_sequence)
    if subset_ids:
        absent = set(subset_ids) - set(frame.variant_id)
        if absent:
            raise ValueError(f"Unknown subset IDs: {sorted(absent)}")
        frame = frame.set_index("variant_id").loc[list(subset_ids)].reset_index()
    return frame.reset_index(drop=True)


def write_sequence_features(frame, wt_sequence, output):
    rows = []
    for _, row in frame.iterrows():
        sequence = list(wt_sequence)
        mutation_names = []
        for wt, pos, mutant in row["parsed_mutations"]:
            sequence[pos - 1] = mutant
            mutation_names.append(f"{wt}{pos}{mutant}")
        positions = {pos for _, pos, _ in row["parsed_mutations"]}
        flags = {
            "has_D99": int(99 in positions),
            "has_F134W": int("F134W" in mutation_names),
            "has_D304": int(304 in positions),
            "has_R330": int(330 in positions),
            "has_D99R": int("D99R" in mutation_names),
            "has_D304M": int("D304M" in mutation_names),
            "has_R330A": int("R330A" in mutation_names),
            "epistasis_D99R_D304": int("D99R" in mutation_names and 304 in positions),
            "hp_like_core": int({"F134W", "D304M", "R330A"}.issubset(mutation_names)),
        }
        rows.append(
            {
                "variant_id": row["variant_id"],
                "sequence": "".join(sequence),
                "mutation_count": len(mutation_names),
                **flags,
                **{f"pos_{p}": 1 for p in positions},
                **{f"mutation_{m}": 1 for m in mutation_names},
                **{f"pair_{a}__{b}": 1 for a, b in combinations(sorted(mutation_names), 2)},
            }
        )
    result = pd.DataFrame(rows).fillna(0)
    result.to_csv(output / "sequence_features.csv", index=False)
    (output / "variants.fasta").write_text(
        "".join(f">{r['variant_id']}\n{r['sequence']}\n" for r in rows), encoding="utf-8"
    )


def load_prepared(path):
    return annotate(pd.read_csv(path))


def training_sem(frame):
    values = frame["activity_sem"].to_numpy(dtype=float)
    usable = values[np.isfinite(values) & (values > 0)]
    return np.where(np.isfinite(values), values, np.median(usable) if len(usable) else 0.0)


def serializable_frame(frame):
    return frame.drop(
        columns=[
            "pocket",
            "parsed_mutations",
            "all_mutations",
            "mutation_positions",
            "aa_tuple",
            "mutation_set",
        ],
        errors="ignore",
    )
