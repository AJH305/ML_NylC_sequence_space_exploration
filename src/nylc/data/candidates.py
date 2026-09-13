"""Candidate mapping and checks required by notebook 04."""

import numpy as np
import pandas as pd
from nylc.features.descriptors import POSITIONS, WT_POCKET, CANONICAL_AA
from nylc.data.lab import mutation_positions, mutation_signature, mutation_list_to_string

BOLTZGEN_POSITION_MAP = [
    ("full_sequence_0", 82, 99),
    ("full_sequence_0", 117, 134),
    ("full_sequence_3", 38, 304),
    ("full_sequence_7", 64, 330),
]

STRUCTURAL_METRICS = ["ptm", "iptm", "design_iiptm", "filter_rmsd"]

HYDROPHOBICITY_METRIC = "design_largest_hydrophobic_patch_refolded"


def robust_z(series, higher_is_better=True):
    s = pd.to_numeric(series, errors="coerce")
    med = s.median()
    mad = (s - med).abs().median()
    if not np.isfinite(mad) or mad == 0:
        std = s.std(ddof=0)
        z = (s - med) / std if std and np.isfinite(std) else pd.Series(0.0, index=s.index)
    else:
        z = 0.6745 * (s - med) / mad
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return z if higher_is_better else -z


def safe_numeric_col(df, col, default=np.nan):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index)


def structural_developability_components(df):
    required = STRUCTURAL_METRICS + [HYDROPHOBICITY_METRIC]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing BoltzGen metrics required for scoring: {missing}")
    structural_block = pd.concat(
        [
            robust_z(safe_numeric_col(df, "ptm"), True),
            robust_z(safe_numeric_col(df, "iptm"), True),
            robust_z(safe_numeric_col(df, "design_iiptm"), True),
            robust_z(safe_numeric_col(df, "filter_rmsd"), False),
        ],
        axis=1,
    ).mean(axis=1)
    hydrophobic_patch = robust_z(
        safe_numeric_col(df, HYDROPHOBICITY_METRIC), higher_is_better=False
    )
    combined = 0.7 * structural_block + 0.3 * hydrophobic_patch
    return (structural_block, hydrophobic_patch, combined)


def validate_required_boltzgen_columns(df, position_map=BOLTZGEN_POSITION_MAP):
    required_cols = sorted(set((col for col, _, _ in position_map)))
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError("Missing required BoltzGen sequence columns: " + ", ".join(missing_cols))


def get_aa_from_chain_sequence(row, sequence_col, local_pos):
    if sequence_col not in row.index:
        return (None, f"missing_column_{sequence_col}")
    seq = row[sequence_col]
    if pd.isna(seq):
        return (None, f"missing_value_{sequence_col}")
    seq = str(seq).strip().upper()
    if len(seq) < local_pos:
        return (None, f"{sequence_col}_too_short_for_position_{local_pos}")
    aa = seq[local_pos - 1]
    if aa not in CANONICAL_AA:
        return (None, f"noncanonical_{aa}_in_{sequence_col}_at_{local_pos}")
    return (aa, "ok")


def extract_boltzgen_pocket_from_chain_sequences(
    row, position_map=BOLTZGEN_POSITION_MAP, wt_pocket=WT_POCKET
):
    pocket = {}
    mutations = []
    status_messages = []
    for sequence_col, local_pos, model_pos in position_map:
        aa, status = get_aa_from_chain_sequence(row, sequence_col, local_pos)
        if status != "ok":
            status_messages.append(status)
            continue
        pocket[model_pos] = aa
        wt_aa = wt_pocket[model_pos]
        if aa != wt_aa:
            mutations.append((wt_aa, model_pos, aa))
    expected_positions = set(POSITIONS)
    if set(pocket.keys()) != expected_positions:
        return (None, "incomplete_pocket", None, None)
    if status_messages:
        return (None, ";".join(status_messages), None, None)
    mutated_positions = tuple((pos for _, pos, _ in mutations))
    return (pocket, "ok", mutations, mutated_positions)


def add_boltzgen_candidates_from_full_sequences(df):
    df = df.copy()
    validate_required_boltzgen_columns(df, BOLTZGEN_POSITION_MAP)
    if "id" in df.columns:
        df["candidate_id"] = df["boltzgen_run"].astype(str) + "__" + df["id"].astype(str)
    else:
        df["candidate_id"] = [f"candidate_{i:05d}" for i in range(len(df))]
    pockets = []
    statuses = []
    mutations = []
    mutation_strings = []
    mutated_positions = []
    for _, row in df.iterrows():
        pocket, status, mutation_list, pos_tuple = extract_boltzgen_pocket_from_chain_sequences(row)
        pockets.append(pocket)
        statuses.append(status)
        mutations.append(mutation_list if mutation_list is not None else [])
        mutation_strings.append(mutation_list_to_string(mutation_list or []))
        mutated_positions.append(pos_tuple)
    df["pocket"] = pockets
    df["candidate_status"] = statuses
    df["parsed_mutations"] = mutations
    df["mutations"] = mutation_strings
    df["mutated_positions"] = mutated_positions
    df_valid = df[df["candidate_status"].eq("ok")].copy()
    df_valid["aa_tuple"] = df_valid["pocket"].apply(
        lambda pocket: tuple((pocket[pos] for pos in POSITIONS))
    )
    for pos in POSITIONS:
        df_valid[f"aa{pos}"] = df_valid["pocket"].apply(lambda pocket, pos=pos: pocket[pos])
    df_valid["n_target_mutations"] = df_valid["parsed_mutations"].apply(len)
    df_valid["n_mutations"] = df_valid["n_target_mutations"]
    df_valid["mutation_positions"] = df_valid["parsed_mutations"].apply(mutation_positions)
    df_valid["mutation_signature"] = df_valid["mutation_positions"].apply(mutation_signature)
    df_valid["mutation_set"] = df_valid["parsed_mutations"].apply(
        lambda muts: frozenset((f"{wt}{pos}{mut}" for wt, pos, mut in muts))
    )
    df_valid["only_gp_target_positions"] = True
    df_valid["n_external_mutations"] = 0
    _, _, df_valid["_dedup_structural_developability_score"] = structural_developability_components(
        df_valid
    )
    df_valid = df_valid.sort_values(
        ["_dedup_structural_developability_score", "candidate_id"], ascending=[False, True]
    )
    df_valid = (
        df_valid.drop_duplicates(subset=["aa_tuple"], keep="first")
        .drop(columns="_dedup_structural_developability_score")
        .reset_index(drop=True)
    )
    return (df, df_valid)


def check_designed_sequence_consistency(df_valid, strict=True):
    if "designed_sequence" not in df_valid.columns:
        pass
        return df_valid
    df_check = df_valid.copy()
    df_check["designed_sequence_clean"] = (
        df_check["designed_sequence"].astype(str).str.strip().str.upper()
    )
    df_check["extracted_designed_sequence"] = (
        df_check["aa99"] + df_check["aa134"] + df_check["aa304"] + df_check["aa330"]
    )
    df_check["designed_sequence_matches"] = (
        df_check["designed_sequence_clean"] == df_check["extracted_designed_sequence"]
    )
    n_mismatches = int((~df_check["designed_sequence_matches"]).sum())
    pass
    pass
    pass
    pass
    pass
    if strict and n_mismatches > 0:
        pass
        raise ValueError(
            "designed_sequence does not match extracted amino acids. Check BOLTZGEN_POSITION_MAP or chain numbering."
        )
    return df_check
