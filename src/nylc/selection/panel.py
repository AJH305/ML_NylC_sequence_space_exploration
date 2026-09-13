"""Activity-constrained DPP selection, migrated from notebook 04.

The kernel is constructed before sorting and remains aligned with matrix indices.
"""

import numpy as np
import pandas as pd
from nylc.selection.dpp import (
    normalize_to_correlation,
    sample_k_dpp as _sample_k_dpp,
    kernel_batch_diagnostics,
)
from nylc.provenance import write_json


def select_panel(candidates, full_candidate_kernel, df_lab, settings, output):
    if full_candidate_kernel.shape != (len(candidates), len(candidates)):
        raise ValueError("Candidate kernel shape mismatch")
    if not candidates["candidate_id"].is_unique:
        raise ValueError("Candidate IDs must be unique")
    if settings["exclude_already_tested"]:
        mask = ~candidates["already_tested"].to_numpy(dtype=bool)
        candidates = candidates.loc[mask].copy()
        full_candidate_kernel = full_candidate_kernel[np.ix_(mask, mask)]
    LAB_PANEL_SIZE = settings["panel_size"]
    N_CONTROLS = settings["controls"]
    N_GENERATED_TO_SELECT = LAB_PANEL_SIZE - N_CONTROLS
    SHORTLIST_SIZES = settings["shortlist_sizes"]
    PRIMARY_SHORTLIST_SIZE = settings["primary_shortlist"]
    ACTIVITY_FRACTION_OF_BEST = settings["activity_fraction"]
    ACTIVITY_FRACTIONS_SENSITIVITY = settings["sensitivity_activity_fractions"]
    DPP_RANDOM_SEED = settings["seed"]
    DPP_SENSITIVITY_N_SEEDS = settings["sensitivity_seeds"]

    def sample_k_dpp(L, k, seed):
        return _sample_k_dpp(
            L,
            k,
            seed,
            jitter=settings["jitter"],
            eigenvalue_tolerance=settings["eigenvalue_tolerance"],
        )

    candidates = candidates.reset_index(drop=True).copy()

    candidates["candidate_matrix_index"] = np.arange(len(candidates), dtype=int)

    candidates = candidates.sort_values(
        ["predicted_activity_gp_mean", "candidate_id"], ascending=[False, True]
    ).reset_index(drop=True)

    candidates["activity_rank"] = np.arange(1, len(candidates) + 1)

    best_predicted_activity = float(candidates["predicted_activity_gp_mean"].max())

    activity_floor = ACTIVITY_FRACTION_OF_BEST * best_predicted_activity

    eligible_candidates = (
        candidates[candidates["predicted_activity_gp_mean"] >= activity_floor]
        .copy()
        .reset_index(drop=True)
    )

    activity_constraint_sensitivity = pd.DataFrame(
        [
            {
                "activity_fraction_of_best": fraction,
                "activity_floor": fraction * best_predicted_activity,
                "n_eligible": int(
                    (
                        candidates["predicted_activity_gp_mean"]
                        >= fraction * best_predicted_activity
                    ).sum()
                ),
            }
            for fraction in ACTIVITY_FRACTIONS_SENSITIVITY
        ]
    )

    if len(eligible_candidates) < max(SHORTLIST_SIZES):
        raise ValueError(
            f"The activity constraint leaves {len(eligible_candidates)} candidates, but the largest shortlist requires {max(SHORTLIST_SIZES)}. Reduce ACTIVITY_FRACTION_OF_BEST or the shortlist sizes."
        )

    pure_top_k_mean = float(
        candidates.head(N_GENERATED_TO_SELECT)["predicted_activity_gp_mean"].mean()
    )

    def select_for_shortlist_size(shortlist_size):
        shortlist = eligible_candidates.head(int(shortlist_size)).copy().reset_index(drop=True)
        matrix_indices = shortlist["candidate_matrix_index"].to_numpy(dtype=int)
        shortlist_kernel = full_candidate_kernel[np.ix_(matrix_indices, matrix_indices)]
        L = normalize_to_correlation(shortlist_kernel)
        seed = int(DPP_RANDOM_SEED + shortlist_size)
        selected_positions = sample_k_dpp(L, N_GENERATED_TO_SELECT, seed)
        selected = shortlist.iloc[selected_positions].copy().reset_index(drop=True)
        selected["selection_bucket"] = f"exploitation_k_dpp_y{shortlist_size}"
        selected["shortlist_size"] = int(shortlist_size)
        selected["dpp_seed"] = seed
        selected["dpp_draw_order"] = np.arange(1, len(selected) + 1)
        selected["panel_rank"] = np.arange(1, len(selected) + 1)
        if len(selected) != N_GENERATED_TO_SELECT:
            raise RuntimeError("k-DPP did not return the requested panel size.")
        if selected["candidate_id"].duplicated().any():
            raise AssertionError("k-DPP returned duplicate candidates.")
        if (selected["predicted_activity_gp_mean"] < activity_floor - 1e-10).any():
            raise AssertionError("Selected candidate violates the activity constraint.")
        selected_matrix_indices = selected["candidate_matrix_index"].to_numpy(dtype=int)
        selected_kernel = full_candidate_kernel[
            np.ix_(selected_matrix_indices, selected_matrix_indices)
        ]
        diagnostics = kernel_batch_diagnostics(selected_kernel)
        diagnostics.update(
            {
                "shortlist_size": int(shortlist_size),
                "n_selected": len(selected),
                "mean_predicted_activity": float(selected["predicted_activity_gp_mean"].mean()),
                "min_predicted_activity": float(selected["predicted_activity_gp_mean"].min()),
                "max_predicted_activity": float(selected["predicted_activity_gp_mean"].max()),
                "mean_activity_retention_vs_top_k": float(
                    selected["predicted_activity_gp_mean"].mean() / pure_top_k_mean
                ),
                "mean_activity_rank": float(selected["activity_rank"].mean()),
                "worst_activity_rank": int(selected["activity_rank"].max()),
                "dpp_seed": seed,
            }
        )
        return (shortlist, selected, diagnostics)

    shortlists = {}

    selected_panels = {}

    diagnostic_rows = []

    for shortlist_size in SHORTLIST_SIZES:
        shortlist, selected, diagnostics = select_for_shortlist_size(shortlist_size)
        shortlists[shortlist_size] = shortlist
        selected_panels[shortlist_size] = selected
        diagnostic_rows.append(diagnostics)

    shortlist_diagnostics = pd.DataFrame(diagnostic_rows).sort_values("shortlist_size")

    lab_panel_generated = selected_panels[PRIMARY_SHORTLIST_SIZE].copy()

    primary_shortlist = shortlists[PRIMARY_SHORTLIST_SIZE]

    primary_matrix_indices = primary_shortlist["candidate_matrix_index"].to_numpy(dtype=int)

    primary_L = normalize_to_correlation(
        full_candidate_kernel[np.ix_(primary_matrix_indices, primary_matrix_indices)]
    )

    primary_ids = set(lab_panel_generated["candidate_id"])

    seed_sensitivity_rows = []

    selection_counts = {candidate_id: 0 for candidate_id in primary_shortlist["candidate_id"]}

    for draw_index in range(DPP_SENSITIVITY_N_SEEDS):
        seed = DPP_RANDOM_SEED + 100000 + draw_index
        positions = sample_k_dpp(primary_L, N_GENERATED_TO_SELECT, seed)
        draw = primary_shortlist.iloc[positions]
        draw_ids = set(draw["candidate_id"])
        for candidate_id in draw_ids:
            selection_counts[candidate_id] += 1
        draw_matrix_indices = draw["candidate_matrix_index"].to_numpy(dtype=int)
        draw_diagnostics = kernel_batch_diagnostics(
            full_candidate_kernel[np.ix_(draw_matrix_indices, draw_matrix_indices)]
        )
        seed_sensitivity_rows.append(
            {
                "draw_index": draw_index,
                "seed": seed,
                "mean_predicted_activity": float(draw["predicted_activity_gp_mean"].mean()),
                "min_predicted_activity": float(draw["predicted_activity_gp_mean"].min()),
                "jaccard_to_primary_panel": len(draw_ids & primary_ids)
                / len(draw_ids | primary_ids),
                **draw_diagnostics,
            }
        )

    dpp_seed_sensitivity = pd.DataFrame(seed_sensitivity_rows)

    dpp_selection_frequency = pd.DataFrame(
        [
            {
                "candidate_id": candidate_id,
                "selection_count": count,
                "selection_frequency": count / DPP_SENSITIVITY_N_SEEDS,
            }
            for candidate_id, count in selection_counts.items()
        ]
    ).sort_values(["selection_frequency", "candidate_id"], ascending=[False, True])

    overlap_rows = []

    for size_a in SHORTLIST_SIZES:
        ids_a = set(selected_panels[size_a]["candidate_id"])
        for size_b in SHORTLIST_SIZES:
            ids_b = set(selected_panels[size_b]["candidate_id"])
            overlap_rows.append(
                {
                    "shortlist_a": size_a,
                    "shortlist_b": size_b,
                    "intersection_count": len(ids_a & ids_b),
                    "jaccard_similarity": len(ids_a & ids_b) / len(ids_a | ids_b),
                }
            )

    panel_overlap = pd.DataFrame(overlap_rows)

    control_rows = []

    control_ids = set()

    def add_control(row, reason):
        variant_id = row["variant_id"]
        if variant_id in control_ids or len(control_rows) >= N_CONTROLS:
            return
        chosen = row.copy()
        chosen["control_reason"] = reason
        control_rows.append(chosen)
        control_ids.add(variant_id)

    if "WT" in set(df_lab["variant_id"]):
        add_control(df_lab[df_lab["variant_id"] == "WT"].iloc[0], "WT")

    add_control(df_lab.sort_values("activity_pa6", ascending=False).iloc[0], "highest_activity")

    median_order = (
        (df_lab["activity_pa6"] - df_lab["activity_pa6"].median()).abs().sort_values().index
    )

    for idx in median_order:
        add_control(df_lab.loc[idx], "median_activity")
        if len(control_rows) >= N_CONTROLS:
            break

    for _, row in df_lab.sort_values(["activity_pa6", "variant_id"]).iterrows():
        add_control(row, "distinct_fallback")
        if len(control_rows) >= N_CONTROLS:
            break

    controls_df = pd.DataFrame(control_rows).head(N_CONTROLS).copy()

    controls_df["selection_bucket"] = "control"

    controls_df["panel_rank"] = np.arange(
        len(lab_panel_generated) + 1, len(lab_panel_generated) + 1 + len(controls_df)
    )

    expected_controls = min(N_CONTROLS, df_lab["variant_id"].nunique())

    if len(controls_df) != expected_controls:
        raise RuntimeError(
            f"Control fill failed: selected {len(controls_df)}, expected {expected_controls}."
        )

    tables = {
        "all_candidates_activity_ranked": candidates,
        "activity_constraint_eligible_candidates": eligible_candidates,
        "activity_constraint_sensitivity": activity_constraint_sensitivity,
        "shortlist_sensitivity_diagnostics": shortlist_diagnostics,
        "lab_test_panel_generated": lab_panel_generated,
        "lab_test_panel_controls": controls_df,
        "shortlist_panel_overlap": panel_overlap,
        "dpp_seed_sensitivity": dpp_seed_sensitivity,
        "dpp_candidate_selection_frequency": dpp_selection_frequency,
    }
    for size in SHORTLIST_SIZES:
        tables[f"shortlist_top_{size}"] = shortlists[size]
        tables[f"panel_k_dpp_shortlist_{size}"] = selected_panels[size]
    from nylc.data.lab import serializable_frame

    for name, table in tables.items():
        serializable_frame(table).to_csv(output / f"{name}.csv", index=False)
    np.savez_compressed(
        output / "primary_kernel.npz",
        kernel=primary_L,
        candidate_ids=primary_shortlist["candidate_id"].to_numpy(dtype=str),
        selected_ids=lab_panel_generated["candidate_id"].to_numpy(dtype=str),
    )
    write_json(
        output / "summary.json",
        {
            "n_candidates": len(candidates),
            "n_eligible": len(eligible_candidates),
            "n_generated": len(lab_panel_generated),
            "n_controls": len(controls_df),
            "activity_floor": activity_floor,
            "kernel_alignment": "kernel built in input order; stable row indices carried through ranking",
            "uncertainty_usage": "sigma scaling changes uncertainty reporting; selection uses predicted mean and prior kernel",
        },
    )
