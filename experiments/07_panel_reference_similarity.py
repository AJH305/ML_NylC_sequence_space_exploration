"""Kernel similarity of the tested panel to the two reference enzymes.

Answers a reviewer question on the thesis draft: the similarity heat map of
Section 6.2.2 shows how similar the designed variants are to one another, but
not how far they lie from the characterized reference enzymes NylC-TS and
NylC-HP.  This script computes that distance in the same prior kernel the
panel selection used, and -- if the processed laboratory data are available --
relates it to the measured 20 h activity.

Inputs (all produced by the packaged pipeline, reference run):
    results/reference_cpu/features/physical.npz      position-wise descriptors
    results/reference_cpu/fit/model.json             fitted hyperparameters
    results/reference_cpu/predict/candidate_kernel.npz   stored candidate kernel
    experiments/tested_variants.csv                  pockets of the 24 constructs
    experiments/derived/lab_variants.csv             optional, from notebook 06

Output:
    experiments/derived/panel_reference_similarity.csv

Length scale
------------
The kernel is k = sigma_main^2 * k_main + sigma_epi^2 * k_epi with
position-wise RBF kernels of length scale l = c * d_train.  The multiplier c is
recorded in fit/model.json, but d_train -- the median positive pairwise
descriptor distance of the training fold -- is not stored as an artifact and
could not be re-derived unambiguously from the feature store.  The length scale
is therefore recovered numerically from the stored candidate kernel and the
reconstruction is asserted to be exact (max deviation < 1e-9).  Any future
change to the kernel or to the hyperparameters is picked up automatically,
because the assertion fails if the reconstruction stops matching.
"""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "reference_cpu"
OUT = ROOT / "experiments" / "derived"
POSITIONS = [99, 134, 304, 330]
# Pockets of the two reference enzymes at positions 99/134/304/330.
# NylC-TS is the parental enzyme; NylC-HP is NylC-F134W/D304M/R330A.
REFERENCES = {"NylC-TS": "DFDR", "NylC-HP": "DWMA"}
N_PERMUTATIONS = 20_000
N_RANDOM_SUBSETS = 5_000
SEED = 0


def load_features():
    data = np.load(RUN / "features" / "physical.npz", allow_pickle=True)
    keys = data["keys"]
    descriptors = np.stack([data[f"component_{p}"] for p in POSITIONS], axis=1)
    return keys, descriptors


def pairwise_distance(a, b):
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)


def kernel_matrix(xa, xb, length_scale, sigma_main, sigma_epi):
    """Combined additive/pairwise kernel between two sets of pockets."""
    n_pos = xa.shape[1]
    per_position = np.stack([
        np.exp(-pairwise_distance(xa[:, p, :], xb[:, p, :]) ** 2 / (2 * length_scale ** 2))
        for p in range(n_pos)
    ])
    k_main = per_position.mean(axis=0)
    k_epi = np.mean(
        [per_position[p] * per_position[q] for p, q in itertools.combinations(range(n_pos), 2)],
        axis=0,
    )
    return sigma_main ** 2 * k_main + sigma_epi ** 2 * k_epi


def recover_length_scale(descriptors, reference_kernel, sigma_main, sigma_epi):
    """Recover l by ternary search against the stored candidate kernel."""
    candidates = descriptors[len(descriptors) - reference_kernel.shape[0]:]

    def deviation(length_scale):
        k = kernel_matrix(candidates, candidates, length_scale, sigma_main, sigma_epi)
        return np.abs(k - reference_kernel).max()

    low, high = 0.05, 10.0
    for _ in range(200):
        m1 = low + (high - low) / 3
        m2 = high - (high - low) / 3
        if deviation(m1) < deviation(m2):
            high = m2
        else:
            low = m1
    length_scale = (low + high) / 2
    return length_scale, deviation(length_scale)


def read_pockets(path):
    """Four-residue pockets from a pipeline selection table."""
    with path.open(newline="", encoding="utf-8") as handle:
        return [row["designed_sequence"] for row in csv.DictReader(handle)]


def panel_statistics(vectors, length_scale, sigma_main, sigma_epi, diagonal):
    """Mean off-diagonal similarity and log-determinant of a set of pockets."""
    matrix = kernel_matrix(vectors, vectors, length_scale, sigma_main, sigma_epi) / diagonal
    n = len(vectors)
    mean_similarity = matrix[np.triu_indices(n, 1)].mean()
    sign, logdet = np.linalg.slogdet(matrix)
    return float(mean_similarity), float(logdet if sign > 0 else np.nan)


def read_tested_variants():
    path = ROOT / "experiments" / "tested_variants.csv"
    designs = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["variant_source"] == "reference_enzyme":
                continue
            pocket = "".join(row[f"aa{p}"] for p in POSITIONS)
            designs.append((row["variant_id"], row["variant_source"],
                            row["panel_variant"].strip().lower() == "true", pocket))
    return designs


def read_measured_activity():
    """20 h specific activities from notebook 06, if it has been run."""
    path = OUT / "lab_variants.csv"
    if not path.exists():
        return {}, f"{path.relative_to(ROOT)} not found -- run notebook 06 first"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}, "lab_variants.csv is empty"
    columns = rows[0].keys()
    id_col = next((c for c in columns if "variant" in c.lower()), None)
    act_col = next((c for c in columns
                    if "20" in c and ("act" in c.lower() or "spec" in c.lower())), None)
    if id_col is None or act_col is None:
        return {}, f"no 20 h activity column found; available columns: {sorted(columns)}"
    activity = {}
    for row in rows:
        try:
            activity[row[id_col]] = float(row[act_col])
        except (TypeError, ValueError):
            continue
    return activity, f"{act_col} from {path.name}"


def ordinal_suffix(value):
    n = int(round(value))
    if 10 <= n % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def spearman(a, b):
    rank_a = np.argsort(np.argsort(a)).astype(float)
    rank_b = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def permutation_p_value(x, y, observed, rng):
    null = np.array([spearman(x, rng.permutation(y)) for _ in range(N_PERMUTATIONS)])
    return float((np.abs(null) >= abs(observed)).mean())


def main():
    keys, descriptors = load_features()
    hyper = json.loads((RUN / "fit" / "model.json").read_text())["hyperparameters"]
    sigma_main = hyper["sigma_main"]
    sigma_epi = hyper["sigma_epi"]
    stored = np.load(RUN / "predict" / "candidate_kernel.npz", allow_pickle=True)["kernel"]

    length_scale, deviation = recover_length_scale(descriptors, stored, sigma_main, sigma_epi)
    assert deviation < 1e-9, (
        f"kernel reconstruction failed (max deviation {deviation:.2e}); "
        "the stored kernel and the hyperparameters are no longer consistent"
    )
    print(f"length scale l = {length_scale:.6f} "
          f"(reconstruction of the stored candidate kernel exact to {deviation:.1e})")

    # residue -> descriptor vector, per position
    lookup = [{} for _ in POSITIONS]
    for row, key in enumerate(keys):
        for p in range(len(POSITIONS)):
            lookup[p].setdefault(key[p], descriptors[row, p, :])

    def embed(pocket):
        return np.stack([lookup[p][aa] for p, aa in enumerate(pocket)])

    designs = read_tested_variants()
    diagonal = sigma_main ** 2 + sigma_epi ** 2  # k(v, v); normalizes similarity to 1
    reference_vectors = {name: embed(pocket) for name, pocket in REFERENCES.items()}

    records = []
    for variant_id, source, in_panel, pocket in designs:
        vector = embed(pocket)
        record = {"variant_id": variant_id, "variant_source": source,
                  "panel_variant": in_panel, "pocket": pocket}
        for name, ref in reference_vectors.items():
            k = kernel_matrix(vector[None, ...], ref[None, ...],
                              length_scale, sigma_main, sigma_epi)[0, 0]
            record[f"similarity_to_{name.replace('NylC-', '').lower()}"] = k / diagonal
        records.append(record)

    # within-panel mean similarity, for comparison with the value reported in 6.2.2
    panel = [embed(p) for _, _, in_panel, p in designs if in_panel]
    panel_matrix = kernel_matrix(np.stack(panel), np.stack(panel),
                                 length_scale, sigma_main, sigma_epi) / diagonal
    off_diagonal = panel_matrix[np.triu_indices(len(panel), 1)]

    ts_hp = kernel_matrix(reference_vectors["NylC-TS"][None, ...],
                          reference_vectors["NylC-HP"][None, ...],
                          length_scale, sigma_main, sigma_epi)[0, 0] / diagonal

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "panel_reference_similarity.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    designs_ts = np.array([r["similarity_to_ts"] for r in records])
    designs_hp = np.array([r["similarity_to_hp"] for r in records])
    panel_ts = np.array([r["similarity_to_ts"] for r in records if r["panel_variant"]])
    panel_hp = np.array([r["similarity_to_hp"] for r in records if r["panel_variant"]])

    print(f"\n{len(records)} designed constructs, {len(panel_ts)} of them drawn by the k-DPP")
    print(f"  mean similarity to NylC-TS: {designs_ts.mean():.3f} (all), "
          f"{panel_ts.mean():.3f} (k-DPP panel)")
    print(f"  mean similarity to NylC-HP: {designs_hp.mean():.3f} (all), "
          f"{panel_hp.mean():.3f} (k-DPP panel)")
    print(f"  mean pairwise similarity within the k-DPP panel: {off_diagonal.mean():.3f}")
    print(f"  similarity between the two reference enzymes: {ts_hp:.3f}")

    # --- diversity of the tested panel against two null sets -------------------
    # Reproduces the third paragraph of Section 6.2.2.
    select_dir = RUN / "select"
    shortlist = read_pockets(select_dir / "shortlist_top_25.csv")
    drawn_panel = read_pockets(select_dir / "panel_k_dpp_shortlist_25.csv")
    tested_pockets = [pocket for _, _, in_panel, pocket in designs if in_panel]
    size = len(tested_pockets)

    tested_mean, tested_logdet = panel_statistics(
        np.stack([embed(p) for p in tested_pockets]),
        length_scale, sigma_main, sigma_epi, diagonal)

    # null set 1: drop one member from the drawn k-DPP panel
    loo = [panel_statistics(np.stack([embed(p) for j, p in enumerate(drawn_panel) if j != i]),
                            length_scale, sigma_main, sigma_epi, diagonal)
           for i in range(len(drawn_panel))]
    loo_similarity = np.array([s for s, _ in loo])
    loo_logdet = np.array([d for _, d in loo])

    # null set 2: random subsets of the same shortlist
    rng_subsets = np.random.default_rng(SEED)
    shortlist_vectors = np.stack([embed(p) for p in shortlist])
    random_means = np.empty(N_RANDOM_SUBSETS)
    for i in range(N_RANDOM_SUBSETS):
        idx = rng_subsets.choice(len(shortlist), size=size, replace=False)
        random_means[i] = panel_statistics(shortlist_vectors[idx], length_scale,
                                           sigma_main, sigma_epi, diagonal)[0]
    percentile = float((random_means <= tested_mean).mean() * 100)

    print(f"\n  tested panel ({size} members): mean similarity {tested_mean:.3f}, "
          f"log-determinant {tested_logdet:.2f}")
    print(f"  leave-one-out subsets of the drawn k-DPP panel (n={len(loo)}): "
          f"similarity {loo_similarity.mean():.3f} "
          f"(range {loo_similarity.min():.3f}-{loo_similarity.max():.3f}), "
          f"log-determinant {loo_logdet.mean():.2f} "
          f"(range {loo_logdet.min():.2f} to {loo_logdet.max():.2f})")
    print(f"  {N_RANDOM_SUBSETS} random {size}-member subsets of the top-{len(shortlist)} "
          f"shortlist: similarity {random_means.mean():.3f} "
          f"(sd {random_means.std(ddof=1):.3f}); the tested panel lies at the "
          f"{percentile:.0f}{ordinal_suffix(percentile)} percentile")

    activity, note = read_measured_activity()
    print(f"\nmeasured activity: {note}")
    if activity:
        shared = [r for r in records if r["variant_id"] in activity]
        if len(shared) >= 5:
            measured = np.array([activity[r["variant_id"]] for r in shared])
            rng = np.random.default_rng(SEED)
            for label, key in (("NylC-TS", "similarity_to_ts"), ("NylC-HP", "similarity_to_hp")):
                values = np.array([r[key] for r in shared])
                rho = spearman(values, measured)
                p = permutation_p_value(values, measured, rho, rng)
                print(f"  similarity to {label} vs 20 h activity: "
                      f"rho = {rho:+.3f}, n = {len(shared)}, permutation p = {p:.3f}")
            print("  NOTE: exploratory; similarity to NylC-TS is confounded with the number "
                  "and the conservativeness of the substitutions.")
        else:
            print("  too few variants matched between the two tables")

    print(f"\nwritten: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
