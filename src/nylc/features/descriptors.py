"""Fixed amino-acid descriptor tables; standardization uses the 20-AA alphabet."""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from peptides import Peptide
from sklearn.preprocessing import StandardScaler

CANONICAL_AA = tuple("ACDEFGHIKLMNPQRSTVWY")

WT_POCKET = {99: "D", 134: "F", 304: "D", 330: "R"}

POSITIONS = tuple(WT_POCKET.keys())

TARGET_POSITIONS = set(POSITIONS)

DESCRIPTOR_SOURCES = {
    "kidera": {
        "method": "kidera_factors",
        "columns": [f"KF{i}" for i in range(1, 11)],
        "description": "Kidera factors; 10-dimensional physicochemical factors.",
    },
    "vhse": {
        "method": "vhse_scales",
        "columns": [f"VHSE{i}" for i in range(1, 9)],
        "description": "VHSE scales; hydrophobic, steric and electronic descriptors.",
    },
    "z_scales": {
        "method": "z_scales",
        "columns": [f"Z{i}" for i in range(1, 6)],
        "description": "Z-scales; compact amino-acid physicochemical descriptors.",
    },
    "atchley": {
        "method": "atchley_factors",
        "columns": [f"AF{i}" for i in range(1, 6)],
        "description": "Atchley factors; multidimensional amino-acid descriptors.",
    },
    "pcp": {
        "method": "pcp_descriptors",
        "columns": [f"PCP{i}" for i in range(1, 6)],
        "description": "Physical-chemical property descriptors from peptides.py.",
    },
    "physical": {
        "method": "physical_descriptors",
        "columns": ["PD1", "PD2"],
        "description": "Two physical descriptors; PD1 relates to volume, PD2 to hydrophilicity.",
    },
}


@dataclass(frozen=True)
class DescriptorSource:
    name: str
    method: str
    description: str
    raw_table: pd.DataFrame
    scaled_table: pd.DataFrame


def residue_descriptor_table(source_name, standardize=True):
    if source_name not in DESCRIPTOR_SOURCES:
        raise KeyError(f"Unknown descriptor source: {source_name}")
    cfg = DESCRIPTOR_SOURCES[source_name]
    rows = {}
    for aa in CANONICAL_AA:
        values = np.asarray(getattr(Peptide(aa), cfg["method"])(), dtype=float)
        if values.shape[0] != len(cfg["columns"]):
            raise ValueError(f"Descriptor length mismatch for {source_name}:{aa}")
        rows[aa] = values
    raw = pd.DataFrame.from_dict(rows, orient="index", columns=cfg["columns"]).sort_index()
    if standardize:
        scaled = pd.DataFrame(
            StandardScaler().fit_transform(raw), index=raw.index, columns=raw.columns
        )
    else:
        scaled = raw.copy()
    return DescriptorSource(source_name, cfg["method"], cfg["description"], raw, scaled)
