from pathlib import Path
import pytest
from nylc.config import load_config
from nylc.data.lab import load_lab, read_reference
from nylc.features.descriptors import residue_descriptor_table
from nylc.models.epistatic_gp import EpistaticGP

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return load_config(ROOT / "configs/smoke.yaml")


@pytest.fixture
def lab():
    return load_lab(ROOT / "inputs/lab_data.csv", read_reference(ROOT / "inputs/WT.fasta"))


@pytest.fixture
def model(config):
    return EpistaticGP(config["gp"], residue_descriptor_table, ["physical"])
