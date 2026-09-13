"""Explicit download of the pinned TabICL checkpoint before offline execution."""

import argparse
from pathlib import Path
import shutil
from huggingface_hub import hf_hub_download
from nylc.config import load_config
from nylc.provenance import sha256_file

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", default="configs/baselines-full.yaml")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
settings = load_config(root / args.config)["baseline_comparison"]["tabicl"]
source = Path(
    hf_hub_download(
        repo_id="jingang/TabICL",
        filename="tabicl-regressor-v2-20260212.ckpt",
        revision=settings["revision"],
        cache_dir=str(root / "cache/huggingface"),
    )
)
if sha256_file(source) != settings["sha256"]:
    raise ValueError("Downloaded TabICL checkpoint checksum mismatch")
destination = root / settings["checkpoint"]
destination.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(source, destination)
print(destination)
