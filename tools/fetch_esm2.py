"""Explicitly provision pinned model weights before offline cluster execution."""

import argparse
from pathlib import Path
from nylc.config import load_config

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", default="configs/esm2.yaml")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
config = load_config(root / args.config)
from huggingface_hub import snapshot_download

seen = set()
zero_shot = config["baseline_comparison"]["zero_shot"]
if config["baseline_comparison"]["enabled"] and zero_shot["enabled"]:
    identity = (zero_shot["model"], zero_shot["revision"])
    print(
        snapshot_download(
            repo_id=identity[0],
            revision=identity[1],
            cache_dir=str(root / "cache/huggingface"),
            allow_patterns=["*.json", "*.txt", "*.safetensors", "pytorch_model.bin"],
        )
    )
    seen.add(identity)
for name, settings in config["features"]["learned"].items():
    if name not in config["features"]["sources"] or settings["kind"] != "esm2":
        continue
    identity = (settings["model"], settings["revision"])
    if identity in seen:
        continue
    seen.add(identity)
    result = snapshot_download(
        repo_id=identity[0],
        revision=identity[1],
        cache_dir=str(root / "cache/huggingface"),
        allow_patterns=["*.json", "*.txt", "*.safetensors", "pytorch_model.bin"],
    )
    print(result)
