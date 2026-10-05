"""Cross-platform command line interface."""

import argparse
import logging
from pathlib import Path
from nylc.config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="ML_NylC: reproducible Epistatic GP pipeline")
    parser.add_argument("command", choices=["run", "stage", "check"])
    parser.add_argument(
        "target",
        nargs="?",
        default="report",
        choices=[
            "prepare",
            "features",
            "diagnostics",
            "evaluate",
            "fit",
            "predict",
            "select",
            "baselines",
            "report",
            "figures",
        ],
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--root", default=".")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--threads", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    if not (root / "pyproject.toml").is_file():
        parser.error("--root must point to the ML_NylC project")
    config = load_config(root / args.config)
    if args.device:
        config["runtime"]["device"] = args.device
    if args.threads is not None:
        if args.threads < 1:
            parser.error("--threads must be positive")
        config["runtime"]["threads"] = args.threads
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from nylc.runner import Runner

    runner = Runner(root, config)
    if args.command == "check":
        import json
        from nylc.provenance import sha256_file

        expected = json.loads((root / config["inputs"]["checksums"]).read_text(encoding="utf-8"))
        for role in ("lab_data", "wt_fasta", "candidates"):
            if sha256_file(root / config["inputs"][role]) != expected[role]:
                raise ValueError(f"Input checksum mismatch: {role}")
        print(f"Configuration valid: {config['experiment']}; Python core ready")
        runner.fingerprint("prepare")
        return 0
    if args.command == "stage":
        runner.run_stage(args.target, args.force)
    else:
        runner.run(args.target, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
