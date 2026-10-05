"""Additive against epistatic kernel across several descriptor sets.

The model-family comparison in the ``evaluate`` stage holds the representation
fixed at one descriptor set, so it answers whether the pairwise terms help *for
that representation*. On its own it leaves open whether the conclusion depends
on the choice. This module repeats the identical comparison -- the same
function, the same outer split -- for any list of descriptor sets, so the
stability of the result across representations can be reported.

The primary comparison stays whatever ``validation.baseline_descriptor`` names.
This sweep is a robustness analysis beside it, not a replacement: it writes its
own files and never touches ``model_family_metrics.csv``.

Reporting obligation: a sweep is only evidence if every descriptor set that was
run is reported, including any for which the additive kernel wins. Selecting the
favourable ones afterwards is exactly what the nested design exists to prevent.

Runtime grows with the number of descriptor sets; each one repeats a nested
leave-one-variant-out comparison over all variants and both kernels. Run a
single set first and read the reported duration before queueing the rest.

    python -m nylc.validation.kernel_family_sweep --descriptors physical
    python -m nylc.validation.kernel_family_sweep --descriptors vhse z_scales

Results accumulate: descriptor sets from earlier runs are kept, a descriptor set
that is run again replaces its own rows. Both files are rewritten after every
descriptor set, so a failure in a later set cannot discard the ones already
computed.
"""

import argparse
import time
from pathlib import Path

import pandas as pd

METRIC_FILE = "kernel_family_sweep_metrics.csv"
PAIRED_FILE = "kernel_family_sweep_paired.csv"


def compare_one(core, descriptor_set):
    """Run the model-family comparison for a single descriptor set."""
    from nylc.features.descriptors import residue_descriptor_table
    from nylc.validation.comparison import nested_loocv_model_comparison

    source = residue_descriptor_table(descriptor_set)
    started = time.perf_counter()
    predictions, metrics, _ = nested_loocv_model_comparison(core, source)
    elapsed = time.perf_counter() - started

    metrics = metrics.assign(descriptor_set=descriptor_set, seconds=round(elapsed, 1))
    paired = (
        predictions.pivot(index="variant_id", columns="model", values="abs_error")
        .reset_index()
        .assign(descriptor_set=descriptor_set)
    )
    paired["error_reduction_epistatic_vs_additive"] = paired["additive"] - paired["epistatic"]
    return metrics, paired, elapsed


def merge(fresh, path):
    """Keep descriptor sets from earlier runs, replace the ones just computed."""
    if path.is_file():
        previous = pd.read_csv(path)
        kept = previous[~previous["descriptor_set"].isin(fresh["descriptor_set"].unique())]
        fresh = pd.concat([kept, fresh], ignore_index=True)
    keys = [c for c in ("descriptor_set", "model", "variant_id") if c in fresh.columns]
    return fresh.sort_values(keys, ignore_index=True) if keys else fresh


def publish(metric_rows, paired_rows, output):
    """Write everything computed so far, so a later failure costs no runtime."""
    metrics = merge(pd.concat(metric_rows, ignore_index=True), output / METRIC_FILE)
    paired = merge(pd.concat(paired_rows, ignore_index=True), output / PAIRED_FILE)
    metrics.to_csv(output / METRIC_FILE, index=False)
    paired.to_csv(output / PAIRED_FILE, index=False)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=".", help="ML_NylC project directory")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--descriptors",
        nargs="+",
        help="descriptor sets to compare; defaults to every set in features.sources",
    )
    args = parser.parse_args(argv)

    from nylc.config import load_config
    from nylc.data.lab import load_prepared
    from nylc.features.descriptors import DESCRIPTOR_SOURCES

    root = Path(args.root).resolve()
    config = load_config(root / args.config)
    base = root / "results" / config["experiment"]
    core_path = base / "prepare" / "core.csv"
    if not core_path.is_file():
        parser.error(f"No prepared variants at {core_path}; run the prepare stage first")

    requested = args.descriptors or [
        name for name in config["features"]["sources"] if name in DESCRIPTOR_SOURCES
    ]
    unknown = [name for name in requested if name not in DESCRIPTOR_SOURCES]
    if unknown:
        parser.error(f"Unknown descriptor sets: {unknown}")

    core = load_prepared(core_path)
    output = base / "evaluate"
    print(f"Variants: {len(core)} | descriptor sets: {', '.join(requested)}\n")

    metric_rows, paired_rows = [], []
    for index, name in enumerate(requested, start=1):
        print(f"[{index}/{len(requested)}] {name} ...", flush=True)
        metrics, paired, elapsed = compare_one(core, name)
        metric_rows.append(metrics)
        paired_rows.append(paired)
        publish(metric_rows, paired_rows, output)
        summary = metrics.set_index("model")["mae"]
        print(
            f"    MAE epistatic {summary['epistatic']:.2f} | additive {summary['additive']:.2f} "
            f"| {elapsed / 60:.1f} min"
        )
        if index == 1 and len(requested) > 1:
            print(
                f"    estimated total for {len(requested)} sets: "
                f"{elapsed * len(requested) / 60:.0f} min",
                flush=True,
            )

    metrics = publish(metric_rows, paired_rows, output)

    print("\n=== all descriptor sets computed so far ===")
    view = metrics.pivot(index="descriptor_set", columns="model", values="mae")
    view["difference"] = view["additive"] - view["epistatic"]
    print(view.sort_values("epistatic").to_string(float_format=lambda v: f"{v:.2f}"))
    print(f"\nWritten: {output / METRIC_FILE}\n         {output / PAIRED_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
