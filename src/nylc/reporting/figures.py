"""Headless publication figures and a concise standalone HTML report."""

from html import escape
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def make_report(config, root, output):
    base = root / "results" / config["experiment"]
    pred = pd.read_csv(base / "evaluate/nested_predictions.csv")
    metrics = pd.read_csv(base / "evaluate/nested_metrics.csv")
    coverage = pd.read_csv(base / "evaluate/coverage.csv")
    panel = pd.read_csv(base / "select/lab_test_panel_generated.csv")
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    ax.errorbar(
        pred.observed,
        pred.predicted,
        yerr=1.96 * pred.predicted_std_observed_calibrated,
        fmt="o",
        alpha=0.7,
        capsize=2,
    )
    bounds = [
        min(pred.observed.min(), pred.predicted.min()),
        max(pred.observed.max(), pred.predicted.max()),
    ]
    ax.plot(bounds, bounds, "--", color="gray")
    ax.set(
        xlabel="Observed PA6 activity (source units)",
        ylabel="Predicted PA6 activity",
        title="Epistatic GP: nested validation",
    )
    fig.savefig(output / "nested_predictions.png", dpi=180)
    fig.savefig(output / "nested_predictions.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    with np.load(base / "select/primary_kernel.npz", allow_pickle=False) as stored:
        kernel = stored["kernel"]
        ids = stored["candidate_ids"]
        selected = set(stored["selected_ids"])
    fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
    plot = ax.imshow(kernel, vmin=0, vmax=1, cmap="viridis")
    labels = [f"{name} *" if name in selected else name for name in ids]
    ax.set_xticks(range(len(ids)), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(ids)), labels, fontsize=7)
    ax.set_title("DPP shortlist similarity (* selected)")
    fig.colorbar(plot, ax=ax, label="Normalized prior kernel")
    fig.savefig(output / "dpp_kernel.png", dpi=180)
    fig.savefig(output / "dpp_kernel.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    summary = json.loads((base / "select/summary.json").read_text())
    note = (
        "Reduced smoke configuration: technical verification only."
        if config["data"]["subset_variant_ids"]
        else "Nested outer predictions estimate generalization; deployment calibration is reported separately."
    )
    html = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>ML_NylC — {escape(config["experiment"])}</title>
<style>body{{font-family:system-ui;max-width:1100px;margin:40px auto;padding:0 24px}} table{{border-collapse:collapse;font-size:14px}}td,th{{padding:8px;border-bottom:1px solid #ddd}}img{{max-width:100%}}</style>
<h1>ML_NylC — Epistatic GP</h1><p>{escape(note)}</p>
<h2>Nested validation</h2>{metrics.to_html(index=False)}<img src="nested_predictions.png" alt="Held-out observed and predicted activity">
<h2>Uncertainty coverage</h2>{coverage.to_html(index=False)}
<h2>Generated laboratory panel</h2><p>{summary["n_generated"]} generated candidates; {summary["n_controls"]} controls. Activity floor: {summary["activity_floor"]:.3f}.</p>
{panel[["candidate_id", "mutations", "predicted_activity_gp_mean"]].to_html(index=False)}
<img src="dpp_kernel.png" alt="DPP kernel, aligned to candidate identifiers">
<p>Each stage contains manifest.json with configuration, software versions and artifact checksums.</p></html>"""
    (output / "report.html").write_text(html, encoding="utf-8")
