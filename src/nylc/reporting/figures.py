"""Headless publication figures and a concise standalone HTML report."""

from html import escape
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def save_figure(fig, output, name):
    """Every figure has a raster preview and a vector publication export."""
    fig.savefig(output / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(
        output / f"{name}.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )


def baseline_figure(metrics, output):
    ordered = metrics.sort_values("mae", na_position="last")
    activity = ordered.loc[ordered.mae.notna()]
    fig, axes = plt.subplots(1, 2, figsize=(17, max(5, len(ordered) * 0.45)), layout="constrained")
    for ax, table, value, title in [
        (axes[0], activity, "mae", "Prediction error"),
        (axes[1], ordered, "spearman", "Variant ranking"),
    ]:
        colors = ["#2f6f73" if name.startswith("gp_") else "#8a8f98" for name in table.model]
        ax.barh(table.model, table[value], color=colors)
        ax.invert_yaxis()
        ax.set_title(title)
    axes[0].set_xlabel("Outer-LOOCV MAE (lower is better)")
    axes[1].set_xlabel("Spearman correlation (higher is better)")
    axes[1].axvline(0, color="black", linewidth=0.8)
    fig.suptitle("NylC: Epistatic GP and baseline models")
    save_figure(fig, output, "model_comparison_mae_spearman")
    plt.close(fig)


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
    save_figure(fig, output, "nested_predictions")
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
    save_figure(fig, output, "dpp_kernel")
    plt.close(fig)
    summary = json.loads((base / "select/summary.json").read_text())
    note = (
        "Reduced smoke configuration: technical verification only."
        if config["data"]["subset_variant_ids"]
        else "Nested outer predictions estimate generalization; deployment calibration is reported separately."
    )
    baseline_html = ""
    if config["baseline_comparison"]["enabled"]:
        baseline_metrics = pd.read_csv(base / "baselines/all_loocv_metrics.csv")
        baseline_figure(baseline_metrics, output)
        baseline_html = (
            "<h2>Baseline notebook comparison</h2>"
            "<p>Hyperparameter grids use nested LOOCV; fixed models use outer LOOCV. "
            "Direct zero-shot scores are ranked only and have no activity-unit error metrics. "
            "The best observed model is descriptive, not an independently validated winner.</p>"
            + baseline_metrics.to_html(index=False, na_rep="N/A")
            + '<img src="model_comparison_mae_spearman.png" alt="Baseline error and rank comparison">'
            + '<p><a href="model_comparison_mae_spearman.pdf">Download comparison PDF</a></p>'
        )
    html = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>ML_NylC — {escape(config["experiment"])}</title>
<style>body{{font-family:system-ui;max-width:1100px;margin:40px auto;padding:0 24px}} table{{border-collapse:collapse;font-size:14px}}td,th{{padding:8px;border-bottom:1px solid #ddd}}img{{max-width:100%}}</style>
<h1>ML_NylC — Epistatic GP</h1><p>{escape(note)}</p>
<h2>Nested validation</h2>{metrics.to_html(index=False)}<img src="nested_predictions.png" alt="Held-out observed and predicted activity">
<h2>Uncertainty coverage</h2>{coverage.to_html(index=False)}
{baseline_html}
<h2>Generated laboratory panel</h2><p>{summary["n_generated"]} generated candidates; {summary["n_controls"]} controls. Activity floor: {summary["activity_floor"]:.3f}.</p>
{panel[["candidate_id", "mutations", "predicted_activity_gp_mean"]].to_html(index=False)}
<img src="dpp_kernel.png" alt="DPP kernel, aligned to candidate identifiers">
<p>Each stage contains manifest.json with configuration, software versions and artifact checksums.</p></html>"""
    (output / "report.html").write_text(html, encoding="utf-8")
