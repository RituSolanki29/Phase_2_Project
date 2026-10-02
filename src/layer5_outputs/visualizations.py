"""
Result Visualizations (Layer 5)
Generates PNG charts from a completed pipeline run: carbon-intensity curves,
NSG distributions, decision-space scatter, predictor quality, feature
importance, and (when a time-sweep was run) a full diurnal comparison of
scheduling behaviour across the day.

All charts are saved to <output_dir>/plots/. Matplotlib's non-interactive
"Agg" backend is used so this runs headless (no display needed).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_carbon_intensity_curve(carbon_feed, regions, output_dir: Path, hours: int = 24):
    """24-hour carbon-intensity curve per region, from the same model the
    pipeline actually uses (not a re-derived approximation)."""
    timestamps = np.linspace(0, 86400, hours * 4)  # 15-min resolution
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for region in regions:
        values = [carbon_feed.get_carbon_intensity(region, t) for t in timestamps]
        ax.plot(timestamps / 3600.0, values, label=region, linewidth=2)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Carbon intensity (gCO2/kWh)")
    ax.set_title("Simulated regional carbon-intensity curves (24h)")
    ax.set_xticks(range(0, 25, 2))
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, output_dir / "carbon_intensity_curve.png")


def plot_nsg_distribution(results_df: pd.DataFrame, output_dir: Path):
    """Histogram of mean NSG per candidate, split by viable vs rejected."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    viable = results_df[results_df["is_statistically_viable"]]["nsg_mean_gco2"]
    rejected = results_df[~results_df["is_statistically_viable"]]["nsg_mean_gco2"]
    bins = np.linspace(results_df["nsg_mean_gco2"].min(), results_df["nsg_mean_gco2"].max(), 40)
    ax.hist(rejected, bins=bins, alpha=0.6, label=f"Rejected (n={len(rejected)})", color="#d94f4f")
    ax.hist(viable, bins=bins, alpha=0.7, label=f"Viable (n={len(viable)})", color="#2f9e5f")
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Mean Net Sustainability Gain (gCO2)")
    ax.set_ylabel("Candidate count")
    ax.set_title("Distribution of candidate NSG, viable vs rejected")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, output_dir / "nsg_distribution.png")


def plot_decision_space(results_df: pd.DataFrame, output_dir: Path,
                         nsg_threshold: float = 0.90, sla_threshold: float = 0.80):
    """Scatter of every candidate in (P(NSG>0), SLA risk) space, with the
    two decision-gate lines drawn in, so the accept/reject boundary is
    visible directly rather than just reported as a count."""
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = results_df["is_statistically_viable"].map({True: "#2f9e5f", False: "#d94f4f"})
    ax.scatter(results_df["p_nsg_positive"], results_df["sla_risk_score"],
               c=colors, alpha=0.6, s=35, edgecolors="none")
    ax.axvline(nsg_threshold, color="black", linestyle="--", linewidth=1)
    ax.axhline(sla_threshold, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("P(NSG > 0)")
    ax.set_ylabel("SLA risk score")
    ax.set_title("Decision space: acceptance gate vs. candidate outcomes")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#2f9e5f", label="Viable"), Patch(color="#d94f4f", label="Rejected")])
    ax.grid(alpha=0.3)
    _save(fig, output_dir / "decision_space.png")


def plot_predictor_quality(pods, predictor, output_dir: Path):
    """Predicted vs. actual duration scatter, on a held-out-style sample,
    to visualize GBR model quality directly rather than only via R²/RMSE."""
    sample = pods[-min(300, len(pods)):]
    actual = [p.duration_sec for p in sample]
    predicted = [predictor.predict(p)[0] for p in sample]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual, predicted, alpha=0.5, s=25, color="#3b6fd6")
    lims = [0, max(max(actual), max(predicted)) * 1.05]
    ax.plot(lims, lims, color="black", linestyle="--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual duration (s)")
    ax.set_ylabel("Predicted duration (s)")
    ax.set_title("GBR predicted vs. actual execution time")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, output_dir / "predictor_quality.png")


def plot_feature_importance(feature_importance: dict, output_dir: Path):
    if not feature_importance:
        return
    names = list(feature_importance.keys())
    values = [feature_importance[n] for n in names]
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh([names[i] for i in order], [values[i] for i in order], color="#3b6fd6")
    ax.set_xlabel("Feature importance")
    ax.set_title("GBR feature importance")
    ax.grid(alpha=0.3, axis="x")
    _save(fig, output_dir / "feature_importance.png")


def plot_hourly_sweep(hourly_df: pd.DataFrame, output_dir: Path):
    """Only produced when the pipeline was run in --time-sweep-hours mode:
    shows mean NSG and viable-% evolving across the day, which is the chart
    that actually demonstrates dynamic, time-varying scheduling behaviour
    rather than two isolated snapshots."""
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(hourly_df["hour"], hourly_df["mean_nsg_gco2"], color="#2f9e5f", marker="o", label="Mean NSG (gCO2)")
    ax1.set_xlabel("Hour of day")
    ax1.set_ylabel("Mean NSG (gCO2)", color="#2f9e5f")
    ax1.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    ax1.tick_params(axis="y", labelcolor="#2f9e5f")

    ax2 = ax1.twinx()
    ax2.plot(hourly_df["hour"], hourly_df["viable_percentage"], color="#d97e2f", marker="s", label="Viable %")
    ax2.set_ylabel("Viable candidates (%)", color="#d97e2f")
    ax2.tick_params(axis="y", labelcolor="#d97e2f")

    ax1.set_title("Scheduling outcomes across a 24h carbon-intensity cycle")
    ax1.set_xticks(range(0, 24, 2))
    fig.legend(loc="upper left", bbox_to_anchor=(0.08, 0.92))
    _save(fig, output_dir / "hourly_sweep.png")


def generate_all_visualizations(
    results_df: pd.DataFrame,
    pods,
    predictor,
    feature_importance: dict,
    carbon_feed,
    regions,
    output_dir: Path,
    hourly_df: pd.DataFrame = None,
):
    """Generates every chart this module knows how to make from one run's
    outputs. Never raises: a plotting failure should not fail the pipeline."""
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    steps = [
        ("carbon_intensity_curve.png", lambda: plot_carbon_intensity_curve(carbon_feed, regions, plots_dir)),
        ("nsg_distribution.png", lambda: plot_nsg_distribution(results_df, plots_dir)),
        ("decision_space.png", lambda: plot_decision_space(results_df, plots_dir)),
        ("predictor_quality.png", lambda: plot_predictor_quality(pods, predictor, plots_dir)),
        ("feature_importance.png", lambda: plot_feature_importance(feature_importance, plots_dir)),
    ]
    if hourly_df is not None and not hourly_df.empty:
        steps.append(("hourly_sweep.png", lambda: plot_hourly_sweep(hourly_df, plots_dir)))

    for name, fn in steps:
        try:
            fn()
            generated.append(str(plots_dir / name))
        except Exception as exc:
            print(f"  [visualization warning] skipped {name}: {exc}")
    return generated
