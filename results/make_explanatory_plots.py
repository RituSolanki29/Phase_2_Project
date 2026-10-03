"""
Explanatory Results Visualizer
Generates 10 publication-ready charts that are easy to understand and present:
  1.  Predicted vs Realized NSG  (Digital Twin accuracy)
  2.  Before vs After PPO Feedback  (learning improvement)
  3.  Cumulative Carbon Saved  (overall benefit over time)
  4.  NSG per Decision colored by Action  (MIGRATE/START/DELAY story)
  5.  SLA Risk Timeline with hard-gate threshold
  6.  Action Distribution Pie  (what the agent does)
  7.  Per-Cycle Mean NSG  (multi-epoch training trend)
  8.  Carbon Intensity vs NSG Correlation  (why CI drives decisions)
  9.  NSG Confidence vs SLA Risk (full decision landscape)
  10. 24-Hour Scheduling Behaviour (CI vs Viable % vs NSG overlay)

Run:
    python results/make_explanatory_plots.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

REWARD_LOG   = ROOT / "reward_log.jsonl"
HOURLY_CSV   = ROOT / "hourly_summary.csv"
L5_CSV       = ROOT / "layer5_candidate_results.csv"

COLORS = {"MIGRATE": "#2ca02c", "START": "#1f77b4", "DELAY": "#d62728"}

def _save(fig, name):
    path = PLOTS_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path}")
    return path

# ── Load reward log ────────────────────────────────────────────────────────
entries = []
if REWARD_LOG.exists():
    with open(REWARD_LOG, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass

decisions = {e["pod_id"]: e for e in entries if e.get("event") == "decision"}
outcomes  = {e["pod_id"]: e for e in entries if e.get("event") == "outcome"}

rows = []
for pid, d in decisions.items():
    cand = d.get("candidate", {})
    real = float(outcomes[pid]["realized_nsg_gco2"]) if pid in outcomes else None
    rows.append({
        "pod_id":        pid,
        "action":        d.get("action", "UNKNOWN"),
        "predicted_nsg": float(cand.get("nsg_mean_gco2", 0)),
        "p_nsg":         float(cand.get("p_nsg_positive", 0)),
        "sla_risk":      float(cand.get("sla_risk_score", 0)),
        "realized_nsg":  real,
        "cycle":         d.get("context", {}).get("cycle", 0),
        "ts":            d.get("timestamp", 0),
    })

df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
df["decision_idx"] = np.arange(1, len(df) + 1)
matched = df.dropna(subset=["realized_nsg"])

print(f"Total decisions: {len(df)} | Matched with outcomes: {len(matched)}")

# ── Load L5 and hourly CSVs ───────────────────────────────────────────────
l5 = pd.read_csv(L5_CSV) if L5_CSV.exists() else pd.DataFrame()
hourly = pd.read_csv(HOURLY_CSV) if HOURLY_CSV.exists() else pd.DataFrame()

# ==========================================================================
# 1. Predicted vs Realized NSG  (Digital Twin Accuracy)
# ==========================================================================
def chart_predicted_vs_realized():
    m = matched[matched["action"] == "MIGRATE"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: scatter
    ax = axes[0]
    colors = [COLORS.get(a, "grey") for a in m["action"]]
    ax.scatter(m["predicted_nsg"], m["realized_nsg"], color="#1f77b4",
               s=60, alpha=0.8, edgecolors="white", linewidths=0.5, zorder=3)
    lim_max = max(m["predicted_nsg"].max(), m["realized_nsg"].max()) * 1.1
    lim_min = min(m["predicted_nsg"].min(), m["realized_nsg"].min()) * 0.9
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=1.5, label="Perfect prediction")
    # ±10% band
    ax.fill_between([lim_min, lim_max],
                    [x * 0.90 for x in [lim_min, lim_max]],
                    [x * 1.10 for x in [lim_min, lim_max]],
                    alpha=0.12, color="grey", label="±10% band")
    ax.set_xlabel("Predicted NSG — Digital Twin (gCO₂)", fontsize=11)
    ax.set_ylabel("Realized NSG — Measured Outcome (gCO₂)", fontsize=11)
    ax.set_title("Digital Twin Accuracy:\nPredicted vs Realized NSG", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Compute error
    err = (m["realized_nsg"] - m["predicted_nsg"]).abs()
    mae = err.mean()
    corr = m["predicted_nsg"].corr(m["realized_nsg"])
    ax.text(0.04, 0.95, f"MAE = {mae:.5f} gCO₂\nPearson r = {corr:.3f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))

    # Right: error distribution
    ax2 = axes[1]
    errors = m["realized_nsg"] - m["predicted_nsg"]
    ax2.hist(errors, bins=15, color="#1f77b4", alpha=0.7, edgecolor="white")
    ax2.axvline(0, color="black", linewidth=1.5, linestyle="--")
    ax2.axvline(errors.mean(), color="#d62728", linewidth=1.5, linestyle="-",
                label=f"Mean error = {errors.mean():+.5f}")
    ax2.set_xlabel("Prediction Error (Realized − Predicted) gCO₂", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Prediction Error Distribution\n(Bias & Spread of Digital Twin)", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    _save(fig, "1_predicted_vs_realized_nsg.png")

# ==========================================================================
# 2. Before vs After PPO Feedback
# ==========================================================================
def chart_before_after_feedback():
    cyc = matched[matched["cycle"] > 0]
    if cyc.empty:
        print("  ⚠ No cycle data, skipping before/after chart")
        return

    # Treat cycle=1 as "before" (first epoch) and cycle=3 as "after" (final)
    before_migrate = cyc[(cyc["cycle"] == 1) & (cyc["action"] == "MIGRATE")]["realized_nsg"]
    after_migrate  = cyc[(cyc["cycle"] == 3) & (cyc["action"] == "MIGRATE")]["realized_nsg"]

    cycles = cyc.groupby("cycle").agg(
        mean_nsg=("realized_nsg", "mean"),
        migrate_count=("action", lambda x: (x == "MIGRATE").sum()),
        delay_count=("action", lambda x: (x == "DELAY").sum()),
    ).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Left: Per-cycle mean realized NSG bar
    ax = axes[0]
    cycle_labels = [f"Cycle {int(c)}" for c in cycles["cycle"]]
    bars = ax.bar(cycle_labels, cycles["mean_nsg"], color=["#aec7e8", "#6baed6", "#08519c"],
                  edgecolor="white", width=0.5)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.0003,
                f"{b.get_height():.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean Realized NSG (gCO₂)", fontsize=11)
    ax.set_title("Policy Improvement Across\nRetraining Cycles", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")

    # Middle: MIGRATE count per cycle (shows policy becoming more decisive)
    ax2 = axes[1]
    x = np.arange(len(cycles))
    ax2.bar(x - 0.2, cycles["migrate_count"], 0.35, label="MIGRATE", color="#2ca02c", alpha=0.85)
    ax2.bar(x + 0.2, cycles["delay_count"],   0.35, label="DELAY",   color="#d62728", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(cycle_labels)
    ax2.set_ylabel("Decision Count", fontsize=11)
    ax2.set_title("Agent Action Profile\nBefore vs After Feedback", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, axis="y")

    # Right: box-whisker before vs after for MIGRATE outcomes
    ax3 = axes[2]
    data_to_plot = [before_migrate.values, after_migrate.values] if len(before_migrate) and len(after_migrate) else [np.array([0]), np.array([0])]
    bp = ax3.boxplot(data_to_plot, patch_artist=True, widths=0.4,
                     labels=["Cycle 1\n(Before)", "Cycle 3\n(After)"])
    colors_box = ["#aec7e8", "#08519c"]
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax3.set_ylabel("Realized NSG — MIGRATE decisions (gCO₂)", fontsize=10)
    ax3.set_title("NSG Distribution:\nBefore vs After Policy Retraining", fontsize=12, fontweight="bold")
    ax3.grid(alpha=0.3, axis="y")

    _save(fig, "2_before_after_feedback.png")

# ==========================================================================
# 3. Cumulative Carbon Saved Over Time
# ==========================================================================
def chart_cumulative_carbon():
    df2 = matched.copy()
    df2["carbon_saved"] = df2["realized_nsg"].clip(lower=0)
    df2["cumulative"]   = df2["carbon_saved"].cumsum()

    fig, ax = plt.subplots(figsize=(10, 5))

    # Background shading by cycle
    cycle_colors = {1: "#e8f5e9", 2: "#c8e6c9", 3: "#a5d6a7"}
    if "cycle" in df2.columns:
        for cyc, color in cycle_colors.items():
            cyc_rows = df2[df2["cycle"] == cyc]
            if len(cyc_rows):
                ax.axvspan(cyc_rows["decision_idx"].min() - 0.5,
                           cyc_rows["decision_idx"].max() + 0.5,
                           alpha=0.3, color=color, label=f"Cycle {cyc}" if cyc == 1 else "")

    # Cumulative line
    ax.fill_between(df2["decision_idx"], df2["cumulative"], alpha=0.2, color="#1f77b4")
    ax.plot(df2["decision_idx"], df2["cumulative"], color="#1f77b4", linewidth=2.5,
            marker="o", markersize=4, label="Cumulative carbon saved")

    # Mark MIGRATE decisions
    migrate = df2[df2["action"] == "MIGRATE"]
    ax.scatter(migrate["decision_idx"], migrate["cumulative"],
               color="#2ca02c", s=70, zorder=5, label="MIGRATE decision", marker="^")

    # DELAY markers
    delay = df2[df2["action"] == "DELAY"]
    if len(delay):
        ax.scatter(delay["decision_idx"], delay["cumulative"],
                   color="#d62728", s=70, zorder=5, label="DELAY (SLA gate)", marker="x")

    ax.set_xlabel("Decision Number", fontsize=12)
    ax.set_ylabel("Cumulative Carbon Saved (gCO₂)", fontsize=12)
    ax.set_title("Cumulative Net Carbon Savings\nAcross All Agent Decisions", fontsize=13, fontweight="bold")

    # Annotate final total
    final = df2["cumulative"].iloc[-1]
    ax.annotate(f"Total: {final:.5f} gCO₂",
                xy=(df2["decision_idx"].iloc[-1], final),
                xytext=(-60, -25), textcoords="offset points",
                fontsize=11, fontweight="bold", color="#1f77b4",
                arrowprops=dict(arrowstyle="->", color="#1f77b4"))

    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    _save(fig, "3_cumulative_carbon_saved.png")

# ==========================================================================
# 4. NSG per Decision, colored by Action type
# ==========================================================================
def chart_nsg_per_decision():
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    for action, color in COLORS.items():
        sub = df[df["action"] == action]
        ax.scatter(sub["decision_idx"], sub["predicted_nsg"],
                   color=color, s=55, alpha=0.85, label=action, zorder=3, edgecolors="white")

    # Overlay realized if available
    ax.scatter(matched["decision_idx"], matched["realized_nsg"],
               color="black", s=20, alpha=0.5, marker="x", label="Realized (measured)", zorder=4)

    ax.axhline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.6)
    ax.set_ylabel("NSG (gCO₂)", fontsize=12)
    ax.set_title("Net Sustainability Gain per Decision\ncolored by Agent Action  (× = measured outcome)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, ncol=4)
    ax.grid(alpha=0.3)

    # Bottom panel: SLA risk
    ax2 = axes[1]
    ax2.bar(df["decision_idx"], df["sla_risk"], color=[COLORS.get(a, "grey") for a in df["action"]],
            alpha=0.7, width=0.6)
    ax2.axhline(0.8, color="red", linewidth=1.5, linestyle="--", label="Hard gate (0.80)")
    ax2.set_xlabel("Decision Number", fontsize=12)
    ax2.set_ylabel("SLA Risk", fontsize=12)
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, axis="y")

    _save(fig, "4_nsg_per_decision_with_sla.png")

# ==========================================================================
# 5. SLA Risk Timeline
# ==========================================================================
def chart_sla_risk_timeline():
    fig, ax = plt.subplots(figsize=(11, 4.5))

    zone_colors = {
        "MIGRATE": "#2ca02c",
        "START":   "#1f77b4",
        "DELAY":   "#d62728",
    }
    for _, row in df.iterrows():
        c = zone_colors.get(row["action"], "grey")
        ax.bar(row["decision_idx"], row["sla_risk"], color=c, alpha=0.75, width=0.7, edgecolor="none")

    ax.axhline(0.80, color="red", linewidth=2, linestyle="--", label="Hard SLA gate (0.80) — DELAY fires above this")
    ax.axhline(0.50, color="orange", linewidth=1, linestyle=":", alpha=0.7, label="50% risk reference")

    # Shade the danger zone
    ax.fill_between([0.4, len(df) + 0.6], [0.80, 0.80], [1.05, 1.05],
                    color="red", alpha=0.06, label="Danger zone — gate blocks dispatch")

    ax.set_xlim(0.4, len(df) + 0.6)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Decision Number", fontsize=12)
    ax.set_ylabel("SLA Risk Score", fontsize=12)
    ax.set_title("SLA Risk per Decision with Hard Safety Gate\n"
                 "(When risk ≥ 0.80 the RL policy is overridden → DELAY)",
                 fontsize=12, fontweight="bold")

    legend_patches = [mpatches.Patch(color=c, label=a) for a, c in zone_colors.items()]
    gate_line = plt.Line2D([0], [0], color="red", linewidth=2, linestyle="--", label="Hard gate (0.80)")
    ax.legend(handles=legend_patches + [gate_line], fontsize=9, ncol=4)
    ax.grid(alpha=0.3, axis="y")
    _save(fig, "5_sla_risk_timeline.png")

# ==========================================================================
# 6. Agent Action Distribution (Pie + Bar)
# ==========================================================================
def chart_action_distribution():
    counts = df["action"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Pie
    ax = axes[0]
    pie_colors = [COLORS.get(a, "grey") for a in counts.index]
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=counts.index, colors=pie_colors,
        autopct="%1.0f%%", startangle=140, pctdistance=0.8,
        wedgeprops=dict(edgecolor="white", linewidth=1.5)
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
    ax.set_title("Agent Action Distribution\n(All Decisions)", fontsize=12, fontweight="bold")

    # Bar with mean NSG
    ax2 = axes[1]
    action_stats = df.groupby("action").agg(
        mean_pred_nsg=("predicted_nsg", "mean"),
        count=("action", "count")
    ).reindex(["MIGRATE", "START", "DELAY"])

    bar_colors = [COLORS.get(a, "grey") for a in action_stats.index]
    bars = ax2.bar(action_stats.index, action_stats["mean_pred_nsg"],
                   color=bar_colors, edgecolor="white", width=0.45, alpha=0.85)
    for b in bars:
        ax2.text(b.get_x() + b.get_width()/2,
                 max(b.get_height(), 0) + 0.0002,
                 f"{b.get_height():.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Mean Predicted NSG (gCO₂)", fontsize=11)
    ax2.set_title("Mean Carbon Benefit by Action\n(MIGRATE = positive NSG, START/DELAY = zero)",
                  fontsize=11, fontweight="bold")
    ax2.axhline(0, color="black", linewidth=1, linestyle="--")
    ax2.grid(alpha=0.3, axis="y")

    # Add count annotations
    for i, (idx, row) in enumerate(action_stats.iterrows()):
        ax2.text(i, -0.0012, f"n={int(row['count'])}", ha="center",
                 fontsize=9, color="grey")

    _save(fig, "6_action_distribution.png")

# ==========================================================================
# 7. Carbon Intensity vs NSG Correlation
# ==========================================================================
def chart_ci_vs_nsg():
    if l5.empty or hourly.empty:
        print("  ⚠ No L5 / hourly data, skipping CI vs NSG chart")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: 24h CI of all 3 regions vs mean NSG per hour
    ax = axes[0]
    ax.plot(hourly["hour"], hourly["DC_EU_ci"], color="#1f77b4", linewidth=2, label="DC_EU CI")
    ax.plot(hourly["hour"], hourly["DC_US_ci"], color="#ff7f0e", linewidth=2, label="DC_US CI")
    ax.plot(hourly["hour"], hourly["DC_ASIA_ci"], color="#d62728", linewidth=2, label="DC_ASIA CI")
    ax.set_xlabel("Hour of Day", fontsize=11)
    ax.set_ylabel("Carbon Intensity (gCO₂/kWh)", fontsize=11)
    ax.set_title("24h Regional Carbon Intensity\n(Drives agent's target selection)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=0.3)

    # Right: CI gap (ASIA-EU) vs NSG achieved
    ax2 = axes[1]
    hourly["ci_gap_asia_eu"] = hourly["DC_ASIA_ci"] - hourly["DC_EU_ci"]
    sc = ax2.scatter(hourly["ci_gap_asia_eu"], hourly["mean_nsg_gco2"],
                     c=hourly["hour"], cmap="twilight", s=70, alpha=0.8,
                     edgecolors="white", linewidths=0.5, zorder=3)
    cbar = plt.colorbar(sc, ax=ax2)
    cbar.set_label("Hour of Day", fontsize=9)
    # Fit trend line
    p = np.polyfit(hourly["ci_gap_asia_eu"], hourly["mean_nsg_gco2"], 1)
    xs = np.linspace(hourly["ci_gap_asia_eu"].min(), hourly["ci_gap_asia_eu"].max(), 100)
    ax2.plot(xs, np.polyval(p, xs), "k--", linewidth=1.5, label="Linear fit")
    corr = hourly["ci_gap_asia_eu"].corr(hourly["mean_nsg_gco2"])
    ax2.set_xlabel("CI Gap: DC_ASIA − DC_EU (gCO₂/kWh)", fontsize=11)
    ax2.set_ylabel("Mean NSG Achieved (gCO₂)", fontsize=11)
    ax2.set_title(f"Wider CI Gap → Higher Carbon Saving\n(r = {corr:.2f})",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    _save(fig, "7_carbon_intensity_vs_nsg.png")

# ==========================================================================
# 8. Full Decision Landscape (P(NSG>0) × SLA risk, sized by NSG magnitude)
# ==========================================================================
def chart_decision_landscape():
    if l5.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 7))

    cmap = {"DC_EU": "#2ca02c", "DC_US": "#1f77b4", "DC_ASIA": "#d62728"}
    for region, grp in l5.groupby("target_region"):
        sizes = (grp["nsg_mean_gco2"].abs() * 5000).clip(lower=20, upper=300)
        ax.scatter(grp["p_nsg_positive"], grp["sla_risk_score"],
                   color=cmap.get(region, "grey"), s=sizes, alpha=0.55,
                   edgecolors="white", linewidths=0.4, label=region, zorder=3)

    # Gate lines
    ax.axvline(0.90, color="black", linestyle="--", linewidth=1.5, label="P(NSG>0) ≥ 0.90 gate")
    ax.axhline(0.80, color="red",   linestyle="--", linewidth=1.5, label="SLA risk < 0.80 gate")

    # Quadrant labels
    ax.text(0.93, 0.05, "✅ ACCEPT\n(MIGRATE)", color="#2ca02c", fontsize=10,
            fontweight="bold", transform=ax.transAxes, ha="right", va="bottom")
    ax.text(0.93, 0.90, "🚫 DELAY\n(SLA breach)", color="#d62728", fontsize=10,
            fontweight="bold", transform=ax.transAxes, ha="right", va="top")
    ax.text(0.02, 0.05, "❌ REJECT\n(low confidence)", color="grey", fontsize=10,
            fontweight="bold", transform=ax.transAxes, va="bottom")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.08)
    ax.set_xlabel("P(NSG > 0)  — statistical confidence green move saves carbon", fontsize=11)
    ax.set_ylabel("SLA Risk Score  — probability of breaching deadline", fontsize=11)
    ax.set_title("Full Decision Landscape\n(bubble size = |NSG|, color = target region)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    _save(fig, "8_full_decision_landscape.png")

# ==========================================================================
# 9. 24h Scheduling Behaviour Triple Overlay
# ==========================================================================
def chart_24h_triple():
    if hourly.empty:
        return
    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Primary: CI for all regions (shaded)
    ax1.fill_between(hourly["hour"], hourly["DC_EU_ci"], alpha=0.15, color="#1f77b4")
    ax1.plot(hourly["hour"], hourly["DC_EU_ci"],  color="#1f77b4", linewidth=2, label="DC_EU CI")
    ax1.plot(hourly["hour"], hourly["DC_US_ci"],  color="#ff7f0e", linewidth=1.5, linestyle="-.", label="DC_US CI")
    ax1.plot(hourly["hour"], hourly["DC_ASIA_ci"], color="#d62728", linewidth=1.5, linestyle=":", label="DC_ASIA CI")
    ax1.set_xlabel("Hour of Day", fontsize=12)
    ax1.set_ylabel("Carbon Intensity (gCO₂/kWh)", fontsize=11, color="#333333")
    ax1.set_xticks(range(0, 24, 2))
    ax1.grid(alpha=0.25)
    ax1.tick_params(axis="y")

    # Secondary: NSG achieved
    ax2 = ax1.twinx()
    ax2.plot(hourly["hour"], hourly["mean_nsg_gco2"] * 1000,
             color="#2ca02c", linewidth=2.5, marker="o", markersize=5,
             label="Mean NSG (×1000, right axis)")
    ax2.fill_between(hourly["hour"], hourly["mean_nsg_gco2"] * 1000,
                     alpha=0.2, color="#2ca02c")
    ax2.set_ylabel("Mean NSG × 1000 (gCO₂)", fontsize=11, color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")

    # Annotate peak
    peak_h = hourly.loc[hourly["mean_nsg_gco2"].idxmax()]
    ax2.annotate(f"Peak NSG\nHour {int(peak_h.hour):02d}:00",
                 xy=(peak_h.hour, peak_h.mean_nsg_gco2 * 1000),
                 xytext=(peak_h.hour + 1.5, peak_h.mean_nsg_gco2 * 1000 * 0.92),
                 fontsize=9, color="#2ca02c", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.2))

    # Collect legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")

    ax1.set_title("24-Hour Autonomous Scheduling Behaviour\n"
                  "Low DC_EU CI (solar peak) → Higher NSG → More MIGRATE decisions",
                  fontsize=12, fontweight="bold")
    _save(fig, "9_24h_scheduling_behaviour.png")

# ==========================================================================
# 10. Summary Dashboard  (2×3 mini-panels, good for a single slide)
# ==========================================================================
def chart_summary_dashboard():
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel A: Action pie ───────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    counts = df["action"].value_counts()
    pie_colors = [COLORS.get(a, "grey") for a in counts.index]
    ax_a.pie(counts.values, labels=counts.index, colors=pie_colors,
             autopct="%1.0f%%", startangle=140, pctdistance=0.78,
             wedgeprops=dict(edgecolor="white"))
    ax_a.set_title("A. Action Distribution", fontweight="bold")

    # ── Panel B: Cumulative carbon ────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    cum = matched.copy()
    cum["cum_saved"] = cum["realized_nsg"].clip(lower=0).cumsum()
    ax_b.fill_between(cum["decision_idx"], cum["cum_saved"], alpha=0.25, color="#1f77b4")
    ax_b.plot(cum["decision_idx"], cum["cum_saved"], color="#1f77b4", linewidth=2.5, marker="o", markersize=4)
    ax_b.set_xlabel("Decision #")
    ax_b.set_ylabel("gCO₂ saved")
    ax_b.set_title("B. Cumulative Carbon Saved", fontweight="bold")
    ax_b.grid(alpha=0.3)

    # ── Panel C: Predicted vs Realized (scatter) ──────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    m = matched[matched["action"] == "MIGRATE"]
    ax_c.scatter(m["predicted_nsg"], m["realized_nsg"], color="#1f77b4", s=45, alpha=0.8, edgecolors="white")
    lim = max(m["predicted_nsg"].max(), m["realized_nsg"].max()) * 1.1
    ax_c.plot([0, lim], [0, lim], "k--", linewidth=1.2, label="y=x")
    ax_c.set_xlabel("Predicted NSG")
    ax_c.set_ylabel("Realized NSG")
    ax_c.set_title("C. Digital Twin Accuracy\n(MIGRATE decisions)", fontweight="bold")
    ax_c.legend(fontsize=8)
    ax_c.grid(alpha=0.3)

    # ── Panel D: Per-cycle improvement ───────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    cyc = matched[matched["cycle"] > 0].groupby("cycle")["realized_nsg"].mean()
    ax_d.bar([f"Cycle {i}" for i in cyc.index], cyc.values,
             color=["#aec7e8", "#6baed6", "#08519c"][:len(cyc)], edgecolor="white")
    for i, v in enumerate(cyc.values):
        ax_d.text(i, v + 0.0002, f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")
    ax_d.set_ylabel("Mean Realized NSG (gCO₂)")
    ax_d.set_title("D. PPO Learning Across Cycles\n(Before → After Feedback)", fontweight="bold")
    ax_d.grid(alpha=0.3, axis="y")

    # ── Panel E: SLA risk histogram ───────────────────────────────────────
    ax_e = fig.add_subplot(gs[1, 1])
    ax_e.hist(df["sla_risk"], bins=20, color="#ff7f0e", alpha=0.75, edgecolor="white")
    ax_e.axvline(0.80, color="red", linewidth=2, linestyle="--", label="Hard gate (0.80)")
    ax_e.set_xlabel("SLA Risk Score")
    ax_e.set_ylabel("Count")
    ax_e.set_title("E. SLA Risk Distribution\n(Right of red = DELAY zone)", fontweight="bold")
    ax_e.legend(fontsize=8)
    ax_e.grid(alpha=0.3)

    # ── Panel F: 24h NSG trend ────────────────────────────────────────────
    if not hourly.empty:
        ax_f = fig.add_subplot(gs[1, 2])
        ax_f.plot(hourly["hour"], hourly["mean_nsg_gco2"] * 1000,
                  color="#2ca02c", linewidth=2.5, marker="o", markersize=4)
        ax_f.fill_between(hourly["hour"], hourly["mean_nsg_gco2"] * 1000, alpha=0.2, color="#2ca02c")
        ax_f.set_xlabel("Hour of Day")
        ax_f.set_ylabel("Mean NSG × 1000")
        ax_f.set_title("F. 24h Carbon Savings Curve\n(Solar-driven peak at noon)", fontweight="bold")
        ax_f.set_xticks(range(0, 24, 4))
        ax_f.grid(alpha=0.3)

    fig.suptitle("Autonomous Sustainability Orchestration Agent — Results Dashboard",
                 fontsize=15, fontweight="bold", y=1.01)
    _save(fig, "10_summary_dashboard.png")

# ==========================================================================
# Run all charts
# ==========================================================================
print("\nGenerating explanatory charts...\n")
for fn, name in [
    (chart_predicted_vs_realized,  "1. Predicted vs Realized NSG"),
    (chart_before_after_feedback,  "2. Before vs After PPO Feedback"),
    (chart_cumulative_carbon,      "3. Cumulative Carbon Saved"),
    (chart_nsg_per_decision,       "4. NSG per Decision with SLA"),
    (chart_sla_risk_timeline,      "5. SLA Risk Timeline"),
    (chart_action_distribution,    "6. Action Distribution"),
    (chart_ci_vs_nsg,              "7. Carbon Intensity vs NSG"),
    (chart_decision_landscape,     "8. Full Decision Landscape"),
    (chart_24h_triple,             "9. 24h Scheduling Behaviour"),
    (chart_summary_dashboard,      "10. Summary Dashboard"),
]:
    print(f"[{name}]")
    try:
        fn()
    except Exception as e:
        print(f"  ⚠ Error: {e}")

print(f"\nAll charts saved to: {PLOTS_DIR}")
