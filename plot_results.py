import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("results/consolidated_results.csv")

out = Path("results/graphs")
out.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 1. Realized NSG per workload
# --------------------------------------------------

plt.figure(figsize=(10, 6))

for experiment in df["experiment"].unique():
    subset = df[df["experiment"] == experiment]

    plt.plot(
        range(1, len(subset) + 1),
        subset["realized_nsg"],
        marker="o",
        label=experiment
    )

plt.xlabel("Workload")
plt.ylabel("Realized NSG (gCO2)")
plt.title("Realized NSG Before and After PPO Feedback")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    out / "realized_nsg_comparison.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 2. Predicted vs realized NSG
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    df["predicted_nsg"],
    df["realized_nsg"]
)

plt.xlabel("Predicted NSG (gCO2)")
plt.ylabel("Realized NSG (gCO2)")
plt.title("Predicted vs Realized NSG")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    out / "predicted_vs_realized_nsg.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 3. Real carbon vs predicted baseline
# --------------------------------------------------

plt.figure(figsize=(10, 6))

x = range(len(df))

plt.bar(
    [i - 0.2 for i in x],
    df["real_carbon"],
    width=0.4,
    label="Real carbon"
)

plt.bar(
    [i + 0.2 for i in x],
    df["predicted_stay_baseline"],
    width=0.4,
    label="Predicted stay baseline"
)

plt.xlabel("Workload")
plt.ylabel("Carbon (gCO2)")
plt.title("Measured Carbon vs Predicted Stay Baseline")
plt.xticks(list(x), df["pod_id"], rotation=45, ha="right")
plt.legend()
plt.tight_layout()

plt.savefig(
    out / "real_vs_baseline_carbon.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 4. Target-node distribution
# --------------------------------------------------

counts = df["target_node"].value_counts()

plt.figure(figsize=(10, 6))

counts.plot(kind="bar")

plt.xlabel("Target Node")
plt.ylabel("Number of Workloads")
plt.title("Workload Placement Across Kubernetes Nodes")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(
    out / "node_distribution.png",
    dpi=300
)

plt.close()


print("Graphs generated in:", out)
