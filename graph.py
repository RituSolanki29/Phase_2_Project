import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/consolidated_results.csv")

after = df[df["experiment"] == "After Feedback"]

# 1. Predicted vs Realized NSG
plt.figure(figsize=(10, 5))
plt.plot(after["pod_id"], after["predicted_nsg"], marker="o", label="Predicted NSG")
plt.plot(after["pod_id"], after["realized_nsg"], marker="x", label="Realized NSG")
plt.axhline(0, linestyle="--")
plt.xticks(rotation=90)
plt.ylabel("NSG (gCO₂)")
plt.title("Predicted vs Realized NSG")
plt.legend()
plt.tight_layout()
plt.savefig("results/predicted_vs_realized_nsg_new.png", dpi=300)
plt.show()


# 2. Realized NSG per workload
plt.figure(figsize=(10, 5))
plt.bar(after["pod_id"], after["realized_nsg"])
plt.axhline(0, linestyle="--")
plt.xticks(rotation=90)
plt.ylabel("Realized NSG (gCO₂)")
plt.title("Realized NSG per Workload")
plt.tight_layout()
plt.savefig("results/realized_nsg_per_workload_new.png", dpi=300)
plt.show()


# 3. Real Carbon vs Predicted Baseline
plt.figure(figsize=(10, 5))
plt.plot(after["pod_id"], after["real_carbon"], marker="o", label="Real Carbon")
plt.plot(after["pod_id"], after["predicted_stay_baseline"],
         marker="x", label="Predicted Stay Baseline")
plt.xticks(rotation=90)
plt.ylabel("Carbon (gCO₂)")
plt.title("Real Carbon vs Predicted Stay Baseline")
plt.legend()
plt.tight_layout()
plt.savefig("results/carbon_vs_baseline_new.png", dpi=300)
plt.show()


# 4. NSG by Region
region_data = after.groupby("target_region")["realized_nsg"].mean()

plt.figure(figsize=(7, 5))
region_data.plot(kind="bar")
plt.axhline(0, linestyle="--")
plt.ylabel("Mean Realized NSG (gCO₂)")
plt.title("Mean Realized NSG by Region")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("results/nsg_by_region_new.png", dpi=300)
plt.show()