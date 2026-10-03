"""
Statistical Significance & Hypothesis Testing Module (Layer 8)
  1. Paired t-test (predicted vs realized NSG)
  2. Wilcoxon signed-rank test
  3. Cohen's d effect size
  4. Required sample size estimate (power analysis)

Fix: outcome entries do not contain the predicted NSG or the action -- those
live in the matching DECISION entry -- so the old version always found N=0.
Decisions and outcomes are now paired by pod_id via RewardStore.load_pairs().
"""

import argparse
import numpy as np
from scipy import stats

from src.layer8_feedback.reward_store import RewardStore


def run_statistical_analysis(log_path: str = "results/reward_log.jsonl"):
    store = RewardStore(log_path)
    pairs = store.load_pairs()

    predicted_nsgs, realized_nsgs, actions = [], [], []
    for pr in pairs:
        cand = pr.get("candidate") or {}
        p_nsg = cand.get("nsg_mean_gco2")
        r_nsg = pr.get("realized_nsg_gco2")
        if p_nsg is not None and r_nsg is not None:
            predicted_nsgs.append(float(p_nsg))
            realized_nsgs.append(float(r_nsg))
            actions.append(pr.get("action", "UNKNOWN"))

    n = len(realized_nsgs)
    print("=" * 80)
    print("           LAYER 8 STATISTICAL SIGNIFICANCE & HYPOTHESIS ANALYSIS")
    print("=" * 80)
    print(f"Total evaluated real outcomes (N): {n}")

    if n < 5:
        print(f"\n[!] Sample size too small (N={n}, minimum 5 required for statistical testing).")
        return None

    p_arr, r_arr = np.array(predicted_nsgs), np.array(realized_nsgs)
    diff = r_arr - p_arr

    unique_actions, counts = np.unique(actions, return_counts=True)
    print("Action Distribution : " + ", ".join(f"{a}: {c}" for a, c in zip(unique_actions, counts)))

    mean_realized, std_realized, mean_predicted = float(np.mean(r_arr)), float(np.std(r_arr, ddof=1)), float(np.mean(p_arr))

    t_stat, p_val_t = stats.ttest_rel(r_arr, p_arr)
    try:
        w_stat, p_val_w = stats.wilcoxon(r_arr, p_arr)
    except Exception:
        w_stat, p_val_w = 0.0, 1.0

    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    cohens_d = mean_diff / max(1e-12, std_diff)

    if abs(cohens_d) > 1e-4:
        z_alpha, z_beta = stats.norm.ppf(0.975), stats.norm.ppf(0.80)
        req_n = int(np.ceil(((z_alpha + z_beta) / cohens_d) ** 2))
    else:
        req_n = 50

    corr = float(np.corrcoef(p_arr, r_arr)[0, 1]) if np.std(p_arr) > 0 and np.std(r_arr) > 0 else float("nan")

    print("\n[Descriptive Statistics]")
    print(f"  Mean Predicted NSG  : {mean_predicted:+.6f} gCO2")
    print(f"  Mean Realized NSG   : {mean_realized:+.6f} gCO2 (std: {std_realized:.6f})")
    print(f"  Mean Discrepancy    : {mean_diff:+.6f} gCO2")
    print(f"  Pearson r (pred vs realized): {corr:+.3f}")

    print("\n[Hypothesis Testing]  H0: realized NSG == predicted NSG")
    print(f"  Paired t-test       : t = {t_stat:+.4f}, p-value = {p_val_t:.4f}")
    print(f"  Wilcoxon test       : W = {w_stat:.1f}, p-value = {p_val_w:.4f}")
    print(f"  Effect Size (d)     : {cohens_d:+.4f} ({'large' if abs(cohens_d)>0.8 else 'medium' if abs(cohens_d)>0.5 else 'small'})")
    print(f"  Est. Required N     : {req_n} samples (80% power, alpha=0.05)")

    signif = "STATISTICALLY SIGNIFICANT" if p_val_t < 0.05 else "NOT SIGNIFICANT (noise / insufficient samples)"
    print(f"\nConclusion: predicted-vs-realized gap is {signif} at the 95% level.")
    print("=" * 80)
    return {"n": n, "mean_pred": mean_predicted, "mean_real": mean_realized, "pearson_r": corr, "p_t": float(p_val_t)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run statistical analysis on accumulated reward logs.")
    parser.add_argument("--log-path", default="results/reward_log.jsonl")
    args = parser.parse_args()
    run_statistical_analysis(args.log_path)