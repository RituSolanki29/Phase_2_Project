"""
Consolidates decision/outcome pairs from the reward logs into
results/consolidated_results.csv and prints summary statistics.

Fixes vs. the previous version:
  * Decisions and outcomes are paired by pod_id in log order via
    RewardStore.load_pairs().
  * Works with both the legacy free-text `detail` string and the new
    structured dict written by post_exec_evaluator.py.
  * The "After" log path is configurable and falls back to the newest
    reward_log*.jsonl instead of silently skipping a missing file.
  * Reports the predicted-vs-realized correlation and the share of pods with
    positive realized NSG.
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np

from src.layer8_feedback.reward_store import RewardStore

_NUM = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"


def extract_detail(detail):
    """Returns (real_carbon, predicted_stay_baseline, migration_carbon)."""
    if isinstance(detail, dict):
        return (detail.get("real_carbon"), detail.get("predicted_stay_baseline"),
                detail.get("migration_carbon"))
    if isinstance(detail, str):
        real = re.search(rf"real_carbon={_NUM}gCO2", detail)
        base = re.search(rf"predicted_stay_baseline={_NUM}gCO2", detail)
        return (float(real.group(1)) if real else None,
                float(base.group(1)) if base else None, None)
    return None, None, None


def _default_after(results_dir: Path) -> Path:
    main = results_dir / "reward_log.jsonl"
    if main.exists():
        return main
    others = sorted(results_dir.glob("reward_log_after*.jsonl"))
    return others[-1] if others else main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--before", default=None)
    ap.add_argument("--after", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    rdir = Path(args.results_dir)
    files = {
        "Before Feedback": Path(args.before) if args.before else rdir / "reward_log_before_final.jsonl",
        "After Feedback": Path(args.after) if args.after else _default_after(rdir),
    }

    rows = []
    for experiment, path in files.items():
        if not path.exists():
            print(f"Missing: {path}")
            continue
        for pr in RewardStore(path).load_pairs():
            cand = pr.get("candidate") or {}
            real_carbon, baseline, migration = extract_detail(pr.get("detail"))
            rows.append({
                "experiment": experiment,
                "pod_id": pr["pod_id"],
                "action": pr.get("action"),
                "target_node": pr.get("bound_node_id"),
                "target_region": cand.get("target_region"),
                "predicted_nsg": cand.get("nsg_mean_gco2"),
                "p_nsg_positive": cand.get("p_nsg_positive"),
                "sla_risk": cand.get("sla_risk_score"),
                "predicted_execution_time": cand.get("predicted_exec_time_sec"),
                "migration_time": cand.get("migration_duration_sec"),
                "realized_nsg": pr.get("realized_nsg_gco2"),
                "real_carbon": real_carbon,
                "migration_carbon": migration,
                "predicted_stay_baseline": baseline,
            })

    if not rows:
        print("No decision/outcome pairs found.")
        return

    output = Path(args.output) if args.output else rdir / "consolidated_results.csv"
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 60 + "\nCONSOLIDATED RESULTS\n" + "=" * 60)
    print(f"Saved: {output}\nTotal observations: {len(rows)}\n")

    for experiment in files:
        subset = [r for r in rows if r["experiment"] == experiment]
        if not subset:
            continue
        nsg = np.array([float(r["realized_nsg"]) for r in subset if r["realized_nsg"] is not None])
        pred = np.array([float(r["predicted_nsg"]) for r in subset if r["realized_nsg"] is not None
                         and r["predicted_nsg"] is not None])
        carbon = [float(r["real_carbon"]) for r in subset if r["real_carbon"] is not None]
        base = [float(r["predicted_stay_baseline"]) for r in subset if r["predicted_stay_baseline"] is not None]

        print(experiment + "\n" + "-" * 40)
        print(f"Number of workloads: {len(nsg)}")
        if len(nsg):
            print(f"Mean realized NSG: {nsg.mean():.6f} gCO2   (positive for {100 * (nsg > 0).mean():.0f}% of pods)")
        if len(pred) == len(nsg) and len(nsg) > 2 and np.std(pred) > 0 and np.std(nsg) > 0:
            print(f"Predicted-vs-realized NSG correlation: {np.corrcoef(pred, nsg)[0, 1]:+.3f}")
        if carbon:
            print(f"Mean real carbon: {np.mean(carbon):.6f} gCO2")
        if base:
            print(f"Mean predicted baseline: {np.mean(base):.6f} gCO2")
        if carbon and base and len(carbon) == len(base):
            print(f"Real / baseline carbon ratio: {np.mean(carbon) / max(np.mean(base), 1e-12):.2f}")
        print()


if __name__ == "__main__":
    main()