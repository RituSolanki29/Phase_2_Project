"""
Sustainability Orchestrator Pipeline Runner
Layers 1 -> 5

Two run modes:
  - Paired two-window mode (default, original behaviour): evaluates the same
    workload batch at a "Night" and a "Mid-Day Solar" timestamp, for a direct
    before/after comparison.
  - Time-sweep mode (--time-sweep-hours N): runs N scheduling cycles spread
    evenly across a 24h cycle, rotating through the workload batch, and
    aggregates results per hour. This is what actually demonstrates the
    system responding dynamically to a changing carbon-intensity curve,
    rather than two isolated snapshots -- see the hourly_sweep.png chart and
    hourly_summary.csv this mode produces.

Produces:
    1. Predictor metrics (predictor_metrics.json)
    2. Per-candidate Layer 5 results (layer5_candidate_results.csv)
    3. Experiment summary (experiment_summary.json)
    4. Hourly aggregation (hourly_summary.csv) -- time-sweep mode only
    5. PNG charts under results/plots/

Compatible with simulated/real Kubernetes, simulated/real Kepler, and
simulated/live carbon-intensity data (see telemetry_source.py).
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from src.layer1_infra.cluster_config import create_default_node_pools
from src.layer1_infra.trace_loader import TraceLoader
from src.layer1_infra import k8s_deployer
from src.layer2_telemetry.telemetry_source import TelemetrySource
from src.layer2_telemetry.feature_store import FeatureStore
from src.layer3_scheduling.nsga2_filter import NSGA2PreFilter
from src.layer4_digital_twin.gbr_predictor import GBRRuntimePredictor
from src.layer4_digital_twin.clark_model import ClarkMigrationModel
from src.layer4_digital_twin.simpy_uncertainty import SimPyCarbonSampler
from src.layer4_digital_twin.monte_carlo import MonteCarloEngine
from src.layer5_outputs.nsg_calculator import Layer5OutputCalculator
from src.layer5_outputs.data_contracts import CandidateEvaluationResult
from src.layer5_outputs.visualizations import generate_all_visualizations

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "borg_traces_data.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results"


def safe_mean(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else 0.0


def safe_percentile(values, pct):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.percentile(values, pct)) if values else 0.0


def _deploy_and_build_frame(pod, node_pools, feature_store, telemetry, timestamp_sec, deploy_real_pods, deployed_pods):
    k8s_pod_name, elapsed_sec = None, None
    if deploy_real_pods and telemetry.k8s_available:
        current_node = next((n for n in node_pools if n.node_id == pod.source_node), node_pools[0])
        try:
            k8s_pod_name = k8s_deployer.deploy_pod(pod, current_node)
            start = time.perf_counter()
            if k8s_deployer.wait_for_pod_running(k8s_pod_name, timeout_sec=20.0):
                elapsed_sec = time.perf_counter() - start
                deployed_pods.append(k8s_pod_name)
            else:
                k8s_pod_name = None
        except Exception:
            k8s_pod_name = None
    return feature_store.build_telemetry_frame(
        pod=pod, nodes=node_pools, timestamp_sec=timestamp_sec,
        k8s_pod_name=k8s_pod_name, elapsed_sec=elapsed_sec,
    )


def _run_one_cycle(cycle_workloads, timestamp_sec, window_label, node_pools, feature_store,
                    telemetry, nsga2, mc_engine, deploy_real_pods, deployed_pods, verbose=True):
    frames = [
        _deploy_and_build_frame(pod, node_pools, feature_store, telemetry, timestamp_sec, deploy_real_pods, deployed_pods)
        for pod in cycle_workloads
    ]
    real_count = sum(1 for f in frames if f.is_real_measurement)

    nsga_start = time.perf_counter()
    candidates_by_pod = nsga2.filter_and_rank_batch(frames)
    nsga_runtime = time.perf_counter() - nsga_start

    cycle_results: List[CandidateEvaluationResult] = []
    for frame in frames:
        pod = frame.pod
        for candidate in candidates_by_pod.get(pod.pod_id, []):
            mc_result = mc_engine.run_simulation(frame=frame, target_node=candidate.target_node)
            result = Layer5OutputCalculator.evaluate_candidate(
                mc_result=mc_result, pod=pod, current_node=frame.current_node,
                target_node=candidate.target_node, scheduling_window=window_label,
                timestamp_sec=timestamp_sec,
            )
            cycle_results.append(result)

    if verbose:
        print(f"  Telemetry frames: {len(frames)} ({real_count} real Kepler, {len(frames) - real_count} model-based)")
        print(f"  NSGA-II runtime: {nsga_runtime:.4f}s | Candidates evaluated: {len(cycle_results)} "
              f"| Raw assignment space: {len(node_pools) ** len(frames):,}")

    return cycle_results, nsga_runtime


def run_sustainability_pipeline(
    csv_path,
    batch_size: int = 100,
    pods_per_cycle: int = 30,
    mc_runs: int = 1000,
    deploy_real_pods: bool = False,
    compression_factor: float = 10.0,
    random_seed: int = 42,
    paired_cycles: bool = True,
    time_sweep_hours: int = 0,
    output_dir=None,
    make_plots: bool = True,
):
    start_time = time.perf_counter()
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found:\n{csv_path}")

    output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("      SUSTAINABILITY ORCHESTRATOR PIPELINE -- LAYERS 1 TO 5")
    print("=" * 80)
    print(f"\nDataset: {csv_path}")

    # -- Layer 2: telemetry mode probe --
    telemetry = TelemetrySource(
        kubeconfig_path=os.environ.get("KUBECONFIG"),
        kepler_endpoint=os.environ.get("KEPLER_ENDPOINT", "http://localhost:9102/metrics"),
        electricitymap_api_key=os.environ.get("ELECTRICITYMAP_API_KEY"),
        sim_seed=random_seed,
    )
    print("\n[Telemetry Mode]")
    print(telemetry.mode_report())

    # -- Layer 1: infra + trace --
    print("\n[Layer 1] Infrastructure & Borg Trace")
    node_pools = create_default_node_pools()
    regions = sorted(set(n.region for n in node_pools))
    print(f"  Nodes: {len(node_pools)} | Regions: {regions}")

    loader = TraceLoader(str(csv_path))
    pods = loader.load_trace_batch(limit=batch_size, compression_factor=compression_factor)
    if not pods:
        raise RuntimeError("No workload records were loaded from the dataset.")
    print(f"  Workloads loaded: {len(pods)}")

    feature_store = FeatureStore(telemetry)

    # -- Layer 4 prep: GBR --
    print("\n[Layer 4 Prep] GBR Runtime Predictor")
    gbr = GBRRuntimePredictor(random_state=random_seed)
    gbr.train_time_split(pods, train_ratio=0.80)
    predictor_metrics = gbr.get_metrics()
    feature_importance = gbr.get_feature_importance()
    print(f"  MAE: {predictor_metrics['mae']:.3f}s | RMSE: {predictor_metrics['rmse']:.3f}s | "
          f"R2: {predictor_metrics['r2']:.3f} | sMAPE: {predictor_metrics['smape_percent']:.1f}% | "
          f"RMSE improvement vs baseline: {predictor_metrics['rmse_improvement_percent']:.1f}%")
    if predictor_metrics["r2"] < 0:
        print("  [!] R2 is negative -- the training sample is likely too small. "
              "Re-run with a larger --batch-size (1000+) for meaningful metrics.")

    # -- Layer 3/4 engines --
    nsga2 = NSGA2PreFilter(telemetry, top_k=2, pop_size=40, n_gen=25, seed=random_seed)
    clark = ClarkMigrationModel()
    simpy_sampler = SimPyCarbonSampler(telemetry._sim_carbon)
    mc_engine = MonteCarloEngine(gbr_predictor=gbr, clark_model=clark, simpy_sampler=simpy_sampler,
                                  kepler=telemetry, num_runs=mc_runs, random_seed=random_seed)

    all_results: List[CandidateEvaluationResult] = []
    deployed_pods: List[str] = []
    nsga2_runtimes: List[float] = []
    hourly_rows = []

    print("\n[Layers 3-5] Scheduling + Digital Twin")

    try:
        if time_sweep_hours and time_sweep_hours > 0:
            # -- Dynamic mode: sweep across the day, rotating the workload batch --
            n_cycles = int(time_sweep_hours)
            timestamps = np.linspace(0, 86400, n_cycles, endpoint=False)
            cycle_size = min(pods_per_cycle, len(pods))
            for i, ts in enumerate(timestamps):
                start_idx = (i * cycle_size) % len(pods)
                cycle_workloads = (pods[start_idx:] + pods[:start_idx])[:cycle_size]
                hour_label = f"Hour {ts / 3600:.1f}"
                print(f"\n--- Cycle {i + 1}/{n_cycles}: {hour_label} (t={ts:.0f}s) ---")
                cycle_results, nsga_runtime = _run_one_cycle(
                    cycle_workloads, float(ts), hour_label, node_pools, feature_store,
                    telemetry, nsga2, mc_engine, deploy_real_pods, deployed_pods,
                )
                nsga2_runtimes.append(nsga_runtime)
                all_results.extend(cycle_results)
                if cycle_results:
                    nsg_vals = [r.nsg_mean_gco2 for r in cycle_results]
                    viable_vals = [r.is_statistically_viable for r in cycle_results]
                    sla_vals = [r.sla_risk_score for r in cycle_results]
                    hourly_rows.append({
                        "hour": round(ts / 3600.0, 2),
                        "mean_nsg_gco2": safe_mean(nsg_vals),
                        "viable_percentage": 100.0 * safe_mean([1.0 if v else 0.0 for v in viable_vals]),
                        "mean_sla_risk": safe_mean(sla_vals),
                        "candidate_count": len(cycle_results),
                    })
        else:
            # -- Original paired two-window mode --
            timestamps = [0.0, 43200.0]
            window_names = ["Night Window", "Mid-Day Solar Window"]
            cycle_size = min(pods_per_cycle, len(pods))
            base_cycle = pods[:cycle_size]
            cycle_pods_list = [base_cycle, base_cycle] if paired_cycles else [
                pods[:cycle_size], pods[cycle_size:cycle_size * 2] or base_cycle
            ]
            for ts, window_name, cycle_workloads in zip(timestamps, window_names, cycle_pods_list):
                print(f"\n--- {window_name} (t={ts}s) ---")
                cycle_results, nsga_runtime = _run_one_cycle(
                    cycle_workloads, ts, window_name, node_pools, feature_store,
                    telemetry, nsga2, mc_engine, deploy_real_pods, deployed_pods,
                )
                nsga2_runtimes.append(nsga_runtime)
                all_results.extend(cycle_results)
    finally:
        for pod_name in deployed_pods:
            try:
                k8s_deployer.delete_pod(pod_name)
            except Exception:
                pass
        if deployed_pods:
            print(f"\n[Cleanup] Deleted {len(deployed_pods)} real Kubernetes pods.")

    results_df = pd.DataFrame([r.to_dict() for r in all_results])
    hourly_df = pd.DataFrame(hourly_rows) if hourly_rows else pd.DataFrame()

    # -- Summary --
    summary = {
        "experiment": {
            "dataset": str(csv_path),
            "workloads_loaded": len(pods),
            "nodes": len(node_pools),
            "regions": regions,
            "mode": "time_sweep" if time_sweep_hours else "paired_two_window",
            "time_sweep_hours": time_sweep_hours,
            "monte_carlo_runs": mc_runs,
            "random_seed": random_seed,
            "candidate_evaluations": len(all_results),
        },
        "predictor": predictor_metrics,
        "predictor_feature_importance": feature_importance,
        "scheduler": {
            "mean_nsga2_runtime_sec": safe_mean(nsga2_runtimes),
            "p95_nsga2_runtime_sec": safe_percentile(nsga2_runtimes, 95),
            "total_pipeline_runtime_sec": time.perf_counter() - start_time,
        },
    }

    if not results_df.empty:
        summary["sustainability"] = {
            "mean_nsg_gco2": float(results_df["nsg_mean_gco2"].mean()),
            "mean_gross_carbon_saving_gco2": float(results_df["gross_carbon_saving_mean_gco2"].mean()),
            "mean_migration_carbon_gco2": float(results_df["migration_carbon_mean_gco2"].mean()),
        }
        summary["decisions"] = {
            "statistically_viable_count": int(results_df["is_statistically_viable"].sum()),
            "rejected_count": int((~results_df["is_statistically_viable"]).sum()),
            "viable_percentage": float(results_df["is_statistically_viable"].mean() * 100.0),
        }

    # -- Save outputs --
    csv_output = output_dir / "layer5_candidate_results.csv"
    json_output = output_dir / "experiment_summary.json"
    predictor_output = output_dir / "predictor_metrics.json"
    results_df.to_csv(csv_output, index=False)
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(predictor_output, "w", encoding="utf-8") as f:
        json.dump({"metrics": predictor_metrics, "feature_importance": feature_importance}, f, indent=2)

    hourly_output = None
    if not hourly_df.empty:
        hourly_output = output_dir / "hourly_summary.csv"
        hourly_df.to_csv(hourly_output, index=False)

    # -- Console summary --
    print("\n" + "=" * 80)
    print("                    EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"Candidate evaluations : {len(all_results)}")
    print(f"Pipeline runtime      : {summary['scheduler']['total_pipeline_runtime_sec']:.2f}s")
    if not results_df.empty:
        print(f"Mean NSG              : {summary['sustainability']['mean_nsg_gco2']:.6f} gCO2")
        print(f"Viable candidates     : {summary['decisions']['statistically_viable_count']} "
              f"({summary['decisions']['viable_percentage']:.1f}%)")

    # -- Visualizations --
    if make_plots:
        print("\n[Visualizations]")
        plots = generate_all_visualizations(
            results_df=results_df, pods=pods, predictor=gbr, feature_importance=feature_importance,
            carbon_feed=telemetry._sim_carbon, regions=regions, output_dir=output_dir,
            hourly_df=hourly_df if not hourly_df.empty else None,
        )
        for p in plots:
            print(f"  {p}")

    print("\n[FILES]")
    print(f"Candidate CSV   : {csv_output}")
    print(f"Experiment JSON : {json_output}")
    print(f"Predictor JSON  : {predictor_output}")
    if hourly_output:
        print(f"Hourly CSV      : {hourly_output}")
    print("=" * 80)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Sustainability Orchestrator Pipeline (Layers 1-5).")
    parser.add_argument("--csv-path", default=str(DEFAULT_CSV_PATH))
    parser.add_argument("--batch-size", type=int, default=1000, help="Rows to load (1000+ recommended for a meaningful GBR R^2).")
    parser.add_argument("--pods-per-cycle", type=int, default=30)
    parser.add_argument("--mc-runs", type=int, default=1000)
    parser.add_argument("--compression-factor", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--no-real-pods", action="store_true")
    parser.add_argument("--unpaired-cycles", action="store_true")
    parser.add_argument("--time-sweep-hours", type=int, default=0,
                         help="If >0, run this many scheduling cycles spread across 24h instead of the paired 2-window comparison.")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    run_sustainability_pipeline(
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        pods_per_cycle=args.pods_per_cycle,
        mc_runs=args.mc_runs,
        deploy_real_pods=not args.no_real_pods,
        compression_factor=args.compression_factor,
        random_seed=args.seed,
        paired_cycles=not args.unpaired_cycles,
        time_sweep_hours=args.time_sweep_hours,
        output_dir=args.output_dir,
        make_plots=not args.no_plots,
    )