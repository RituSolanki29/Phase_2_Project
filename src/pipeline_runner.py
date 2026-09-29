"""
Sustainability Orchestrator Pipeline Runner (Layers 1 to 5)
Executes full data, telemetry, pre-filtering, digital twin simulation,
and statistical evaluation pipeline. Uses REAL Kubernetes + Kepler + live
carbon-intensity data wherever reachable, and transparently falls back to
the simulated models otherwise (see telemetry_source.py for the rule on
which parts are allowed to be real vs. must stay model-based).
"""

import os
import time
from typing import List

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

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV_PATH = os.path.join(_PROJECT_ROOT, "data", "borg_traces_data.csv")


def run_sustainability_pipeline(
    csv_path: str = DEFAULT_CSV_PATH,
    batch_size: int = 100,
    pods_per_cycle: int = 30,
    mc_runs: int = 1000,
    deploy_real_pods: bool = True,
) -> List[CandidateEvaluationResult]:
    """Runs end-to-end processing pipeline across Layers 1-5."""
    print("=" * 70)
    print("      SUSTAINABILITY ORCHESTRATOR PIPELINE (LAYERS 1 TO 5)")
    print("=" * 70)

    # Layer 2 Prep: probe what real infrastructure is actually reachable.
    telemetry = TelemetrySource(
        kubeconfig_path=os.environ.get("KUBECONFIG"),
        kepler_endpoint=os.environ.get("KEPLER_ENDPOINT", "http://localhost:9102/metrics"),
        electricitymap_api_key=os.environ.get("ELECTRICITYMAP_API_KEY"),
    )
    print("\n[Telemetry Mode]")
    print(telemetry.mode_report())

    # Layer 1: Infrastructure & Trace Loading
    print("\n[Layer 1] Initializing Infrastructure & Replaying Borg Trace Batch...")
    node_pools = create_default_node_pools()
    print(f"  Loaded {len(node_pools)} heterogeneous nodes across regions: {[n.region for n in node_pools]}")

    loader = TraceLoader(csv_path)
    pods = loader.load_trace_batch(limit=batch_size, compression_factor=10.0)
    print(f"  Loaded {len(pods)} trace workload pods.")

    feature_store = FeatureStore(telemetry)

    # Layer 4 Prep: Train GBR Model with time-based train/test split
    print("\n[Layer 4 Prep] Training GBR Execution Time Predictor (80/20 Time Split)...")
    gbr_predictor = GBRRuntimePredictor(random_state=42)
    rmse, r2 = gbr_predictor.train_time_split(pods, train_ratio=0.8)
    print(f"  GBR Model Trained -> Test RMSE: {rmse:.3f}s, Test R\u00b2 Score: {r2:.3f}")

    # Layer 3 & 4 Engines. `telemetry` is passed wherever a KeplerEmulator
    # was previously expected -- TelemetrySource exposes the same
    # `estimate_pod_power(...)` method (always model-based for these
    # hypothetical-candidate evaluations; see telemetry_source.py).
    nsga2_filter = NSGA2PreFilter(telemetry, top_k=2, pop_size=40, n_gen=25, seed=42)
    clark_model = ClarkMigrationModel()
    simpy_sampler = SimPyCarbonSampler(telemetry._sim_carbon)
    mc_engine = MonteCarloEngine(gbr_predictor, clark_model, simpy_sampler, telemetry, num_runs=mc_runs)

    all_evaluations: List[CandidateEvaluationResult] = []
    deployed_k8s_pods = []  # (k8s_pod_name, namespace) to clean up at the end

    print("\n[Layer 3 - 5] Processing Pod Scheduling Cycles & Digital Twin Simulations...")
    timestamps = [0.0, 43200.0]  # Night vs Mid-day Solar Peak

    try:
        for cycle_idx, timestamp_sec in enumerate(timestamps):
            time_label = "Mid-Day Solar Peak (12 PM)" if timestamp_sec == 43200.0 else "Baseline Night Window"
            print(f"\n--- Scheduling Cycle #{cycle_idx + 1}: {time_label} (t={timestamp_sec}s) ---")

            cycle_pods = pods[:pods_per_cycle]
            frames = []
            for pod in cycle_pods:
                k8s_pod_name = None
                elapsed_sec = None
                if deploy_real_pods and telemetry.k8s_available:
                    # Layer 1: actually deploy this pod onto its CURRENT
                    # (source) node on the real cluster, so Layer 2 can read
                    # REAL Kepler telemetry for it below.
                    current_node = next((n for n in node_pools if n.node_id == pod.source_node), node_pools[0])
                    try:
                        k8s_pod_name = k8s_deployer.deploy_pod(pod, current_node)
                        deploy_start = time.time()
                        if k8s_deployer.wait_for_pod_running(k8s_pod_name, timeout_sec=20.0):
                            elapsed_sec = time.time() - deploy_start
                            deployed_k8s_pods.append(k8s_pod_name)
                        else:
                            k8s_pod_name = None  # failed to come up; fall back to model
                    except Exception:
                        k8s_pod_name = None

                frame = feature_store.build_telemetry_frame(
                    pod, node_pools, timestamp_sec=timestamp_sec,
                    k8s_pod_name=k8s_pod_name, elapsed_sec=elapsed_sec,
                )
                frames.append(frame)

            real_count = sum(1 for f in frames if f.is_real_measurement)
            print(f"  Built {len(frames)} telemetry frames ({real_count} with REAL Kepler measurement, "
                  f"{len(frames) - real_count} model-based).")

            # Layer 3: ONE batched NSGA-II run across every pod in this cycle.
            candidates_by_pod = nsga2_filter.filter_and_rank_batch(frames)
            print(f"  NSGA-II batch-optimized {len(frames)} pods over {len(node_pools)} nodes "
                  f"({len(node_pools) ** len(frames):,} raw combinations).")

            for frame in frames:
                pod = frame.pod
                for cand in candidates_by_pod.get(pod.pod_id, []):
                    target_node = cand.target_node
                    # Layer 4: Digital Twin 1000-Run Monte Carlo Simulation
                    # (always model-based for the candidate target node).
                    mc_result = mc_engine.run_simulation(frame, target_node)
                    # Layer 5: Statistical Model Outputs (NSG & SLA Risk)
                    eval_result = Layer5OutputCalculator.evaluate_candidate(
                        mc_result=mc_result, pod=pod,
                        current_node=frame.current_node, target_node=target_node,
                    )
                    all_evaluations.append(eval_result)
    finally:
        # Best-effort cleanup of any real pods we deployed for measurement.
        for k8s_pod_name in deployed_k8s_pods:
            k8s_deployer.delete_pod(k8s_pod_name)
        if deployed_k8s_pods:
            print(f"\n[Cleanup] Deleted {len(deployed_k8s_pods)} real Kubernetes pods.")

    print(f"\nCompleted evaluation of {len(all_evaluations)} pod-target candidate pairs.")
    return all_evaluations


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the sustainability orchestrator pipeline (Layers 1-5).")
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH, help="Path to the trace CSV file.")
    parser.add_argument("--batch-size", type=int, default=100, help="Number of trace rows to load.")
    parser.add_argument("--pods-per-cycle", type=int, default=30, help="Pods per scheduling cycle batch.")
    parser.add_argument("--mc-runs", type=int, default=1000, help="Monte Carlo runs per candidate.")
    parser.add_argument("--no-real-pods", action="store_true", help="Never deploy real K8s pods even if a cluster is reachable.")
    args = parser.parse_args()

    results = run_sustainability_pipeline(
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        pods_per_cycle=args.pods_per_cycle,
        mc_runs=args.mc_runs,
        deploy_real_pods=not args.no_real_pods,
    )

    print("\n" + "=" * 70)
    print("                SAMPLE LAYER 5 MODEL OUTPUTS (NIGHT vs SOLAR PEAK)")
    print("=" * 70)

    viable = [r for r in results if r.is_statistically_viable]
    rejected = [r for r in results if not r.is_statistically_viable]
    samples_to_show = (viable[:3] + rejected[:2]) if viable else results[:5]

    for res in samples_to_show:
        status_str = "[VIABLE]" if res.is_statistically_viable else "[REJECTED]"
        print(f"\nPod: {res.pod_id:<25} Status: {status_str}")
        print(f"  Current Node  : {res.current_node_id:<12} Region: {res.current_region}")
        print(f"  Target Node   : {res.target_node_id:<12} Region: {res.target_region}")
        print(f"  Net Sust Gain : {res.nsg_mean_gco2:+.4f} gCO2 (90% CI: [{res.nsg_p05_gco2:+.4f}, {res.nsg_p95_gco2:+.4f}])")
        print(f"  P(NSG > 0)    : {res.p_nsg_positive * 100:.1f}%  (Confidence Threshold >= 90.0%)")
        print(f"  SLA Risk Score: {res.sla_risk_score:.4f}  (Safety Gate Threshold < 0.8000)")
        print(f"  Est Exec Time : {res.predicted_exec_time_sec:.2f}s | Migration Overhead: {res.migration_duration_sec:.2f}s")