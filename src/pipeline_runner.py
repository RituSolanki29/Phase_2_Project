"""
Sustainability Orchestrator Pipeline Runner (Layers 1 to 5)
Executes full data, telemetry, pre-filtering, digital twin simulation,
and statistical evaluation pipeline.
"""

import os
import json
from typing import List, Dict, Any

from src.layer1_infra.cluster_config import create_default_node_pools
from src.layer1_infra.trace_loader import TraceLoader
from src.layer1_infra.kepler_emulator import KeplerEmulator
from src.layer2_telemetry.carbon_feed import CarbonFeed
from src.layer2_telemetry.feature_store import FeatureStore
from src.layer3_scheduling.nsga2_filter import NSGA2PreFilter
from src.layer4_digital_twin.gbr_predictor import GBRRuntimePredictor
from src.layer4_digital_twin.clark_model import ClarkMigrationModel
from src.layer4_digital_twin.simpy_uncertainty import SimPyCarbonSampler
from src.layer4_digital_twin.monte_carlo import MonteCarloEngine
from src.layer5_outputs.nsg_calculator import Layer5OutputCalculator
from src.layer5_outputs.data_contracts import CandidateEvaluationResult

def run_sustainability_pipeline(
    csv_path: str = "f:/Phase 2/data/borg_traces_data.csv",
    batch_size: int = 100,
    mc_runs: int = 1000
) -> List[CandidateEvaluationResult]:
    """Runs end-to-end processing pipeline across Layers 1-5."""
    print("=" * 70)
    print("      SUSTAINABILITY ORCHESTRATOR PIPELINE (LAYERS 1 TO 5)")
    print("=" * 70)
    
    # Layer 1: Infrastructure & Trace Loading
    print("\n[Layer 1] Initializing Infrastructure & Replaying Borg Trace Batch...")
    node_pools = create_default_node_pools()
    print(f"  Loaded {len(node_pools)} heterogeneous nodes across regions: {[n.region for n in node_pools]}")
    
    loader = TraceLoader(csv_path)
    pods = loader.load_trace_batch(limit=batch_size, compression_factor=10.0)
    print(f"  Loaded {len(pods)} trace workload pods.")
    
    # Layer 1 & 2: Kepler Telemetry & Carbon Feed Setup
    kepler = KeplerEmulator()
    carbon_feed = CarbonFeed(seed=42)
    feature_store = FeatureStore(carbon_feed, kepler)
    
    # Layer 4 Prep: Train GBR Model with time-based train/test split
    print("\n[Layer 4 Prep] Training GBR Execution Time Predictor (80/20 Time Split)...")
    gbr_predictor = GBRRuntimePredictor(random_state=42)
    rmse, r2 = gbr_predictor.train_time_split(pods, train_ratio=0.8)
    print(f"  GBR Model Trained -> Test RMSE: {rmse:.3f}s, Test R² Score: {r2:.3f}")
    
    # Layer 3 & 4 Engines
    nsga2_filter = NSGA2PreFilter(kepler, top_k=2)
    clark_model = ClarkMigrationModel()
    simpy_sampler = SimPyCarbonSampler(carbon_feed)
    mc_engine = MonteCarloEngine(gbr_predictor, clark_model, simpy_sampler, kepler, num_runs=mc_runs)
    
    all_evaluations: List[CandidateEvaluationResult] = []
    
    print("\n[Layer 3 - 5] Processing Pod Scheduling Cycles & Digital Twin Simulations...")
    # Evaluate scheduling cycles at baseline and mid-day solar peak (12 PM)
    timestamps = [0.0, 43200.0]  # Night vs Mid-day Solar Peak
    
    for cycle_idx, timestamp_sec in enumerate(timestamps):
        time_label = "Mid-Day Solar Peak (12 PM)" if timestamp_sec == 43200.0 else "Baseline Night Window"
        print(f"\n--- Scheduling Cycle #{cycle_idx + 1}: {time_label} (t={timestamp_sec}s) ---")
        
        for idx, pod in enumerate(pods[:30]):
            frame = feature_store.build_telemetry_frame(pod, node_pools, timestamp_sec=timestamp_sec)
            
            # Layer 3: NSGA-II Pre-Filtering
            top_candidates = nsga2_filter.filter_and_rank(frame)
            
            for cand in top_candidates:
                target_node = cand.target_node
                
                # Layer 4: Digital Twin 1000-Run Monte Carlo Simulation
                mc_result = mc_engine.run_simulation(frame, target_node)
                
                # Layer 5: Statistical Model Outputs (NSG & SLA Risk)
                eval_result = Layer5OutputCalculator.evaluate_candidate(
                    mc_result=mc_result,
                    pod=pod,
                    current_node=frame.current_node,
                    target_node=target_node
                )
                all_evaluations.append(eval_result)
                
    print(f"\nCompleted evaluation of {len(all_evaluations)} pod-target candidate pairs.")
    return all_evaluations

if __name__ == "__main__":
    results = run_sustainability_pipeline()
    
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
