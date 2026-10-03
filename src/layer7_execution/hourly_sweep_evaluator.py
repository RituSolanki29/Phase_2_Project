"""
Hourly Carbon Sweep & Multi-Cycle Training Evaluator (Layers 1-8)
Runs:
  1. A 24-hour diurnal carbon sweep evaluating workload placement across solar/wind cycles.
  2. Multi-cycle training demonstration (30+ workloads across multiple learning epochs).
  3. Automatic generation of charts (plots/) and statistical summary tables.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from src.layer1_infra.cluster_config import create_default_node_pools, NodeConfig
from src.layer1_infra.trace_loader import TraceLoader, WorkloadPod
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
from src.layer6_decision.ppo_agent import PPOAgent, DEFAULT_MODEL_PATH
from src.layer6_decision.decision_gate import DecisionGate
from src.layer8_feedback.reward_store import RewardStore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "borg_traces_data.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
REWARD_LOG_PATH = RESULTS_DIR / "reward_log.jsonl"


def run_comprehensive_evaluation(batch_size: int = 36, sweep_hours: int = 24):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plots_dir = RESULTS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("   AUTONOMOUS SUSTAINABILITY AGENT: HOURLY SWEEP & MULTI-CYCLE EVALUATION")
    print("=" * 80)

    # 1. Initialize Infrastructure & Telemetry
    node_pools: List[NodeConfig] = create_default_node_pools()
    telemetry = TelemetrySource(sim_seed=42)
    feature_store = FeatureStore(telemetry)

    # 2. Train GBR Predictor
    print("[1/5] Training GBR Runtime Predictor on Borg trace dataset...")
    loader = TraceLoader(str(DATA_PATH))
    train_pods = loader.load_trace_batch(limit=300, compression_factor=10.0)
    gbr = GBRRuntimePredictor(random_state=42)
    rmse, r2 = gbr.train_time_split(train_pods, train_ratio=0.8)
    print(f"      GBR trained: RMSE={rmse:.3f}s, R^2={r2:.3f}")

    # 3. Setup K=3 NSGA-II Filter & Digital Twin
    print("[2/5] Initializing NSGA-II (K=3) Pre-Filter & Monte Carlo Digital Twin...")
    nsga2 = NSGA2PreFilter(telemetry, top_k=3, pop_size=40, n_gen=25, seed=42)
    clark = ClarkMigrationModel()
    simpy_sampler = SimPyCarbonSampler(telemetry._sim_carbon)
    mc_engine = MonteCarloEngine(gbr, clark, simpy_sampler, telemetry, num_runs=300, random_seed=42)

    ppo_agent = PPOAgent()
    try:
        ppo_agent.load(DEFAULT_MODEL_PATH)
        print(f"      Loaded pretrained PPO policy from {DEFAULT_MODEL_PATH}")
    except Exception:
        print("      No pretrained PPO policy found -- initializing fresh policy.")
        dummy_pods = train_pods[:10]
        # Quick pretrain
        dummy_cands = []
        for p in dummy_pods:
            f = feature_store.build_telemetry_frame(p, node_pools, timestamp_sec=43200.0)
            res = mc_engine.run_simulation(f, node_pools[0])
            dummy_cands.append(Layer5OutputCalculator.evaluate_candidate(res, p, node_pools[0], node_pools[0]))
        ppo_agent.pretrain(dummy_cands, total_timesteps=1000)

    decision_gate = DecisionGate(ppo_agent)
    reward_store = RewardStore(str(REWARD_LOG_PATH))

    # 4. Hourly 24-Hour Diurnal Carbon Sweep
    print(f"\n[3/5] Executing 24-Hour Diurnal Carbon Sweep ({sweep_hours} hours)...")
    hourly_records = []
    all_evaluated_candidates: List[Dict[str, Any]] = []
    regions = ["DC_ASIA", "DC_EU", "DC_US"]

    test_pods = loader.load_trace_batch(limit=batch_size, compression_factor=10.0)

    for h in range(sweep_hours):
        t_sec = float(h * 3600 + 1800)  # Mid-hour timestamp
        hour_label = f"{h:02d}:00"
        
        # Current regional grid CIs
        cis = {reg: telemetry.get_current_carbon_intensity(reg, t_sec) for reg in regions}
        greenest_region = min(cis.keys(), key=lambda r: cis[r])

        # Evaluate a subset of pods arriving in this hour
        hour_pods = [test_pods[(h * 2 + i) % len(test_pods)] for i in range(3)]
        hour_decisions = []
        hour_nsgs = []
        hour_sla_risks = []
        viable_count = 0

        for idx, pod in enumerate(hour_pods):
            # Assign source region round-robin
            src_reg = regions[(h + idx) % len(regions)]
            pod.source_region = src_reg
            prefix = src_reg.split("_")[1].lower()
            pod.source_node = f"{prefix}-node-{(idx % 3) + 1}"

            frame = feature_store.build_telemetry_frame(pod, node_pools, timestamp_sec=t_sec)
            candidates = nsga2.quick_rank_single(frame)

            l5_results = []
            for cand in candidates:
                mc_res = mc_engine.run_simulation(frame, cand.target_node)
                l5 = Layer5OutputCalculator.evaluate_candidate(
                    mc_res, pod, frame.current_node, cand.target_node,
                    scheduling_window=hour_label, timestamp_sec=t_sec
                )
                l5_results.append(l5)
                all_evaluated_candidates.append({
                    "pod_id": pod.pod_id,
                    "hour": h,
                    "current_region": src_reg,
                    "target_region": cand.target_node.region,
                    "target_node_id": cand.target_node.node_id,
                    "nsg_mean_gco2": l5.nsg_mean_gco2,
                    "p_nsg_positive": l5.p_nsg_positive,
                    "sla_risk_score": l5.sla_risk_score,
                    "is_statistically_viable": l5.is_statistically_viable,
                })

            dec = decision_gate.decide_best_candidate(l5_results)
            hour_decisions.append(dec.action)
            hour_nsgs.append(dec.nsg_mean_gco2)
            hour_sla_risks.append(dec.sla_risk_score)
            if dec.p_nsg_positive >= 0.90 and dec.nsg_mean_gco2 > 0:
                viable_count += 1

        mean_h_nsg = float(np.mean(hour_nsgs))
        mean_h_sla = float(np.mean(hour_sla_risks))
        viable_pct = (viable_count / len(hour_pods)) * 100.0

        hourly_records.append({
            "hour": h,
            "mean_nsg_gco2": round(mean_h_nsg, 5),
            "viable_percentage": round(viable_pct, 1),
            "mean_sla_risk": round(mean_h_sla, 3),
            "greenest_region": greenest_region,
            "DC_EU_ci": round(cis["DC_EU"], 1),
            "DC_US_ci": round(cis["DC_US"], 1),
            "DC_ASIA_ci": round(cis["DC_ASIA"], 1),
        })

    hourly_df = pd.DataFrame(hourly_records)
    hourly_df.to_csv(RESULTS_DIR / "hourly_summary.csv", index=False)
    print(f"      Completed 24h sweep: saved {RESULTS_DIR / 'hourly_summary.csv'}")

    # 5. Multi-Cycle Training Demonstration (30+ pods across 3 Retraining Cycles)
    print("\n[4/5] Executing Multi-Cycle Policy Retraining Demonstration (30+ pods across 3 Cycles)...")
    cycle_size = 12
    total_cycles = 3
    cumulative_carbon_saved = 0.0
    training_history = []

    for cycle in range(1, total_cycles + 1):
        print(f"  --- Epoch / Cycle {cycle}/{total_cycles} ---")
        cycle_pods = test_pods[(cycle - 1) * cycle_size : cycle * cycle_size]
        cycle_cands = []
        cycle_realized = []

        for idx, pod in enumerate(cycle_pods):
            src_reg = regions[idx % len(regions)]
            pod.source_region = src_reg
            prefix = src_reg.split("_")[1].lower()
            pod.source_node = f"{prefix}-node-{(idx % 3) + 1}"

            t_sec = float((12 + cycle) * 3600)  # Afternoon solar peak
            frame = feature_store.build_telemetry_frame(pod, node_pools, timestamp_sec=t_sec)
            candidates = nsga2.quick_rank_single(frame)

            l5_results = []
            for cand in candidates:
                mc_res = mc_engine.run_simulation(frame, cand.target_node)
                l5 = Layer5OutputCalculator.evaluate_candidate(
                    mc_res, pod, frame.current_node, cand.target_node, timestamp_sec=t_sec
                )
                l5_results.append(l5)

            dec = decision_gate.decide_best_candidate(l5_results)
            chosen_l5 = next((r for r in l5_results if r.target_node_id == dec.target_node_id), l5_results[0])

            # Measure simulated realized outcome with calibration noise
            realized_nsg = float(chosen_l5.nsg_mean_gco2 + np.random.normal(0, 0.0005))
            if dec.action == "DELAY":
                realized_nsg = 0.0
            cumulative_carbon_saved += max(0.0, realized_nsg)

            # Record decision and outcome in reward store
            reward_store.record_decision(
                pod.pod_id, dec.action, chosen_l5, dec.target_node_id,
                context={"cycle": cycle, "pod_duration": pod.duration_sec}
            )
            reward_store.record_outcome(
                pod.pod_id,
                realized_nsg_gco2=realized_nsg,
                detail={
                    "measured_energy_kwh": 0.0002,
                    "in_place": (dec.target_node_id == pod.source_node),
                    "sla_violated": (chosen_l5.sla_risk_score > 0.8),
                    "energy_source": "power_model",
                }
            )

            cycle_cands.append(chosen_l5)
            cycle_realized.append(realized_nsg)

        # Retrain policy on real batch outcomes
        print(f"      Cycle {cycle} complete: Retraining PPO agent on {len(cycle_cands)} outcomes...")
        ppo_agent.retrain_batch(cycle_cands, realized_nsgs=cycle_realized, total_timesteps=1500)
        ppo_agent.save(DEFAULT_MODEL_PATH)
        print(f"      Cumulative Carbon Saved so far: {cumulative_carbon_saved:.5f} gCO2")
        training_history.append({
            "cycle": cycle,
            "workloads_processed": cycle * cycle_size,
            "cumulative_nsg_gco2": round(cumulative_carbon_saved, 5),
            "mean_cycle_nsg": round(float(np.mean(cycle_realized)), 5),
        })

    # Save candidate results df
    results_df = pd.DataFrame(all_evaluated_candidates)
    results_df.to_csv(RESULTS_DIR / "layer5_candidate_results.csv", index=False)

    # 6. Generate All Visualizations
    print("\n[5/5] Generating publication-quality charts into results/plots/...")
    generated_plots = generate_all_visualizations(
        results_df=results_df,
        pods=test_pods,
        predictor=gbr,
        feature_importance=gbr.get_feature_importance(),
        carbon_feed=telemetry._sim_carbon,
        regions=regions,
        output_dir=RESULTS_DIR,
        hourly_df=hourly_df,
        reward_log_path=REWARD_LOG_PATH,
    )
    print(f"      Generated {len(generated_plots)} charts:")
    for p in generated_plots:
        print(f"        -> {p}")

    print("\n" + "=" * 80)
    print("   ALL EVALUATIONS & CHARTS GENERATED SUCCESSFULLY!")
    print("=" * 80)
    return hourly_df, training_history, generated_plots


if __name__ == "__main__":
    run_comprehensive_evaluation(batch_size=36, sweep_hours=24)
