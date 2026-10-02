"""
Offline PPO pretraining entry point (Layer 6).
Runs the full Layer 1-5 pipeline once to generate a batch of Digital Twin
candidate evaluations, then pretrains the PPO policy on them and saves it
to disk for api.py / orchestrator_loop.py to load.

Run:
    python -m src.layer6_decision.train_policy --batch-size 1000 --timesteps 20000
"""

import argparse
from pathlib import Path

from src.layer1_infra.cluster_config import create_default_node_pools
from src.layer1_infra.trace_loader import TraceLoader
from src.layer2_telemetry.telemetry_source import TelemetrySource
from src.layer2_telemetry.feature_store import FeatureStore
from src.layer3_scheduling.nsga2_filter import NSGA2PreFilter
from src.layer4_digital_twin.gbr_predictor import GBRRuntimePredictor
from src.layer4_digital_twin.clark_model import ClarkMigrationModel
from src.layer4_digital_twin.simpy_uncertainty import SimPyCarbonSampler
from src.layer4_digital_twin.monte_carlo import MonteCarloEngine
from src.layer5_outputs.nsg_calculator import Layer5OutputCalculator
from src.layer6_decision.ppo_agent import PPOAgent, DEFAULT_MODEL_PATH
from src.pipeline_runner import DEFAULT_CSV_PATH


def generate_training_candidates(csv_path, batch_size, pods_per_cycle, mc_runs, seed=42):
    telemetry = TelemetrySource(kubeconfig_path=None, sim_seed=seed)
    node_pools = create_default_node_pools()
    loader = TraceLoader(str(csv_path))
    pods = loader.load_trace_batch(limit=batch_size, compression_factor=10.0)

    feature_store = FeatureStore(telemetry)
    gbr = GBRRuntimePredictor(random_state=seed)
    gbr.train_time_split(pods, train_ratio=0.8)

    nsga2 = NSGA2PreFilter(telemetry, top_k=2, pop_size=40, n_gen=25, seed=seed)
    clark = ClarkMigrationModel()
    simpy_sampler = SimPyCarbonSampler(telemetry._sim_carbon)
    mc_engine = MonteCarloEngine(gbr, clark, simpy_sampler, telemetry, num_runs=mc_runs, random_seed=seed)

    results = []
    # Sweep across a few timestamps so the policy sees varied carbon-intensity
    # conditions during pretraining, not just one snapshot.
    for ts in [0.0, 21600.0, 43200.0, 64800.0]:
        cycle = pods[:pods_per_cycle]
        frames = [feature_store.build_telemetry_frame(p, node_pools, timestamp_sec=ts) for p in cycle]
        candidates_by_pod = nsga2.filter_and_rank_batch(frames)
        for frame in frames:
            for cand in candidates_by_pod.get(frame.pod.pod_id, []):
                mc_result = mc_engine.run_simulation(frame, cand.target_node)
                results.append(Layer5OutputCalculator.evaluate_candidate(
                    mc_result, frame.pod, frame.current_node, cand.target_node,
                    scheduling_window=f"t={ts}", timestamp_sec=ts,
                ))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain the Layer 6 PPO scheduling policy offline.")
    parser.add_argument("--csv-path", default=str(DEFAULT_CSV_PATH))
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--pods-per-cycle", type=int, default=30)
    parser.add_argument("--mc-runs", type=int, default=500)
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    args = parser.parse_args()

    print(f"[1/3] Generating training candidates from {args.batch_size} trace workloads...")
    candidates = generate_training_candidates(args.csv_path, args.batch_size, args.pods_per_cycle, args.mc_runs)
    print(f"      Generated {len(candidates)} candidate evaluations.")

    print(f"[2/3] Pretraining PPO policy for {args.timesteps} timesteps...")
    agent = PPOAgent()
    agent.pretrain(candidates, total_timesteps=args.timesteps)
    print(f"      Auto-scaled nsg_scale = {agent.nsg_scale:.6f}")

    print(f"[3/3] Saving policy to {args.model_path}...")
    agent.save(Path(args.model_path))
    print("Done. Start the API with:\n  uvicorn src.layer6_decision.api:app --port 8080")