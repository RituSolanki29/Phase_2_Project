"""
Custom Scheduler Watch Loop (Layer 7: "Scheduler Extender")
A real controller that watches for Pods submitted with
spec.schedulerName == "sustainability-scheduler" and status.phase ==
Pending, runs each one through Layers 2-6 (telemetry -> NSGA-II shortlist
-> Digital Twin -> Layer 5 outputs -> PPO decision gate), and BINDS the pod
to the chosen node via the real Kubernetes Binding API
(k8s_deployer.bind_pod_to_node) -- the same mechanism the default
kube-scheduler uses. This is why this qualifies as a real scheduler
component, not a wrapper around kubectl.

See k8s_deployer.build_pod_spec()'s docstring for how a pod opts into this
path, and layer7_execution/pod_submission.py for how to submit demo pods
that way.

Run as a long-lived process for the live demo:
    python -m src.layer7_execution.custom_scheduler
"""

import time
from typing import List

from src.layer1_infra import k8s_deployer
from src.layer1_infra.cluster_config import create_default_node_pools, NodeConfig
from src.layer2_telemetry.telemetry_source import TelemetrySource
from src.layer2_telemetry.feature_store import FeatureStore
from src.layer3_scheduling.nsga2_filter import NSGA2PreFilter
from src.layer4_digital_twin.gbr_predictor import GBRRuntimePredictor
from src.layer4_digital_twin.clark_model import ClarkMigrationModel
from src.layer4_digital_twin.simpy_uncertainty import SimPyCarbonSampler
from src.layer4_digital_twin.monte_carlo import MonteCarloEngine
from src.layer5_outputs.nsg_calculator import Layer5OutputCalculator
from src.layer6_decision.ppo_agent import PPOAgent, DEFAULT_MODEL_PATH
from src.layer6_decision.decision_gate import DecisionGate
from src.layer7_execution.cooldown_manager import CooldownManager
from src.layer8_feedback.reward_store import RewardStore

SCHEDULER_NAME = "sustainability-scheduler"
POLL_INTERVAL_SEC = 5.0
MAX_DELAY_CYCLES = 3


class CustomScheduler:
    def __init__(self, csv_path_for_gbr_training: str, train_batch_size: int = 500, seed: int = 42):
        self.telemetry = TelemetrySource(sim_seed=seed)
        self.node_pools: List[NodeConfig] = create_default_node_pools()
        self.feature_store = FeatureStore(self.telemetry)

        print("[CustomScheduler] Training GBR predictor for live decisions...")
        from src.layer1_infra.trace_loader import TraceLoader
        pods = TraceLoader(csv_path_for_gbr_training).load_trace_batch(limit=train_batch_size)
        self.gbr = GBRRuntimePredictor(random_state=seed)
        self.gbr.train_time_split(pods, train_ratio=0.8)

        self.nsga2 = NSGA2PreFilter(self.telemetry, top_k=3, pop_size=40, n_gen=25, seed=seed)
        self.clark = ClarkMigrationModel()
        self.simpy_sampler = SimPyCarbonSampler(self.telemetry._sim_carbon)
        self.mc_engine = MonteCarloEngine(self.gbr, self.clark, self.simpy_sampler, self.telemetry,
                                           num_runs=300, random_seed=seed)

        self.ppo_agent = PPOAgent()
        try:
            self.ppo_agent.load(DEFAULT_MODEL_PATH)
            print(f"[CustomScheduler] Loaded trained policy from {DEFAULT_MODEL_PATH}")
        except FileNotFoundError:
            print(f"[CustomScheduler] No trained policy found at {DEFAULT_MODEL_PATH} -- "
                  f"run `python -m src.layer6_decision.train_policy` first before starting this scheduler.")
            raise

        self.gate = DecisionGate(self.ppo_agent)
        self.cooldown = CooldownManager()
        self.reward_store = RewardStore()
        self._delay_counts = {}

    def _evaluate_and_bind(self, k8s_pod) -> None:
        pod = k8s_deployer.workload_pod_from_annotations(k8s_pod)
        current_node = next((n for n in self.node_pools if n.node_id == pod.source_node), self.node_pools[0])

        timestamp_sec = time.time() % 86400
        frame = self.feature_store.build_telemetry_frame(pod, self.node_pools, timestamp_sec=timestamp_sec)
        candidates = self.nsga2.quick_rank_single(frame)
        if not candidates:
            print(f"[CustomScheduler] No feasible candidates for {pod.pod_id}; skipping this cycle.")
            return

        layer5_results = []
        for cand in candidates:
            mc_result = self.mc_engine.run_simulation(frame, cand.target_node)
            layer5_results.append(Layer5OutputCalculator.evaluate_candidate(
                mc_result, pod, frame.current_node, cand.target_node,
                timestamp_sec=timestamp_sec,
            ))

        decision = self.gate.decide_best_candidate(layer5_results)

        # A pod that keeps getting DELAYed must not stay Pending forever:
        # after MAX_DELAY_CYCLES, force-bind the lowest-SLA-risk candidate.
        if decision.action == "DELAY":
            n = self._delay_counts.get(pod.pod_id, 0) + 1
            self._delay_counts[pod.pod_id] = n
            if n >= MAX_DELAY_CYCLES:
                best = min(layer5_results, key=lambda r: r.sla_risk_score)
                decision.action = "START"
                decision.target_node_id = best.target_node_id
                decision.rationale += f" Forced start after {n} delayed cycles (lowest SLA risk candidate)."

        chosen_result = next((r for r in layer5_results if r.target_node_id == decision.target_node_id),
                             layer5_results[0])
        target_node = next(n for n in self.node_pools if n.node_id == decision.target_node_id)

        print(f"[CustomScheduler] {pod.pod_id}: {decision.action} -> {target_node.node_id}. {decision.rationale}")

        action = decision.action
        # Cooldown only guards MIGRATIONS (thrash prevention). If the target is
        # cooling down, start the pod in place instead of skipping it.
        if action == "MIGRATE" and self.cooldown.is_in_cooldown(target_node.node_id):
            print(f"[CustomScheduler]   -> {target_node.node_id} in cooldown; starting in place on {current_node.node_id}.")
            action = "START"
            target_node = current_node
            chosen_result = next((r for r in layer5_results if r.target_node_id == current_node.node_id),
                                 chosen_result)

        if action in ("MIGRATE", "START"):
            try:
                k8s_deployer.bind_pod_to_node(k8s_pod.metadata.name, target_node.node_id, k8s_deployer.DEFAULT_NAMESPACE)
                if action == "MIGRATE":
                    self.cooldown.start_cooldown(target_node.node_id)
                print(f"[CustomScheduler]   -> Bound {k8s_pod.metadata.name} to node {target_node.node_id}.")
                self._delay_counts.pop(pod.pod_id, None)
                # Only log a decision once the bind actually succeeded.
                self.reward_store.record_decision(
                    pod.pod_id, action, chosen_result, target_node.node_id,
                    context={
                        "cpu_request": pod.cpu_request,
                        "ram_request_gb": pod.ram_request_gb,
                        "assigned_memory_gb": pod.assigned_memory_gb,
                        "duration_sec": pod.duration_sec,
                        "timestamp_sec": timestamp_sec,
                        "nominal_node_id": current_node.node_id,
                    },
                )
            except Exception as exc:
                print(f"[CustomScheduler]   -> Bind API reported: {exc}")
        else:
            print(f"[CustomScheduler]   -> {action}: not binding this cycle "
                  f"(delay {self._delay_counts.get(pod.pod_id, 0)}/{MAX_DELAY_CYCLES}).")

    def run_forever(self) -> None:
        print(f"[CustomScheduler] Watching for Pending pods with schedulerName='{SCHEDULER_NAME}'...")
        while True:
            try:
                pending = k8s_deployer.list_pending_unscheduled_pods(SCHEDULER_NAME)
                for k8s_pod in pending:
                    self._evaluate_and_bind(k8s_pod)
            except Exception as exc:
                print(f"[CustomScheduler] Watch loop error (will retry): {exc}")
            time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the live custom scheduler loop.")
    parser.add_argument("--csv-path", default="data/borg_traces_data.csv")
    parser.add_argument("--train-batch-size", type=int, default=500)
    args = parser.parse_args()

    scheduler = CustomScheduler(args.csv_path, args.train_batch_size)
    scheduler.run_forever()