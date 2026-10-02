"""
Post-Execution Evaluator (Layer 8: "Actual Energy & Carbon", "Performance Impact")
Closes the feedback loop: after a real pod has actually run on its bound
node, this measures the REAL achieved energy/carbon via TelemetrySource
(real Kepler when reachable; see telemetry_source.py), computes the
REALIZED Net Sustainability Gain against the Digital Twin's predicted
"stay" baseline, and records it via RewardStore so the PPO policy can be
fine-tuned on reality (see retrain_policy.py).

Run as a companion process to custom_scheduler.py during a live demo:
    python -m src.layer8_feedback.post_exec_evaluator
"""

import json
import time

from src.layer1_infra import k8s_deployer
from src.layer2_telemetry.telemetry_source import TelemetrySource
from src.layer8_feedback.reward_store import RewardStore, DEFAULT_LOG_PATH

POLL_INTERVAL_SEC = 5.0


class PostExecEvaluator:
    def __init__(self, seed: int = 42):
        self.telemetry = TelemetrySource(sim_seed=seed)
        self.reward_store = RewardStore()
        self._processed_pod_ids = set()

    def _pending_decisions(self):
        """Reads decision entries from the reward log that don't yet have a
        matching outcome entry, so we know what to go measure."""
        if not DEFAULT_LOG_PATH.exists():
            return []
        decisions, outcomes_seen = {}, set()
        for line in DEFAULT_LOG_PATH.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry["event"] == "decision":
                decisions[entry["pod_id"]] = entry
            elif entry["event"] == "outcome":
                outcomes_seen.add(entry["pod_id"])
        return [d for pod_id, d in decisions.items()
                if pod_id not in outcomes_seen and pod_id not in self._processed_pod_ids]

    def evaluate_one(self, decision_entry: dict) -> None:
        pod_id = decision_entry["pod_id"]
        candidate = decision_entry["candidate"]
        bound_node_id = decision_entry["bound_node_id"]
        # The k8s pod name is deterministic from pod_id via the same
        # sanitization k8s_deployer uses when it created the pod.
        k8s_pod_name = pod_id.lower().replace("_", "-")[:63]

        predicted_exec_sec = float(candidate.get("predicted_exec_time_sec", 30.0))
        migration_sec = float(candidate.get("migration_duration_sec", 0.0))
        wait_sec = predicted_exec_sec + migration_sec + 3.0  # small safety margin

        print(f"[PostExec] Waiting ~{wait_sec:.0f}s for {pod_id} to finish on {bound_node_id} before measuring...")
        time.sleep(min(wait_sec, 120.0))  # cap the wait for demo responsiveness

        region = candidate.get("target_region", "DC_EU")
        real_metrics = self.telemetry.get_current_state_energy(
            pod_id=pod_id, cpu_util=0.5, ram_used_gb=float(candidate.get("ram_request_gb", 2.0)),
            duration_sec=predicted_exec_sec, idle_power_w=90.0, max_power_w=380.0,
            k8s_pod_name=k8s_pod_name, elapsed_sec=wait_sec,
        )
        real_ci = self.telemetry.get_current_carbon_intensity(region, time.time() % 86400)
        real_energy_kwh = real_metrics.total_energy_joules / 3_600_000.0
        real_carbon_gco2 = real_energy_kwh * real_ci

        predicted_stay_carbon = float(candidate.get("stay_carbon_mean_gco2", 0.0))
        realized_nsg = predicted_stay_carbon - real_carbon_gco2

        self.reward_store.record_outcome(
            pod_id, realized_nsg_gco2=realized_nsg,
            detail=f"real_carbon={real_carbon_gco2:.6f}gCO2 (measured={real_metrics.pod_id!r}), "
                   f"predicted_stay_baseline={predicted_stay_carbon:.6f}gCO2",
        )
        print(f"[PostExec] {pod_id}: realized NSG = {realized_nsg:+.6f} gCO2 "
              f"(real carbon {real_carbon_gco2:.6f} vs predicted stay baseline {predicted_stay_carbon:.6f})")
        self._processed_pod_ids.add(pod_id)

    def run_forever(self) -> None:
        print("[PostExec] Watching reward log for decisions awaiting real-outcome measurement...")
        while True:
            for decision_entry in self._pending_decisions():
                try:
                    self.evaluate_one(decision_entry)
                except Exception as exc:
                    print(f"[PostExec] Failed to evaluate {decision_entry.get('pod_id')}: {exc}")
            time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    PostExecEvaluator().run_forever()