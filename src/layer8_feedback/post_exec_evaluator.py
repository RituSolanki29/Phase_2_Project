"""
Post-Execution Evaluator (Layer 8: "Actual Energy & Carbon", "Performance Impact")
Closes the feedback loop: after a real pod has run on its bound node, this
computes the REALIZED Net Sustainability Gain with exactly the same
definition the Digital Twin predicted:

    realized NSG = predicted_stay_carbon
                   - (realized execution carbon on the bound node
                      + migration carbon actually incurred)

What was wrong before:
  * Realized carbon used hard-coded cpu_util=0.5, idle/max power 90/380 W and
    a default 2 GB of RAM, whereas the twin used the pod's real CPU request,
    its real memory and the bound node's own power curve.
  * Migration carbon was predicted but never subtracted from the realized
    value, so predicted vs realized NSG were anti-correlated (about -0.87).
  * The carbon intensity was read at wall-clock time rather than at the
    decision's own time axis.

Energy source (in priority order):
  1. REAL Kepler measurement for the pod, if Kepler is reachable.
  2. Otherwise the same node-specific power model the twin uses, applied to
     the pod's OBSERVED runtime (Kubernetes container start/finish) when
     available, else the predicted runtime.
"""

import json
import time
from typing import Optional

from src.layer1_infra import k8s_deployer
from src.layer1_infra.cluster_config import create_default_node_pools
from src.layer2_telemetry.telemetry_source import TelemetrySource
from src.layer8_feedback.reward_store import RewardStore, DEFAULT_LOG_PATH

POLL_INTERVAL_SEC = 5.0


class PostExecEvaluator:
    def __init__(self, seed: int = 42, log_path=DEFAULT_LOG_PATH, wait: bool = True):
        self.telemetry = TelemetrySource(sim_seed=seed)
        self.reward_store = RewardStore(log_path)
        self.log_path = log_path
        self.nodes = {n.node_id: n for n in create_default_node_pools()}
        self.wait = wait
        self._processed_pod_ids = set()

    def _pending_decisions(self):
        """Decision entries that don't yet have a later outcome entry."""
        if not self.reward_store.log_path.exists():
            return []
        pending = {}
        for line in self.reward_store.log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry["event"] == "decision":
                pending[entry["pod_id"]] = entry
            elif entry["event"] == "outcome":
                pending.pop(entry["pod_id"], None)
        return [d for pod_id, d in pending.items() if pod_id not in self._processed_pod_ids]

    # ------------------------------------------------------------------
    def compute_realized(self, decision_entry: dict, observed_runtime_sec: Optional[float] = None,
                         measured_energy_joules: Optional[float] = None) -> dict:
        """Pure computation (no sleeping / no cluster calls) so it can be unit-tested."""
        candidate = decision_entry["candidate"]
        ctx = decision_entry.get("context", {}) or {}
        bound_node_id = decision_entry["bound_node_id"]
        node = self.nodes.get(bound_node_id) or self.nodes[candidate["target_node_id"]]

        predicted_exec_sec = float(candidate.get("predicted_exec_time_sec", 30.0))
        migration_sec = float(candidate.get("migration_duration_sec", 0.0))
        in_place = bound_node_id == candidate.get("current_node_id")
        if in_place:
            migration_sec = 0.0
        # Runtime on the bound node: observed if we have it, else predicted.
        runtime_sec = float(observed_runtime_sec) if observed_runtime_sec else predicted_exec_sec

        # Energy: real measurement if provided, else the twin's own power model
        # evaluated on the BOUND node with the pod's real parameters.
        if measured_energy_joules is not None and measured_energy_joules > 0:
            energy_joules, energy_source = float(measured_energy_joules), "kepler"
        else:
            cpu_request = float(ctx.get("cpu_request", 1.0))
            assigned_mem = float(ctx.get("assigned_memory_gb", ctx.get("ram_request_gb", 2.0)))
            cpu_util = min(1.0, cpu_request / max(node.mips_rating / 1000.0, 0.001))
            power_w = self.telemetry.estimate_pod_power(cpu_util, assigned_mem, node.idle_power_w, node.max_power_w)
            energy_joules, energy_source = power_w * runtime_sec, "power_model"

        # Carbon intensity on the same time axis as the decision.
        decision_ts = float(ctx.get("timestamp_sec", candidate.get("timestamp_sec", 0.0)))
        ci = self.telemetry.get_current_carbon_intensity(node.region, decision_ts + migration_sec)

        exec_carbon = energy_joules / 3_600_000.0 * ci
        migration_carbon = 0.0 if in_place else float(candidate.get("migration_carbon_mean_gco2", 0.0))
        stay_baseline = float(candidate.get("stay_carbon_mean_gco2", 0.0))
        realized_nsg = stay_baseline - (exec_carbon + migration_carbon)

        return {
            "realized_nsg_gco2": realized_nsg,
            "detail": {
                "real_carbon": round(exec_carbon + migration_carbon, 6),
                "real_execution_carbon": round(exec_carbon, 6),
                "migration_carbon": round(migration_carbon, 6),
                "predicted_stay_baseline": round(stay_baseline, 6),
                "energy_source": energy_source,
                "runtime_sec": round(runtime_sec, 4),
                "carbon_intensity": round(ci, 3),
                "in_place": in_place,
                "bound_node_id": bound_node_id,
            },
        }

    def evaluate_one(self, decision_entry: dict) -> None:
        pod_id = decision_entry["pod_id"]
        candidate = decision_entry["candidate"]
        bound_node_id = decision_entry["bound_node_id"]
        k8s_pod_name = pod_id.lower().replace("_", "-")[:63]

        predicted_exec_sec = float(candidate.get("predicted_exec_time_sec", 30.0))
        migration_sec = float(candidate.get("migration_duration_sec", 0.0))
        wait_sec = predicted_exec_sec + migration_sec + 3.0

        observed_runtime, measured_energy = None, None
        if self.wait:
            print(f"[PostExec] Waiting ~{wait_sec:.0f}s for {pod_id} on {bound_node_id}...")
            time.sleep(min(wait_sec, 120.0))
            if self.telemetry.k8s_available:
                observed_runtime = k8s_deployer.get_pod_runtime_sec(k8s_pod_name)
                if self.telemetry._kepler_reachable:
                    real = self.telemetry._real_kepler.measure_pod(
                        pod_name=k8s_pod_name, namespace=k8s_deployer.DEFAULT_NAMESPACE,
                        elapsed_sec=observed_runtime or wait_sec)
                    if real is not None:
                        measured_energy = real.total_energy_joules

        out = self.compute_realized(decision_entry, observed_runtime, measured_energy)
        self.reward_store.record_outcome(pod_id, out["realized_nsg_gco2"], out["detail"])
        d = out["detail"]
        print(f"[PostExec] {pod_id}: realized NSG = {out['realized_nsg_gco2']:+.6f} gCO2 "
              f"(real carbon {d['real_carbon']:.6f} vs predicted stay baseline {d['predicted_stay_baseline']:.6f}; "
              f"source={d['energy_source']})")
        self._processed_pod_ids.add(pod_id)

    def run_forever(self) -> None:
        print("[PostExec] Watching reward log for decisions awaiting real-outcome measurement...")
        while True:
            for decision_entry in self._pending_decisions():
                try:
                    self.evaluate_one(decision_entry)
                except Exception as exc:
                    print(f"[PostExec] Failed to evaluate {decision_entry.get('pod_id')}: {exc}")
                    self._processed_pod_ids.add(decision_entry.get("pod_id"))
            time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    PostExecEvaluator().run_forever()