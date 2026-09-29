"""
Dynamic Feature Store & Telemetry Collector (Layer 2)
Aggregates workload parameters, Kepler energy telemetry (real when
available, modeled otherwise), node capacities, and carbon intensity
(real when available, modeled otherwise) into unified state frames.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from src.layer1_infra.trace_loader import WorkloadPod
from src.layer1_infra.cluster_config import NodeConfig
from src.layer1_infra.kepler_emulator import PodEnergyMetrics
from src.layer2_telemetry.telemetry_source import TelemetrySource

@dataclass
class UnifiedTelemetryFrame:
    timestamp_sec: float
    pod: WorkloadPod
    current_node: NodeConfig
    kepler_metrics: PodEnergyMetrics
    current_region_ci: float
    target_node_states: List[NodeConfig]
    target_region_cis: Dict[str, float]
    k8s_pod_name: Optional[str] = None          # set only if a real pod was deployed
    is_real_measurement: bool = False           # True if kepler_metrics came from real Kepler


class FeatureStore:
    """Central Telemetry Feature Store for scheduling and simulation layers."""

    def __init__(self, telemetry: TelemetrySource):
        self.telemetry = telemetry

    def build_telemetry_frame(
        self,
        pod: WorkloadPod,
        nodes: List[NodeConfig],
        timestamp_sec: float = 0.0,
        k8s_pod_name: Optional[str] = None,
        elapsed_sec: Optional[float] = None,
    ) -> UnifiedTelemetryFrame:
        """
        Constructs a complete unified telemetry snapshot for a pod evaluation
        cycle. If `k8s_pod_name` is given (pipeline_runner actually deployed
        this pod to a real cluster), current-node energy telemetry is REAL
        Kepler measurement when reachable; otherwise it is the same
        model-based estimate as before. Candidate/target nodes are always
        evaluated by the model in Layers 3-4 (see telemetry_source.py).
        """
        current_node = next((n for n in nodes if n.node_id == pod.source_node), nodes[0])
        current_ci = self.telemetry.get_current_carbon_intensity(current_node.region, timestamp_sec)

        cpu_util = min(1.0, pod.cpu_request / (current_node.mips_rating / 1000.0))
        kepler_metrics = self.telemetry.get_current_state_energy(
            pod_id=pod.pod_id,
            cpu_util=cpu_util,
            ram_used_gb=pod.assigned_memory_gb,
            duration_sec=pod.duration_sec,
            idle_power_w=current_node.idle_power_w,
            max_power_w=current_node.max_power_w,
            k8s_pod_name=k8s_pod_name,
            elapsed_sec=elapsed_sec,
        )
        is_real = bool(k8s_pod_name) and self.telemetry._kepler_reachable and kepler_metrics.total_energy_joules > 0

        region_cis = {}
        for n in nodes:
            if n.region not in region_cis:
                region_cis[n.region] = self.telemetry.get_current_carbon_intensity(n.region, timestamp_sec)

        return UnifiedTelemetryFrame(
            timestamp_sec=timestamp_sec,
            pod=pod,
            current_node=current_node,
            kepler_metrics=kepler_metrics,
            current_region_ci=current_ci,
            target_node_states=nodes,
            target_region_cis=region_cis,
            k8s_pod_name=k8s_pod_name,
            is_real_measurement=is_real,
        )