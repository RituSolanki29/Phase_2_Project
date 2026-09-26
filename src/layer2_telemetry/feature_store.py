"""
Dynamic Feature Store & Telemetry Collector (Layer 2)
Aggregates workload parameters, Kepler energy telemetry, node capacities,
and real-time grid carbon intensity into unified state frames.
"""

from dataclasses import dataclass
from typing import Dict, List, Any
from src.layer1_infra.trace_loader import WorkloadPod
from src.layer1_infra.cluster_config import NodeConfig
from src.layer1_infra.kepler_emulator import KeplerEmulator, PodEnergyMetrics
from src.layer2_telemetry.carbon_feed import CarbonFeed

@dataclass
class UnifiedTelemetryFrame:
    timestamp_sec: float
    pod: WorkloadPod
    current_node: NodeConfig
    kepler_metrics: PodEnergyMetrics
    current_region_ci: float
    target_node_states: List[NodeConfig]
    target_region_cis: Dict[str, float]


class FeatureStore:
    """Central Telemetry Feature Store for scheduling and simulation layers."""
    
    def __init__(self, carbon_feed: CarbonFeed, kepler_emulator: KeplerEmulator):
        self.carbon_feed = carbon_feed
        self.kepler = kepler_emulator

    def build_telemetry_frame(
        self,
        pod: WorkloadPod,
        nodes: List[NodeConfig],
        timestamp_sec: float = 0.0
    ) -> UnifiedTelemetryFrame:
        """Constructs a complete unified telemetry snapshot for a pod evaluation cycle."""
        # Find current node and region
        current_node = next((n for n in nodes if n.node_id == pod.source_node), nodes[0])
        current_ci = self.carbon_feed.get_carbon_intensity(current_node.region, timestamp_sec)
        
        # Calculate Kepler energy telemetry on current node
        cpu_util = min(1.0, pod.cpu_request / (current_node.mips_rating / 1000.0))
        kepler_metrics = self.kepler.measure_pod(
            pod_id=pod.pod_id,
            cpu_util=cpu_util,
            ram_used_gb=pod.assigned_memory_gb,
            duration_sec=pod.duration_sec,
            idle_power_w=current_node.idle_power_w,
            max_power_w=current_node.max_power_w
        )
        
        # Collect target region carbon intensity snapshots
        region_cis = {}
        for n in nodes:
            if n.region not in region_cis:
                region_cis[n.region] = self.carbon_feed.get_carbon_intensity(n.region, timestamp_sec)

        return UnifiedTelemetryFrame(
            timestamp_sec=timestamp_sec,
            pod=pod,
            current_node=current_node,
            kepler_metrics=kepler_metrics,
            current_region_ci=current_ci,
            target_node_states=nodes,
            target_region_cis=region_cis
        )
