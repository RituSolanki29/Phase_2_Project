"""
Clark Live-Migration Overhead Model (Layer 4)
Quantifies workload migration network transfer duration, memory checkpointing energy,
and net migration carbon overhead.

Note: Migration cost only applies to LIVE migrations of already-running pods.
Fresh pod placements (new submissions) incur zero migration overhead — they are
simply bound to a node and started there.
"""

from dataclasses import dataclass
from src.layer1_infra.trace_loader import WorkloadPod
from src.layer1_infra.cluster_config import NodeConfig

@dataclass
class MigrationCostEstimate:
    migration_duration_sec: float
    checkpoint_size_gb: float
    network_energy_joules: float
    migration_carbon_gco2: float


class ClarkMigrationModel:
    """Models live migration cost based on Clark et al. live memory migration equations."""
    
    def __init__(
        self,
        bandwidth_gbps: float = 1.0,           # Network bandwidth in Gbps (0.125 GB/s)
        network_energy_j_per_gb: float = 12.0,  # Energy per GB transferred across regions
        switch_power_w: float = 15.0,           # Intermediate network switch power draw
        fixed_latency_sec: float = 1.5          # Pre-copy memory checkpoint & handshake overhead
    ):
        self.bandwidth_gb_per_sec = bandwidth_gbps / 8.0
        self.network_energy_j_per_gb = network_energy_j_per_gb
        self.switch_power_w = switch_power_w
        self.fixed_latency_sec = fixed_latency_sec

    def estimate_migration_cost(
        self,
        pod: WorkloadPod,
        source_node: NodeConfig,
        target_node: NodeConfig,
        source_region_ci: float,
        is_new_placement: bool = True,
    ) -> MigrationCostEstimate:
        """
        Calculates network transfer duration, payload transmission energy,
        and net carbon cost incurred by performing remote scheduling / migration.
        """
        if source_node.node_id == target_node.node_id:
            # Same node: zero migration overhead
            return MigrationCostEstimate(0.0, 0.0, 0.0, 0.0)

        is_cross_region = source_node.region != target_node.region
        checkpoint_size_gb = max(0.05, float(pod.assigned_memory_gb))

        if is_new_placement:
            # New placement: container image & input payload synchronization
            # Cross-region has WAN RTT, handshake, and container pull overhead (~1.5s to 2.2s)
            # Intra-region has local LAN dispatch overhead (~0.2s to 0.4s)
            region_latency = 1.6 if is_cross_region else 0.1
            fixed_latency = 0.5 if is_cross_region else 0.1
            transfer_time_sec = (checkpoint_size_gb / self.bandwidth_gb_per_sec) * 0.4
            migration_duration = transfer_time_sec + fixed_latency + region_latency

            # Incremental switch power draw for the flow (1.5W for WAN routers, 0.5W local switch)
            incremental_switch_w = 1.5 if is_cross_region else 0.3
            transfer_energy_j = (checkpoint_size_gb * 4.0) + (migration_duration * incremental_switch_w)
        else:
            # Live migration of in-flight container state (Clark pre-copy)
            region_latency = 2.0 if is_cross_region else 0.0
            transfer_time_sec = checkpoint_size_gb / self.bandwidth_gb_per_sec
            migration_duration = transfer_time_sec + self.fixed_latency_sec + region_latency
            transfer_energy_j = (checkpoint_size_gb * self.network_energy_j_per_gb) + (migration_duration * 3.0)

        # Convert Joules to kWh and compute emitted carbon
        energy_kwh = transfer_energy_j / 3_600_000.0
        migration_carbon_gco2 = energy_kwh * source_region_ci

        return MigrationCostEstimate(
            migration_duration_sec=round(float(migration_duration), 2),
            checkpoint_size_gb=round(float(checkpoint_size_gb), 4),
            network_energy_joules=round(float(transfer_energy_j), 4),
            migration_carbon_gco2=round(float(migration_carbon_gco2), 6),
        )

