"""
Clark Live-Migration Overhead Model (Layer 4)
Quantifies workload migration network transfer duration, memory checkpointing energy,
and net migration carbon overhead.
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
        bandwidth_gbps: float = 1.0,          # Network bandwidth in Gbps (0.125 GB/s)
        network_energy_j_per_gb: float = 12.0, # Energy per GB transferred across regions
        switch_power_w: float = 15.0,         # Intermediate network switch power draw
        fixed_latency_sec: float = 1.5        # Pre-copy memory checkpoint & handshake overhead
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
        source_region_ci: float
    ) -> MigrationCostEstimate:
        """
        Calculates migration duration, checkpoint transfer energy,
        and total carbon cost incurred by performing the migration.
        """
        if source_node.node_id == target_node.node_id:
            # Same node: zero migration overhead
            return MigrationCostEstimate(0.0, 0.0, 0.0, 0.0)

        # Checkpoint size (dirty memory pages to transfer)
        checkpoint_size_gb = max(0.1, pod.assigned_memory_gb)
        
        # Transmission time + pre-copy dirty memory rounds + handshake overhead
        transfer_time_sec = checkpoint_size_gb / self.bandwidth_gb_per_sec
        
        # Inter-region migrations add cross-datacenter latency
        region_latency = 0.0 if source_node.region == target_node.region else 2.5
        migration_duration = transfer_time_sec + self.fixed_latency_sec + region_latency

        # Network transfer energy (J) = memory transferred * energy_per_gb + duration * switch_power
        transfer_energy_j = (checkpoint_size_gb * self.network_energy_j_per_gb) + (migration_duration * self.switch_power_w)
        
        # Convert Joules to kWh (1 kWh = 3.6e6 Joules)
        energy_kwh = transfer_energy_j / 3.6e6
        
        # Migration carbon footprint emitted in source region during transfer
        migration_carbon_gco2 = energy_kwh * source_region_ci

        return MigrationCostEstimate(
            migration_duration_sec=round(migration_duration, 2),
            checkpoint_size_gb=round(checkpoint_size_gb, 2),
            network_energy_joules=round(transfer_energy_j, 2),
            migration_carbon_gco2=round(migration_carbon_gco2, 4)
        )
