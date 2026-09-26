"""
Hardware Heterogeneity & MIPS Normalization Module (Layer 3)
Calculates performance scaling, execution duration adjustment,
and power multipliers across heterogeneous CPU hardware architectures.
"""

from src.layer1_infra.cluster_config import NodeConfig

class HardwareHeterogeneityModel:
    """Models hardware speedup/slowdown based on MIPS ratings."""
    
    @staticmethod
    def get_execution_scaling_factor(source_node: NodeConfig, target_node: NodeConfig) -> float:
        """
        Calculates execution time scaling ratio.
        If target node has higher MIPS, execution time decreases (< 1.0).
        If target node has lower MIPS, execution time increases (> 1.0).
        """
        if target_node.mips_rating <= 0:
            return 1.0
        return round(source_node.mips_rating / target_node.mips_rating, 4)

    @staticmethod
    def scale_duration(base_duration_sec: float, source_node: NodeConfig, target_node: NodeConfig) -> float:
        """Adjusts workload execution duration for the target node's MIPS capacity."""
        scaling_factor = HardwareHeterogeneityModel.get_execution_scaling_factor(source_node, target_node)
        return round(base_duration_sec * scaling_factor, 2)
