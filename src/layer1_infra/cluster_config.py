"""
Cluster & Node Configuration Module (Layer 1)
Defines node pools, hardware heterogeneity (MIPS ratings), regional parameters,
and Kubernetes Kind cluster manifest generation.
"""

from dataclasses import dataclass, field
from typing import Dict, List
import yaml

@dataclass
class NodeConfig:
    node_id: str
    region: str
    mips_rating: float  # Million Instructions Per Second
    total_ram_gb: float
    idle_power_w: float = 100.0  # Baseline idle power consumption in Watts
    max_power_w: float = 400.0   # Max load power consumption in Watts
    free_ram_gb: float = field(init=False)

    def __post_init__(self):
        self.free_ram_gb = self.total_ram_gb


@dataclass
class RegionConfig:
    region_id: str
    display_name: str
    baseline_mips: float
    avg_carbon_intensity: float  # gCO2/kWh baseline

DEFAULT_REGIONS: Dict[str, RegionConfig] = {
    "DC_EU": RegionConfig("DC_EU", "EU Ireland (Green Solar/Wind)", 4000.0, 180.0),
    "DC_US": RegionConfig("DC_US", "US Virginia (Standard Grid Mix)", 3500.0, 420.0),
    "DC_ASIA": RegionConfig("DC_ASIA", "APAC Tokyo (High Baseline Grid)", 2800.0, 520.0),
}

def create_default_node_pools() -> List[NodeConfig]:
    """Generates synthetic multi-region node pools simulating heterogeneous hardware."""
    nodes = []
    # EU Region (High performance, high green energy window)
    for i in range(1, 4):
        nodes.append(NodeConfig(f"eu-node-{i}", "DC_EU", mips_rating=4000.0, total_ram_gb=64.0, idle_power_w=90.0, max_power_w=380.0))

    # US Region (Standard performance)
    for i in range(1, 4):
        nodes.append(NodeConfig(f"us-node-{i}", "DC_US", mips_rating=3500.0, total_ram_gb=48.0, idle_power_w=100.0, max_power_w=400.0))

    # ASIA Region (Slower/Legacy hardware)
    for i in range(1, 4):
        nodes.append(NodeConfig(f"asia-node-{i}", "DC_ASIA", mips_rating=2800.0, total_ram_gb=32.0, idle_power_w=110.0, max_power_w=420.0))

    return nodes

def generate_kind_config(nodes: List[NodeConfig]) -> str:
    """
    Generates a Kind (Kubernetes in Docker) YAML cluster manifest
    with topology.kubernetes.io/region node labels.
    """
    kind_doc = {
        "kind": "Cluster",
        "apiVersion": "kind.x-k8s.io/v1alpha4",
        "nodes": [
            {"role": "control-plane"}
        ]
    }
    
    for n in nodes:
        kind_doc["nodes"].append({
            "role": "worker",
            "kubeadmConfigPatches": [
                f"kind: JoinConfiguration\nnodeRegistration:\n  kubeletExtraArgs:\n    node-labels: \"topology.kubernetes.io/region={n.region},mips={n.mips_rating},node-id={n.node_id}\""
            ]
        })
    
    return yaml.dump(kind_doc, sort_keys=False)
