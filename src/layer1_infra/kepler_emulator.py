"""
Kepler eBPF Energy Sensor & Power Calculation Module (Layer 1)
Implements per-pod energy and power metrics based on Kepler eBPF estimation models.
"""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class PodEnergyMetrics:
    pod_id: str
    instant_power_watts: float
    total_energy_joules: float
    cpu_utilization: float
    ram_utilization: float


class KeplerEmulator:
    """Emulates Kepler eBPF power metrics exporter for Kubernetes Pods."""
    
    def __init__(self, dram_power_w_per_gb: float = 0.35):
        self.dram_power_w_per_gb = dram_power_w_per_gb

    def estimate_pod_power(
        self,
        cpu_util: float,          # 0.0 to 1.0 fraction of core capacity
        ram_used_gb: float,
        node_idle_power_w: float = 90.0,
        node_max_power_w: float = 380.0,
        node_cpu_cores: int = 16
    ) -> float:
        """
        Computes dynamic pod power draw in Watts using Kepler eBPF linear dynamic formula.
        """
        # Per-core power dynamic range
        core_power_range = (node_max_power_w - node_idle_power_w) / max(1, node_cpu_cores)
        
        # CPU dynamic component
        cpu_power = cpu_util * core_power_range
        
        # DRAM memory power component
        dram_power = ram_used_gb * self.dram_power_w_per_gb
        
        # Static baseline allocation per active pod
        pod_idle_baseline = node_idle_power_w / node_cpu_cores
        
        total_power_w = pod_idle_baseline + cpu_power + dram_power
        return max(1.0, round(total_power_w, 3))

    def estimate_pod_energy(
        self,
        power_watts: float,
        duration_seconds: float
    ) -> float:
        """Computes total energy consumed in Joules (1 Watt * 1 Second = 1 Joule)."""
        return round(power_watts * duration_seconds, 3)

    def measure_pod(
        self,
        pod_id: str,
        cpu_util: float,
        ram_used_gb: float,
        duration_sec: float,
        idle_power_w: float = 90.0,
        max_power_w: float = 380.0
    ) -> PodEnergyMetrics:
        """Creates complete energy telemetry record for a given pod execution."""
        power_w = self.estimate_pod_power(cpu_util, ram_used_gb, idle_power_w, max_power_w)
        energy_j = self.estimate_pod_energy(power_w, duration_sec)
        
        return PodEnergyMetrics(
            pod_id=pod_id,
            instant_power_watts=power_w,
            total_energy_joules=energy_j,
            cpu_utilization=cpu_util,
            ram_utilization=ram_used_gb
        )
