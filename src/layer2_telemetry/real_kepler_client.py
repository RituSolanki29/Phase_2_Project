"""
Real Kepler eBPF Telemetry Client (Layer 2)
Reads ACTUAL measured energy for an ALREADY-RUNNING pod from a real Kepler
exporter's Prometheus `/metrics` endpoint. This is only meaningful for a pod
that genuinely exists on real hardware right now (see k8s_deployer.py) --
it cannot and does not attempt to answer "what if this pod ran elsewhere",
which is exactly what the Digital Twin's power model (Layer 4) is for.

Deploy real Kepler on your cluster yourself, e.g.:
    kubectl apply --server-side -f https://github.com/sustainable-computing-io/kepler/releases/latest/download/kepler.yaml
Then port-forward or expose its metrics port (default 9102):
    kubectl port-forward -n kepler svc/kepler 9102:9102
"""

import re
from typing import Optional
import requests

from src.layer1_infra.kepler_emulator import PodEnergyMetrics

# Kepler exposes several joule counters; sum the ones that exist for total energy.
_JOULE_METRIC_NAMES = [
    "kepler_container_joules_total",
    "kepler_container_package_joules_total",
    "kepler_container_dram_joules_total",
]


def is_available(endpoint: str, timeout_sec: float = 2.0) -> bool:
    """Fast health check for a real Kepler /metrics endpoint."""
    try:
        resp = requests.get(endpoint, timeout=timeout_sec)
        return resp.status_code == 200 and "kepler_" in resp.text
    except Exception:
        return False


def _parse_metric_value(metrics_text: str, metric_name: str, pod_name: str, namespace: str) -> Optional[float]:
    """
    Parses a Prometheus text-format metric line for the given pod, e.g.:
      kepler_container_joules_total{pod_name="pod-123",container_namespace="sustainability-sim",...} 4821.3
    Returns the value if found, else None.
    """
    pattern = re.compile(
        rf'{re.escape(metric_name)}\{{[^}}]*pod_name="{re.escape(pod_name)}"[^}}]*'
        rf'(?:container_namespace|namespace)="{re.escape(namespace)}"[^}}]*\}}\s+([0-9.eE+-]+)'
    )
    match = pattern.search(metrics_text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


class RealKeplerClient:
    """Thin client over a real, already-deployed Kepler exporter."""

    def __init__(self, endpoint: str = "http://localhost:9102/metrics", timeout_sec: float = 3.0):
        self.endpoint = endpoint
        self.timeout_sec = timeout_sec

    def is_available(self) -> bool:
        return is_available(self.endpoint, self.timeout_sec)

    def measure_pod(
        self,
        pod_name: str,
        namespace: str,
        cpu_util: float = 0.0,
        ram_used_gb: float = 0.0,
        elapsed_sec: Optional[float] = None,
    ) -> Optional[PodEnergyMetrics]:
        """
        Fetches the REAL measured cumulative energy for an already-running
        pod. `kepler_container_joules_total` is a cumulative counter since
        the pod started, not an instantaneous power reading -- pass
        `elapsed_sec` (how long the pod has been running) if you want an
        average power figure derived from it; otherwise `instant_power_watts`
        is left as 0.0 and only `total_energy_joules` should be trusted.
        Returns None (never raises) if the endpoint is unreachable or the
        pod's metric hasn't appeared yet (e.g. scraped too soon after
        creation) -- callers must fall back to the model-based estimate.
        """
        try:
            resp = requests.get(self.endpoint, timeout=self.timeout_sec)
            resp.raise_for_status()
        except Exception:
            return None

        text = resp.text
        total_joules = 0.0
        found_any = False
        for metric_name in _JOULE_METRIC_NAMES:
            value = _parse_metric_value(text, metric_name, pod_name, namespace)
            if value is not None:
                total_joules += value
                found_any = True

        if not found_any:
            return None

        avg_power_w = (total_joules / elapsed_sec) if elapsed_sec and elapsed_sec > 0 else 0.0
        return PodEnergyMetrics(
            pod_id=pod_name,
            instant_power_watts=round(avg_power_w, 3),
            total_energy_joules=round(total_joules, 3),
            cpu_utilization=cpu_util,
            ram_utilization=ram_used_gb,
        )