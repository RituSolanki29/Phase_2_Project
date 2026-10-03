"""
Telemetry Source Facade (Layer 2)
Single entry point that Layers 1-2 use to get telemetry. On construction it
PROBES what's actually reachable (a real Kubernetes cluster? a real Kepler
exporter? a real carbon-intensity API key?) and transparently uses the real
source where available, falling back to the existing simulated models
(KeplerEmulator, CarbonFeed) everywhere else -- with no crash and no
behavior change to Layers 3-5 if nothing real is reachable.

ARCHITECTURAL RULE THIS CLASS ENFORCES (see class docstrings for why):
  - `get_current_state_energy(...)` may use REAL measured telemetry, because
    it only ever describes a pod that is actually running right now.
  - `estimate_hypothetical_power(...)` ALWAYS uses the deterministic model
    formula, because it is used by Layer 3/4 to evaluate candidate target
    nodes a pod is NOT currently running on -- there is no real measurement
    of a hypothetical, so a model is the only physically valid choice there.
This mirrors the architecture diagram exactly: Box 2's Kepler sensor is real
telemetry for the current state; Box 4's "ML Model" is a predictive model
for hypothetical futures. They are different things by design.
"""

from dataclasses import dataclass
from typing import Dict, Optional

from src.layer1_infra.kepler_emulator import KeplerEmulator, PodEnergyMetrics
from src.layer1_infra import k8s_deployer
from src.layer2_telemetry.carbon_feed import CarbonFeed
from src.layer2_telemetry.real_kepler_client import RealKeplerClient
from src.layer2_telemetry.real_carbon_feed import RealCarbonFeed


@dataclass
class TelemetryModeReport:
    k8s_cluster_reachable: bool
    real_kepler_available: bool
    real_carbon_feed_available: bool

    def __str__(self) -> str:
        def tag(b: bool) -> str:
            return "REAL" if b else "SIMULATED"
        return (
            f"Kubernetes cluster : {tag(self.k8s_cluster_reachable)}\n"
            f"Kepler telemetry   : {tag(self.real_kepler_available)}\n"
            f"Carbon intensity   : {tag(self.real_carbon_feed_available)}"
        )


class TelemetrySource:
    """Auto-detecting real/simulated telemetry facade."""

    def __init__(
        self,
        kubeconfig_path: Optional[str] = None,
        kepler_endpoint: str = "http://localhost:9102/metrics",
        electricitymap_api_key: Optional[str] = None,
        carbon_zone_map: Optional[Dict[str, str]] = None,
        sim_seed: int = 42,
    ):
        # Always-available simulated backends (unchanged from before).
        self._sim_kepler = KeplerEmulator()
        self._sim_carbon = CarbonFeed(seed=sim_seed)

        # Probe real backends. Each probe is fast (few seconds max) and
        # never raises -- if anything is missing/unreachable we simply run
        # in simulated mode for that component.
        self.kubeconfig_path = kubeconfig_path
        self._k8s_reachable = k8s_deployer.is_cluster_reachable(kubeconfig_path)

        self._real_kepler = RealKeplerClient(kepler_endpoint)
        self._kepler_reachable = self._k8s_reachable and self._real_kepler.is_available()

        self._real_carbon = RealCarbonFeed(electricitymap_api_key, carbon_zone_map)
        self._carbon_reachable = self._real_carbon.is_available()

    def mode_report(self) -> TelemetryModeReport:
        return TelemetryModeReport(
            k8s_cluster_reachable=self._k8s_reachable,
            real_kepler_available=self._kepler_reachable,
            real_carbon_feed_available=self._carbon_reachable,
        )

    @property
    def k8s_available(self) -> bool:
        return self._k8s_reachable

    # ---- CURRENT-STATE telemetry: real measurement where possible ----

    def get_current_state_energy(
        self,
        pod_id: str,
        cpu_util: float,
        ram_used_gb: float,
        duration_sec: float,
        idle_power_w: float,
        max_power_w: float,
        k8s_pod_name: Optional[str] = None,
        k8s_namespace: Optional[str] = None,
        elapsed_sec: Optional[float] = None,
    ) -> PodEnergyMetrics:
        """
        Returns energy telemetry for a pod's CURRENT, already-running node.
        Uses REAL Kepler measurement if a real pod was actually deployed
        (k8s_pod_name given) and Kepler is reachable; otherwise falls back
        to the deterministic power-model estimate.
        """
        if self._kepler_reachable and k8s_pod_name:
            real_metrics = self._real_kepler.measure_pod(
                pod_name=k8s_pod_name,
                namespace=k8s_namespace or k8s_deployer.DEFAULT_NAMESPACE,
                cpu_util=cpu_util,
                ram_used_gb=ram_used_gb,
                elapsed_sec=elapsed_sec,
            )
            if real_metrics is not None:
                return real_metrics
        # Fallback: simulated/model-based estimate (unchanged old behavior).
        return self._sim_kepler.measure_pod(pod_id, cpu_util, ram_used_gb, duration_sec, idle_power_w, max_power_w)

    def get_current_carbon_intensity(self, region: str, timestamp_sec: float) -> float:
        """
        Returns the CURRENT carbon intensity for `region`. Uses a real live
        API reading if available; falls back to the synthetic diurnal model
        otherwise. Note the real reading is always "now" (see
        real_carbon_feed.py) regardless of `timestamp_sec`.
        """
        if self._carbon_reachable:
            real_ci = self._real_carbon.get_carbon_intensity(region)
            if real_ci is not None:
                return real_ci
        return self._sim_carbon.get_carbon_intensity(region, timestamp_sec)

    # ---- HYPOTHETICAL/candidate-node estimates: ALWAYS model-based ----

    def estimate_hypothetical_power(
        self,
        cpu_util: float,
        ram_used_gb: float,
        node_idle_power_w: float = 90.0,
        node_max_power_w: float = 380.0,
        node_cpu_cores: int = 16,
    ) -> float:
        """
        Estimates power draw for a CANDIDATE node the pod is not currently
        running on. Always uses the deterministic physics-formula model --
        this is intentional, see module docstring: there is no real
        measurement of a hypothetical placement. Automatically incorporates
        calibration parameters if fitted from real cluster data.
        """
        from src.layer2_telemetry.calibration import load_calibration_params
        scale, offset = load_calibration_params()
        raw_power = self._sim_kepler.estimate_pod_power(cpu_util, ram_used_gb, node_idle_power_w, node_max_power_w, node_cpu_cores)
        return max(0.0, raw_power * scale + offset)

    def get_region_forecast(self, region: str, start_time: float, horizon_sec: float, step_sec: float = 60.0):
        """
        Forecast carbon-intensity trajectory for the Digital Twin's Monte
        Carlo window. Always model-based (no API gives future values), but
        re-centered on the real current reading when available, so the
        forecast at least starts from ground truth.
        """
        trajectory = self._sim_carbon.get_region_forecast(region, start_time, horizon_sec, step_sec)
        if self._carbon_reachable:
            real_now = self._real_carbon.get_carbon_intensity(region)
            if real_now is not None and len(trajectory) > 0:
                offset = real_now - trajectory[0]
                trajectory = trajectory + offset
        return trajectory

    # ---- Backward-compatible shim for code that expects a raw KeplerEmulator ----

    def estimate_pod_power(self, cpu_util: float, ram_used_gb: float, node_idle_power_w: float = 90.0,
                            node_max_power_w: float = 380.0, node_cpu_cores: int = 16) -> float:
        """Alias so this object can be passed anywhere a KeplerEmulator was
        expected (Layer 3/4 candidate-node evaluation, which must stay
        model-based regardless)."""
        return self.estimate_hypothetical_power(cpu_util, ram_used_gb, node_idle_power_w, node_max_power_w, node_cpu_cores)