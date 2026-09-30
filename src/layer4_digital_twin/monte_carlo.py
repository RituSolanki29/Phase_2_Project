"""
Monte Carlo Digital Twin Engine (Layer 4)

Combines:

1. GBR execution-time prediction
2. Hardware heterogeneity scaling
3. Clark migration overhead
4. SimPy carbon trajectories
5. Stochastic execution uncertainty
6. Migration-carbon uncertainty

The stay and migration scenarios use separate execution distributions.
"""

from dataclasses import dataclass

from typing import Any

import numpy as np

from src.layer1_infra.trace_loader import WorkloadPod
from src.layer1_infra.cluster_config import NodeConfig
from src.layer2_telemetry.feature_store import UnifiedTelemetryFrame
from src.layer3_scheduling.heterogeneity import HardwareHeterogeneityModel
from src.layer4_digital_twin.gbr_predictor import GBRRuntimePredictor
from src.layer4_digital_twin.clark_model import ClarkMigrationModel
from src.layer4_digital_twin.simpy_uncertainty import (
    SimPyCarbonSampler,
)


@dataclass
class MonteCarloSimulationResult:

    pod_id: str
    target_node_id: str
    num_runs: int

    stay_carbon_samples_gco2: np.ndarray

    target_carbon_samples_gco2: np.ndarray

    target_execution_carbon_samples_gco2: np.ndarray

    migration_carbon_samples_gco2: np.ndarray

    net_sustainability_gain_samples: np.ndarray

    gross_carbon_saving_samples_gco2: np.ndarray

    stay_execution_time_samples_sec: np.ndarray

    target_execution_time_samples_sec: np.ndarray

    migration_time_samples_sec: np.ndarray

    total_duration_samples_sec: np.ndarray

    predicted_exec_time_sec: float

    baseline_exec_time_sec: float

    execution_std_sec: float

    migration_time_sec: float

    migration_carbon_gco2: float


class MonteCarloEngine:
    """Uncertainty-aware Digital Twin simulator."""

    def __init__(
        self,
        gbr_predictor: GBRRuntimePredictor,
        clark_model: ClarkMigrationModel,
        simpy_sampler: SimPyCarbonSampler,
        kepler,
        num_runs: int = 1000,
        random_seed: int = 42,
    ):

        self.gbr = gbr_predictor
        self.clark = clark_model
        self.simpy_sampler = simpy_sampler
        self.kepler = kepler

        self.num_runs = int(num_runs)
        self.seed = int(random_seed)

    def _candidate_rng(
        self,
        frame: UnifiedTelemetryFrame,
        target_node: NodeConfig,
    ) -> np.random.Generator:

        """
        Create a stable but candidate-specific RNG.

        Different candidates therefore do not receive exactly the same
        random stream merely because the global seed is the same.
        """

        text = (
            f"{self.seed}|"
            f"{frame.timestamp_sec}|"
            f"{frame.pod.pod_id}|"
            f"{target_node.node_id}"
        )

        # Stable deterministic integer hash independent of Python's
        # randomized string hash.
        value = 0

        for char in text:
            value = (
                value * 131
                + ord(char)
            ) % (2**32)

        return np.random.default_rng(
            value
        )

    def run_simulation(
        self,
        frame: UnifiedTelemetryFrame,
        target_node: NodeConfig,
    ) -> MonteCarloSimulationResult:

        pod = frame.pod

        current_node = frame.current_node

        rng = self._candidate_rng(
            frame,
            target_node,
        )

        # ---------------------------------------------------------
        # 1. BASELINE GBR PREDICTION
        # ---------------------------------------------------------

        baseline_exec_time, baseline_std = (
            self.gbr.predict(pod)
        )

        # Target hardware scaling.
        mips_ratio = (
            HardwareHeterogeneityModel
            .get_execution_scaling_factor(
                current_node,
                target_node,
            )
        )

        predicted_target_exec_time = (
            baseline_exec_time
            * mips_ratio
        )

        target_exec_std = max(
            0.1,
            baseline_std * mips_ratio,
        )

        # ---------------------------------------------------------
        # 2. STAY EXECUTION DISTRIBUTION
        # ---------------------------------------------------------

        # IMPORTANT:
        # Stay uses the baseline/source prediction and its own
        # uncertainty. It does NOT use target-scaled uncertainty.

        stay_exec_time_samples = rng.normal(
            loc=baseline_exec_time,
            scale=max(0.1, baseline_std),
            size=self.num_runs,
        )

        stay_exec_time_samples = np.clip(
            stay_exec_time_samples,
            2.0,
            None,
        )

        # ---------------------------------------------------------
        # 3. TARGET EXECUTION DISTRIBUTION
        # ---------------------------------------------------------

        target_exec_time_samples = rng.normal(
            loc=predicted_target_exec_time,
            scale=target_exec_std,
            size=self.num_runs,
        )

        target_exec_time_samples = np.clip(
            target_exec_time_samples,
            2.0,
            None,
        )

        # ---------------------------------------------------------
        # 4. CLARK MIGRATION MODEL
        # ---------------------------------------------------------

        migration_estimate = (
            self.clark.estimate_migration_cost(
                pod=pod,
                source_node=current_node,
                target_node=target_node,
                source_region_ci=frame.current_region_ci,
            )
        )

        migration_time_samples = rng.normal(
            loc=migration_estimate.migration_duration_sec,
            scale=max(
                0.05,
                migration_estimate.migration_duration_sec
                * 0.08,
            ),
            size=self.num_runs,
        )

        migration_time_samples = np.clip(
            migration_time_samples,
            0.0,
            None,
        )

        total_duration_samples = (
            migration_time_samples
            + target_exec_time_samples
        )

        # ---------------------------------------------------------
        # 5. SIMPY CARBON TRAJECTORIES
        # ---------------------------------------------------------

        current_duration_for_forecast = float(
            np.mean(stay_exec_time_samples)
        )

        target_duration_for_forecast = float(
            np.mean(target_exec_time_samples)
        )

        current_trajectory = np.asarray(
            self.simpy_sampler.sample_carbon_trajectory(
                region=current_node.region,
                start_time_sec=frame.timestamp_sec,
                duration_sec=current_duration_for_forecast,
                random_seed=self.seed + 11,
            ),
            dtype=float,
        )

        target_trajectory = np.asarray(
            self.simpy_sampler.sample_carbon_trajectory(
                region=target_node.region,
                start_time_sec=(
                    frame.timestamp_sec
                    + migration_estimate.migration_duration_sec
                ),
                duration_sec=target_duration_for_forecast,
                random_seed=self.seed + 29,
            ),
            dtype=float,
        )

        current_ci_mean = float(
            np.mean(current_trajectory)
        )

        target_ci_mean = float(
            np.mean(target_trajectory)
        )

        current_ci_std = max(
            5.0,
            float(np.std(current_trajectory)),
        )

        target_ci_std = max(
            5.0,
            float(np.std(target_trajectory)),
        )

        sampled_current_cis = np.clip(
            rng.normal(
                current_ci_mean,
                current_ci_std,
                self.num_runs,
            ),
            50.0,
            700.0,
        )

        sampled_target_cis = np.clip(
            rng.normal(
                target_ci_mean,
                target_ci_std,
                self.num_runs,
            ),
            50.0,
            700.0,
        )

        # ---------------------------------------------------------
        # 6. POWER
        # ---------------------------------------------------------

        cpu_util_stay = min(
            1.0,
            pod.cpu_request
            / max(
                current_node.mips_rating / 1000.0,
                0.001,
            ),
        )

        power_stay_w = (
            self.kepler.estimate_pod_power(
                cpu_util_stay,
                pod.assigned_memory_gb,
                current_node.idle_power_w,
                current_node.max_power_w,
            )
        )

        cpu_util_target = min(
            1.0,
            pod.cpu_request
            / max(
                target_node.mips_rating / 1000.0,
                0.001,
            ),
        )

        power_target_w = (
            self.kepler.estimate_pod_power(
                cpu_util_target,
                pod.assigned_memory_gb,
                target_node.idle_power_w,
                target_node.max_power_w,
            )
        )

        # ---------------------------------------------------------
        # 7. ENERGY + CARBON
        # ---------------------------------------------------------

        stay_energy_kwh = (
            power_stay_w
            * stay_exec_time_samples
            / 3_600_000.0
        )

        target_energy_kwh = (
            power_target_w
            * target_exec_time_samples
            / 3_600_000.0
        )

        stay_carbon_samples = (
            stay_energy_kwh
            * sampled_current_cis
        )

        target_execution_carbon_samples = (
            target_energy_kwh
            * sampled_target_cis
        )

        # Migration carbon uncertainty.
        migration_carbon_std = max(
            0.0001,
            migration_estimate.migration_carbon_gco2
            * 0.10,
        )

        migration_carbon_samples = np.clip(
            rng.normal(
                migration_estimate.migration_carbon_gco2,
                migration_carbon_std,
                self.num_runs,
            ),
            0.0,
            None,
        )

        target_total_carbon_samples = (
            target_execution_carbon_samples
            + migration_carbon_samples
        )

        # ---------------------------------------------------------
        # 8. FINAL SUSTAINABILITY METRICS
        # ---------------------------------------------------------

        gross_carbon_saving_samples = (
            stay_carbon_samples
            - target_execution_carbon_samples
        )

        nsg_samples = (
            stay_carbon_samples
            - target_total_carbon_samples
        )

        return MonteCarloSimulationResult(
            pod_id=pod.pod_id,
            target_node_id=target_node.node_id,
            num_runs=self.num_runs,

            stay_carbon_samples_gco2=(
                stay_carbon_samples
            ),

            target_carbon_samples_gco2=(
                target_total_carbon_samples
            ),

            target_execution_carbon_samples_gco2=(
                target_execution_carbon_samples
            ),

            migration_carbon_samples_gco2=(
                migration_carbon_samples
            ),

            net_sustainability_gain_samples=(
                nsg_samples
            ),

            gross_carbon_saving_samples_gco2=(
                gross_carbon_saving_samples
            ),

            stay_execution_time_samples_sec=(
                stay_exec_time_samples
            ),

            target_execution_time_samples_sec=(
                target_exec_time_samples
            ),

            migration_time_samples_sec=(
                migration_time_samples
            ),

            total_duration_samples_sec=(
                total_duration_samples
            ),

            predicted_exec_time_sec=round(
                float(predicted_target_exec_time),
                4,
            ),

            baseline_exec_time_sec=round(
                float(baseline_exec_time),
                4,
            ),

            execution_std_sec=round(
                float(target_exec_std),
                4,
            ),

            migration_time_sec=round(
                float(
                    migration_estimate
                    .migration_duration_sec
                ),
                4,
            ),

            migration_carbon_gco2=round(
                float(
                    migration_estimate
                    .migration_carbon_gco2
                ),
                6,
            ),
        )