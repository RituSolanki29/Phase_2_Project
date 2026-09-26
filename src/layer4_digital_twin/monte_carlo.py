"""
1000-Run Monte Carlo Simulation Engine (Layer 4)
Combines GBR execution time uncertainty, Clark migration overhead variance,
and SimPy renewable carbon trajectory fluctuations into empirical outcome distributions.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np

from src.layer1_infra.trace_loader import WorkloadPod
from src.layer1_infra.cluster_config import NodeConfig
from src.layer1_infra.kepler_emulator import KeplerEmulator
from src.layer2_telemetry.feature_store import UnifiedTelemetryFrame
from src.layer3_scheduling.heterogeneity import HardwareHeterogeneityModel
from src.layer4_digital_twin.gbr_predictor import GBRRuntimePredictor
from src.layer4_digital_twin.clark_model import ClarkMigrationModel
from src.layer4_digital_twin.simpy_uncertainty import SimPyCarbonSampler

@dataclass
class MonteCarloSimulationResult:
    pod_id: str
    target_node_id: str
    num_runs: int
    stay_carbon_samples_gco2: np.ndarray
    target_carbon_samples_gco2: np.ndarray
    migration_carbon_samples_gco2: np.ndarray
    net_sustainability_gain_samples: np.ndarray
    total_duration_samples_sec: np.ndarray
    predicted_exec_time_sec: float
    migration_time_sec: float


class MonteCarloEngine:
    """1000-Run Monte Carlo Simulation Engine for Digital Twin evaluation."""
    
    def __init__(
        self,
        gbr_predictor: GBRRuntimePredictor,
        clark_model: ClarkMigrationModel,
        simpy_sampler: SimPyCarbonSampler,
        kepler: KeplerEmulator,
        num_runs: int = 1000,
        random_seed: int = 42
    ):
        self.gbr = gbr_predictor
        self.clark = clark_model
        self.simpy_sampler = simpy_sampler
        self.kepler = kepler
        self.num_runs = num_runs
        self.seed = random_seed

    def run_simulation(
        self,
        frame: UnifiedTelemetryFrame,
        target_node: NodeConfig
    ) -> MonteCarloSimulationResult:
        """
        Runs 1000 stochastic simulation iterations evaluating the Net Sustainability Gain
        and SLA latency risk of migrating frame.pod from frame.current_node to target_node.
        """
        np.random.seed(self.seed)
        pod = frame.pod
        current_node = frame.current_node
        
        # 1. GBR Execution Time Prediction
        mips_ratio = HardwareHeterogeneityModel.get_execution_scaling_factor(current_node, target_node)
        predicted_exec_time, exec_std = self.gbr.predict(pod, mips_ratio)
        
        # 2. Clark Migration Overhead Model
        migration_est = self.clark.estimate_migration_cost(
            pod=pod,
            source_node=current_node,
            target_node=target_node,
            source_region_ci=frame.current_region_ci
        )
        
        # 3. Monte Carlo Vectorized Sampling
        # Sample execution time distribution
        exec_time_samples = np.random.normal(predicted_exec_time, exec_std, self.num_runs)
        exec_time_samples = np.clip(exec_time_samples, 2.0, None)
        
        # Sample migration duration distribution
        mig_time_samples = np.random.normal(migration_est.migration_duration_sec, 0.4, self.num_runs)
        mig_time_samples = np.clip(mig_time_samples, 0.0, None)
        
        # Total duration = migration overhead + target execution time
        total_duration_samples = mig_time_samples + exec_time_samples
        
        # 4. Carbon Trajectory Sampling
        current_ci_samples = np.array(self.simpy_sampler.sample_carbon_trajectory(
            region=current_node.region,
            start_time_sec=frame.timestamp_sec,
            duration_sec=predicted_exec_time,
            random_seed=self.seed
        ))
        avg_current_ci = float(np.mean(current_ci_samples))
        
        target_ci_samples = np.array(self.simpy_sampler.sample_carbon_trajectory(
            region=target_node.region,
            start_time_sec=frame.timestamp_sec + migration_est.migration_duration_sec,
            duration_sec=predicted_exec_time,
            random_seed=self.seed + 1
        ))
        avg_target_ci = float(np.mean(target_ci_samples))
        
        # Dynamic CI noise per sample
        ci_noise_current = np.random.normal(0, 15.0, self.num_runs)
        ci_noise_target = np.random.normal(0, 15.0, self.num_runs)
        
        sampled_current_cis = np.clip(avg_current_ci + ci_noise_current, 50.0, 700.0)
        sampled_target_cis = np.clip(avg_target_ci + ci_noise_target, 50.0, 700.0)
        
        # 5. Energy and Carbon Calculations across 1000 samples
        # Current node (STAY) execution energy (kWh)
        cpu_util_stay = min(1.0, pod.cpu_request / (current_node.mips_rating / 1000.0))
        power_stay_w = self.kepler.estimate_pod_power(cpu_util_stay, pod.assigned_memory_gb, current_node.idle_power_w, current_node.max_power_w)
        
        # Unscaled duration for stay
        stay_exec_time_samples = np.random.normal(pod.duration_sec, exec_std, self.num_runs)
        stay_exec_time_samples = np.clip(stay_exec_time_samples, 2.0, None)
        stay_energy_kwh = (power_stay_w * stay_exec_time_samples) / 3600000.0
        stay_carbon_samples = stay_energy_kwh * sampled_current_cis
        
        # Target node (MIGRATE) execution energy (kWh)
        cpu_util_target = min(1.0, pod.cpu_request / (target_node.mips_rating / 1000.0))
        power_target_w = self.kepler.estimate_pod_power(cpu_util_target, pod.assigned_memory_gb, target_node.idle_power_w, target_node.max_power_w)
        target_energy_kwh = (power_target_w * exec_time_samples) / 3600000.0
        target_exec_carbon_samples = target_energy_kwh * sampled_target_cis
        
        # Migration carbon overhead
        mig_carbon_noise = np.random.normal(0, migration_est.migration_carbon_gco2 * 0.1, self.num_runs)
        mig_carbon_samples = np.clip(migration_est.migration_carbon_gco2 + mig_carbon_noise, 0.0, None)
        
        total_target_carbon_samples = target_exec_carbon_samples + mig_carbon_samples
        
        # Net Sustainability Gain (NSG) = STAY Carbon - MIGRATE Carbon
        nsg_samples = stay_carbon_samples - total_target_carbon_samples

        return MonteCarloSimulationResult(
            pod_id=pod.pod_id,
            target_node_id=target_node.node_id,
            num_runs=self.num_runs,
            stay_carbon_samples_gco2=stay_carbon_samples,
            target_carbon_samples_gco2=total_target_carbon_samples,
            migration_carbon_samples_gco2=mig_carbon_samples,
            net_sustainability_gain_samples=nsg_samples,
            total_duration_samples_sec=total_duration_samples,
            predicted_exec_time_sec=predicted_exec_time,
            migration_time_sec=migration_est.migration_duration_sec
        )
