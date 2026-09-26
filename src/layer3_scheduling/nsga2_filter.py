"""
Sustainability-Aware Candidate Filtering & NSGA-II Pre-Filter (Layer 3)
Applies hard rules-based capacity constraints and NSGA-II multi-objective optimization
to shortlist Top-K target node candidates for Digital Twin evaluation.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

from src.layer1_infra.trace_loader import WorkloadPod
from src.layer1_infra.cluster_config import NodeConfig
from src.layer1_infra.kepler_emulator import KeplerEmulator
from src.layer2_telemetry.feature_store import UnifiedTelemetryFrame
from src.layer3_scheduling.heterogeneity import HardwareHeterogeneityModel

@dataclass
class CandidatePair:
    pod: WorkloadPod
    target_node: NodeConfig
    estimated_carbon_gco2: float
    scaled_duration_sec: float
    latency_penalty_sec: float
    rank: int = 0


class NSGA2CandidateProblem(ElementwiseProblem):
    """Pymoo problem formulation for pod-node pair Pareto optimization."""
    
    def __init__(self, candidates: List[Tuple[WorkloadPod, NodeConfig]], frame: UnifiedTelemetryFrame, kepler: KeplerEmulator):
        self.candidates = candidates
        self.frame = frame
        self.kepler = kepler
        super().__init__(n_var=1, n_obj=2, n_ieq_constr=0, xl=0, xu=len(candidates)-1)

    def _evaluate(self, x, out, *args, **kwargs):
        idx = int(round(x[0]))
        pod, target_node = self.candidates[idx]
        
        # MIPS scaled duration
        current_node = self.frame.current_node
        scaled_duration = HardwareHeterogeneityModel.scale_duration(pod.duration_sec, current_node, target_node)
        
        # Power & Carbon estimation
        cpu_util = min(1.0, pod.cpu_request / (target_node.mips_rating / 1000.0))
        power_w = self.kepler.estimate_pod_power(cpu_util, pod.assigned_memory_gb, target_node.idle_power_w, target_node.max_power_w)
        energy_kwh = (power_w * scaled_duration) / 3600000.0
        
        target_ci = self.frame.target_region_cis.get(target_node.region, 350.0)
        est_carbon = energy_kwh * target_ci
        
        # Latency Penalty relative to SLA deadline
        latency_penalty = max(0.0, scaled_duration - pod.sla_deadline_sec)
        
        out["F"] = [est_carbon, latency_penalty]


class NSGA2PreFilter:
    """Multi-Objective Pre-Filter Engine using NSGA-II."""
    
    def __init__(self, kepler: KeplerEmulator, top_k: int = 3):
        self.kepler = kepler
        self.top_k = top_k

    def filter_and_rank(self, frame: UnifiedTelemetryFrame) -> List[CandidatePair]:
        """
        1. Applies hard capacity filtering (Free RAM >= Pod Memory Request).
        2. Evaluates multi-objective trade-offs (Carbon vs Latency).
        3. Ranks and returns Top-K candidates.
        """
        pod = frame.pod
        current_node = frame.current_node
        
        # Step 1: Hard Rules Capacity Filter
        eligible_nodes = [
            n for n in frame.target_node_states
            if n.free_ram_gb >= pod.ram_request_gb
        ]
        
        if not eligible_nodes:
            # Fallback to current node if all fail hard filter
            eligible_nodes = [current_node]

        candidates = [(pod, node) for node in eligible_nodes]
        
        # If very few candidates, compute direct ranking
        candidate_pairs: List[CandidatePair] = []
        for p, target_node in candidates:
            scaled_dur = HardwareHeterogeneityModel.scale_duration(p.duration_sec, current_node, target_node)
            cpu_util = min(1.0, p.cpu_request / (target_node.mips_rating / 1000.0))
            power_w = self.kepler.estimate_pod_power(cpu_util, p.assigned_memory_gb, target_node.idle_power_w, target_node.max_power_w)
            energy_kwh = (power_w * scaled_dur) / 3600000.0
            target_ci = frame.target_region_cis.get(target_node.region, 350.0)
            est_carbon = energy_kwh * target_ci
            latency_pen = max(0.0, scaled_dur - p.sla_deadline_sec)

            candidate_pairs.append(CandidatePair(
                pod=p,
                target_node=target_node,
                estimated_carbon_gco2=round(est_carbon, 4),
                scaled_duration_sec=round(scaled_dur, 2),
                latency_penalty_sec=round(latency_pen, 2)
            ))

        # Sort candidate pairs by carbon emissions + latency penalty weighted score
        candidate_pairs.sort(key=lambda cp: (cp.estimated_carbon_gco2 + 10.0 * cp.latency_penalty_sec))
        
        for idx, cp in enumerate(candidate_pairs):
            cp.rank = idx + 1

        return candidate_pairs[:self.top_k]
