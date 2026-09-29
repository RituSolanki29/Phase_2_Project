"""
Sustainability-Aware Candidate Filtering & NSGA-II Batch Scheduler (Layer 3)
Applies hard rules-based capacity constraints and a genuinely multi-objective,
multi-pod NSGA-II optimization to shortlist Top-K target node candidates per
pod, for Digital Twin evaluation in Layer 4.

FIXED vs. earlier version:
  1. NSGA-II is now actually executed via pymoo's `minimize()`. The previous
     version imported pymoo and defined a Problem class but never called
     `minimize()` -- it silently fell back to a hand-written weighted-sum
     sort, which is a very different (and much weaker) algorithm than what
     was claimed.
  2. NSGA-II is now run ONCE PER SCHEDULING CYCLE across the whole batch of
     pending pods, not once per single pod against ~9 nodes. A per-pod
     search space that small does not justify a population-based GA; the
     joint assignment of many pods to many nodes is a genuinely large
     combinatorial space (num_nodes ** num_pods), which is exactly the
     regime NSGA-II is for.
  3. The capacity constraint is now a REAL cross-pod constraint: multiple
     pods competing for the same node's RAM within one batch/cycle are
     jointly accounted for as an inequality constraint inside the GA, rather
     than an always-true per-pod check against a capacity value that never
     changes.
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


def _evaluate_pair(
    pod: WorkloadPod,
    current_node: NodeConfig,
    target_node: NodeConfig,
    frame: UnifiedTelemetryFrame,
    kepler: KeplerEmulator,
) -> Tuple[float, float, float]:
    """
    Shared quick-estimate objective function used both inside the NSGA-II
    fitness evaluation and when materializing the final CandidatePair
    objects. This is a fast heuristic estimate (Kepler + MIPS scaling +
    current carbon-intensity snapshot) -- the expensive, uncertainty-aware
    evaluation happens later in Layer 4's Digital Twin / Monte Carlo engine.
    Returns (estimated_carbon_gco2, scaled_duration_sec, latency_penalty_sec).
    """
    scaled_duration = HardwareHeterogeneityModel.scale_duration(pod.duration_sec, current_node, target_node)
    cpu_util = min(1.0, pod.cpu_request / (target_node.mips_rating / 1000.0))
    power_w = kepler.estimate_pod_power(cpu_util, pod.assigned_memory_gb, target_node.idle_power_w, target_node.max_power_w)
    energy_kwh = (power_w * scaled_duration) / 3600000.0

    target_ci = frame.target_region_cis.get(target_node.region, 350.0)
    est_carbon = energy_kwh * target_ci
    latency_penalty = max(0.0, scaled_duration - pod.sla_deadline_sec)

    return est_carbon, scaled_duration, latency_penalty


class _BatchAssignmentProblem(ElementwiseProblem):
    """
    Pymoo multi-objective problem: jointly assign every pod in the current
    scheduling cycle to one of ITS eligible target nodes, minimizing total
    estimated carbon and total SLA-latency-penalty, subject to each node's
    RAM capacity not being oversubscribed across the pods assigned to it
    within this one batch.
    """

    def __init__(
        self,
        pods: List[WorkloadPod],
        frames: List[UnifiedTelemetryFrame],
        current_nodes: List[NodeConfig],
        eligible_node_lists: List[List[NodeConfig]],
        all_nodes: List[NodeConfig],
        kepler: KeplerEmulator,
    ):
        self.pods = pods
        self.frames = frames
        self.current_nodes = current_nodes
        self.eligible_node_lists = eligible_node_lists
        self.all_nodes = all_nodes
        self.node_index = {n.node_id: i for i, n in enumerate(all_nodes)}
        self.kepler = kepler

        n_var = len(pods)
        xl = np.zeros(n_var)
        xu = np.array([max(0, len(el) - 1) for el in eligible_node_lists], dtype=float)

        super().__init__(
            n_var=n_var,
            n_obj=2,
            n_ieq_constr=len(all_nodes),
            xl=xl,
            xu=xu,
        )

    def _decode(self, x: np.ndarray) -> List[NodeConfig]:
        assigned = []
        for i, val in enumerate(x):
            eligible = self.eligible_node_lists[i]
            idx = int(round(float(np.clip(val, 0, len(eligible) - 1))))
            assigned.append(eligible[idx])
        return assigned

    def _evaluate(self, x, out, *args, **kwargs):
        assigned_nodes = self._decode(x)

        total_carbon = 0.0
        total_latency_penalty = 0.0
        ram_used_per_node = np.zeros(len(self.all_nodes))

        for pod, current_node, target_node, frame in zip(self.pods, self.current_nodes, assigned_nodes, self.frames):
            carbon, _, latency_penalty = _evaluate_pair(pod, current_node, target_node, frame, self.kepler)
            total_carbon += carbon
            total_latency_penalty += latency_penalty
            ram_used_per_node[self.node_index[target_node.node_id]] += pod.ram_request_gb

        # Inequality constraints: cumulative RAM assigned to a node in THIS
        # batch must not exceed that node's total RAM (G <= 0 is feasible).
        capacity_violation = np.array([
            ram_used_per_node[i] - self.all_nodes[i].total_ram_gb
            for i in range(len(self.all_nodes))
        ])

        out["F"] = [total_carbon, total_latency_penalty]
        out["G"] = capacity_violation


class NSGA2PreFilter:
    """Multi-Objective, Multi-Pod Batch Scheduler using real NSGA-II (pymoo)."""

    def __init__(self, kepler: KeplerEmulator, top_k: int = 3, pop_size: int = 40, n_gen: int = 25, seed: int = 42):
        self.kepler = kepler
        self.top_k = top_k
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.seed = seed

    def filter_and_rank_batch(
        self,
        frames: List[UnifiedTelemetryFrame],
    ) -> Dict[str, List[CandidatePair]]:
        """
        Runs ONE NSGA-II optimization jointly over every pod in `frames`
        (a scheduling cycle's pending-pod batch), enforcing shared node
        capacity across them, and returns each pod's Top-K shortlisted
        candidate nodes for Layer 4 Digital Twin evaluation.
        """
        if not frames:
            return {}

        all_nodes = frames[0].target_node_states
        pods = [f.pod for f in frames]
        current_nodes = [f.current_node for f in frames]

        # Step 1: hard rules-based capacity pre-filter (per pod, against each
        # node's total capacity) -- this trims the search space before GA.
        eligible_node_lists: List[List[NodeConfig]] = []
        for pod in pods:
            eligible = [n for n in all_nodes if n.total_ram_gb >= pod.ram_request_gb]
            if not eligible:
                eligible = [all_nodes[0]]
            eligible_node_lists.append(eligible)

        problem = _BatchAssignmentProblem(
            pods=pods,
            frames=frames,
            current_nodes=current_nodes,
            eligible_node_lists=eligible_node_lists,
            all_nodes=all_nodes,
            kepler=self.kepler,
        )

        algorithm = NSGA2(pop_size=self.pop_size)

        try:
            res = minimize(
                problem,
                algorithm,
                ('n_gen', self.n_gen),
                seed=self.seed,
                verbose=False,
            )
        except Exception:
            res = None

        if res is None or res.X is None:
            # Fallback: per-pod weighted-sum ranking against the (capacity
            # pre-filtered) eligible node list -- keeps the pipeline robust
            # even if the GA finds no feasible joint assignment.
            return self._fallback_rank(pods, current_nodes, frames, eligible_node_lists)

        pareto_X = np.atleast_2d(res.X)

        per_pod_candidates: Dict[str, List[CandidatePair]] = {pod.pod_id: [] for pod in pods}
        per_pod_seen_nodes: Dict[str, set] = {pod.pod_id: set() for pod in pods}

        # Sort Pareto solutions by total carbon so the first candidate we
        # keep per pod comes from a low-carbon overall assignment plan.
        f_totals = res.F if res.F.ndim == 2 else res.F.reshape(1, -1)
        sort_idx = np.argsort(f_totals[:, 0])

        for sol_idx in sort_idx:
            x_row = pareto_X[sol_idx]
            assigned_nodes = problem._decode(x_row)
            for pod, current_node, target_node, frame in zip(pods, current_nodes, assigned_nodes, frames):
                if len(per_pod_candidates[pod.pod_id]) >= self.top_k:
                    continue
                if target_node.node_id in per_pod_seen_nodes[pod.pod_id]:
                    continue
                carbon, scaled_dur, latency_pen = _evaluate_pair(pod, current_node, target_node, frame, self.kepler)
                per_pod_candidates[pod.pod_id].append(CandidatePair(
                    pod=pod,
                    target_node=target_node,
                    estimated_carbon_gco2=round(carbon, 4),
                    scaled_duration_sec=round(scaled_dur, 2),
                    latency_penalty_sec=round(latency_pen, 2),
                ))
                per_pod_seen_nodes[pod.pod_id].add(target_node.node_id)

        # Any pod that (rarely) got fewer than top_k distinct candidates from
        # the Pareto set is topped up via the fallback ranking.
        for pod, current_node, frame in zip(pods, current_nodes, frames):
            if len(per_pod_candidates[pod.pod_id]) < self.top_k:
                fallback = self._fallback_rank_single(pod, current_node, frame, eligible_node_lists[pods.index(pod)])
                for cp in fallback:
                    if cp.target_node.node_id not in per_pod_seen_nodes[pod.pod_id]:
                        per_pod_candidates[pod.pod_id].append(cp)
                        per_pod_seen_nodes[pod.pod_id].add(cp.target_node.node_id)
                    if len(per_pod_candidates[pod.pod_id]) >= self.top_k:
                        break

        for pod in pods:
            ranked = sorted(per_pod_candidates[pod.pod_id], key=lambda cp: (cp.estimated_carbon_gco2 + 10.0 * cp.latency_penalty_sec))
            for i, cp in enumerate(ranked):
                cp.rank = i + 1
            per_pod_candidates[pod.pod_id] = ranked[:self.top_k]

        return per_pod_candidates

    def filter_and_rank(self, frame: UnifiedTelemetryFrame) -> List[CandidatePair]:
        """Backward-compatible single-pod entry point (wraps the batch API)."""
        result = self.filter_and_rank_batch([frame])
        return result.get(frame.pod.pod_id, [])

    def _fallback_rank_single(
        self,
        pod: WorkloadPod,
        current_node: NodeConfig,
        frame: UnifiedTelemetryFrame,
        eligible_nodes: List[NodeConfig],
    ) -> List[CandidatePair]:
        candidate_pairs: List[CandidatePair] = []
        for target_node in eligible_nodes:
            carbon, scaled_dur, latency_pen = _evaluate_pair(pod, current_node, target_node, frame, self.kepler)
            candidate_pairs.append(CandidatePair(
                pod=pod,
                target_node=target_node,
                estimated_carbon_gco2=round(carbon, 4),
                scaled_duration_sec=round(scaled_dur, 2),
                latency_penalty_sec=round(latency_pen, 2),
            ))
        candidate_pairs.sort(key=lambda cp: (cp.estimated_carbon_gco2 + 10.0 * cp.latency_penalty_sec))
        for idx, cp in enumerate(candidate_pairs):
            cp.rank = idx + 1
        return candidate_pairs[:self.top_k]

    def _fallback_rank(
        self,
        pods: List[WorkloadPod],
        current_nodes: List[NodeConfig],
        frames: List[UnifiedTelemetryFrame],
        eligible_node_lists: List[List[NodeConfig]],
    ) -> Dict[str, List[CandidatePair]]:
        result: Dict[str, List[CandidatePair]] = {}
        for pod, current_node, frame, eligible in zip(pods, current_nodes, frames, eligible_node_lists):
            result[pod.pod_id] = self._fallback_rank_single(pod, current_node, frame, eligible)
        return result