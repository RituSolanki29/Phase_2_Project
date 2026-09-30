"""
Sustainability-Aware NSGA-II Candidate Filter (Layer 3)

Objective 1:
    Estimated target execution carbon
    + estimated migration carbon

Objective 2:
    SLA excess time after migration + target execution

Constraint:
    Aggregate RAM assigned to each node must not exceed node capacity.

Layer 4 Monte Carlo remains the final uncertainty-aware evaluation.
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

from src.layer2_telemetry.feature_store import (
    UnifiedTelemetryFrame,
)

from src.layer3_scheduling.heterogeneity import (
    HardwareHeterogeneityModel,
)

from src.layer4_digital_twin.clark_model import (
    ClarkMigrationModel,
)


@dataclass
class CandidatePair:

    pod: WorkloadPod

    target_node: NodeConfig

    estimated_carbon_gco2: float

    estimated_migration_carbon_gco2: float

    estimated_total_carbon_gco2: float

    scaled_duration_sec: float

    migration_duration_sec: float

    total_duration_sec: float

    latency_penalty_sec: float

    rank: int = 0


def _evaluate_pair(
    pod: WorkloadPod,
    current_node: NodeConfig,
    target_node: NodeConfig,
    frame: UnifiedTelemetryFrame,
    kepler,
    clark_model: ClarkMigrationModel,
) -> Tuple[float, float, float, float, float, float]:

    # ---------------------------------------------------------
    # Target execution time
    # ---------------------------------------------------------

    scaled_duration = (
        HardwareHeterogeneityModel
        .scale_duration(
            pod.duration_sec,
            current_node,
            target_node,
        )
    )

    # ---------------------------------------------------------
    # Target power
    # ---------------------------------------------------------

    cpu_util = min(
        1.0,
        pod.cpu_request
        / max(
            target_node.mips_rating / 1000.0,
            0.001,
        ),
    )

    power_w = kepler.estimate_pod_power(
        cpu_util,
        pod.assigned_memory_gb,
        target_node.idle_power_w,
        target_node.max_power_w,
    )

    execution_energy_kwh = (
        power_w
        * scaled_duration
        / 3_600_000.0
    )

    target_ci = frame.target_region_cis.get(
        target_node.region,
        350.0,
    )

    execution_carbon = (
        execution_energy_kwh
        * target_ci
    )

    # ---------------------------------------------------------
    # Migration cost
    # ---------------------------------------------------------

    migration = (
        clark_model.estimate_migration_cost(
            pod=pod,
            source_node=current_node,
            target_node=target_node,
            source_region_ci=frame.current_region_ci,
        )
    )

    migration_carbon = (
        migration.migration_carbon_gco2
    )

    migration_duration = (
        migration.migration_duration_sec
    )

    total_carbon = (
        execution_carbon
        + migration_carbon
    )

    # ---------------------------------------------------------
    # SLA
    # ---------------------------------------------------------

    total_duration = (
        migration_duration
        + scaled_duration
    )

    latency_penalty = max(
        0.0,
        total_duration
        - pod.sla_deadline_sec,
    )

    return (
        float(execution_carbon),
        float(migration_carbon),
        float(total_carbon),
        float(scaled_duration),
        float(migration_duration),
        float(latency_penalty),
    )


class _BatchAssignmentProblem(
    ElementwiseProblem
):

    def __init__(
        self,
        pods: List[WorkloadPod],
        frames: List[UnifiedTelemetryFrame],
        current_nodes: List[NodeConfig],
        eligible_node_lists: List[List[NodeConfig]],
        all_nodes: List[NodeConfig],
        kepler,
        clark_model: ClarkMigrationModel,
    ):

        self.pods = pods
        self.frames = frames
        self.current_nodes = current_nodes
        self.eligible_node_lists = eligible_node_lists
        self.all_nodes = all_nodes

        self.node_index = {
            n.node_id: i
            for i, n in enumerate(all_nodes)
        }

        self.kepler = kepler
        self.clark_model = clark_model

        n_var = len(pods)

        xl = np.zeros(
            n_var,
            dtype=float,
        )

        xu = np.asarray(
            [
                max(
                    0,
                    len(nodes) - 1,
                )
                for nodes in eligible_node_lists
            ],
            dtype=float,
        )

        super().__init__(
            n_var=n_var,
            n_obj=2,
            n_ieq_constr=len(all_nodes),
            xl=xl,
            xu=xu,
        )

    def _decode(
        self,
        x: np.ndarray,
    ) -> List[NodeConfig]:

        assigned = []

        for i, value in enumerate(x):

            eligible = (
                self.eligible_node_lists[i]
            )

            index = int(
                round(
                    float(
                        np.clip(
                            value,
                            0,
                            len(eligible) - 1,
                        )
                    )
                )
            )

            assigned.append(
                eligible[index]
            )

        return assigned

    def _evaluate(
        self,
        x,
        out,
        *args,
        **kwargs,
    ):

        assigned_nodes = (
            self._decode(x)
        )

        total_carbon = 0.0
        total_latency_penalty = 0.0

        ram_used = np.zeros(
            len(self.all_nodes),
            dtype=float,
        )

        for (
            pod,
            current_node,
            target_node,
            frame,
        ) in zip(
            self.pods,
            self.current_nodes,
            assigned_nodes,
            self.frames,
        ):

            (
                _execution_carbon,
                _migration_carbon,
                total_pair_carbon,
                _scaled_duration,
                _migration_duration,
                latency_penalty,
            ) = _evaluate_pair(
                pod=pod,
                current_node=current_node,
                target_node=target_node,
                frame=frame,
                kepler=self.kepler,
                clark_model=self.clark_model,
            )

            total_carbon += (
                total_pair_carbon
            )

            total_latency_penalty += (
                latency_penalty
            )

            node_idx = self.node_index[
                target_node.node_id
            ]

            ram_used[node_idx] += (
                pod.ram_request_gb
            )

        capacity_violation = (
            ram_used
            - np.asarray(
                [
                    node.total_ram_gb
                    for node in self.all_nodes
                ]
            )
        )

        out["F"] = [
            total_carbon,
            total_latency_penalty,
        ]

        out["G"] = (
            capacity_violation
        )


class NSGA2PreFilter:
    """
    Multi-pod NSGA-II candidate filter.
    """

    def __init__(
        self,
        kepler,
        top_k: int = 2,
        pop_size: int = 40,
        n_gen: int = 25,
        seed: int = 42,
    ):

        self.kepler = kepler

        self.top_k = int(top_k)
        self.pop_size = int(pop_size)
        self.n_gen = int(n_gen)
        self.seed = int(seed)

        self.clark_model = (
            ClarkMigrationModel()
        )

    def _make_candidates(
        self,
        pods,
        current_nodes,
        frames,
        assignments,
    ):

        result = {
            pod.pod_id: []
            for pod in pods
        }

        seen = {
            pod.pod_id: set()
            for pod in pods
        }

        for (
            pod,
            current_node,
            frame,
            target_node,
        ) in zip(
            pods,
            current_nodes,
            frames,
            assignments,
        ):

            if (
                target_node.node_id
                in seen[pod.pod_id]
            ):
                continue

            (
                execution_carbon,
                migration_carbon,
                total_carbon,
                scaled_duration,
                migration_duration,
                latency_penalty,
            ) = _evaluate_pair(
                pod=pod,
                current_node=current_node,
                target_node=target_node,
                frame=frame,
                kepler=self.kepler,
                clark_model=self.clark_model,
            )

            result[pod.pod_id].append(
                CandidatePair(
                    pod=pod,
                    target_node=target_node,

                    estimated_carbon_gco2=round(
                        execution_carbon,
                        6,
                    ),

                    estimated_migration_carbon_gco2=round(
                        migration_carbon,
                        6,
                    ),

                    estimated_total_carbon_gco2=round(
                        total_carbon,
                        6,
                    ),

                    scaled_duration_sec=round(
                        scaled_duration,
                        4,
                    ),

                    migration_duration_sec=round(
                        migration_duration,
                        4,
                    ),

                    total_duration_sec=round(
                        migration_duration
                        + scaled_duration,
                        4,
                    ),

                    latency_penalty_sec=round(
                        latency_penalty,
                        4,
                    ),
                )
            )

            seen[pod.pod_id].add(
                target_node.node_id
            )

        return result

    @staticmethod
    def _candidate_sort_key(
        candidate: CandidatePair,
    ):

        # First prefer candidates with no SLA excess.
        # Among those, lower estimated total carbon is preferred.
        # Remaining ties are resolved using latency penalty.
        return (
            candidate.latency_penalty_sec > 0.0,
            candidate.estimated_total_carbon_gco2,
            candidate.latency_penalty_sec,
        )

    def _fallback_rank_single(
        self,
        pod,
        current_node,
        frame,
        eligible_nodes,
    ):

        candidates = []

        for target_node in eligible_nodes:

            (
                execution_carbon,
                migration_carbon,
                total_carbon,
                scaled_duration,
                migration_duration,
                latency_penalty,
            ) = _evaluate_pair(
                pod=pod,
                current_node=current_node,
                target_node=target_node,
                frame=frame,
                kepler=self.kepler,
                clark_model=self.clark_model,
            )

            candidates.append(
                CandidatePair(
                    pod=pod,
                    target_node=target_node,

                    estimated_carbon_gco2=round(
                        execution_carbon,
                        6,
                    ),

                    estimated_migration_carbon_gco2=round(
                        migration_carbon,
                        6,
                    ),

                    estimated_total_carbon_gco2=round(
                        total_carbon,
                        6,
                    ),

                    scaled_duration_sec=round(
                        scaled_duration,
                        4,
                    ),

                    migration_duration_sec=round(
                        migration_duration,
                        4,
                    ),

                    total_duration_sec=round(
                        migration_duration
                        + scaled_duration,
                        4,
                    ),

                    latency_penalty_sec=round(
                        latency_penalty,
                        4,
                    ),
                )
            )

        candidates.sort(
            key=self._candidate_sort_key
        )

        for index, candidate in enumerate(
            candidates
        ):
            candidate.rank = index + 1

        return candidates[
            : self.top_k
        ]

    def _fallback_rank_batch(
        self,
        pods,
        current_nodes,
        frames,
        eligible_node_lists,
    ):

        result = {}

        for (
            pod,
            current_node,
            frame,
            eligible,
        ) in zip(
            pods,
            current_nodes,
            frames,
            eligible_node_lists,
        ):

            result[pod.pod_id] = (
                self._fallback_rank_single(
                    pod,
                    current_node,
                    frame,
                    eligible,
                )
            )

        return result

    def filter_and_rank_batch(
        self,
        frames: List[UnifiedTelemetryFrame],
    ) -> Dict[str, List[CandidatePair]]:

        if not frames:
            return {}

        all_nodes = (
            frames[0].target_node_states
        )

        pods = [
            frame.pod
            for frame in frames
        ]

        current_nodes = [
            frame.current_node
            for frame in frames
        ]

        # -----------------------------------------------------
        # Capacity eligibility
        # -----------------------------------------------------

        eligible_node_lists = []

        for pod in pods:

            eligible = [
                node
                for node in all_nodes
                if node.total_ram_gb
                >= pod.ram_request_gb
            ]

            if not eligible:
                eligible = [
                    all_nodes[0]
                ]

            eligible_node_lists.append(
                eligible
            )

        # -----------------------------------------------------
        # NSGA-II
        # -----------------------------------------------------

        problem = (
            _BatchAssignmentProblem(
                pods=pods,
                frames=frames,
                current_nodes=current_nodes,
                eligible_node_lists=eligible_node_lists,
                all_nodes=all_nodes,
                kepler=self.kepler,
                clark_model=self.clark_model,
            )
        )

        algorithm = NSGA2(
            pop_size=self.pop_size
        )

        try:

            result = minimize(
                problem,
                algorithm,
                ("n_gen", self.n_gen),
                seed=self.seed,
                verbose=False,
            )

        except Exception:

            result = None

        if (
            result is None
            or result.X is None
        ):

            return self._fallback_rank_batch(
                pods,
                current_nodes,
                frames,
                eligible_node_lists,
            )

        pareto_X = np.atleast_2d(
            result.X
        )

        # -----------------------------------------------------
        # Collect distinct candidates
        # -----------------------------------------------------

        candidate_map = {
            pod.pod_id: []
            for pod in pods
        }

        seen = {
            pod.pod_id: set()
            for pod in pods
        }

        # Sort Pareto solutions by total carbon.
        if result.F is None:
            order = range(
                len(pareto_X)
            )
        else:
            objectives = np.atleast_2d(
                result.F
            )

            order = np.argsort(
                objectives[:, 0]
            )

        for solution_index in order:

            assignment = (
                problem._decode(
                    pareto_X[
                        solution_index
                    ]
                )
            )

            for (
                pod,
                current_node,
                target_node,
                frame,
            ) in zip(
                pods,
                current_nodes,
                assignment,
                frames,
            ):

                if (
                    len(
                        candidate_map[
                            pod.pod_id
                        ]
                    )
                    >= self.top_k
                ):
                    continue

                if (
                    target_node.node_id
                    in seen[pod.pod_id]
                ):
                    continue

                candidates = self._make_candidates(
                    [pod],
                    [current_node],
                    [frame],
                    [target_node],
                )

                if candidates[
                    pod.pod_id
                ]:

                    candidate = (
                        candidates[
                            pod.pod_id
                        ][0]
                    )

                    candidate_map[
                        pod.pod_id
                    ].append(
                        candidate
                    )

                    seen[
                        pod.pod_id
                    ].add(
                        target_node.node_id
                    )

        # -----------------------------------------------------
        # Top-up missing candidates
        # -----------------------------------------------------

        for (
            pod,
            current_node,
            frame,
            eligible,
        ) in zip(
            pods,
            current_nodes,
            frames,
            eligible_node_lists,
        ):

            if (
                len(
                    candidate_map[
                        pod.pod_id
                    ]
                )
                >= self.top_k
            ):
                continue

            fallback = (
                self._fallback_rank_single(
                    pod,
                    current_node,
                    frame,
                    eligible,
                )
            )

            for candidate in fallback:

                if (
                    candidate.target_node.node_id
                    not in seen[pod.pod_id]
                ):

                    candidate_map[
                        pod.pod_id
                    ].append(
                        candidate
                    )

                    seen[
                        pod.pod_id
                    ].add(
                        candidate.target_node.node_id
                    )

                if (
                    len(
                        candidate_map[
                            pod.pod_id
                        ]
                    )
                    >= self.top_k
                ):
                    break

        # -----------------------------------------------------
        # Final ranking
        # -----------------------------------------------------

        for pod in pods:

            candidates = candidate_map[
                pod.pod_id
            ]

            candidates.sort(
                key=self._candidate_sort_key
            )

            for index, candidate in enumerate(
                candidates
            ):
                candidate.rank = index + 1

            candidate_map[
                pod.pod_id
            ] = candidates[
                : self.top_k
            ]

        return candidate_map

    def filter_and_rank(
        self,
        frame: UnifiedTelemetryFrame,
    ) -> List[CandidatePair]:

        result = (
            self.filter_and_rank_batch(
                [frame]
            )
        )

        return result.get(
            frame.pod.pod_id,
            [],
        )