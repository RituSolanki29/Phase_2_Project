"""
Net Sustainability Gain and SLA Risk Calculator (Layer 5).

Calculates empirical statistics from the Monte Carlo distributions.

Project decision thresholds:
    P(NSG > 0) >= 0.90
    SLA risk < 0.80

These are project-level decision thresholds, not universal industry limits.
"""

import numpy as np

from src.layer1_infra.trace_loader import WorkloadPod
from src.layer1_infra.cluster_config import NodeConfig

from src.layer4_digital_twin.monte_carlo import (
    MonteCarloSimulationResult,
)

from src.layer5_outputs.data_contracts import (
    CandidateEvaluationResult,
)


class Layer5OutputCalculator:
    """Computes statistical Layer 5 outputs."""

    @staticmethod
    def evaluate_candidate(
        mc_result: MonteCarloSimulationResult,
        pod: WorkloadPod,
        current_node: NodeConfig,
        target_node: NodeConfig,
        nsg_confidence_threshold: float = 0.90,
        max_sla_risk_threshold: float = 0.80,
        scheduling_window: str = "",
        timestamp_sec: float = 0.0,
    ) -> CandidateEvaluationResult:

        nsg_samples = np.asarray(
            mc_result.net_sustainability_gain_samples,
            dtype=float,
        )

        duration_samples = np.asarray(
            mc_result.total_duration_samples_sec,
            dtype=float,
        )

        gross_saving_samples = np.asarray(
            mc_result.gross_carbon_saving_samples_gco2,
            dtype=float,
        )

        target_exec_carbon_samples = np.asarray(
            mc_result.target_execution_carbon_samples_gco2,
            dtype=float,
        )

        migration_carbon_samples = np.asarray(
            mc_result.migration_carbon_samples_gco2,
            dtype=float,
        )

        stay_carbon_samples = np.asarray(
            mc_result.stay_carbon_samples_gco2,
            dtype=float,
        )

        # ---------------------------------------------------------
        # NSG
        # ---------------------------------------------------------

        mean_nsg = float(
            np.mean(nsg_samples)
        )

        p05_nsg = float(
            np.percentile(
                nsg_samples,
                5.0,
            )
        )

        p95_nsg = float(
            np.percentile(
                nsg_samples,
                95.0,
            )
        )

        nsg_std = float(
            np.std(nsg_samples)
        )

        ci_width = (
            p95_nsg
            - p05_nsg
        )

        p_nsg_positive = float(
            np.mean(
                nsg_samples > 0.0
            )
        )

        # ---------------------------------------------------------
        # SLA
        # ---------------------------------------------------------

        deadline = float(
            pod.sla_deadline_sec
        )

        sla_violations = (
            duration_samples > deadline
        )

        sla_risk = float(
            np.mean(
                sla_violations
            )
        )

        # Same quantity as SLA risk, exposed separately for
        # easier reporting.
        deadline_miss_rate = sla_risk

        total_duration_mean = float(
            np.mean(duration_samples)
        )

        total_duration_p95 = float(
            np.percentile(
                duration_samples,
                95.0,
            )
        )

        # ---------------------------------------------------------
        # Carbon decomposition
        # ---------------------------------------------------------

        gross_saving_mean = float(
            np.mean(
                gross_saving_samples
            )
        )

        target_exec_carbon_mean = float(
            np.mean(
                target_exec_carbon_samples
            )
        )

        migration_carbon_mean = float(
            np.mean(
                migration_carbon_samples
            )
        )

        stay_carbon_mean = float(
            np.mean(
                stay_carbon_samples
            )
        )

        # ---------------------------------------------------------
        # Derived decision-analysis metrics
        # ---------------------------------------------------------

        confidence_margin = (
            p_nsg_positive
            - nsg_confidence_threshold
        )

        sla_headroom = (
            max_sla_risk_threshold
            - sla_risk
        )

        if abs(gross_saving_mean) > 1e-12:
            migration_overhead_ratio = (
                migration_carbon_mean
                / abs(gross_saving_mean)
            )
        else:
            migration_overhead_ratio = 0.0

        # ---------------------------------------------------------
        # Project gate
        # ---------------------------------------------------------

        is_viable = (
            p_nsg_positive
            >= nsg_confidence_threshold
            and
            sla_risk
            < max_sla_risk_threshold
        )

        return CandidateEvaluationResult(

            pod_id=pod.pod_id,

            current_node_id=current_node.node_id,
            current_region=current_node.region,

            target_node_id=target_node.node_id,
            target_region=target_node.region,

            nsg_mean_gco2=round(
                mean_nsg,
                6,
            ),

            nsg_p05_gco2=round(
                p05_nsg,
                6,
            ),

            nsg_p95_gco2=round(
                p95_nsg,
                6,
            ),

            p_nsg_positive=round(
                p_nsg_positive,
                6,
            ),

            sla_risk_score=round(
                sla_risk,
                6,
            ),

            predicted_exec_time_sec=round(
                mc_result.predicted_exec_time_sec,
                4,
            ),

            migration_duration_sec=round(
                mc_result.migration_time_sec,
                4,
            ),

            is_statistically_viable=is_viable,

            nsg_std_gco2=round(
                nsg_std,
                6,
            ),

            nsg_ci_width_gco2=round(
                ci_width,
                6,
            ),

            gross_carbon_saving_mean_gco2=round(
                gross_saving_mean,
                6,
            ),

            target_execution_carbon_mean_gco2=round(
                target_exec_carbon_mean,
                6,
            ),

            migration_carbon_mean_gco2=round(
                migration_carbon_mean,
                6,
            ),

            stay_carbon_mean_gco2=round(
                stay_carbon_mean,
                6,
            ),

            total_duration_mean_sec=round(
                total_duration_mean,
                4,
            ),

            total_duration_p95_sec=round(
                total_duration_p95,
                4,
            ),

            deadline_miss_rate=round(
                deadline_miss_rate,
                6,
            ),

            sla_headroom=round(
                sla_headroom,
                6,
            ),

            confidence_margin=round(
                confidence_margin,
                6,
            ),

            migration_overhead_ratio=round(
                migration_overhead_ratio,
                6,
            ),

            scheduling_window=scheduling_window,

            timestamp_sec=float(
                timestamp_sec
            ),
        )