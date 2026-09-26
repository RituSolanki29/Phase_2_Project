"""
Net Sustainability Gain (NSG) & SLA Risk Calculator (Layer 5)
Computes empirical statistical metrics, 90% confidence intervals,
SLA risk scores, and statistical viability flags from Monte Carlo distributions.
"""

import numpy as np
from src.layer1_infra.trace_loader import WorkloadPod
from src.layer1_infra.cluster_config import NodeConfig
from src.layer4_digital_twin.monte_carlo import MonteCarloSimulationResult
from src.layer5_outputs.data_contracts import CandidateEvaluationResult

class Layer5OutputCalculator:
    """Computes Layer 5 statistical model outputs from Monte Carlo empirical samples."""
    
    @staticmethod
    def evaluate_candidate(
        mc_result: MonteCarloSimulationResult,
        pod: WorkloadPod,
        current_node: NodeConfig,
        target_node: NodeConfig,
        nsg_confidence_threshold: float = 0.90,
        max_sla_risk_threshold: float = 0.80
    ) -> CandidateEvaluationResult:
        """
        Analyzes 1000 Monte Carlo samples to compute:
        1. Mean Net Sustainability Gain (NSG)
        2. 90% Confidence Interval [5th percentile, 95th percentile]
        3. Statistical Confidence P(NSG > 0)
        4. SLA Risk Score P(Total Duration > SLA Deadline)
        """
        nsg_samples = mc_result.net_sustainability_gain_samples
        duration_samples = mc_result.total_duration_samples_sec
        
        mean_nsg = float(np.mean(nsg_samples))
        p05_nsg = float(np.percentile(nsg_samples, 5.0))
        p95_nsg = float(np.percentile(nsg_samples, 95.0))
        
        # Empirical probability P(NSG > 0)
        num_positive_nsg = np.sum(nsg_samples > 0.0)
        p_nsg_positive = float(num_positive_nsg / len(nsg_samples))
        
        # Empirical SLA Risk Score P(T_total > SLA_deadline)
        num_sla_violations = np.sum(duration_samples > pod.sla_deadline_sec)
        sla_risk_score = float(num_sla_violations / len(duration_samples))
        
        # Hard Statistical Viability Gate: P(NSG > 0) >= 0.90 and SLA Risk < 0.80
        is_viable = (p_nsg_positive >= nsg_confidence_threshold) and (sla_risk_score < max_sla_risk_threshold)

        return CandidateEvaluationResult(
            pod_id=pod.pod_id,
            current_node_id=current_node.node_id,
            current_region=current_node.region,
            target_node_id=target_node.node_id,
            target_region=target_node.region,
            nsg_mean_gco2=round(mean_nsg, 4),
            nsg_p05_gco2=round(p05_nsg, 4),
            nsg_p95_gco2=round(p95_nsg, 4),
            p_nsg_positive=round(p_nsg_positive, 4),
            sla_risk_score=round(sla_risk_score, 4),
            predicted_exec_time_sec=mc_result.predicted_exec_time_sec,
            migration_duration_sec=mc_result.migration_time_sec,
            is_statistically_viable=is_viable
        )
