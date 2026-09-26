"""
Layer 5 Output Data Contracts & Schemas
Defines structured evaluation data types consumed by downstream Layer 6 Decision Engine.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any

@dataclass
class CandidateEvaluationResult:
    pod_id: str
    current_node_id: str
    current_region: str
    target_node_id: str
    target_region: str
    nsg_mean_gco2: float
    nsg_p05_gco2: float               # 5th percentile NSG
    nsg_p95_gco2: float               # 95th percentile NSG
    p_nsg_positive: float             # Statistical confidence P(NSG > 0)
    sla_risk_score: float             # P(T_total > SLA_deadline)
    predicted_exec_time_sec: float
    migration_duration_sec: float
    is_statistically_viable: bool    # True if P(NSG > 0) >= 0.90 and SLA Risk < 0.80

    def to_dict(self) -> Dict[str, Any]:
        """Converts dataclass to JSON-compatible dictionary."""
        return asdict(self)
