"""
Layer 5 Output Data Contracts.

The original fields are retained for compatibility.

Additional fields expose metrics useful for the experimental results section:
- gross carbon saving
- target execution carbon
- migration carbon
- NSG standard deviation
- confidence interval width
- SLA deadline
- scheduling window
- timestamp
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class CandidateEvaluationResult:

    # ---------------------------------------------------------
    # Existing core identifiers
    # ---------------------------------------------------------

    pod_id: str

    current_node_id: str
    current_region: str

    target_node_id: str
    target_region: str

    # ---------------------------------------------------------
    # Existing Layer 5 metrics
    # ---------------------------------------------------------

    nsg_mean_gco2: float
    nsg_p05_gco2: float
    nsg_p95_gco2: float

    p_nsg_positive: float

    sla_risk_score: float

    predicted_exec_time_sec: float

    migration_duration_sec: float

    is_statistically_viable: bool

    # ---------------------------------------------------------
    # New experimental metrics
    # ---------------------------------------------------------

    nsg_std_gco2: float = 0.0

    nsg_ci_width_gco2: float = 0.0

    gross_carbon_saving_mean_gco2: float = 0.0

    target_execution_carbon_mean_gco2: float = 0.0

    migration_carbon_mean_gco2: float = 0.0

    stay_carbon_mean_gco2: float = 0.0

    total_duration_mean_sec: float = 0.0

    total_duration_p95_sec: float = 0.0

    deadline_miss_rate: float = 0.0

    sla_headroom: float = 0.0

    confidence_margin: float = 0.0

    migration_overhead_ratio: float = 0.0

    scheduling_window: str = ""

    timestamp_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)