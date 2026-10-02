"""
Decision FastAPI Engine (Layer 6: "DECISION OUTPUT -- FastAPI Engine")
Exposes the hard-gate + PPO decision as an HTTP service, matching the
diagram's "/PRIORITIZE" webhook and STAY/MIGRATE/DELAY/START output.
This is what Layer 7's custom scheduler calls to get a real decision for a
real pending pod, and what a human (or a demo dashboard) can call directly.

Run with:
    uvicorn src.layer6_decision.api:app --host 0.0.0.0 --port 8080
"""

from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.layer6_decision.ppo_agent import PPOAgent, DEFAULT_MODEL_PATH
from src.layer6_decision.decision_gate import DecisionGate
from src.layer5_outputs.data_contracts import CandidateEvaluationResult

app = FastAPI(title="Sustainability Orchestrator -- Decision Engine", version="1.0")

_agent = PPOAgent()
_gate: Optional[DecisionGate] = None


class CandidateIn(BaseModel):
    pod_id: str
    current_node_id: str
    current_region: str
    target_node_id: str
    target_region: str
    p_nsg_positive: float
    sla_risk_score: float
    predicted_exec_time_sec: float
    migration_duration_sec: float
    is_statistically_viable: bool
    nsg_mean_gco2: float
    nsg_std_gco2: float = 0.0
    nsg_ci_width_gco2: float = 0.0
    gross_carbon_saving_mean_gco2: float = 0.0
    migration_carbon_mean_gco2: float = 0.0
    target_execution_carbon_mean_gco2: float = 0.0
    stay_carbon_mean_gco2: float = 0.0
    nsg_p05_gco2: float = 0.0
    nsg_p95_gco2: float = 0.0
    total_duration_mean_sec: float = 0.0
    total_duration_p95_sec: float = 0.0
    deadline_miss_rate: float = 0.0
    sla_headroom: float = 0.0
    confidence_margin: float = 0.0
    migration_overhead_ratio: float = 0.0
    scheduling_window: str = ""
    timestamp_sec: float = 0.0

    def to_dataclass(self) -> CandidateEvaluationResult:
        return CandidateEvaluationResult(**self.dict())


class PrioritizeRequest(BaseModel):
    candidates: List[CandidateIn]


class DecisionOut(BaseModel):
    pod_id: str
    target_node_id: str
    action: str
    hard_gate_triggered: bool
    sla_risk_score: float
    p_nsg_positive: float
    nsg_mean_gco2: float
    rationale: str


@app.on_event("startup")
def _load_policy():
    global _gate
    try:
        _agent.load(DEFAULT_MODEL_PATH)
    except FileNotFoundError:
        # No trained policy yet -- /prioritize will return 503 until one is
        # trained via `python -m src.layer6_decision.train_policy`.
        pass
    _gate = DecisionGate(_agent)


@app.get("/health")
def health():
    return {"status": "ok", "policy_loaded": _agent.model is not None}


@app.post("/prioritize", response_model=DecisionOut)
def prioritize(request: PrioritizeRequest):
    """Matches the diagram's '/PRIORITIZE' webhook: given one pod's
    candidate target nodes (from Layer 3's Top-K shortlist, evaluated by
    Layer 4/5), returns the single best decision."""
    if _agent.model is None:
        raise HTTPException(status_code=503, detail="No trained policy loaded. Run train_policy.py first.")
    if not request.candidates:
        raise HTTPException(status_code=400, detail="At least one candidate is required.")

    candidates = [c.to_dataclass() for c in request.candidates]
    decision = _gate.decide_best_candidate(candidates)
    return DecisionOut(**decision.__dict__)


@app.post("/decide_one", response_model=DecisionOut)
def decide_one(candidate: CandidateIn):
    """Decide on a single candidate directly (no Top-K comparison)."""
    if _agent.model is None:
        raise HTTPException(status_code=503, detail="No trained policy loaded. Run train_policy.py first.")
    decision = _gate.decide(candidate.to_dataclass())
    return DecisionOut(**decision.__dict__)