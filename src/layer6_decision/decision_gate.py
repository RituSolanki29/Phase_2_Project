"""
Decision Gate (Layer 6: "Hard SLA Constraint Gate" + "PPO RL Agent Tuning")

Decision logic for ONE candidate (pod -> target node):

  1. Hard SLA gate (deterministic, the agent cannot override it):
         SLA risk >= SLA_RISK_HARD_LIMIT  ->  DELAY
  2. In-place candidate (target == current node):
         -> START in place.  This is the "stay" baseline; its NSG is exactly 0
            by construction, so no statistical test applies.
  3. Migration candidate:
         requires P(NSG>0) >= NSG_CONFIDENCE_TARGET (0.90) AND mean NSG > 0
         (i.e. the candidate is statistically viable).  Otherwise -> STAY.
         If it is viable the PPO policy gets the final say (it may veto).

Why this replaced the old gate:
  * The old gate used a hard-coded 0.40 confidence threshold (the project
    spec says 0.90) and then let PPO choose among START/MIGRATE, which have
    identical rewards -- so every pod was logged as "START" regardless of
    whether the candidate was viable.
  * A non-viable migration must fall back to starting in place, NOT to DELAY:
    DELAY leaves a Pending pod unscheduled forever.
"""

from dataclasses import dataclass
from typing import List

from src.layer5_outputs.data_contracts import CandidateEvaluationResult
from src.layer6_decision.policy_env import (
    Action, ACTION_NAMES, SLA_RISK_HARD_LIMIT, NSG_CONFIDENCE_TARGET,
)
from src.layer6_decision.ppo_agent import PPOAgent


@dataclass
class DecisionResult:
    pod_id: str
    target_node_id: str
    action: str                 # STAY | MIGRATE | DELAY | START
    hard_gate_triggered: bool   # True if a deterministic rule decided (policy not consulted)
    sla_risk_score: float
    p_nsg_positive: float
    nsg_mean_gco2: float
    rationale: str


class DecisionGate:
    def __init__(self, ppo_agent: PPOAgent,
                 confidence_threshold: float = NSG_CONFIDENCE_TARGET,
                 sla_risk_limit: float = SLA_RISK_HARD_LIMIT):
        self.ppo_agent = ppo_agent
        self.confidence_threshold = confidence_threshold
        self.sla_risk_limit = sla_risk_limit

    def _result(self, c: CandidateEvaluationResult, action: Action, target_node_id: str,
                hard: bool, rationale: str) -> DecisionResult:
        return DecisionResult(
            pod_id=c.pod_id, target_node_id=target_node_id, action=action.name,
            hard_gate_triggered=hard, sla_risk_score=c.sla_risk_score,
            p_nsg_positive=c.p_nsg_positive, nsg_mean_gco2=c.nsg_mean_gco2, rationale=rationale,
        )

    def decide(self, candidate: CandidateEvaluationResult) -> DecisionResult:
        c = candidate
        in_place = c.target_node_id == c.current_node_id

        # 1. Hard SLA gate
        if c.sla_risk_score >= self.sla_risk_limit:
            return self._result(
                c, Action.DELAY, c.target_node_id, True,
                f"Hard gate: SLA risk {c.sla_risk_score:.2f} >= {self.sla_risk_limit:.2f}; policy not consulted.",
            )

        # 2. In-place start = stay baseline
        if in_place:
            return self._result(
                c, Action.START, c.current_node_id, False,
                f"In-place start on {c.current_node_id} (stay baseline, NSG=0 by definition; "
                f"SLA risk={c.sla_risk_score:.2f}).",
            )

        # 3. Migration candidate: statistical viability first
        reasons = []
        if c.p_nsg_positive < self.confidence_threshold:
            reasons.append(f"P(NSG>0)={c.p_nsg_positive:.2f} < {self.confidence_threshold:.2f}")
        if c.nsg_mean_gco2 <= 0:
            reasons.append(f"mean NSG {c.nsg_mean_gco2:+.5f} gCO2 <= 0")
        if reasons:
            return self._result(
                c, Action.STAY, c.current_node_id, True,
                f"Not statistically viable ({'; '.join(reasons)}); staying on {c.current_node_id}.",
            )

        # Viable: learned policy has the final say.
        policy_action = ACTION_NAMES[self.ppo_agent.predict(c)]
        if policy_action in (Action.MIGRATE.name, Action.START.name):
            return self._result(
                c, Action.MIGRATE, c.target_node_id, False,
                f"PPO approved migration (P(NSG>0)={c.p_nsg_positive:.2f} >= {self.confidence_threshold:.2f}; "
                f"NSG mean={c.nsg_mean_gco2:+.5f} gCO2; SLA risk={c.sla_risk_score:.2f}).",
            )
        return self._result(
            c, Action.STAY, c.current_node_id, False,
            f"PPO vetoed a viable migration (chose {policy_action}); staying on {c.current_node_id}.",
        )

    def decide_best_candidate(self, candidates: List[CandidateEvaluationResult]) -> DecisionResult:
        """Given Layer 3's Top-K shortlist for ONE pod, return the final decision.

        Preference order:
          1. the approved MIGRATE with the highest mean NSG;
          2. otherwise start in place (the stay candidate, or an equivalent
             STAY decision converted to START on the current node);
          3. otherwise (everything violates the hard SLA gate) DELAY, reporting
             the lowest-SLA-risk candidate so the scheduler can force-bind it
             after a bounded number of retries.
        """
        decisions = [self.decide(c) for c in candidates]

        migrations = [d for d in decisions if d.action == Action.MIGRATE.name]
        if migrations:
            return max(migrations, key=lambda d: d.nsg_mean_gco2)

        starts = [d for d in decisions if d.action == Action.START.name]
        if starts:
            return starts[0]

        stays = [d for d in decisions if d.action == Action.STAY.name]
        if stays:
            d = stays[0]
            d.action = Action.START.name  # pod must start somewhere: in place
            return d

        return min(decisions, key=lambda d: d.sla_risk_score)