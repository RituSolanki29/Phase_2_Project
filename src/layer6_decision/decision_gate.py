"""
Decision Gate (Layer 6: "Hard SLA Constraint Gate" + "PPO RL Agent Tuning")
Combines a deterministic hard safety rule with the learned PPO policy to
produce the final decision for one candidate. The hard gate is enforced
HERE, independently of the agent -- so even if the policy hasn't fully
learned to respect the SLA threshold yet, the system never migrates/starts
a placement the Digital Twin flagged as high SLA risk. This is "defense in
depth": a safety rule the agent cannot override, plus a learned policy that
also receives a penalty for violating it (see policy_env.compute_reward).
"""

from dataclasses import dataclass

from src.layer5_outputs.data_contracts import CandidateEvaluationResult
from src.layer6_decision.policy_env import Action, ACTION_NAMES, SLA_RISK_HARD_LIMIT, NSG_CONFIDENCE_TARGET
from src.layer6_decision.ppo_agent import PPOAgent


@dataclass
class DecisionResult:
    pod_id: str
    target_node_id: str
    action: str                 # STAY | MIGRATE | DELAY | START
    hard_gate_triggered: bool   # True if the deterministic SLA rule overrode the policy
    sla_risk_score: float
    p_nsg_positive: float
    nsg_mean_gco2: float
    rationale: str


class DecisionGate:
    def __init__(self, ppo_agent: PPOAgent):
        self.ppo_agent = ppo_agent

    def decide(self, candidate: CandidateEvaluationResult) -> DecisionResult:
        # --- Hard, non-negotiable safety rule (matches diagram: "< 0.80") ---
        if candidate.sla_risk_score >= SLA_RISK_HARD_LIMIT:
            return DecisionResult(
                pod_id=candidate.pod_id, target_node_id=candidate.target_node_id,
                action=Action.DELAY.name, hard_gate_triggered=True,
                sla_risk_score=candidate.sla_risk_score, p_nsg_positive=candidate.p_nsg_positive,
                nsg_mean_gco2=candidate.nsg_mean_gco2,
                rationale=f"Hard SLA gate: risk {candidate.sla_risk_score:.2f} >= {SLA_RISK_HARD_LIMIT:.2f}; "
                          f"policy was not consulted.",
            )

        # --- Learned policy decides among the remaining safe actions ---
        action_value = self.ppo_agent.predict(candidate)
        action_name = ACTION_NAMES[action_value]

        rationale = (
            f"PPO policy chose {action_name} "
            f"(P(NSG>0)={candidate.p_nsg_positive:.2f}, target={NSG_CONFIDENCE_TARGET:.0%}; "
            f"NSG mean={candidate.nsg_mean_gco2:.5f} gCO2; SLA risk={candidate.sla_risk_score:.2f})."
        )
        return DecisionResult(
            pod_id=candidate.pod_id, target_node_id=candidate.target_node_id,
            action=action_name, hard_gate_triggered=False,
            sla_risk_score=candidate.sla_risk_score, p_nsg_positive=candidate.p_nsg_positive,
            nsg_mean_gco2=candidate.nsg_mean_gco2, rationale=rationale,
        )

    def decide_best_candidate(self, candidates: list) -> DecisionResult:
        """Convenience: given several candidate target nodes for the same
        pod (Layer 3's Top-K shortlist), decide on each and return the one
        with the most favourable outcome (prefers MIGRATE/START with the
        highest NSG, falling back to STAY/DELAY if none are favourable)."""
        decisions = [self.decide(c) for c in candidates]
        actionable = [d for d in decisions if d.action in (Action.MIGRATE.name, Action.START.name)]
        if actionable:
            return max(actionable, key=lambda d: d.nsg_mean_gco2)
        return decisions[0]