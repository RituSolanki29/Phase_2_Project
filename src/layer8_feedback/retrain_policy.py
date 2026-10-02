"""
Online PPO Fine-Tuning from Real Outcomes (Layer 8 -> Layer 6 feedback loop)
Reads (decision, real outcome) pairs accumulated by post_exec_evaluator.py
in the reward log, reconstructs the original candidates, and runs a batched
PPO fine-tuning step using the REALIZED NSG instead of the Digital Twin's
simulated prediction -- this is what makes the policy improve from actual
live-cluster behaviour over time, matching the diagram's "PPO Policy
Update" feedback arrow from Box 8 back into Box 6.

Run periodically during/after a live demo:
    python -m src.layer8_feedback.retrain_policy
"""

from src.layer5_outputs.data_contracts import CandidateEvaluationResult
from src.layer6_decision.ppo_agent import PPOAgent, DEFAULT_MODEL_PATH
from src.layer6_decision.policy_env import Action
from src.layer8_feedback.reward_store import RewardStore


def _candidate_from_dict(d: dict) -> CandidateEvaluationResult:
    valid_fields = CandidateEvaluationResult.__dataclass_fields__.keys()
    return CandidateEvaluationResult(**{k: v for k, v in d.items() if k in valid_fields})


if __name__ == "__main__":
    store = RewardStore()
    pairs = store.load_unprocessed_outcomes()
    if not pairs:
        print("[Retrain] No new real outcomes to learn from yet. Run the live demo first.")
        raise SystemExit(0)

    print(f"[Retrain] Found {len(pairs)} real outcomes. Loading current policy...")
    agent = PPOAgent()
    try:
        agent.load(DEFAULT_MODEL_PATH)
    except FileNotFoundError:
        print("[Retrain] No existing policy found -- training a fresh one from these real outcomes only.")

    for pair in pairs:
        candidate = _candidate_from_dict(pair["candidate"])
        action_value = Action[pair["action"]].value if pair["action"] in Action.__members__ else Action.MIGRATE.value
        reward = agent.learn_from_real_outcome(candidate, action_value, pair["realized_nsg_gco2"])
        print(f"  {pair['pod_id']}: action={pair['action']}, realized_nsg={pair['realized_nsg_gco2']:+.6f}, "
              f"fine-tune reward={reward:+.6f}")

    agent.save(DEFAULT_MODEL_PATH)
    print(f"[Retrain] Updated policy saved to {DEFAULT_MODEL_PATH}. "
          f"Restart the API (or it will reload on next request) to use it.")