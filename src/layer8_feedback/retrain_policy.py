"""
Online PPO Fine-Tuning from Real Outcomes (Layer 8 -> Layer 6 feedback loop)
Reads (decision, real outcome) pairs from the reward log, reconstructs the
original candidates and fine-tunes PPO with the REALIZED NSG substituted for
the Digital Twin's prediction in the reward.

Fixes vs. the previous version:
  * The realized NSG now actually reaches the training environment.
  * Only NEW outcomes are used each run.
"""

import argparse

from src.layer5_outputs.data_contracts import CandidateEvaluationResult
from src.layer6_decision.ppo_agent import PPOAgent, DEFAULT_MODEL_PATH
from src.layer8_feedback.reward_store import RewardStore, DEFAULT_LOG_PATH


def _candidate_from_dict(d: dict) -> CandidateEvaluationResult:
    valid_fields = CandidateEvaluationResult.__dataclass_fields__.keys()
    return CandidateEvaluationResult(**{k: v for k, v in d.items() if k in valid_fields})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune the PPO policy from real outcomes.")
    parser.add_argument("--log-path", default=str(DEFAULT_LOG_PATH))
    parser.add_argument("--timesteps", type=int, default=2000)
    parser.add_argument("--all", action="store_true", help="Use all outcomes, not just unprocessed ones.")
    args = parser.parse_args()

    store = RewardStore(args.log_path)
    pairs = store.load_pairs(only_unprocessed=not args.all)
    if not pairs:
        print("[Retrain] No new real outcomes to learn from yet. Run the live demo first.")
        raise SystemExit(0)

    print(f"[Retrain] Found {len(pairs)} real outcomes. Loading current policy...")
    agent = PPOAgent()
    try:
        agent.load(DEFAULT_MODEL_PATH)
    except FileNotFoundError:
        print("[Retrain] No existing policy found -- training a fresh one from these real outcomes only.")

    candidates = [_candidate_from_dict(p["candidate"]) for p in pairs]
    realized = [float(p["realized_nsg_gco2"]) for p in pairs]
    for p in pairs:
        print(f"  {p['pod_id']}: action={p['action']}, realized_nsg={p['realized_nsg_gco2']:+.6f}")

    agent.retrain_batch(candidates, realized_nsgs=realized, total_timesteps=args.timesteps)
    agent.save(DEFAULT_MODEL_PATH)
    store.mark_processed([p["key"] for p in pairs])
    print(f"[Retrain] Updated policy saved to {DEFAULT_MODEL_PATH}.")