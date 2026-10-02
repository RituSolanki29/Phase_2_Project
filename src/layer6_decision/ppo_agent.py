"""
PPO Scheduling Policy (Layer 6: "PPO RL Agent Tuning")
Thin wrapper over stable-baselines3's PPO for this project's specific
observation/action contract (see policy_env.py). Supports:
  - Offline pretraining on Layer 5's Digital Twin outputs.
  - Saving/loading a trained policy to disk.
  - Online fine-tuning from real post-execution outcomes (Layer 8).
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.layer5_outputs.data_contracts import CandidateEvaluationResult
from src.layer6_decision.policy_env import SchedulingDecisionEnv, Action, observation_for, compute_reward

DEFAULT_MODEL_PATH = Path("models/ppo_scheduling_policy.zip")


class PPOAgent:
    def __init__(self, nsg_scale: float = 50.0, seed: int = 42):
        self.nsg_scale = nsg_scale
        self.seed = seed
        self.model: Optional[PPO] = None

    def pretrain(self, candidates: List[CandidateEvaluationResult], total_timesteps: int = 20_000,
                 auto_scale: bool = True) -> None:
        """Offline pretraining on the Digital Twin's simulated NSG outcomes.
        `auto_scale`: if True (default), derive nsg_scale from the actual
        magnitude of NSG values in `candidates` rather than trusting a fixed
        constant -- NSG magnitude depends heavily on workload size and trace
        compression factor, and a mismatched fixed scale silently flattens
        the reward signal the policy learns from."""
        if auto_scale:
            magnitudes = [abs(c.nsg_mean_gco2) for c in candidates if c.nsg_mean_gco2 != 0]
            if magnitudes:
                self.nsg_scale = max(1e-9, float(np.mean(magnitudes)) * 3.0)
        env = DummyVecEnv([lambda: SchedulingDecisionEnv(candidates, self.nsg_scale, self.seed)])
        self.model = PPO("MlpPolicy", env, verbose=0, seed=self.seed,
                          n_steps=256, batch_size=64, learning_rate=3e-4)
        self.model.learn(total_timesteps=total_timesteps)

    def predict(self, candidate: CandidateEvaluationResult) -> int:
        if self.model is None:
            raise RuntimeError("PPOAgent has no trained/loaded model. Call pretrain() or load() first.")
        obs = observation_for(candidate, self.nsg_scale)
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)

    def learn_from_real_outcome(self, candidate: CandidateEvaluationResult, action_taken: int, realized_nsg_gco2: float) -> None:
        """
        Online fine-tuning step: given a REAL measured outcome (from Layer 8)
        for a decision the policy already made, run one small gradient update
        so the policy incorporates reality rather than staying frozen on the
        Digital Twin's simulated predictions. This is a lightweight on-policy
        nudge (a single-transition rollout), not a full retrain -- call
        periodically (e.g. after every N real outcomes) via retrain_batch().
        """
        if self.model is None:
            raise RuntimeError("PPOAgent has no trained/loaded model.")
        reward = compute_reward(candidate, action_taken, realized_nsg_gco2=realized_nsg_gco2)
        env = DummyVecEnv([lambda: SchedulingDecisionEnv([candidate], self.nsg_scale, self.seed)])
        self.model.set_env(env)
        self.model.learn(total_timesteps=256, reset_num_timesteps=False)
        return reward

    def retrain_batch(self, candidates: List[CandidateEvaluationResult], total_timesteps: int = 2_000) -> None:
        """Periodic batched fine-tune from a buffer of real (candidate, outcome) pairs
        -- call this from reward_store.py once enough real samples have accumulated,
        rather than fine-tuning one transition at a time."""
        if self.model is None:
            self.pretrain(candidates, total_timesteps=total_timesteps)
            return
        env = DummyVecEnv([lambda: SchedulingDecisionEnv(candidates, self.nsg_scale, self.seed)])
        self.model.set_env(env)
        self.model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)

    def save(self, path: Path = DEFAULT_MODEL_PATH) -> None:
        if self.model is None:
            raise RuntimeError("Nothing to save -- train a model first.")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))

    def load(self, path: Path = DEFAULT_MODEL_PATH) -> None:
        if not Path(path).exists():
            raise FileNotFoundError(f"No saved policy at {path}. Run pretraining first.")
        self.model = PPO.load(str(path))