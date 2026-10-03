"""
PPO Scheduling Policy (Layer 6: "PPO RL Agent Tuning")
Thin wrapper over stable-baselines3's PPO for this project's specific
observation/action contract (see policy_env.py). Supports:
  - Offline pretraining on Layer 5's Digital Twin outputs.
  - Saving/loading a trained policy to disk (including the nsg_scale the
    policy was trained with -- see save()/load()).
  - Online fine-tuning from real post-execution outcomes (Layer 8).
"""

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.layer5_outputs.data_contracts import CandidateEvaluationResult
from src.layer6_decision.policy_env import SchedulingDecisionEnv, observation_for, compute_reward

DEFAULT_MODEL_PATH = Path("models/ppo_scheduling_policy.zip")


def _meta_path(model_path: Path) -> Path:
    model_path = Path(model_path)
    return model_path.with_suffix(".meta.json")


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
                          n_steps=256, batch_size=64, learning_rate=3e-4, ent_coef=0.02)
        self.model.learn(total_timesteps=total_timesteps)

    def predict(self, candidate: CandidateEvaluationResult) -> int:
        if self.model is None:
            raise RuntimeError("PPOAgent has no trained/loaded model. Call pretrain() or load() first.")
        obs = observation_for(candidate, self.nsg_scale)
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)

    def learn_from_real_outcome(self, candidate: CandidateEvaluationResult, action_taken: int,
                                realized_nsg_gco2: float) -> float:
        """Single-transition convenience wrapper around retrain_batch().
        Prefer retrain_batch() with many outcomes at once."""
        if self.model is None:
            raise RuntimeError("PPOAgent has no trained/loaded model.")
        reward = compute_reward(candidate, action_taken, realized_nsg_gco2=realized_nsg_gco2,
                                nsg_scale=self.nsg_scale)
        self.retrain_batch([candidate], realized_nsgs=[realized_nsg_gco2], total_timesteps=256)
        return reward

    def retrain_batch(self, candidates: List[CandidateEvaluationResult],
                      realized_nsgs: Optional[List[Optional[float]]] = None,
                      total_timesteps: int = 2_000) -> None:
        """Periodic batched fine-tune from a buffer of (candidate, outcome) pairs.
        `realized_nsgs[i]` (if given and not None) REPLACES the Digital Twin's
        predicted NSG in the reward for candidates[i], so the policy really
        learns from measured outcomes."""
        if self.model is None:
            self.pretrain(candidates, total_timesteps=total_timesteps)
            return
        env = DummyVecEnv([lambda: SchedulingDecisionEnv(candidates, self.nsg_scale, self.seed)])
        self.model.set_env(env)
        self.model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)

    def save(self, path: Path = DEFAULT_MODEL_PATH) -> None:
        if self.model is None:
            raise RuntimeError("Nothing to save -- train a model first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        # The observation scaling used in training MUST be reused at inference.
        _meta_path(path).write_text(json.dumps({"nsg_scale": self.nsg_scale}, indent=2), encoding="utf-8")

    def load(self, path: Path = DEFAULT_MODEL_PATH) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No saved policy at {path}. Run pretraining first.")
        self.model = PPO.load(str(path))
        meta = _meta_path(path)
        if meta.exists():
            self.nsg_scale = float(json.loads(meta.read_text(encoding="utf-8"))["nsg_scale"])
        else:
            print(f"[PPOAgent] WARNING: {meta} not found. This policy was saved before nsg_scale was "
                  f"persisted, so observations at inference may not match training. Re-run "
                  f"`python -m src.layer6_decision.train_policy`.")