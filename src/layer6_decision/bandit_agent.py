"""
Contextual Bandit Scheduling Policy (LinUCB Baseline / Alternative for Layer 6)
A swappable contextual bandit alternative to PPO. Designed for low-sample regimes
(15-50 samples) where deep RL policy gradient algorithms may suffer from high variance
or mode collapse. Uses linear upper confidence bounds over extracted candidate features.
"""

from pathlib import Path
from typing import List, Optional
import numpy as np

from src.layer5_outputs.data_contracts import CandidateEvaluationResult
from src.layer6_decision.policy_env import Action, observation_for, ACTION_NAMES, OBS_DIM
DEFAULT_BANDIT_PATH = Path("models/bandit_policy.npz")


class BanditAgent:
    """LinUCB Contextual Bandit Agent."""

    def __init__(self, n_actions: int = 4, alpha: float = 1.0, seed: int = 42):
        self.n_actions = n_actions
        self.alpha = alpha
        self.seed = seed
        self.feature_dim = OBS_DIM # matches policy_env.observation_for vector length

        # LinUCB parameters per action: A = d x d identity, b = d x 1 zeros
        self.A = [np.identity(self.feature_dim, dtype=float) for _ in range(n_actions)]
        self.b = [np.zeros((self.feature_dim, 1), dtype=float) for _ in range(n_actions)]

    def _extract_feature(self, candidate: CandidateEvaluationResult) -> np.ndarray:
        obs = observation_for(candidate, nsg_scale=50.0)
        return obs.reshape(-1, 1)

    def predict(self, candidate: CandidateEvaluationResult) -> int:
        x = self._extract_feature(candidate)
        ucb_values = []
        for a in range(self.n_actions):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            p_a = (theta.T @ x)[0, 0] + self.alpha * np.sqrt((x.T @ A_inv @ x)[0, 0])
            ucb_values.append(p_a)
        return int(np.argmax(ucb_values))

    def pretrain(self, candidates: List[CandidateEvaluationResult], total_timesteps: int = 1000) -> None:
        """Pretrain LinUCB by simulating decisions on candidate evaluations."""
        rng = np.random.default_rng(self.seed)
        for _ in range(total_timesteps):
            candidate = candidates[rng.integers(0, len(candidates))]
            action = self.predict(candidate)
            # Simulated reward: +1 for positive NSG with low SLA risk, -1 for risk/loss
            reward = float(candidate.nsg_mean_gco2 * 50.0 - candidate.sla_risk_score * 2.0)
            if action in (Action.STAY.value, Action.DELAY.value):
                reward = max(0.0, reward)
            x = self._extract_feature(candidate)
            self.A[action] += x @ x.T
            self.b[action] += reward * x

    def update_from_real_outcome(self, candidate: CandidateEvaluationResult, action: int, realized_nsg: float):
        x = self._extract_feature(candidate)
        reward = float(realized_nsg * 50.0 - candidate.sla_risk_score * 2.0)
        self.A[action] += x @ x.T
        self.b[action] += reward * x

    def save(self, path: Path = DEFAULT_BANDIT_PATH):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            A=np.array(self.A),
            b=np.array(self.b),
            alpha=self.alpha,
            feature_dim=self.feature_dim,
        )
        print(f"[BanditAgent] Saved bandit policy to {path}")

    def load(self, path: Path = DEFAULT_BANDIT_PATH):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No bandit policy at {path}")
        data = np.load(path)
        self.A = [data["A"][i] for i in range(self.n_actions)]
        self.b = [data["b"][i] for i in range(self.n_actions)]
        self.alpha = float(data["alpha"])
        print(f"[BanditAgent] Loaded bandit policy from {path}")
