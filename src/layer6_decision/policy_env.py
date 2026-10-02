"""
Scheduling Decision Environment (Layer 6)
A Gymnasium environment that frames one candidate (pod, target-node)
evaluation from Layer 5 as a single-step decision problem: given the
Digital Twin's statistical outputs for this candidate, choose one of
STAY / MIGRATE / DELAY / START.

Each episode is exactly one decision (horizon = 1). This is a standard way
to train PPO as a contextual-bandit-style policy: the "episode" is short,
but PPO's clipped objective and value baseline still provide stable,
sample-efficient learning here, and -- importantly -- it means the same
agent architecture can later be extended to multi-step horizons without
changing the action/observation contract.

TRAINING SIGNAL (important, state this in your report):
  - Offline/pretraining: reward is the Digital Twin's SIMULATED Net
    Sustainability Gain (Layer 5's nsg_mean_gco2) for this candidate --
    i.e. the agent first learns from the Monte Carlo model's predictions,
    not from reality.
  - Online fine-tuning: once running against a real cluster, Layer 8
    (post_exec_evaluator.py) measures the ACTUAL achieved NSG after a real
    migration/placement and feeds that back through this same environment's
    reward function via PPOAgent.learn_from_real_outcome(), so the policy
    keeps improving on real data rather than staying frozen on simulation.
This sim-to-real split is a deliberate, standard RL pattern -- not a
shortcut -- and should be described as such.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.layer5_outputs.data_contracts import CandidateEvaluationResult

SLA_RISK_HARD_LIMIT = 0.80   # matches diagram Box 6: "Hard SLA Constraint Gate < 0.80"
NSG_CONFIDENCE_TARGET = 0.90  # matches diagram Box 5: "90% Confidence Interval"


class Action(IntEnum):
    STAY = 0
    MIGRATE = 1
    DELAY = 2
    START = 3


ACTION_NAMES = {a.value: a.name for a in Action}

# Observation feature order -- keep this in sync with _vectorize() below.
OBS_FEATURES = [
    "p_nsg_positive", "sla_risk_score", "nsg_mean_scaled", "confidence_margin",
    "sla_headroom", "migration_overhead_ratio", "deadline_miss_rate",
]
OBS_DIM = len(OBS_FEATURES)


def _vectorize(c: CandidateEvaluationResult, nsg_scale: float = 50.0) -> np.ndarray:
    """Turns one Layer 5 candidate result into a bounded observation vector.
    nsg_mean is divided by `nsg_scale` (gCO2) and clipped to keep PPO's
    inputs roughly in [-1, 1] -- adjust nsg_scale to your workload's typical
    NSG magnitude if your trace produces much larger/smaller values."""
    return np.array([
        float(np.clip(c.p_nsg_positive, 0.0, 1.0)),
        float(np.clip(c.sla_risk_score, 0.0, 1.0)),
        float(np.clip(c.nsg_mean_gco2 / nsg_scale, -3.0, 3.0)),
        float(np.clip(c.confidence_margin, -1.0, 1.0)),
        float(np.clip(c.sla_headroom, -1.0, 1.0)),
        float(np.clip(c.migration_overhead_ratio, 0.0, 5.0)),
        float(np.clip(c.deadline_miss_rate, 0.0, 1.0)),
    ], dtype=np.float32)


def compute_reward(c: CandidateEvaluationResult, action: int, realized_nsg_gco2: Optional[float] = None) -> float:
    """
    Reward function shared by offline pretraining and online fine-tuning.
    `realized_nsg_gco2`: pass the Digital Twin's nsg_mean_gco2 for offline
    pretraining, or the REAL measured NSG from Layer 8 for online updates.
    """
    nsg = c.nsg_mean_gco2 if realized_nsg_gco2 is None else realized_nsg_gco2
    sla_violation = c.sla_risk_score >= SLA_RISK_HARD_LIMIT

    if action == Action.MIGRATE:
        reward = nsg
        if sla_violation:
            reward -= 5.0  # strong penalty: policy should learn to respect the hard gate itself
        return reward
    if action == Action.START:
        # No "stay" baseline exists for a brand-new pod -- reward is simply
        # the NSG this placement choice achieves relative to a null (no
        # power-aware placement) baseline, which Layer 5 already encodes.
        reward = nsg
        if sla_violation:
            reward -= 5.0
        return reward
    if action == Action.STAY:
        return 0.0  # explicit no-op baseline; NSG is defined relative to this
    if action == Action.DELAY:
        # Small opportunity-cost penalty for delaying a good opportunity,
        # but a small reward for correctly delaying a risky one.
        return 0.3 if sla_violation else -0.05
    return 0.0


class SchedulingDecisionEnv(gym.Env):
    """Gymnasium env over a fixed list of CandidateEvaluationResult rows,
    one of which is sampled per episode (horizon = 1)."""

    metadata = {"render_modes": []}

    def __init__(self, candidates: List[CandidateEvaluationResult], nsg_scale: float = 50.0, seed: int = 42):
        super().__init__()
        if not candidates:
            raise ValueError("SchedulingDecisionEnv requires at least one candidate.")
        self.candidates = candidates
        self.nsg_scale = nsg_scale
        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(OBS_DIM,), dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._current: Optional[CandidateEvaluationResult] = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._current = self.candidates[self._rng.integers(0, len(self.candidates))]
        obs = _vectorize(self._current, self.nsg_scale)
        return obs, {}

    def step(self, action: int):
        reward = compute_reward(self._current, int(action))
        obs = _vectorize(self._current, self.nsg_scale)  # terminal obs == current (horizon=1)
        terminated = True
        truncated = False
        info = {"pod_id": self._current.pod_id, "target_node_id": self._current.target_node_id,
                 "action_name": ACTION_NAMES[int(action)]}
        return obs, reward, terminated, truncated, info


def observation_for(candidate: CandidateEvaluationResult, nsg_scale: float = 50.0) -> np.ndarray:
    """Public helper so decision_gate.py can build an observation without
    instantiating a full Gym env."""
    return _vectorize(candidate, nsg_scale)