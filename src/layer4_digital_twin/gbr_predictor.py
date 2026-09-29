"""
GradientBoostingRegressor Runtime Predictor (Layer 4)
Predicts baseline workload execution runtime (on hardware equivalent to the
node the workload historically ran on) from trace features, using a strict
time-based train/test split to prevent leakage.

IMPORTANT DESIGN NOTE (fixed vs. earlier version):
MIPS/hardware-heterogeneity scaling is intentionally NOT passed into the GBR
as a feature. A tree ensemble can only learn a relationship from a feature
that actually varies across training examples; every historical trace row
only tells us the duration achieved on ONE piece of hardware, so there is no
ground truth for "how would this same job run at a different MIPS rating"
to train against. Feeding a MIPS ratio into the model here would appear
principled but the model could not have learned anything from it if that
input were held constant during training, and if we synthesize varying
ratios we'd just be baking our deterministic formula into noisy labels for
the model to loosely re-learn.

The correct separation of concerns is:
  - GBR predicts the baseline duration from workload characteristics alone.
  - HardwareHeterogeneityModel.scale_duration() deterministically projects
    that baseline duration onto the target node's MIPS rating afterwards.
This keeps a physically-known deterministic relationship deterministic, and
lets the learned model focus on what it can actually learn from the data.
"""

from typing import List, Tuple
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, r2_score

from src.layer1_infra.trace_loader import WorkloadPod


class GBRRuntimePredictor:
    """Predicts baseline workload duration and its prediction-error variance."""

    def __init__(self, random_state: int = 42):
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            random_state=random_state
        )
        self.is_trained = False
        self.residual_std = 5.0  # Default fallback residual standard deviation (seconds)

    def _extract_features(self, pods: List[WorkloadPod]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Constructs feature matrix X and target vector y from pods.
        Target is the pod's actual historical duration on its own (source)
        node -- no MIPS ratio is folded in here; see module docstring.
        """
        X = []
        y = []
        for pod in pods:
            X.append([
                pod.cpu_request,
                pod.ram_request_gb,
                pod.assigned_memory_gb,
                pod.cpi,
                float(pod.priority),
            ])
            y.append(pod.duration_sec)

        return np.array(X), np.array(y)

    def train_time_split(self, pods: List[WorkloadPod], train_ratio: float = 0.8) -> Tuple[float, float]:
        """
        Trains GBR model using time-ordered split (first train_ratio train,
        remaining test) to avoid data leakage. Returns (RMSE, R2 score).
        """
        if len(pods) < 10:
            self.is_trained = True
            return 0.0, 1.0

        # Sort pods chronologically by start_time_micro -- critical to avoid leakage
        sorted_pods = sorted(pods, key=lambda p: p.start_time_micro)

        X, y = self._extract_features(sorted_pods)

        split_idx = int(len(X) * train_ratio)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        self.model.fit(X_train, y_train)
        self.is_trained = True

        y_pred = self.model.predict(X_test)
        rmse = float(root_mean_squared_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        residuals = y_test - y_pred
        self.residual_std = max(1.0, float(np.std(residuals)))

        return round(rmse, 3), round(r2, 3)

    def predict(self, pod: WorkloadPod) -> Tuple[float, float]:
        """
        Predicts baseline expected duration (mean) and uncertainty standard
        deviation (std) for `pod`, AT THE HARDWARE CLASS IT WAS HISTORICALLY
        OBSERVED ON. Callers must apply
        HardwareHeterogeneityModel.scale_duration() themselves to project
        this baseline onto a different target node's MIPS rating.
        """
        if not self.is_trained:
            # Baseline estimate if untrained
            return round(pod.duration_sec, 2), self.residual_std

        X_single = np.array([[
            pod.cpu_request,
            pod.ram_request_gb,
            pod.assigned_memory_gb,
            pod.cpi,
            float(pod.priority),
        ]])

        predicted_mean = float(self.model.predict(X_single)[0])
        predicted_mean = max(2.0, predicted_mean)
        return round(predicted_mean, 2), self.residual_std