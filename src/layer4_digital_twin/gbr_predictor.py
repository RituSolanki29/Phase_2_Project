"""
Gradient Boosting Runtime Predictor (Layer 4)

Predicts workload execution duration from workload characteristics.

Important methodological rule:
MIPS is NOT used as a learned feature because the historical trace does not
provide repeated observations of the same workload across different hardware.
Hardware scaling is therefore applied deterministically after prediction.

The model additionally reports:
    - MAE
    - RMSE
    - R²
    - sMAPE
    - baseline MAE
    - baseline RMSE
    - baseline R²
    - improvement over mean baseline
    - residual standard deviation
"""

from typing import List, Tuple, Dict

import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.layer1_infra.trace_loader import WorkloadPod


class GBRRuntimePredictor:
    """Predicts baseline workload duration and prediction uncertainty."""

    def __init__(self, random_state: int = 42):

        self.random_state = int(random_state)

        self.model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=2,
            min_samples_split=5,
            min_samples_leaf=3,
            loss="huber",
            random_state=self.random_state,
        )

        self.is_trained = False

        self.residual_std = 5.0

        self.metrics: Dict[str, float] = {}

        self.feature_names = [
            "cpu_request",
            "ram_request_gb",
            "assigned_memory_gb",
            "memory_ratio",
            "cpi",
            "priority",
            "cpu_memory_product",
            "log_duration_independent_features",
        ]

    def _extract_features(
        self,
        pods: List[WorkloadPod],
    ) -> Tuple[np.ndarray, np.ndarray]:

        X = []
        y = []

        for pod in pods:

            ram_request = max(float(pod.ram_request_gb), 0.001)
            assigned_memory = max(
                float(pod.assigned_memory_gb),
                0.001,
            )

            memory_ratio = assigned_memory / ram_request

            cpu_memory_product = (
                float(pod.cpu_request) * assigned_memory
            )

            # This feature is deliberately independent of the target.
            independent_log_feature = np.log1p(
                max(float(pod.cpu_request), 0.0)
                + max(float(pod.ram_request_gb), 0.0)
                + max(float(pod.assigned_memory_gb), 0.0)
                + max(float(pod.cpi), 0.0)
            )

            X.append(
                [
                    float(pod.cpu_request),
                    float(pod.ram_request_gb),
                    float(pod.assigned_memory_gb),
                    float(memory_ratio),
                    float(pod.cpi),
                    float(pod.priority),
                    float(cpu_memory_product),
                    float(independent_log_feature),
                ]
            )

            y.append(float(pod.duration_sec))

        return (
            np.asarray(X, dtype=float),
            np.asarray(y, dtype=float),
        )

    @staticmethod
    def _smape(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:

        denominator = (
            np.abs(y_true) + np.abs(y_pred)
        )

        valid = denominator > 0

        if not np.any(valid):
            return 0.0

        value = (
            2.0
            * np.abs(
                y_true[valid] - y_pred[valid]
            )
            / denominator[valid]
        )

        return float(np.mean(value) * 100.0)

    def train_time_split(
        self,
        pods: List[WorkloadPod],
        train_ratio: float = 0.8,
    ) -> Tuple[float, float]:

        if len(pods) < 10:
            raise ValueError(
                "At least 10 workload records are required "
                "for a meaningful train/test split."
            )

        sorted_pods = sorted(
            pods,
            key=lambda p: p.start_time_micro,
        )

        X, y = self._extract_features(sorted_pods)

        split_idx = int(
            len(X) * train_ratio
        )

        if split_idx <= 0 or split_idx >= len(X):
            raise ValueError(
                "train_ratio must leave samples in both "
                "training and testing sets."
            )

        X_train = X[:split_idx]
        X_test = X[split_idx:]

        y_train = y[:split_idx]
        y_test = y[split_idx:]

        self.model.fit(
            X_train,
            y_train,
        )

        self.is_trained = True

        y_pred = self.model.predict(X_test)

        mae = float(
            mean_absolute_error(
                y_test,
                y_pred,
            )
        )

        mse = float(
            mean_squared_error(
                y_test,
                y_pred,
            )
        )

        rmse = float(
            np.sqrt(mse)
        )

        r2 = float(
            r2_score(
                y_test,
                y_pred,
            )
        )

        smape = self._smape(
            y_test,
            y_pred,
        )

        # Mean-duration baseline.
        baseline_value = float(
            np.mean(y_train)
        )

        baseline_pred = np.full(
            shape=len(y_test),
            fill_value=baseline_value,
            dtype=float,
        )

        baseline_mae = float(
            mean_absolute_error(
                y_test,
                baseline_pred,
            )
        )

        baseline_rmse = float(
            np.sqrt(
                mean_squared_error(
                    y_test,
                    baseline_pred,
                )
            )
        )

        baseline_r2 = float(
            r2_score(
                y_test,
                baseline_pred,
            )
        )

        if baseline_rmse > 0:
            rmse_improvement_pct = (
                1.0
                - rmse / baseline_rmse
            ) * 100.0
        else:
            rmse_improvement_pct = 0.0

        residuals = (
            y_test - y_pred
        )

        self.residual_std = max(
            1.0,
            float(np.std(residuals)),
        )

        self.metrics = {
            "train_samples": float(len(X_train)),
            "test_samples": float(len(X_test)),
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "smape_percent": smape,
            "baseline_mae": baseline_mae,
            "baseline_rmse": baseline_rmse,
            "baseline_r2": baseline_r2,
            "rmse_improvement_percent": rmse_improvement_pct,
            "residual_std": self.residual_std,
        }

        return round(rmse, 3), round(r2, 3)

    def get_metrics(self) -> Dict[str, float]:
        """Return all predictor evaluation metrics."""

        return dict(self.metrics)

    def get_feature_importance(self) -> Dict[str, float]:
        """Return GBR feature importance values."""

        if not self.is_trained:
            return {}

        return {
            name: float(value)
            for name, value in zip(
                self.feature_names,
                self.model.feature_importances_,
            )
        }

    def predict(
        self,
        pod: WorkloadPod,
    ) -> Tuple[float, float]:

        if not self.is_trained:
            return (
                round(
                    max(2.0, pod.duration_sec),
                    2,
                ),
                self.residual_std,
            )

        X_single, _ = self._extract_features(
            [pod]
        )

        predicted_mean = float(
            self.model.predict(
                X_single
            )[0]
        )

        predicted_mean = max(
            2.0,
            predicted_mean,
        )

        return (
            round(
                predicted_mean,
                2,
            ),
            round(
                self.residual_std,
                4,
            ),
        )