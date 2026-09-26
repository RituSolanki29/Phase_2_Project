"""
GradientBoostingRegressor Runtime Predictor (Layer 4)
Predicts workload execution runtime on target nodes based on trace features,
using a strict time-based train/test split to prevent leakage.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, r2_score

from src.layer1_infra.trace_loader import WorkloadPod

class GBRRuntimePredictor:
    """Predicts workload duration and computes prediction error variance."""
    
    def __init__(self, random_state: int = 42):
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            random_state=random_state
        )
        self.is_trained = False
        self.residual_std = 5.0  # Default fallback residual standard deviation (seconds)

    def _extract_features(self, pods: List[WorkloadPod], mips_ratios: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Constructs feature matrix X and target vector y from pods."""
        X = []
        y = []
        for pod, mips_ratio in zip(pods, mips_ratios):
            X.append([
                pod.cpu_request,
                pod.ram_request_gb,
                pod.assigned_memory_gb,
                pod.cpi,
                float(pod.priority),
                mips_ratio
            ])
            # Target is actual duration scaled by MIPS ratio
            y.append(pod.duration_sec * mips_ratio)
            
        return np.array(X), np.array(y)

    def train_time_split(self, pods: List[WorkloadPod], train_ratio: float = 0.8) -> Tuple[float, float]:
        """
        Trains GBR model using time-ordered split (first 80% train, remaining 20% test).
        Returns (RMSE, R2 score).
        """
        if len(pods) < 10:
            self.is_trained = True
            return 0.0, 1.0

        # Sort pods chronologically by start_time_micro
        sorted_pods = sorted(pods, key=lambda p: p.start_time_micro)
        mips_ratios = [1.0] * len(sorted_pods)
        
        X, y = self._extract_features(sorted_pods, mips_ratios)
        
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

    def predict(self, pod: WorkloadPod, mips_ratio: float = 1.0) -> Tuple[float, float]:
        """
        Predicts expected duration (mean) and uncertainty standard deviation (std).
        """
        if not self.is_trained:
            # Baseline estimate if untrained
            predicted_mean = pod.duration_sec * mips_ratio
            return round(predicted_mean, 2), self.residual_std

        X_single = np.array([[
            pod.cpu_request,
            pod.ram_request_gb,
            pod.assigned_memory_gb,
            pod.cpi,
            float(pod.priority),
            mips_ratio
        ]])
        
        predicted_mean = float(self.model.predict(X_single)[0])
        predicted_mean = max(2.0, predicted_mean)
        return round(predicted_mean, 2), self.residual_std
