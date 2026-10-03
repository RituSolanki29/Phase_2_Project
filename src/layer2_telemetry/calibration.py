"""
Telemetry Calibration Tool (Layer 2)
Fits a linear model (scale factor + offset) mapping modeled "predicted stay baseline"
estimates to actual real Kepler energy measurements collected from the cluster.
Fixes the systematic ~1.5–2× baseline calibration offset between model estimates
and physical cluster measurements.
"""

import argparse
import json
from pathlib import Path
import numpy as np

CALIBRATION_FILE = Path("results/calibration_params.json")


def load_calibration_params():
    """Loads fitted scale and offset parameters. Defaults to scale=1.0, offset=0.0."""
    if CALIBRATION_FILE.exists():
        try:
            data = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
            return float(data.get("scale", 1.0)), float(data.get("offset", 0.0))
        except Exception:
            pass
    return 1.0, 0.0


def save_calibration_params(scale: float, offset: float):
    CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_FILE.write_text(json.dumps({"scale": float(scale), "offset": float(offset)}, indent=2), encoding="utf-8")
    print(f"[Calibration] Saved parameters: scale={scale:.4f}, offset={offset:.4f} to {CALIBRATION_FILE}")


def calibrate_from_reward_log(log_path: str = "results/reward_log.jsonl"):
    """
    Fits scale and offset by comparing predicted_stay_baseline vs real outcome
    from recorded reward log entries.
    """
    path = Path(log_path)
    if not path.exists():
        print(f"[Calibration] No reward log found at {path}. Run live demo first to collect samples.")
        return 1.0, 0.0

    predicted_vals = []
    real_vals = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            if entry.get("event") == "outcome":
                detail = entry.get("detail", {})
                if not isinstance(detail, dict):
                    continue  # legacy free-text details cannot be parsed reliably
                p_stay = detail.get("predicted_stay_baseline")
                r_energy = detail.get("real_execution_carbon", detail.get("real_carbon"))
                if p_stay is not None and r_energy is not None:
                    predicted_vals.append(float(p_stay))
                    real_vals.append(float(r_energy))
        except Exception:
            continue

    if len(predicted_vals) < 3:
        print(f"[Calibration] Insufficient samples ({len(predicted_vals)}) in reward log to fit model.")
        return load_calibration_params()

    p_arr = np.array(predicted_vals)
    r_arr = np.array(real_vals)

    # Fit linear regression: real = scale * predicted + offset
    if np.std(p_arr) > 1e-9:
        scale, offset = np.polyfit(p_arr, r_arr, 1)
    else:
        scale = float(np.mean(r_arr) / max(1e-9, np.mean(p_arr)))
        offset = 0.0

    print(f"[Calibration] Fitted from {len(predicted_vals)} real samples:")
    print(f"  Scale factor : {scale:.4f}")
    print(f"  Offset       : {offset:.6f}")
    save_calibration_params(scale, offset)
    return float(scale), float(offset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate modeled energy against real Kepler measurements.")
    parser.add_argument("--log-path", default="results/reward_log.jsonl")
    args = parser.parse_args()
    calibrate_from_reward_log(args.log_path)
