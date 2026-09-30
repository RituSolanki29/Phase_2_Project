"""
Carbon Intensity Data Feed (Layer 2)

Provides deterministic, reproducible simulated carbon-intensity trajectories
for the synthetic regions used by the project.

Important:
- This is a MODEL, not measured grid data.
- Randomness is local to each call; the global NumPy RNG is never reset.
- A caller can provide an independent Generator for reproducible Monte Carlo
  experiments without coupling different candidates to the same random stream.
"""

from typing import Optional
import numpy as np


class CarbonFeed:
    """Simulates multi-region carbon-intensity streams in gCO2/kWh."""

    def __init__(self, seed: int = 42):
        self.seed = int(seed)

    def _rng(self, rng: Optional[np.random.Generator] = None):
        """
        Return the caller's RNG when supplied.

        If no RNG is supplied, create a deterministic local generator.
        This avoids changing NumPy's global random state.
        """
        if rng is not None:
            return rng

        return np.random.default_rng(self.seed)

    def get_carbon_intensity(
        self,
        region: str,
        timestamp_sec: float,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """
        Calculate simulated carbon intensity for a region and timestamp.

        The regional curves are the same conceptual curves used by the
        original implementation. Only the random-number handling has been
        changed so that separate simulations do not accidentally reuse or
        reset one global random stream.
        """

        local_rng = self._rng(rng)

        time_of_day = timestamp_sec % 86400.0

        if region == "DC_EU":
            # Lower intensity around the middle of the simulated day.
            solar_effect = -140.0 * np.sin(
                np.pi * time_of_day / 86400.0
            )
            noise = local_rng.normal(0.0, 10.0)

            ci = 280.0 + solar_effect + noise

            return float(np.clip(ci, 90.0, 450.0))

        elif region == "DC_US":
            # Simulated demand-driven variation.
            demand_effect = 70.0 * np.cos(
                2.0 * np.pi * (time_of_day - 28800.0) / 86400.0
            )
            noise = local_rng.normal(0.0, 15.0)

            ci = 420.0 + demand_effect + noise

            return float(np.clip(ci, 250.0, 580.0))

        elif region == "DC_ASIA":
            # Higher simulated baseline intensity.
            fluctuation = 40.0 * np.sin(
                2.0 * np.pi * time_of_day / 86400.0
            )
            noise = local_rng.normal(0.0, 12.0)

            ci = 540.0 + fluctuation + noise

            return float(np.clip(ci, 380.0, 680.0))

        else:
            return 350.0

    def get_region_forecast(
        self,
        region: str,
        start_time: float,
        horizon_sec: float,
        step_sec: float = 60.0,
        random_seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate a deterministic simulated carbon trajectory.

        A separate RNG stream is created for every forecast request when
        random_seed is provided.
        """

        if random_seed is None:
            random_seed = self.seed

        rng = np.random.default_rng(int(random_seed))

        timestamps = np.arange(
            start_time,
            start_time + max(horizon_sec, step_sec),
            step_sec,
        )

        values = [
            self.get_carbon_intensity(
                region=region,
                timestamp_sec=float(t),
                rng=rng,
            )
            for t in timestamps
        ]

        return np.asarray(values, dtype=float)