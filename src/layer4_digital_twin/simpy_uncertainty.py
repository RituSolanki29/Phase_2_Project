"""
SimPy Carbon Trajectory Stochastic Sampler (Layer 4)

Simulates carbon-intensity fluctuations over the predicted execution window.

The simulator is intentionally MODEL-BASED for future values. It does not
claim to be real future grid telemetry.
"""

import simpy
import numpy as np
from typing import List

from src.layer2_telemetry.carbon_feed import CarbonFeed


class SimPyCarbonSampler:
    """Discrete-event simulation engine for future carbon trajectories."""

    def __init__(
        self,
        carbon_feed: CarbonFeed,
        event_noise_std: float = 8.0,
    ):
        self.carbon_feed = carbon_feed
        self.event_noise_std = float(event_noise_std)

    def sample_carbon_trajectory(
        self,
        region: str,
        start_time_sec: float,
        duration_sec: float,
        step_sec: float = 10.0,
        random_seed: int = 42,
    ) -> List[float]:
        """
        Run a SimPy discrete-event simulation.

        Each call owns its own random generator. This makes separate
        candidates reproducible but statistically independent.
        """

        duration_sec = max(float(duration_sec), step_sec)

        rng = np.random.default_rng(int(random_seed))

        env = simpy.Environment(initial_time=float(start_time_sec))

        samples: List[float] = []

        def carbon_monitor_process(environment):
            while environment.now < start_time_sec + duration_sec:

                base_ci = self.carbon_feed.get_carbon_intensity(
                    region=region,
                    timestamp_sec=float(environment.now),
                    rng=rng,
                )

                event_noise = rng.normal(
                    0.0,
                    self.event_noise_std,
                )

                sampled_ci = float(
                    np.clip(
                        base_ci + event_noise,
                        50.0,
                        700.0,
                    )
                )

                samples.append(sampled_ci)

                yield environment.timeout(step_sec)

        env.process(carbon_monitor_process(env))

        env.run(
            until=start_time_sec + duration_sec + step_sec
        )

        if not samples:
            fallback = self.carbon_feed.get_carbon_intensity(
                region=region,
                timestamp_sec=start_time_sec,
                rng=rng,
            )

            samples.append(float(fallback))

        return samples