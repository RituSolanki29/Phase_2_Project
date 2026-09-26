"""
SimPy Carbon Trajectory Stochastic Sampler (Layer 4)
Models stochastic discrete-event grid carbon intensity fluctuations
across future workload execution windows.
"""

import simpy
import numpy as np
from typing import List
from src.layer2_telemetry.carbon_feed import CarbonFeed

class SimPyCarbonSampler:
    """Discrete-event simulation engine for future grid carbon trajectory sampling."""
    
    def __init__(self, carbon_feed: CarbonFeed):
        self.carbon_feed = carbon_feed

    def sample_carbon_trajectory(
        self,
        region: str,
        start_time_sec: float,
        duration_sec: float,
        step_sec: float = 10.0,
        random_seed: int = 42
    ) -> List[float]:
        """
        Runs SimPy discrete-event environment to sample carbon intensity
        fluctuations over [start_time, start_time + duration].
        """
        np.random.seed(random_seed)
        env = simpy.Environment(initial_time=start_time_sec)
        samples: List[float] = []

        def carbon_monitor_process(env):
            while env.now < start_time_sec + duration_sec:
                # Get baseline grid carbon intensity
                base_ci = self.carbon_feed.get_carbon_intensity(region, env.now)
                
                # Dynamic renewable cloud cover / wind gust event fluctuation
                event_noise = np.random.normal(0, 8.0)
                sampled_ci = float(np.clip(base_ci + event_noise, 50.0, 700.0))
                samples.append(sampled_ci)
                
                yield env.timeout(step_sec)

        env.process(carbon_monitor_process(env))
        env.run(until=start_time_sec + duration_sec + 1.0)
        
        if not samples:
            samples.append(self.carbon_feed.get_carbon_intensity(region, start_time_sec))

        return samples
