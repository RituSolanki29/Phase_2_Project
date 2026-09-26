"""
Carbon Intensity Data Feed (Layer 2)
Provides dynamic real-time and forecast grid Carbon Intensity (gCO2/kWh)
for simulated cloud regions (DC_EU, DC_US, DC_ASIA).
"""

import numpy as np
from typing import Dict

class CarbonFeed:
    """Simulates multi-region real-time carbon intensity streams (gCO2/kWh)."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)

    def get_carbon_intensity(self, region: str, timestamp_sec: float) -> float:
        """
        Calculates time-dependent carbon intensity (gCO2/kWh) based on regional grid mix models.
        """
        # Diurnal solar/wind cycle (24-hour period = 86400s)
        time_of_day = timestamp_sec % 86400.0
        
        if region == "DC_EU":
            # EU: High solar penetration; low carbon around mid-day (t ~ 43200)
            solar_effect = -140.0 * np.sin(np.pi * time_of_day / 86400.0)
            noise = np.random.normal(0, 10.0)
            ci = 280.0 + solar_effect + noise
            return float(np.clip(ci, 90.0, 450.0))
            
        elif region == "DC_US":
            # US: Gas/Nuclear grid mix; peak demand hours (morning/evening) increase carbon
            demand_effect = 70.0 * np.cos(2.0 * np.pi * (time_of_day - 28800) / 86400.0)
            noise = np.random.normal(0, 15.0)
            ci = 420.0 + demand_effect + noise
            return float(np.clip(ci, 250.0, 580.0))
            
        elif region == "DC_ASIA":
            # ASIA: Fossil/Coal dominant; high baseline intensity
            fluctuation = 40.0 * np.sin(2.0 * np.pi * time_of_day / 86400.0)
            noise = np.random.normal(0, 12.0)
            ci = 540.0 + fluctuation + noise
            return float(np.clip(ci, 380.0, 680.0))
            
        else:
            return 350.0

    def get_region_forecast(self, region: str, start_time: float, horizon_sec: float, step_sec: float = 60.0) -> np.ndarray:
        """Generates forecast trajectory array over a future time horizon."""
        timestamps = np.arange(start_time, start_time + horizon_sec, step_sec)
        return np.array([self.get_carbon_intensity(region, t) for t in timestamps])
