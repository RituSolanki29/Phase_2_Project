"""
Real-Time Grid Carbon Intensity Client (Layer 2)
Fetches the ACTUAL current carbon intensity (gCO2/kWh) for a real grid zone
from the Electricity Maps API, in place of the synthetic sinusoidal model in
carbon_feed.py.

IMPORTANT LIMITATION (by nature of the data source, not a bug):
This API only reports the LIVE "now" value for a zone -- there is no
historical/arbitrary-timestamp lookup on the free tier, and no future value
either. That means:
  - For the CURRENT scheduling decision, a real live reading is genuinely
    "real-time" and is used as-is.
  - For the Digital Twin's Monte Carlo forecast window (how carbon intensity
    might drift over the pod's predicted execution time), there is no way to
    fetch a real "future" reading -- that remains necessarily a MODEL
    (SimPyCarbonSampler), which we optionally re-center on this real current
    reading instead of a synthetic baseline (see telemetry_source.py).
Get a free API key at https://www.electricitymaps.com/free-tier-api and set
it as the ELECTRICITYMAP_API_KEY environment variable.
"""

from typing import Dict, Optional
import requests

DEFAULT_ZONE_MAP: Dict[str, str] = {
    # Maps this project's synthetic regions to real Electricity Maps zone
    # codes. Verify/adjust these to zones your API key has access to.
    "DC_EU": "IE",              # Ireland
    "DC_US": "US-MIDA-PJM",     # US PJM Interconnection (Virginia area)
    "DC_ASIA": "SG",            # Singapore
}

_API_URL = "https://api.electricitymap.org/v3/carbon-intensity/latest"


class RealCarbonFeed:
    """Live grid carbon-intensity client backed by a real external API."""

    def __init__(self, api_key: Optional[str], zone_map: Optional[Dict[str, str]] = None, timeout_sec: float = 3.0):
        self.api_key = api_key
        self.zone_map = zone_map or DEFAULT_ZONE_MAP
        self.timeout_sec = timeout_sec

    def is_available(self) -> bool:
        return bool(self.api_key)

    def get_carbon_intensity(self, region: str) -> Optional[float]:
        """
        Returns the REAL, current carbon intensity (gCO2/kWh) for the real
        zone mapped to `region`, or None on any failure (missing key, zone
        not mapped, network error, rate limit) -- callers must fall back to
        the synthetic CarbonFeed model.
        """
        if not self.api_key:
            return None
        zone = self.zone_map.get(region)
        if not zone:
            return None
        try:
            resp = requests.get(
                _API_URL,
                params={"zone": zone},
                headers={"auth-token": self.api_key},
                timeout=self.timeout_sec,
            )
            resp.raise_for_status()
            data = resp.json()
            value = data.get("carbonIntensity")
            return float(value) if value is not None else None
        except Exception:
            return None