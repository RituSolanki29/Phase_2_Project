"""
Cooldown Manager (Layer 7: "Scaling / Cooldown")
Prevents migration thrashing: once a node has just received or lost a
workload, it is locked out of further migrations for a configurable window.
Persisted to a small JSON file so cooldown state survives process restarts
(important for a "live demo" you might stop and resume).
"""

import json
import time
from pathlib import Path
from threading import Lock
from typing import Dict

DEFAULT_COOLDOWN_SEC = 30.0
DEFAULT_STATE_PATH = Path("results/cooldown_state.json")


class CooldownManager:
    def __init__(self, cooldown_sec: float = DEFAULT_COOLDOWN_SEC, state_path: Path = DEFAULT_STATE_PATH):
        self.cooldown_sec = cooldown_sec
        self.state_path = Path(state_path)
        self._lock = Lock()
        self._until: Dict[str, float] = self._load()

    def _load(self) -> Dict[str, float]:
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text())
            except Exception:
                return {}
        return {}

    def _persist(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self._until, indent=2))

    def is_in_cooldown(self, node_id: str) -> bool:
        with self._lock:
            return time.time() < self._until.get(node_id, 0.0)

    def remaining_sec(self, node_id: str) -> float:
        with self._lock:
            return max(0.0, self._until.get(node_id, 0.0) - time.time())

    def start_cooldown(self, node_id: str, duration_sec: float = None) -> None:
        with self._lock:
            self._until[node_id] = time.time() + (duration_sec or self.cooldown_sec)
            self._persist()

    def clear(self, node_id: str = None) -> None:
        with self._lock:
            if node_id is None:
                self._until = {}
            else:
                self._until.pop(node_id, None)
            self._persist()