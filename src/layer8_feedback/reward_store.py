from pathlib import Path
import json
import time
from typing import Any, Dict, List, Optional


DEFAULT_LOG_PATH = Path("results/reward_log.jsonl")


class RewardStore:
    def __init__(self, log_path: Path = DEFAULT_LOG_PATH, processed_path: Optional[Path] = None):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.processed_path = Path(processed_path) if processed_path else self.log_path.with_suffix(".processed.json")

    def _append(self, entry: Dict[str, Any]):
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def record_decision(self, pod_id: str, action, candidate, bound_node_id: str,
                        context: Optional[Dict[str, Any]] = None):
        """`context` carries what Layer 8 needs to re-compute the REAL outcome
        with the same physics as the Digital Twin (pod CPU/memory, decision
        timestamp, ...)."""
        action_value = action.name if hasattr(action, "name") else str(action)
        candidate_data = candidate.to_dict() if hasattr(candidate, "to_dict") else candidate
        self._append({
            "event": "decision",
            "timestamp": time.time(),
            "pod_id": pod_id,
            "action": action_value,
            "candidate": candidate_data,
            "bound_node_id": bound_node_id,
            "context": context or {},
        })

    def record_outcome(self, pod_id: str, realized_nsg_gco2: float, detail: Optional[Dict[str, Any]] = None):
        self._append({
            "event": "outcome",
            "timestamp": time.time(),
            "pod_id": pod_id,
            "realized_nsg_gco2": float(realized_nsg_gco2),
            "detail": detail or {},
        })

    # ------------------------------------------------------------------
    def _read_entries(self) -> List[Dict[str, Any]]:
        if not self.log_path.exists():
            return []
        entries = []
        with self.log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries

    def _load_processed(self) -> set:
        if self.processed_path.exists():
            try:
                return set(json.loads(self.processed_path.read_text(encoding="utf-8")))
            except Exception:
                return set()
        return set()

    def mark_processed(self, keys) -> None:
        done = self._load_processed() | set(keys)
        self.processed_path.write_text(json.dumps(sorted(done)), encoding="utf-8")

    def load_pairs(self, only_unprocessed: bool = False) -> List[Dict[str, Any]]:
        """Pairs every outcome with the decision that CAUSED it (the most recent
        decision for that pod logged BEFORE the outcome).  Each pair has a
        unique `key` so it can be marked as processed."""
        entries = self._read_entries()
        last_decision: Dict[str, Dict[str, Any]] = {}
        processed = self._load_processed() if only_unprocessed else set()
        pairs = []
        for e in entries:
            pod_id = e.get("pod_id")
            if not pod_id:
                continue
            if e.get("event") == "decision":
                last_decision[pod_id] = e
            elif e.get("event") == "outcome" and pod_id in last_decision:
                d = last_decision[pod_id]
                key = f"{pod_id}@{d.get('timestamp')}"
                if key in processed:
                    continue
                pairs.append({
                    "key": key,
                    "pod_id": pod_id,
                    "action": d.get("action"),
                    "candidate": d.get("candidate"),
                    "bound_node_id": d.get("bound_node_id"),
                    "context": d.get("context", {}),
                    "realized_nsg_gco2": e.get("realized_nsg_gco2", 0.0),
                    "detail": e.get("detail", {}),
                })
        return pairs

    def load_unprocessed_outcomes(self) -> List[Dict[str, Any]]:
        """Kept for backward compatibility: only pairs NOT yet used for retraining."""
        return self.load_pairs(only_unprocessed=True)