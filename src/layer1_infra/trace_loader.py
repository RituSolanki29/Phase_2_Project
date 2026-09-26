"""
Borg Trace Ingestion & Timeline Compressor (Layer 1)
Parses Google Borg cluster traces, applies timeline compression,
and converts trace records into Workload Pod structures & K8s Pod specifications.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd
import json

@dataclass
class WorkloadPod:
    pod_id: str
    collection_id: int
    instance_index: int
    user: str
    priority: int
    cpu_request: float         # Scaled core units (e.g., 0.1 to 4.0 cores)
    ram_request_gb: float      # Memory requested in GB
    assigned_memory_gb: float   # Actual memory assigned/used in GB
    cpi: float                 # Cycles per instruction
    duration_sec: float        # Workload duration in seconds
    start_time_micro: int
    end_time_micro: int
    source_node: str
    source_region: str
    sla_deadline_sec: float    # Computed SLA deadline (e.g., duration * 1.5)

    def to_k8s_manifest(self) -> Dict[str, Any]:
        """Converts pod dataclass into a Kubernetes Pod spec manifest dictionary."""
        return {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": self.pod_id,
                "labels": {
                    "app.kubernetes.io/managed-by": "sustainability-orchestrator",
                    "priority-class": str(self.priority),
                    "source-region": self.source_region
                }
            },
            "spec": {
                "containers": [{
                    "name": "workload-container",
                    "image": "synthetic-workload-runner:latest",
                    "resources": {
                        "requests": {
                            "cpu": f"{self.cpu_request:.2f}",
                            "memory": f"{int(self.ram_request_gb * 1024)}Mi"
                        }
                    }
                }]
            }
        }


class TraceLoader:
    """Loads Borg trace dataset and extracts sanitized pod workloads."""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def load_trace_batch(
        self,
        limit: int = 500,
        compression_factor: float = 100.0,
        node_pool_regions: Optional[List[str]] = None
    ) -> List[WorkloadPod]:
        """
        Reads CSV trace file, cleans records, applies time compression,
        and constructs WorkloadPod objects.
        """
        if node_pool_regions is None:
            node_pool_regions = ["DC_EU", "DC_US", "DC_ASIA"]

        df = pd.read_csv(self.csv_path, nrows=limit)
        pods: List[WorkloadPod] = []
        
        for idx, row in df.iterrows():
            # Parse resource request safely
            cpu_req = 1.0
            ram_req = 2.0
            if pd.notna(row.get("resource_request")):
                try:
                    res_str = str(row["resource_request"]).replace("'", '"')
                    res_dict = json.loads(res_str)
                    cpu_req = float(res_dict.get("cpus", 1.0)) * 4.0  # Scale to core count
                    ram_req = float(res_dict.get("memory", 0.1)) * 16.0  # Scale to GB
                except Exception:
                    pass
            
            assigned_mem = float(row.get("assigned_memory", 0.1)) * 16.0
            if assigned_mem <= 0:
                assigned_mem = ram_req

            cpi = float(row.get("cycles_per_instruction", 1.2))
            if pd.isna(cpi) or cpi <= 0:
                cpi = 1.2

            start_t = int(row.get("start_time", 0))
            end_t = int(row.get("end_time", start_t + 60000000))
            
            # Compute raw duration in seconds from microseconds
            raw_duration = (end_t - start_t) / 1e6
            if raw_duration <= 0:
                raw_duration = 60.0  # Default 60 seconds
            
            # Apply timeline compression
            compressed_duration = max(5.0, raw_duration / compression_factor)
            
            # Assign source region and node deterministically based on machine_id / row index
            region_idx = idx % len(node_pool_regions)
            source_region = node_pool_regions[region_idx]
            machine_num = (int(row.get("machine_id", idx)) % 3) + 1
            region_prefix = source_region.split("_")[1].lower()
            source_node = f"{region_prefix}-node-{machine_num}"
            
            pod_id = f"pod-{row.get('collection_id', idx)}-{row.get('instance_index', 0)}"
            sla_deadline = compressed_duration * 1.5

            pod = WorkloadPod(
                pod_id=pod_id,
                collection_id=int(row.get("collection_id", idx)),
                instance_index=int(row.get("instance_index", 0)),
                user=str(row.get("user", "user_default")),
                priority=int(row.get("priority", 0)),
                cpu_request=round(cpu_req, 2),
                ram_request_gb=round(ram_req, 2),
                assigned_memory_gb=round(assigned_mem, 2),
                cpi=round(cpi, 2),
                duration_sec=round(compressed_duration, 2),
                start_time_micro=start_t,
                end_time_micro=end_t,
                source_node=source_node,
                source_region=source_region,
                sla_deadline_sec=round(sla_deadline, 2)
            )
            pods.append(pod)

        return pods
