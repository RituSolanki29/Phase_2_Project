"""
Demo Pod Submission Helper (Layer 7)
Submits a real, unscheduled Kubernetes pod carrying this project's workload
annotations, with schedulerName="sustainability-scheduler", so that
custom_scheduler.py's watch loop will pick it up, evaluate it through
Layers 2-6, and bind it to a real node. This is the "front door" for a live
demo: run this to inject a workload, then watch custom_scheduler.py's log
output make and execute a real decision.

Run:
    python -m src.layer7_execution.pod_submission --from-trace data/borg_traces_data.csv --count 5
"""

import argparse
import time

from src.layer1_infra import k8s_deployer
from src.layer1_infra.cluster_config import create_default_node_pools
from src.layer1_infra.trace_loader import TraceLoader, WorkloadPod
from src.layer7_execution.custom_scheduler import SCHEDULER_NAME


def submit(pod: WorkloadPod) -> str:
    nodes = create_default_node_pools()
    nominal_node = next((n for n in nodes if n.node_id == pod.source_node), nodes[0])
    return k8s_deployer.deploy_pod(pod, nominal_node, scheduler_name=SCHEDULER_NAME)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit demo workload pods for the live custom scheduler.")
    parser.add_argument("--from-trace", default="data/borg_traces_data.csv")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--interval-sec", type=float, default=3.0, help="Delay between submissions, to watch the scheduler react one at a time.")
    parser.add_argument("--balance-regions", action="store_true", help="Balance workload nominal regions evenly across DC_ASIA, DC_EU, and DC_US.")
    args = parser.parse_args()

    pods = TraceLoader(args.from_trace).load_trace_batch(limit=args.count, compression_factor=10.0)
    regions = ["DC_ASIA", "DC_EU", "DC_US"]
    for idx, pod in enumerate(pods):
        if args.balance_regions:
            region = regions[idx % len(regions)]
            pod.source_region = region
            region_prefix = region.split("_")[1].lower()
            pod.source_node = f"{region_prefix}-node-{(idx % 3) + 1}"
        name = submit(pod)
        print(f"Submitted {name} (pod_id={pod.pod_id}, nominal node={pod.source_node}, region={pod.source_region}) -- unscheduled, awaiting custom_scheduler.py")
        time.sleep(args.interval_sec)