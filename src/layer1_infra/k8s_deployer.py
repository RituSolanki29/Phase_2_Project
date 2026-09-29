"""
Real Kubernetes Deployment Client (Layer 1)
Actually creates, monitors, and tears down real Pods on a real Kubernetes
cluster (e.g. a local Kind cluster) using the official `kubernetes` Python
client -- replacing pure manifest-dict generation with a real API call.

This is intentionally isolated from the rest of the pipeline: if no cluster
is reachable, `is_cluster_reachable()` returns False quickly (short timeout)
and every caller (see telemetry_source.py, pipeline_runner.py) falls back
to the existing simulated path. Nothing else in Layers 3-5 depends on this
module -- they only ever use the deterministic power MODEL (see module note
in telemetry_source.py for why that's correct, not a compromise).
"""

import math
import time
from typing import Optional, Tuple

try:
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException
    _K8S_CLIENT_AVAILABLE = True
except ImportError:
    _K8S_CLIENT_AVAILABLE = False

from src.layer1_infra.trace_loader import WorkloadPod
from src.layer1_infra.cluster_config import NodeConfig

DEFAULT_NAMESPACE = "sustainability-sim"
STRESS_IMAGE = "polinux/stress"  # small, widely-available CPU/mem stress image


def is_cluster_reachable(kubeconfig_path: Optional[str] = None, timeout_sec: float = 3.0) -> bool:
    """
    Fast health check: can we actually talk to a live Kubernetes API server?
    Returns False immediately (no exception) if the `kubernetes` package
    isn't installed, no kubeconfig is found, or the API server doesn't
    respond within `timeout_sec`. This is the single gate every other real
    integration in this project depends on.
    """
    if not _K8S_CLIENT_AVAILABLE:
        return False
    try:
        if kubeconfig_path:
            config.load_kube_config(config_file=kubeconfig_path)
        else:
            config.load_kube_config()
        v1 = client.CoreV1Api()
        v1.api_client.configuration.timeout = timeout_sec
        v1.list_namespace(_request_timeout=timeout_sec)
        return True
    except Exception:
        return False


def ensure_namespace(namespace: str = DEFAULT_NAMESPACE) -> None:
    """Creates the working namespace if it doesn't already exist."""
    v1 = client.CoreV1Api()
    try:
        v1.read_namespace(name=namespace)
    except ApiException as e:
        if e.status == 404:
            v1.create_namespace(client.V1Namespace(metadata=client.V1ObjectMeta(name=namespace)))
        else:
            raise


def _k8s_pod_name(pod: WorkloadPod) -> str:
    # K8s pod names must be lowercase RFC-1123; the trace-derived pod_id
    # already is, but sanitize defensively.
    return pod.pod_id.lower().replace("_", "-")[:63]


def build_pod_spec(pod: WorkloadPod, node: NodeConfig, namespace: str = DEFAULT_NAMESPACE) -> "client.V1Pod":
    """
    Builds a REAL Pod spec that actually consumes CPU/RAM proportional to
    the trace-derived request, for approximately `pod.duration_sec` seconds,
    scheduled onto a node in the target synthetic region via nodeSelector
    (matching the `topology.kubernetes.io/region` label applied by
    generate_kind_config()'s kubeadm node-labels patch).
    """
    cpu_cores = max(1, math.ceil(pod.cpu_request))
    timeout_sec = max(1, int(pod.duration_sec))

    container = client.V1Container(
        name="workload",
        image=STRESS_IMAGE,
        args=["--cpu", str(cpu_cores), "--vm", "1", "--vm-bytes",
              f"{max(16, int(pod.ram_request_gb * 1024))}M", "--timeout", f"{timeout_sec}s"],
        resources=client.V1ResourceRequirements(
            requests={"cpu": f"{pod.cpu_request:.2f}", "memory": f"{int(pod.ram_request_gb * 1024)}Mi"},
            limits={"cpu": f"{pod.cpu_request:.2f}", "memory": f"{int(pod.ram_request_gb * 1024)}Mi"},
        ),
    )

    spec = client.V1PodSpec(
        containers=[container],
        node_selector={"topology.kubernetes.io/region": node.region},
        restart_policy="Never",
    )

    metadata = client.V1ObjectMeta(
        name=_k8s_pod_name(pod),
        namespace=namespace,
        labels={
            "app.kubernetes.io/managed-by": "sustainability-orchestrator",
            "pod-id": pod.pod_id,
            "region": node.region,
        },
    )

    return client.V1Pod(metadata=metadata, spec=spec)


def deploy_pod(pod: WorkloadPod, node: NodeConfig, namespace: str = DEFAULT_NAMESPACE) -> str:
    """Creates the real Pod on the cluster. Returns the K8s pod name."""
    ensure_namespace(namespace)
    v1 = client.CoreV1Api()
    pod_spec = build_pod_spec(pod, node, namespace)
    k8s_name = _k8s_pod_name(pod)
    try:
        v1.create_namespaced_pod(namespace=namespace, body=pod_spec)
    except ApiException as e:
        if e.status == 409:  # already exists (e.g. re-run) -- reuse it
            pass
        else:
            raise
    return k8s_name


def wait_for_pod_running(k8s_pod_name: str, namespace: str = DEFAULT_NAMESPACE, timeout_sec: float = 30.0) -> bool:
    """Polls until the pod reaches Running (or Succeeded), or times out."""
    v1 = client.CoreV1Api()
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            pod = v1.read_namespaced_pod_status(name=k8s_pod_name, namespace=namespace)
            phase = pod.status.phase
            if phase in ("Running", "Succeeded"):
                return True
            if phase == "Failed":
                return False
        except ApiException:
            pass
        time.sleep(1.0)
    return False


def get_pod_resource_usage(k8s_pod_name: str, namespace: str = DEFAULT_NAMESPACE) -> Optional[Tuple[float, float]]:
    """
    Reads REAL live CPU/memory usage from the Kubernetes Metrics API
    (requires metrics-server to be installed in the cluster). Returns
    (cpu_cores_used, memory_gb_used), or None if metrics-server isn't
    available -- callers should fall back to the request-based estimate.
    """
    try:
        custom_api = client.CustomObjectsApi()
        metrics = custom_api.get_namespaced_custom_object(
            group="metrics.k8s.io", version="v1beta1",
            namespace=namespace, plural="pods", name=k8s_pod_name,
        )
        containers = metrics.get("containers", [])
        if not containers:
            return None
        usage = containers[0]["usage"]
        cpu_str = usage.get("cpu", "0n")
        mem_str = usage.get("memory", "0Ki")
        cpu_cores = _parse_cpu_quantity(cpu_str)
        mem_gb = _parse_memory_quantity(mem_str)
        return cpu_cores, mem_gb
    except Exception:
        return None


def delete_pod(k8s_pod_name: str, namespace: str = DEFAULT_NAMESPACE) -> None:
    """Deletes the real pod (cleanup). Never raises -- best-effort."""
    try:
        v1 = client.CoreV1Api()
        v1.delete_namespaced_pod(name=k8s_pod_name, namespace=namespace, grace_period_seconds=0)
    except Exception:
        pass


def _parse_cpu_quantity(q: str) -> float:
    if q.endswith("n"):
        return float(q[:-1]) / 1e9
    if q.endswith("u"):
        return float(q[:-1]) / 1e6
    if q.endswith("m"):
        return float(q[:-1]) / 1e3
    return float(q)


def _parse_memory_quantity(q: str) -> float:
    units = {"Ki": 1 / (1024 * 1024), "Mi": 1 / 1024, "Gi": 1.0, "K": 1e3 / (1024 ** 3), "M": 1e6 / (1024 ** 3), "G": 1e9 / (1024 ** 3)}
    for suffix, factor in units.items():
        if q.endswith(suffix):
            return float(q[: -len(suffix)]) * factor
    return float(q) / (1024 ** 3)