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
    config.load_kube_config()
    v1 = client.CoreV1Api()

    try:
        v1.read_namespace(name=namespace)
    except ApiException as e:
        if e.status == 404:
            v1.create_namespace(
                client.V1Namespace(
                    metadata=client.V1ObjectMeta(name=namespace)
                )
            )
        else:
            raise


def _k8s_pod_name(pod: WorkloadPod) -> str:
    # K8s pod names must be lowercase RFC-1123; the trace-derived pod_id
    # already is, but sanitize defensively.
    return pod.pod_id.lower().replace("_", "-")[:63]


def build_pod_spec(pod: WorkloadPod, node: NodeConfig, namespace: str = DEFAULT_NAMESPACE,
                    scheduler_name: str = None) -> "client.V1Pod":
    """
    Builds a REAL Pod spec that actually consumes CPU/RAM proportional to
    the trace-derived request, for approximately `pod.duration_sec` seconds.

    Two submission modes:
      - scheduler_name=None (default): scheduled immediately by the default
        Kubernetes scheduler via nodeSelector on `node.region` -- used for
        direct telemetry-measurement deployments (pipeline_runner.py).
      - scheduler_name="sustainability-scheduler": submitted UNSCHEDULED
        (no nodeSelector, no nodeName) with that schedulerName, so the
        default scheduler ignores it and our custom_scheduler.py picks it
        up and binds it explicitly after running it through Layers 2-6.
        Workload parameters are encoded as annotations so the scheduler can
        reconstruct a WorkloadPod purely from the live cluster object.
    """
    cpu_cores = max(1, math.ceil(pod.cpu_request))
    timeout_sec = max(1, int(pod.duration_sec))

    memory_mb = 64

    container = client.V1Container(
        name="workload",
        image="busybox:latest",
        command=["sh", "-c", f"sleep {timeout_sec}"],
        resources=client.V1ResourceRequirements(
            requests={
                "cpu": f"{pod.cpu_request:.2f}",
                "memory": "256Mi"
            },
            limits={
                "cpu": f"{pod.cpu_request:.2f}",
                "memory": "512Mi"
            },
        ),
    )

    spec_kwargs = dict(containers=[container], restart_policy="Never")
    if scheduler_name:
        spec_kwargs["scheduler_name"] = scheduler_name
        # Deliberately NO node_selector: this pod is meant to stay Pending
        # until custom_scheduler.py evaluates it and binds it explicitly.
    else:
        spec_kwargs["node_selector"] = {"topology.kubernetes.io/region": node.region}
    spec = client.V1PodSpec(**spec_kwargs)

    annotations = {
        "sustainability.orchestrator/cpu-request": str(pod.cpu_request),
        "sustainability.orchestrator/ram-request-gb": str(pod.ram_request_gb),
        "sustainability.orchestrator/assigned-memory-gb": str(pod.assigned_memory_gb),
        "sustainability.orchestrator/cpi": str(pod.cpi),
        "sustainability.orchestrator/duration-sec": str(pod.duration_sec),
        "sustainability.orchestrator/sla-deadline-sec": str(pod.sla_deadline_sec),
        "sustainability.orchestrator/priority": str(pod.priority),
        "sustainability.orchestrator/source-node": pod.source_node,
        "sustainability.orchestrator/source-region": pod.source_region,
    } if scheduler_name else {}

    metadata = client.V1ObjectMeta(
        name=_k8s_pod_name(pod),
        namespace=namespace,
        labels={
            "app.kubernetes.io/managed-by": "sustainability-orchestrator",
            "pod-id": pod.pod_id,
            "region": node.region,
        },
        annotations=annotations,
    )

    return client.V1Pod(metadata=metadata, spec=spec)


def deploy_pod(pod: WorkloadPod, node: NodeConfig, namespace: str = DEFAULT_NAMESPACE,
               scheduler_name: str = None) -> str:
    """Creates the real Pod on the cluster. Returns the K8s pod name.
    Pass scheduler_name to submit it unscheduled for custom_scheduler.py to
    pick up and bind (see build_pod_spec)."""
    ensure_namespace(namespace)
    v1 = client.CoreV1Api()
    pod_spec = build_pod_spec(pod, node, namespace, scheduler_name=scheduler_name)
    k8s_name = _k8s_pod_name(pod)
    try:
        v1.create_namespaced_pod(namespace=namespace, body=pod_spec)
    except ApiException as e:
        if e.status == 409:  # already exists from previous run -- recreate cleanly
            try:
                v1.delete_namespaced_pod(name=k8s_name, namespace=namespace, grace_period_seconds=0)
                time.sleep(0.5)
            except Exception:
                pass
            v1.create_namespaced_pod(namespace=namespace, body=pod_spec)
        else:
            raise
    return k8s_name


def workload_pod_from_annotations(k8s_pod) -> "WorkloadPod":
    """Reconstructs a WorkloadPod from a live Kubernetes pod object's
    annotations (see build_pod_spec) -- used by custom_scheduler.py so it
    can evaluate a real pending pod without needing the original Python
    WorkloadPod instance that submitted it."""
    a = k8s_pod.metadata.annotations or {}
    return WorkloadPod(
        pod_id=k8s_pod.metadata.labels.get("pod-id", k8s_pod.metadata.name),
        collection_id=0, instance_index=0, user="demo",
        priority=int(a.get("sustainability.orchestrator/priority", 0)),
        cpu_request=float(a.get("sustainability.orchestrator/cpu-request", 1.0)),
        ram_request_gb=float(a.get("sustainability.orchestrator/ram-request-gb", 2.0)),
        assigned_memory_gb=float(a.get("sustainability.orchestrator/assigned-memory-gb", 2.0)),
        cpi=float(a.get("sustainability.orchestrator/cpi", 1.2)),
        duration_sec=float(a.get("sustainability.orchestrator/duration-sec", 30.0)),
        start_time_micro=0, end_time_micro=0,
        source_node=a.get("sustainability.orchestrator/source-node", "eu-node-1"),
        source_region=a.get("sustainability.orchestrator/source-region", "DC_EU"),
        sla_deadline_sec=float(a.get("sustainability.orchestrator/sla-deadline-sec", 45.0)),
    )


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


def bind_pod_to_node(
    pod_name: str,
    node_name: str,
    namespace: str = DEFAULT_NAMESPACE,
    verify_timeout_sec: float = 5.0,
):
    """
    Bind a Pending pod to a Kubernetes node.

    ``node_name`` may be either the real Kubernetes node name or the
    project's logical ``node-id`` label.

    Important compatibility detail:
    recent versions of the Kubernetes Python client can successfully submit
    a Binding but then fail while deserializing the HTTP response into
    ``V1Binding``. That produces the misleading:

        ValueError: Invalid value for `target`, must not be `None`

    The Binding API does not need the response body, so we deliberately use
    ``_preload_content=False`` and then verify the pod's ``spec.node_name``.
    We also use the pod-specific Binding endpoint when available.
    """
    if not _K8S_CLIENT_AVAILABLE:
        raise RuntimeError(
            "The Kubernetes Python client is not installed. "
            "Install the dependencies from requirements.txt."
        )

    v1 = client.CoreV1Api()

    # Resolve the project's logical node-id to the real Kubernetes node name.
    actual_node_name = None
    for node in v1.list_node().items:
        k8s_name = node.metadata.name
        if k8s_name == node_name:
            actual_node_name = k8s_name
            break

        labels = node.metadata.labels or {}
        if labels.get("node-id") == node_name:
            actual_node_name = k8s_name
            break

    if actual_node_name is None:
        raise ValueError(
            f"Could not find Kubernetes node for logical node '{node_name}'"
        )

    # Fail early with a useful error if the scheduler is trying to bind a
    # pod that disappeared or is already assigned.
    pod = v1.read_namespaced_pod(name=pod_name, namespace=namespace)
    existing_node = getattr(pod.spec, "node_name", None)
    if existing_node:
        if existing_node == actual_node_name:
            # Idempotent success: another scheduler cycle may already have
            # completed this bind.
            return
        raise RuntimeError(
            f"Pod '{pod_name}' is already assigned to Kubernetes node "
            f"'{existing_node}', cannot re-bind it to '{actual_node_name}'."
        )

    target = client.V1ObjectReference(
        api_version="v1",
        kind="Node",
        name=actual_node_name,
    )

    metadata = client.V1ObjectMeta(name=pod_name)

    # Explicitly supply target. It is required by the Kubernetes Binding API.
    body = client.V1Binding(
        api_version="v1",
        kind="Binding",
        metadata=metadata,
        target=target,
    )

    # The pod-specific endpoint is the canonical endpoint for binding a pod.
    # _preload_content=False is intentional: it prevents the Python client
    # from trying to deserialize the Binding response and triggering the
    # misleading V1Binding.target == None validation error.
    try:
        response = v1.create_namespaced_pod_binding(
            name=pod_name,
            namespace=namespace,
            body=body,
            _preload_content=False,
        )
    except AttributeError:
        # Compatibility fallback for older clients that do not expose the
        # pod-specific convenience method.
        response = v1.create_namespaced_binding(
            namespace=namespace,
            body=body,
            _preload_content=False,
        )

    status_code = getattr(response, "status", None)
    if status_code is not None and status_code not in (200, 201, 202):
        raise RuntimeError(
            f"Kubernetes Binding API returned HTTP {status_code} "
            f"for pod '{pod_name}'."
        )

    # Binding is asynchronous from the client's perspective. Verify that
    # Kubernetes actually recorded the node assignment before reporting
    # success to the scheduler.
    deadline = time.time() + max(0.0, verify_timeout_sec)
    last_seen_node = None
    while time.time() < deadline:
        try:
            current = v1.read_namespaced_pod(
                name=pod_name,
                namespace=namespace,
            )
            last_seen_node = getattr(current.spec, "node_name", None)
            if last_seen_node == actual_node_name:
                print(
                    f"Bound pod '{pod_name}' to logical node '{node_name}' "
                    f"(Kubernetes node: '{actual_node_name}')"
                )
                return
            if last_seen_node and last_seen_node != actual_node_name:
                raise RuntimeError(
                    f"Pod '{pod_name}' was assigned to unexpected Kubernetes "
                    f"node '{last_seen_node}' instead of '{actual_node_name}'."
                )
        except ApiException as exc:
            # A transient 404 immediately after the binding request is not
            # treated as success; keep polling until the verification timeout.
            if exc.status not in (404, 409):
                raise
        time.sleep(0.2)

    raise TimeoutError(
        f"Binding request for pod '{pod_name}' was submitted, but Kubernetes "
        f"did not report spec.nodeName='{actual_node_name}' within "
        f"{verify_timeout_sec:.1f}s (last seen: {last_seen_node!r})."
    )


def list_pending_unscheduled_pods(scheduler_name: str, namespace: str = DEFAULT_NAMESPACE):
    """Returns pods with spec.schedulerName == scheduler_name that are still
    Pending and unbound (no nodeName yet) -- the real queue a custom
    scheduler watches and binds from."""
    v1 = client.CoreV1Api()
    pods = v1.list_namespaced_pod(namespace=namespace).items
    return [
        p for p in pods
        if p.spec.scheduler_name == scheduler_name and p.status.phase == "Pending" and not p.spec.node_name
    ]

def get_pod_runtime_sec(k8s_pod_name: str, namespace: str = DEFAULT_NAMESPACE) -> Optional[float]:
    """Observed wall-clock runtime of a finished pod (container start -> finish),
    or None if the pod hasn't finished / the API is unavailable."""
    try:
        v1 = client.CoreV1Api()
        pod = v1.read_namespaced_pod_status(name=k8s_pod_name, namespace=namespace)
        for cs in (pod.status.container_statuses or []):
            term = cs.state.terminated if cs.state else None
            if term and term.started_at and term.finished_at:
                return float((term.finished_at - term.started_at).total_seconds())
    except Exception:
        pass
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