"""
Migration Executor (Layer 7: "Checkpoint & Migrate", "Scheduler Extender")
Executes a Layer 6 decision (STAY / MIGRATE / DELAY / START) against a REAL
Kubernetes cluster via the existing k8s_deployer client.

HONEST SCOPE NOTE (read before claiming "live migration" in your report):
"MIGRATE" here means delete-and-recreate on the target node, not CRIU-style
live process migration. True live migration needs checkpoint/restore kernel
support (CRIU) with specific container-runtime configuration that Kind on
Docker Desktop does not reliably provide, especially on Windows. This
project's workloads are short-lived, restartable batch/stress jobs replayed
from a trace (see trace_loader.py), which are a legitimate, common case
where restart-based migration is an accurate and sufficient strategy -- it
would NOT be an accurate strategy for long-running stateful services, and
you should say so explicitly if asked.
"""

from dataclasses import dataclass
from typing import Optional

from src.layer1_infra import k8s_deployer
from src.layer1_infra.trace_loader import WorkloadPod
from src.layer1_infra.cluster_config import NodeConfig
from src.layer7_execution.cooldown_manager import CooldownManager


@dataclass
class ExecutionResult:
    pod_id: str
    action: str
    executed: bool              # False if skipped (e.g. cooldown, no real cluster)
    k8s_pod_name: Optional[str]
    node_id: Optional[str]
    detail: str


class MigrationExecutor:
    def __init__(self, cooldown: CooldownManager, cluster_available: bool):
        self.cooldown = cooldown
        self.cluster_available = cluster_available

    def execute(self, action: str, pod: WorkloadPod, current_node: NodeConfig,
                target_node: Optional[NodeConfig] = None) -> ExecutionResult:
        if not self.cluster_available:
            return ExecutionResult(pod.pod_id, action, executed=False, k8s_pod_name=None,
                                    node_id=None, detail="No real cluster reachable; decision logged only.")

        if action == "STAY":
            return ExecutionResult(pod.pod_id, action, executed=True, k8s_pod_name=None,
                                    node_id=current_node.node_id, detail="No action taken; pod remains in place.")

        if action == "DELAY":
            return ExecutionResult(pod.pod_id, action, executed=True, k8s_pod_name=None,
                                    node_id=current_node.node_id,
                                    detail="Decision deferred; will be re-evaluated next scheduling cycle.")

        if action == "START":
            return self._start(pod, target_node or current_node)

        if action == "MIGRATE":
            if target_node is None:
                return ExecutionResult(pod.pod_id, action, executed=False, k8s_pod_name=None,
                                        node_id=None, detail="MIGRATE requested with no target node.")
            if self.cooldown.is_in_cooldown(target_node.node_id):
                return ExecutionResult(pod.pod_id, action, executed=False, k8s_pod_name=None,
                                        node_id=target_node.node_id,
                                        detail=f"Target node in cooldown for "
                                               f"{self.cooldown.remaining_sec(target_node.node_id):.0f}s more; skipped.")
            return self._migrate(pod, current_node, target_node)

        return ExecutionResult(pod.pod_id, action, executed=False, k8s_pod_name=None,
                                node_id=None, detail=f"Unknown action '{action}'.")

    def _start(self, pod: WorkloadPod, target_node: NodeConfig) -> ExecutionResult:
        try:
            k8s_pod_name = k8s_deployer.deploy_pod(pod, target_node)
            self.cooldown.start_cooldown(target_node.node_id)
            return ExecutionResult(pod.pod_id, "START", executed=True, k8s_pod_name=k8s_pod_name,
                                    node_id=target_node.node_id, detail=f"Deployed fresh pod onto {target_node.node_id}.")
        except Exception as exc:
            return ExecutionResult(pod.pod_id, "START", executed=False, k8s_pod_name=None,
                                    node_id=target_node.node_id, detail=f"Deployment failed: {exc}")

    def _migrate(self, pod: WorkloadPod, current_node: NodeConfig, target_node: NodeConfig) -> ExecutionResult:
        """Delete-and-recreate migration: remove the pod from its current
        node, then create a fresh instance on the target node. See the
        module docstring for why this is the right strategy here."""
        try:
            source_k8s_name = k8s_deployer._k8s_pod_name(pod)
            k8s_deployer.delete_pod(source_k8s_name)
            new_k8s_name = k8s_deployer.deploy_pod(pod, target_node)
            self.cooldown.start_cooldown(current_node.node_id)
            self.cooldown.start_cooldown(target_node.node_id)
            return ExecutionResult(
                pod.pod_id, "MIGRATE", executed=True, k8s_pod_name=new_k8s_name, node_id=target_node.node_id,
                detail=f"Deleted from {current_node.node_id}, recreated on {target_node.node_id} "
                       f"(restart-based migration; see module docstring).",
            )
        except Exception as exc:
            return ExecutionResult(pod.pod_id, "MIGRATE", executed=False, k8s_pod_name=None,
                                    node_id=target_node.node_id, detail=f"Migration failed: {exc}")