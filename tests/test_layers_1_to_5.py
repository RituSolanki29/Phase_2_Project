"""
Automated Pytest Suite for Sustainability Orchestrator (Layers 1 to 5)
Tests trace ingestion, Kepler energy telemetry, carbon feature store,
batched NSGA-II pre-filtering, GBR runtime prediction, Clark migration model,
SimPy sampler, 1000-run Monte Carlo simulation, and Layer 5 statistical model
output interfaces.
"""

import os
import pytest
import numpy as np

from src.layer1_infra.cluster_config import create_default_node_pools, generate_kind_config
from src.layer1_infra.trace_loader import TraceLoader, WorkloadPod
from src.layer1_infra.kepler_emulator import KeplerEmulator
from src.layer2_telemetry.carbon_feed import CarbonFeed
from src.layer2_telemetry.telemetry_source import TelemetrySource
from src.layer2_telemetry.feature_store import FeatureStore
from src.layer3_scheduling.heterogeneity import HardwareHeterogeneityModel
from src.layer3_scheduling.nsga2_filter import NSGA2PreFilter
from src.layer4_digital_twin.gbr_predictor import GBRRuntimePredictor
from src.layer4_digital_twin.clark_model import ClarkMigrationModel
from src.layer4_digital_twin.simpy_uncertainty import SimPyCarbonSampler
from src.layer4_digital_twin.monte_carlo import MonteCarloEngine
from src.layer5_outputs.nsg_calculator import Layer5OutputCalculator

# Portable path: <project_root>/data/borg_traces_data.csv
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(_PROJECT_ROOT, "data", "borg_traces_data.csv")


def test_cluster_config():
    nodes = create_default_node_pools()
    assert len(nodes) == 9
    regions = set(n.region for n in nodes)
    assert regions == {"DC_EU", "DC_US", "DC_ASIA"}

    kind_yaml = generate_kind_config(nodes)
    assert "kind: Cluster" in kind_yaml
    assert "topology.kubernetes.io/region=DC_EU" in kind_yaml


def test_trace_loader():
    loader = TraceLoader(CSV_PATH)
    pods = loader.load_trace_batch(limit=20, compression_factor=100.0)
    assert len(pods) == 20
    pod = pods[0]
    assert isinstance(pod, WorkloadPod)
    assert pod.cpu_request > 0
    assert pod.ram_request_gb > 0

    # Test K8s manifest conversion
    manifest = pod.to_k8s_manifest()
    assert manifest["kind"] == "Pod"
    assert manifest["metadata"]["name"] == pod.pod_id


def test_trace_loader_region_is_deterministic_per_machine():
    """Same machine_id must always resolve to the same synthetic region,
    regardless of which row/position it appears in."""
    loader = TraceLoader(CSV_PATH)
    pods_a = loader.load_trace_batch(limit=200)
    by_machine = {}
    for pod in pods_a:
        by_machine.setdefault(pod.source_node, set()).add(pod.source_region)
    for node, regions in by_machine.items():
        assert len(regions) == 1, f"source_node {node} mapped to multiple regions: {regions}"


def test_kepler_emulator():
    kepler = KeplerEmulator()
    power_w = kepler.estimate_pod_power(cpu_util=0.5, ram_used_gb=4.0)
    assert power_w > 0

    energy_j = kepler.estimate_pod_energy(power_w, duration_seconds=10.0)
    assert energy_j == power_w * 10.0

    metrics = kepler.measure_pod("test-pod", cpu_util=0.8, ram_used_gb=8.0, duration_sec=5.0)
    assert metrics.pod_id == "test-pod"
    assert metrics.total_energy_joules > 0


def test_carbon_feed():
    feed = CarbonFeed(seed=42)
    eu_ci = feed.get_carbon_intensity("DC_EU", timestamp_sec=43200.0)
    us_ci = feed.get_carbon_intensity("DC_US", timestamp_sec=43200.0)
    asia_ci = feed.get_carbon_intensity("DC_ASIA", timestamp_sec=43200.0)

    assert 50.0 <= eu_ci <= 500.0
    assert 200.0 <= us_ci <= 600.0
    assert 300.0 <= asia_ci <= 700.0


def test_feature_store():
    nodes = create_default_node_pools()
    loader = TraceLoader(CSV_PATH)
    pod = loader.load_trace_batch(limit=1)[0]

    kepler = KeplerEmulator()
    carbon_feed = CarbonFeed()
    telemetry = TelemetrySource(kubeconfig_path="/nonexistent")  # force simulated mode in tests
    feature_store = FeatureStore(telemetry)

    frame = feature_store.build_telemetry_frame(pod, nodes, timestamp_sec=100.0)
    assert frame.pod.pod_id == pod.pod_id
    assert "DC_EU" in frame.target_region_cis


def test_heterogeneity_scaling():
    nodes = create_default_node_pools()
    eu_node = next(n for n in nodes if n.region == "DC_EU")     # MIPS 4000
    asia_node = next(n for n in nodes if n.region == "DC_ASIA")  # MIPS 2800

    ratio = HardwareHeterogeneityModel.get_execution_scaling_factor(eu_node, asia_node)
    assert ratio > 1.0  # Slower node -> ratio > 1.0

    scaled_dur = HardwareHeterogeneityModel.scale_duration(10.0, eu_node, asia_node)
    assert scaled_dur > 10.0


def test_nsga2_single_pod_wrapper():
    """Backward-compatible single-pod entry point still works."""
    nodes = create_default_node_pools()
    loader = TraceLoader(CSV_PATH)
    pod = loader.load_trace_batch(limit=1)[0]

    kepler = KeplerEmulator()
    carbon_feed = CarbonFeed()
    telemetry = TelemetrySource(kubeconfig_path="/nonexistent")  # force simulated mode in tests
    feature_store = FeatureStore(telemetry)
    frame = feature_store.build_telemetry_frame(pod, nodes)

    nsga2_filter = NSGA2PreFilter(kepler, top_k=3, pop_size=20, n_gen=5, seed=42)
    candidates = nsga2_filter.filter_and_rank(frame)
    assert len(candidates) <= 3
    assert candidates[0].rank == 1


def test_nsga2_batch_scheduling_respects_capacity():
    """
    Real regression test for the fixed capacity constraint: pack many pods
    each requesting most of a node's RAM into one cycle, across few nodes,
    and confirm no single node is assigned more cumulative RAM than it has.
    """
    nodes = create_default_node_pools()
    loader = TraceLoader(CSV_PATH)
    pods = loader.load_trace_batch(limit=8)

    kepler = KeplerEmulator()
    carbon_feed = CarbonFeed()
    telemetry = TelemetrySource(kubeconfig_path="/nonexistent")  # force simulated mode in tests
    feature_store = FeatureStore(telemetry)

    # Force a genuinely binding capacity scenario: each node can fit AT MOST
    # ONE of these pods (20 GB request vs. 25 GB node capacity), so any
    # feasible assignment must spread the 8 pods across at least 8 of the 9
    # nodes. The old (unfixed) implementation never tracked cumulative
    # per-node usage across pods, so it could happily pile several of these
    # onto the single cheapest node -- this test would have caught that.
    for node in nodes:
        node.total_ram_gb = 25.0
        node.free_ram_gb = 25.0
    for pod in pods:
        pod.ram_request_gb = 20.0

    frames = [feature_store.build_telemetry_frame(pod, nodes, timestamp_sec=0.0) for pod in pods]

    nsga2_filter = NSGA2PreFilter(kepler, top_k=1, pop_size=40, n_gen=20, seed=42)
    candidates_by_pod = nsga2_filter.filter_and_rank_batch(frames)

    # Reconstruct the top-1 (best) assignment per pod and check aggregate
    # RAM assigned to each physical node does not exceed that node's total.
    ram_used = {n.node_id: 0.0 for n in nodes}
    for pod in pods:
        top = candidates_by_pod[pod.pod_id][0]
        ram_used[top.target_node.node_id] += pod.ram_request_gb

    node_by_id = {n.node_id: n for n in nodes}
    for node_id, used in ram_used.items():
        assert used <= node_by_id[node_id].total_ram_gb + 1e-6, (
            f"Node {node_id} oversubscribed: {used} GB used vs {node_by_id[node_id].total_ram_gb} GB capacity"
        )


def test_gbr_predictor():
    loader = TraceLoader(CSV_PATH)
    pods = loader.load_trace_batch(limit=50)

    predictor = GBRRuntimePredictor()
    rmse, r2 = predictor.train_time_split(pods, train_ratio=0.8)
    assert predictor.is_trained

    pred_mean, std = predictor.predict(pods[0])
    assert pred_mean > 0
    assert std > 0


def test_clark_migration_model():
    nodes = create_default_node_pools()
    loader = TraceLoader(CSV_PATH)
    pod = loader.load_trace_batch(limit=1)[0]

    eu_node = next(n for n in nodes if n.region == "DC_EU")
    asia_node = next(n for n in nodes if n.region == "DC_ASIA")

    clark = ClarkMigrationModel()
    cost = clark.estimate_migration_cost(pod, eu_node, asia_node, source_region_ci=200.0)
    assert cost.migration_duration_sec > 0
    assert cost.migration_carbon_gco2 > 0


def test_simpy_sampler():
    feed = CarbonFeed()
    sampler = SimPyCarbonSampler(feed)
    samples = sampler.sample_carbon_trajectory("DC_EU", start_time_sec=0.0, duration_sec=60.0)
    assert len(samples) > 0


def test_monte_carlo_and_layer5_outputs():
    nodes = create_default_node_pools()
    loader = TraceLoader(CSV_PATH)
    pods = loader.load_trace_batch(limit=30)
    pod = pods[0]

    kepler = KeplerEmulator()
    carbon_feed = CarbonFeed()
    telemetry = TelemetrySource(kubeconfig_path="/nonexistent")  # force simulated mode in tests
    feature_store = FeatureStore(telemetry)
    frame = feature_store.build_telemetry_frame(pod, nodes)

    gbr = GBRRuntimePredictor()
    gbr.train_time_split(pods)
    clark = ClarkMigrationModel()
    simpy_sampler = SimPyCarbonSampler(carbon_feed)

    mc_engine = MonteCarloEngine(gbr, clark, simpy_sampler, kepler, num_runs=100)
    target_node = nodes[1]

    mc_result = mc_engine.run_simulation(frame, target_node)
    assert len(mc_result.net_sustainability_gain_samples) == 100

    output = Layer5OutputCalculator.evaluate_candidate(mc_result, pod, frame.current_node, target_node)
    assert output.pod_id == pod.pod_id
    assert 0.0 <= output.p_nsg_positive <= 1.0
    assert 0.0 <= output.sla_risk_score <= 1.0
    assert isinstance(output.is_statistically_viable, bool)