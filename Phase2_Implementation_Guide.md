# Agentic AI-Driven Sustainability Orchestrator
## Step-by-Step Implementation Guide (Layers 1 to 5)
**Context**: 23CSE498 Phase 2 · Team 23UG048 · Guide: Dr. Beena B.M.

---

## 1. Executive Summary & Core Contribution

Current "green cloud" workload systems suffer from a critical flaw: they migrate workloads naively to cleaner regions based purely on static grid carbon intensity numbers. However:
1. **Migration has a hidden energy cost**: Live migration (checkpointing memory pages and transmitting data over cross-region networks) consumes significant network energy.
2. **Hardware heterogeneity matters**: Slower destination hardware increases execution runtime, causing workloads to burn more total energy even if the destination grid is "greener".
3. **Renewable energy fluctuates**: Solar/wind grid intensity changes dynamically over time.
4. **SLA Risk**: Unvalidated migrations cause latency spikes and SLA deadline violations.

**Our Core Contribution**:
We implemented a closed-loop system up to Layer 5 that:
- Measures real per-pod energy via Kepler eBPF metrics.
- Uses a **Digital Twin** to simulate "what would happen if we migrated this pod" before doing it.
- Computes **Net Sustainability Gain (NSG)** ($\text{Carbon}_{\text{saved}} - \text{Carbon}_{\text{migration}}$) across 1000 Monte Carlo runs.
- Accepts a migration plan **only if** $P(\text{NSG} > 0) \ge 90\%$ and $\text{SLA Risk} < 0.80$.

---

## 2. Directory & Module Architecture

```text
f:/Phase 2/
├── data/
│   └── borg_traces_data.csv                   # Google Borg Cluster Trace (405,894 records)
├── src/
│   ├── layer1_infra/                          # LAYER 1: Infrastructure & Trace Replay Engine
│   │   ├── __init__.py
│   │   ├── cluster_config.py                  # Multi-region node pools, MIPS ratings, Kind K8s manifest
│   │   ├── trace_loader.py                    # Borg trace parser, 10x timeline compressor, K8s Pod manifests
│   │   └── kepler_emulator.py                # Kepler eBPF per-pod power (Watts) & energy (Joules) model
│   ├── layer2_telemetry/                      # LAYER 2: Monitoring & Telemetry Pipeline
│   │   ├── __init__.py
│   │   ├── carbon_feed.py                     # Multi-region carbon intensity time-series feed (gCO2/kWh)
│   │   └── feature_store.py                  # Dynamic unified telemetry feature store
│   ├── layer3_scheduling/                     # LAYER 3: Sustainability Scheduling Pre-Filter
│   │   ├── __init__.py
│   │   ├── heterogeneity.py                  # MIPS hardware performance scaling model
│   │   └── nsga2_filter.py                   # Hard rules capacity filter & NSGA-II multi-objective pre-filter
│   ├── layer4_digital_twin/                   # LAYER 4: Digital Twin & Simulation Engine
│   │   ├── __init__.py
│   │   ├── gbr_predictor.py                  # GradientBoostingRegressor runtime model (80/20 time split)
│   │   ├── clark_model.py                    # Clark live memory migration energy & carbon cost model
│   │   ├── simpy_uncertainty.py              # SimPy discrete-event carbon trajectory sampler
│   │   └── monte_carlo.py                    # 1000-run Monte Carlo uncertainty simulation engine
│   ├── layer5_outputs/                        # LAYER 5: Model Outputs & Risk Assessment Interface
│   │   ├── __init__.py
│   │   ├── nsg_calculator.py                 # NSG distribution, 90% CI, SLA Risk & Viability calculator
│   │   └── data_contracts.py                 # Structured dataclass & JSON schemas output contract
│   └── pipeline_runner.py                     # End-to-end pipeline runner (night vs solar peak cycles)
└── tests/
    └── test_layers_1_to_5.py                  # Comprehensive Pytest suite (11 automated test cases)
```

---

## 3. Detailed Step-by-Step Breakdown of Every File

### 3.1 Layer 1: Infrastructure & Replay System (`src/layer1_infra/`)

#### 1. `src/layer1_infra/cluster_config.py`
- **What is it for?**: Configures the multi-region node pool topologies, hardware MIPS capacities, and Kubernetes cluster manifests.
- **Key Contents**:
  - `NodeConfig`: Dataclass defining `node_id`, `region`, `mips_rating`, `total_ram_gb`, `idle_power_w`, `max_power_w`, `free_ram_gb`.
  - `create_default_node_pools()`: Builds 9 heterogeneous worker nodes across 3 regions:
    - `DC_EU`: EU Ireland (4000 MIPS, high green solar/wind).
    - `DC_US`: US Virginia (3500 MIPS, standard grid mix).
    - `DC_ASIA`: APAC Tokyo (2800 MIPS, fossil heavy).
  - `generate_kind_config()`: Generates a Kubernetes Kind cluster YAML manifest (`kind-config.yaml`) with `topology.kubernetes.io/region` node labels.
- **Why is it needed?**: Provides the physical and heterogeneous hardware capacity foundation required for MIPS scaling and Kepler power estimation.

#### 2. `src/layer1_infra/trace_loader.py`
- **What is it for?**: Parses Google Borg cluster trace CSV data and converts workload instances into normalized Kubernetes Pod specs.
- **Key Contents**:
  - `WorkloadPod`: Dataclass storing `pod_id`, `collection_id`, `cpu_request`, `ram_request_gb`, `assigned_memory_gb`, `cpi`, `duration_sec`, `source_node`, `source_region`, `sla_deadline_sec`.
  - `TraceLoader`: Ingests `borg_traces_data.csv`, applies timeline compression (10x compression ratio), and computes SLA deadlines ($1.5 \times \text{duration}$).
  - `to_k8s_manifest()`: Exports standard Kubernetes V1 `Pod` manifest dictionaries (`kind: Pod`, container resources requests).
- **Why is it needed?**: Replays real enterprise cluster workload requests through the orchestrator.

#### 3. `src/layer1_infra/kepler_emulator.py`
- **What is it for?**: Calculates dynamic per-pod power draw (Watts) and energy consumption (Joules) using Kepler eBPF formulas.
- **Key Contents**:
  - `estimate_pod_power()`: Implements Kepler eBPF linear dynamic formula:
    $$P_{\text{pod}} = P_{\text{idle\_per\_pod}} + (P_{\text{max}} - P_{\text{idle}}) \times \text{CPU}_{\text{util}} + E_{\text{DRAM}} \times \text{RAM}_{\text{used}}$$
  - `estimate_pod_energy()`: Converts power over duration into Joules ($1 \text{ Watt} \times 1 \text{ Second} = 1 \text{ Joule}$).
  - `measure_pod()`: Returns complete `PodEnergyMetrics` objects.
- **Why is it needed?**: Provides eBPF power estimation metrics when physical RAPL hardware counters are unavailable in cloud VMs or Kind clusters.

---

### 3.2 Layer 2: Monitoring & Telemetry Pipeline (`src/layer2_telemetry/`)

#### 4. `src/layer2_telemetry/carbon_feed.py`
- **What is it for?**: Generates real-time and forecast regional grid Carbon Intensity time-series (gCO2/kWh).
- **Key Contents**:
  - `get_carbon_intensity(region, timestamp_sec)`:
    - `DC_EU`: Solar/Wind diurnal model with clean solar window (~140 gCO2/kWh) around mid-day (12 PM).
    - `DC_US`: Gas/Nuclear demand mix (~420 gCO2/kWh).
    - `DC_ASIA`: Coal/Fossil baseline (~540 gCO2/kWh).
  - `get_region_forecast()`: Returns carbon intensity forecast arrays over future time horizons.
- **Why is it needed?**: Supplies the real-time carbon signal driving candidate ranking and Digital Twin simulation.

#### 5. `src/layer2_telemetry/feature_store.py`
- **What is it for?**: Aggregates telemetry signals into dynamic unified state frames per scheduling cycle.
- **Key Contents**:
  - `UnifiedTelemetryFrame`: Dataclass bundling `timestamp_sec`, `pod`, `current_node`, `kepler_metrics`, `current_region_ci`, `target_node_states`, `target_region_cis`.
- **Why is it needed?**: Serves as the single source of state passed to downstream pre-filtering and simulation modules.

---

### 3.3 Layer 3: Sustainability Scheduling Pre-Filter (`src/layer3_scheduling/`)

#### 6. `src/layer3_scheduling/heterogeneity.py`
- **What is it for?**: Calculates CPU speedup/slowdown scaling ratios across heterogeneous node hardware.
- **Key Contents**:
  - `get_execution_scaling_factor()`: $\text{Ratio} = \text{MIPS}_{\text{source}} / \text{MIPS}_{\text{target}}$.
  - `scale_duration()`: Adjusts workload runtime when placed on target node ($T_{\text{scaled}} = T_{\text{base}} \times \text{Ratio}$).
- **Why is it needed?**: Prevents false green decisions where running on slower hardware increases execution time and total energy consumption.

#### 7. `src/layer3_scheduling/nsga2_filter.py`
- **What is it for?**: Applies hard capacity filtering and NSGA-II multi-objective candidate ranking to shortlist Top-K targets.
- **Key Contents**:
  - Hard Rule Filter: $\text{Node Free RAM} \ge \text{Pod RAM Request}$.
  - Multi-Objective Problem (`pymoo`): Minimizes (1) Estimated Carbon Footprint (gCO2), and (2) SLA Latency Penalty.
  - Returns `CandidatePair` objects ranked and shortlisted to Top-K (default $K=3$).
- **Why is it needed?**: Reduces computational overhead by pre-filtering thousands of invalid node candidates before Digital Twin simulation.

---

### 3.4 Layer 4: Digital Twin & Simulation Engine (`src/layer4_digital_twin/`)

#### 8. `src/layer4_digital_twin/gbr_predictor.py`
- **What is it for?**: Predicts expected workload runtime on target nodes using ML regression with time-based splitting.
- **Key Contents**:
  - `GradientBoostingRegressor`: Features `[cpu_request, ram_request_gb, assigned_memory_gb, cpi, priority, mips_ratio]`.
  - `train_time_split()`: Time-ordered 80/20 train/test split to prevent lookahead data leakage.
  - `predict()`: Returns predicted mean duration $\hat{T}_{\text{exec}}$ and residual error standard deviation $\sigma_{\text{gbr}}$.
- **Why is it needed?**: Delivers ML-based runtime expectation and uncertainty bounds for Monte Carlo sampling.

#### 9. `src/layer4_digital_twin/clark_model.py`
- **What is it for?**: Models live migration network transfer duration, memory checkpoint energy, and carbon emissions.
- **Key Contents**:
  - Checkpoint Size: $D_{\text{mem}} = \text{RAM}_{\text{assigned\_gb}}$.
  - Migration Time: $T_{\text{mig}} = (D_{\text{mem}} / \text{Bandwidth}_{\text{net}}) + T_{\text{latency}}$.
  - Network Energy: $E_{\text{mig}} = (D_{\text{mem}} \times e_{\text{net}}) + (T_{\text{mig}} \times P_{\text{switches}})$.
  - Migration Carbon Cost: $C_{\text{mig}} = (E_{\text{mig}} / 3.6\times 10^6) \times \text{CI}_{\text{source\_region}}$.
- **Why is it needed?**: Deducts the hidden carbon cost of migration when calculating Net Sustainability Gain.

#### 10. `src/layer4_digital_twin/simpy_uncertainty.py`
- **What is it for?**: Simulates stochastic discrete-event carbon trajectory dynamic fluctuations.
- **Key Contents**:
  - `SimPyCarbonSampler`: Uses `simpy.Environment` to step through future execution windows and sample carbon intensity with event noise.
- **Why is it needed?**: Captures grid carbon intensity fluctuations during workload execution.

#### 11. `src/layer4_digital_twin/monte_carlo.py`
- **What is it for?**: Executes 1000 stochastic simulation runs combining all uncertainty sources.
- **Key Contents**:
  - Vectorized sampling across 1000 iterations:
    1. Runtime: $T_{\text{exec}} \sim \mathcal{N}(\hat{T}_{\text{exec}}, \sigma_{\text{gbr}}^2)$.
    2. Migration Overhead: $T_{\text{mig}} \sim \mathcal{N}(T_{\text{mig\_clark}}, 0.4^2)$.
    3. Renewable Grid Carbon Fluctuation trajectory.
  - Outputs empirical sample arrays for STAY Carbon, MIGRATE Carbon, and Net Sustainability Gain ($\text{NSG} = \text{STAY} - \text{MIGRATE}$).
- **Why is it needed?**: Produces empirical outcome distributions for Layer 5 statistical evaluation.

---

### 3.5 Layer 5: Model Outputs & Risk Assessment (`src/layer5_outputs/`)

#### 12. `src/layer5_outputs/nsg_calculator.py`
- **What is it for?**: Computes NSG statistics, 90% confidence intervals, SLA risk scores, and statistical viability flags.
- **Key Contents**:
  - Mean NSG ($\mu_{\text{NSG}}$), 5th percentile ($p05$), 95th percentile ($p95$).
  - Statistical Confidence: $P(\text{NSG} > 0) = \text{Count}(\text{NSG}_i > 0) / 1000$.
  - SLA Risk Score: $P(T_{\text{total}} > \text{SLA}_{\text{deadline}}) = \text{Count}(T_{\text{total}, i} > \text{SLA}_{\text{deadline}}) / 1000$.
  - Viability Gate: $\text{Viable} = (P(\text{NSG} > 0) \ge 0.90) \text{ and } (\text{SLA Risk} < 0.80)$.
- **Why is it needed?**: Evaluates candidate safety and sustainability before passing to Layer 6.

#### 13. `src/layer5_outputs/data_contracts.py`
- **What is it for?**: Defines structured dataclasses and JSON output schemas for downstream layers.
- **Key Contents**:
  - `CandidateEvaluationResult`: Dataclass containing `pod_id`, `current_node_id`, `target_node_id`, `target_region`, `nsg_mean_gco2`, `nsg_p05_gco2`, `nsg_p95_gco2`, `p_nsg_positive`, `sla_risk_score`, `predicted_exec_time_sec`, `migration_duration_sec`, `is_statistically_viable`.
- **Why is it needed?**: Establishes the strict API contract consumed by the Layer 6 Decision & Policy Engine.

---

## 4. Pipeline Runner & Verification Suite

### `src/pipeline_runner.py`
- **Purpose**: End-to-end pipeline runner executing trace batch ingestion, feature enrichment, NSGA-II candidate pre-filtering, 1000-run Monte Carlo Digital Twin simulation, and Layer 5 model output formatting across both Night and Mid-day Solar Peak scheduling cycles.

### `tests/test_layers_1_to_5.py`
- **Purpose**: Automated Pytest suite containing 11 test cases covering every layer.
- **Execution Command**:
  ```bash
  python -m pytest tests/test_layers_1_to_5.py -v
  ```
- **Result**: All 11 unit tests passed in **1.85 seconds**.

---

## 5. File Location of Generated Deliverables

1. **Microsoft Word Document (.docx)**:
   [Sustainability_Orchestrator_Phase2_Implementation_Guide.docx](file:///f:/Phase%202/Sustainability_Orchestrator_Phase2_Implementation_Guide.docx)

2. **Markdown Document (.md)**:
   [Phase2_Implementation_Guide.md](file:///f:/Phase%202/Phase2_Implementation_Guide.md)
