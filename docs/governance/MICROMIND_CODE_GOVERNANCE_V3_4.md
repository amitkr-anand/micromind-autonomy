# MICROMIND CODE GOVERNANCE MANUAL

**Version 3.4 — Handoff Execution Protocol & Adversarial QA Standard**  
**Classification:** Programme Confidential / Project MicroMind Internal  
**Authority:** TASL-MM-GOV-3.4  
**Effective Date:** 12 April 2026  
**Supersedes:** v3.3

---

## Preamble: Founder's Intent

> *"Western technology is built for Air Superiority. Indian reality is Air Denial. MicroMind is not built for the parade ground; it is built for the GPS-dead, spectrum-denied reality of a two-front war where the enemy knows our frequencies and watches us from space. It is the brain that survives when the network dies."*

> *"MicroMind is an onboard autonomy and GNSS-denied navigation stack for tactical UAVs and loitering munitions. It is NOT a flight controller. It operates above PX4, issuing high-level mission intent while PX4 handles vehicle stabilisation."*

**The User's Metric:** Precision delivery at 150+ km under full GNSS denial and RF severance from km 50. **One Mission — One Hit.** Everything else is implementation detail.

**Core Principles:**
1. **Assured Reliability** — One mission, one hit. System reliability is non-negotiable.
2. **Bounded Deterministic Autonomy** — Every decision must be traceable to a specific input condition and a specific rule. No free-form AI decision-making occurs in-mission (AD-07). The system implements the Mission Commander's Intent.
3. **Hardware Sovereignty** — No single foreign-controlled dependency shall create strategic failure.
4. **Meaningful Human Control** — L10s-SE enforces ROE. IHL compliance is designed in, not bolted on.

---

## 1. Module Sovereignty & Ownership

The system is governed by the principle of **Logical Isolation**. No module may leak its internal logic, state, or dependencies into another. The Unified State Vector (USV) is the sole data transport between modules.

### 1.1 Module Ownership Matrix

| Module | Primary Owner | Layer | Authoritative Source |
|--------|---------------|-------|---------------------|
| **Navigation Manager** | L1/L2/L3 Stack (AD-01) | MicroMind-X | `core/navigation/` |
| **Mission Manager** | NanoCorteX FSM | NanoCorteX | `core/state_machine/` |
| **EW Manager** | Threat/Signature Maps | MicroMind-X | `core/ew/` |
| **Route Planner** | Hybrid A* / Kinematics | NanoCorteX | `core/route_planner/` |
| **PX4 Bridge** | MAVLink / Hardware HAL | Integration | `integration/mavlink_bridge.py` |
| **BIM** | GNSS Trust Arbitration | MicroMind-X | `core/bim/` |
| **DMRL** | Terminal Guidance | MicroMind-X | `core/dmrl/` |
| **L10s-SE** | ROE Enforcement | Layer 1 Authority | `core/l10s_se/` |

### 1.2 Three-Layer Navigation Architecture (AD-01)

Per the architecture decision of 03 April 2026, navigation uses a three-layer stack:

| Layer | Name | Mechanism | Owner |
|-------|------|-----------|-------|
| **L1** | Relative Tracking | IMU + VIO (OpenVINS) | Navigation Manager |
| **L2** | Absolute Reset | Orthophoto Image Matching vs preloaded satellite tiles | Navigation Manager |
| **L3** | Vertical Stability | Baro-INS fusion | Navigation Manager |

**RADALT is retained for terminal phase only** (0–300 m AGL, final 5 km).

### 1.3 Forbidden Behaviours (Agent Prohibitions)

These are **Hard Fails**. Any violation results in immediate rejection.

| Module | Forbidden Behaviour | Rationale |
|--------|---------------------|-----------|
| **Navigation Manager** | NO direct PX4 mode commands | AD-08: Autonomy as Payload |
| **Navigation Manager** | NO mission-logic state changes | Authority Chain Layer 3 |
| **Navigation Manager** | NO ROE evaluation | L10s-SE is sole ROE authority |
| **Mission Manager** | NO raw sensor fusion | Authority Chain Layer 2 |
| **Mission Manager** | NO bypass of Mission Envelope hashes | Layer 0 is root authority |
| **Mission Manager** | NO modification of navigation state vectors | USV is read-only to consumers |
| **EW Manager** | NO modification of navigation state vectors | EW is advisory to Route Planner |
| **EW Manager** | NO direct PX4 commands | All autopilot via PX4 Bridge |
| **Route Planner** | NO direct `time.time()` calls | AD-11: Mission Clock abstraction |
| **Route Planner** | NO hardcoded mission thresholds | Config governance §4 |
| **Route Planner** | NO ROE evaluation | L10s-SE is sole ROE authority |
| **PX4 Bridge** | NO mission logic | AD-08: Driver abstraction |
| **PX4 Bridge** | NO storage of mutable global state | USV is sole state transport |
| **PX4 Bridge** | NO modification of core navigation code | AD-09: ABC interfaces only |
| **BIM** | NO direct ESKF state modification | BIM outputs trust score only |
| **BIM** | NO navigation mode decisions | Navigation Manager arbitrates |
| **DMRL** | NO abort/continue decision | L10s-SE decides; DMRL provides confidence |
| **L10s-SE** | NO machine learning | FR-105: Deterministic tree only |
| **L10s-SE** | NO modification of mission envelope | Layer 0 is immutable |
| **L10s-SE** | NO bypass of civilian detection | IHL compliance requirement |

### 1.4 The "No-Go" List (Universal Hard Fails)

These apply to **all modules without exception**:

| Rule | Violation | Detection |
|------|-----------|-----------|
| **No Mutable Global State** | All data must pass through USV | Static analysis: grep for `global` keyword |
| **No Silent Exceptions** | Every `except` block must generate structured JSON log | AST scan for bare `except:` without logging |
| **No Magic Numbers** | Every threshold must reside in versioned config | grep for numeric literals in conditionals |
| **No Implicit Timing** | No thread creation outside SystemScheduler | grep for `threading.Thread`, `multiprocessing.Process` |
| **No Code Entropy** | `TODO`, `FIXME`, `HACK` markers prohibited in production | grep + CI gate |
| **No Velocity-Dependent Control Logic** | System Rule 1.8 (AD-19) — no control logic may depend on `state.v` | Code review gate |

---

## 2. Multi-Agent Workflow: The Two-Deputy Hierarchy

To prevent context-entropy and ensure mission success, authority is split into two autonomous branches. Development follows a heterogeneous multi-agent pipeline. **The two Deputies are equal in authority but distinct in mandate.**

### 2.1 Deputy 1: Design & Architecture (Agents 1 & 2)

**Mandate:** Implementation of the "Founder's Vision."

| Agent | Role | Responsibility | Environment |
|-------|------|----------------|-------------|
| **Agent 1: Architect (Lead)** | Strategic Design | Enforces Founder's vision. Plans sprints based on QA Log and SRS v1.3. Keeps development aligned to Part Two V7.2, Part One V6, and DECISIONS.md. Owns SRS mapping, ADRs, and Daily Mission planning. Defines interface specifications and performance requirements. | Claude Opus 4.5/4.6 — Web interface. Amit bridges prompts to Agent 2. |
| **Agent 2: Implementer** | Production Logic | Owns production code in `/core` and `/src`. Writes logic strictly following SRS v1.3 and this manual. Follows Tiered Complexity Model and sprints planned by Deputy 1. Delivers implementation code with header citations, tested and ready for QA. Maintains GIT repository, updates Status and `TECHNICAL_NOTES.md`. | Claude Sonnet 4.6 — Claude-Code on development workstation. Receives commands from Deputy 1 via Amit (paste to terminal). |

**Power:** Final authority on *how* the system is built.

### 2.2 Deputy 2: Quality & Audit (Agents 3 & 4)

**Mandate:** Enforcement of "System Quality, Integrity & Performance Truth."

| Agent | Role | Responsibility | Environment |
|-------|------|----------------|-------------|
| **Agent 4: Sentinel (Lead)** | Quality Enforcement | Actively looks for logic-bleed, dependency violations, SRS non-compliance, and performance leaks. Flags any violations of Forbidden Behaviours (§1.3). Flags potential operational issues. Helps Deputy 1 assess practical limits of their product. | Gemini Flash — Web interface / CLI. Git Auditor: commits AUDIT_LOG.md and workflow documents. Initiated by Amit. |
| **Agent 3: Tester** | Adversarial Stress | Owns "Breaking Limits" (adversarial testing). Focuses on AT-6 Repeatability and Compound Fault Injection. Generates Integration, Performance & Stress Tests following 18-field SRS format. Delivers test files, coverage reports, fault matrix results. | Gemini Pro — Git Contributor: commits test files and evidence to `/tests` and `/docs/qa`. |

**Power:** Final authority on *if* the system is ready. Works with Deputy 1 to unblock development if the same module fails tests for more than 3 times.

### 2.3 The Logic Box Framework (Domain Isolation)

Crossing box boundaries without an integration bridge is a **Hard Fail**.

| Box | Name | Contents | Rules |
|-----|------|----------|-------|
| **Orchestration Box** | `simulation/` | "The Marketing Jazz" — SITL, visualization, non-deterministic demo logic | MAVLink/Gazebo dependencies allowed |
| **Navigation Core Box** | `core/` | "The Frozen Baseline" — Mathematical estimators (L1/L2/L3), NanoCorteX FSM, navigation resets | **FORBIDDEN:** MAVLink or Gazebo dependencies |
| **Governance Box** | `docs/` | "The Sovereign Truth" — MHM artifacts, ADRs, SRS | Controlled documents only |

### 2.4 Document Ownership & Update Responsibility

| Document | Owner | Update Trigger | Ratification |
|----------|-------|----------------|--------------|
| `MICROMIND_PROJECT_CONTEXT.md` | **Agent 1 (Deputy 1)** | Sprint completion, architecture changes, baseline updates | Deputy 1 authority |
| `MICROMIND_QA_LOG.md` | **Agent 4 (Deputy 2)** | Test completion, findings, OI status changes | Deputy 2 authority |
| `AUDIT_LOG.md` | **Agent 4 (Deputy 2)** | Codebase drift/gap audit, sovereignty checks | Deputy 2 authority |
| `MICROMIND_DECISIONS.md` | **Agent 1 (Deputy 1)** | ADR creation/update | Can be raised by either Deputy; **must be ratified by Programme Director**; final version written by Deputy 1 |
| `DAILY_MISSION.md` | **Agent 1 (Deputy 1)** | Daily sprint planning | Deputy 1 authority; consumed by Agent 2 |
| `TECHNICAL_NOTES.md` (per module) | **Agent 2 (Deputy 1)** | Module implementation/update | Deputy 1 authority |
| Governing Documents (SRS, Part Two, Part One) | **Programme Director** | Major revision only | Programme Director authority |

### 2.5 The Human-Readiness Mandate

Every functional module must contain a `TECHNICAL_NOTES.md` file:

- **OODA-Loop Rationale:** Documentation must explain *why* a state transition occurred.
- **Conflict Resolution:** Example: Explain why the system trusts the **L2 Orthophoto Reset** (Absolute) over the **L1 VIO** (Relative) when BIM scores GNSS as "Red."
- **Design Decisions:** Any non-obvious implementation choice must be documented with rationale.

---

## 3. Inter-Deputy Communication Protocol

### 3.1 Communication Channels

| Channel | Format | Purpose |
|---------|--------|---------|
| **Module Readiness Manifest (MRM)** | YAML | Deputy 1 → Deputy 2: Signal code ready for QA |
| **QA Findings Report (QFR)** | YAML | Deputy 2 → Deputy 1: Test results and feedback |
| **Live Documents** | Markdown | Human-readable project state (see §2.4) |
| **Governing Documents** | DOCX/PDF | Authoritative requirements and architecture |

### 3.2 Module Readiness Manifest (MRM) — Deputy 1 → Deputy 2

When Deputy 1 completes a module, Agent 1 commits an MRM to `docs/handoffs/`:

```yaml
# File: docs/handoffs/MRM_<MODULE>_<DATE>.yaml
manifest_version: "1.0"
handoff_type: DEPUTY1_TO_DEPUTY2
timestamp: "2026-04-10T14:30:00Z"

# IDENTIFICATION
deputy1_session_id: "<session-id>"
module:
  name: "Orthophoto Matching Stub"
  path: "core/trn/orthophoto_matching_stub.py"
  complexity_tier: MATHEMATICAL  # DECISION | MATHEMATICAL | DRIVER

# FILES CHANGED
files_changed:
  - path: "core/trn/orthophoto_matching_stub.py"
    action: CREATED  # CREATED | MODIFIED | DELETED
    lines: 287
    complexity_measured: 18
  - path: "tests/test_sprint_c_om_stub.py"
    action: CREATED
    lines: 156

# REQUIREMENTS ADDRESSED
requirements:
  - req_id: "FR-107"
    description: "L2 Absolute Reset via orthophoto matching"
    srs_section: "§1.7.2"
  - req_id: "NAV-02"
    description: "TRN Accuracy (mechanism changed)"
    srs_section: "§2.2"
    note: "Replaces RADALT-NCC per AD-01"

# TEST GUIDANCE FOR DEPUTY 2
test_guidance:
  unit_tests:
    - file: "tests/test_sprint_c_om_stub.py"
      gates: ["OM-01", "OM-02", "OM-03", "OM-04", "OM-05", "OM-06", "OM-07", "OM-08"]
      
  integration_tests_required:
    - id: "IT-NAV-02"
      description: "OM reset integrated with ESKF"
      stimulus: "Inject 5 km drift, verify OM correction < 7 m MAE"
      threshold: 7.0
      unit: "m"
      
  fault_injection_applicable:
    - "FI-01"  # GNSS Spoofing + VIO Outage
    
  adversarial_scenarios:
    - scenario: "Featureless terrain (14 km zone)"
      expected: "Zero corrections applied, NAV_DEGRADED_INS_ONLY logged"
      test_id: "OM-08"
      threshold: 0
      unit: "corrections"

# FROZEN CONSTANTS USED
frozen_constants_used:
  - name: "OM_R_NORTH"
    value: 81.0
    source: "AD-01"

# CONFIG CHANGES
config_keys_added:
  - key: "sigma_terrain_threshold"
    file: "config/tunable_mission.yaml"
    default: 30.0

# KNOWN LIMITATIONS
caveats:
  - "Stub only — no real satellite tile provider"
  - "LWIR dual-use (ingress OM + terminal DMRL) not yet integrated"

# REGRESSION IMPACT
regression_suite:
  baseline_gates: 552
  expected_gates_after: 560
  frozen_files_modified: []
  bcmp_impact: "None"

# DEPUTY 1 SELF-ASSESSMENT (§2.6 Checklist)
deputy1_checklist:
  - item: "Req ID cited in all headers"
    status: PASS
  - item: "Complexity ≤ tier limit"
    status: PASS
    measured: 18
    limit: 20
  - item: "No magic numbers"
    status: PASS
  - item: "Logging schema compliant"
    status: PASS
  - item: "mission_clock.py used"
    status: PASS
  - item: "Forbidden behaviours checked"
    status: PASS
  - item: "USV access pattern correct"
    status: PASS
  - item: "TECHNICAL_NOTES.md present"
    status: PASS

# DECLARATION
deputy1_declaration: |
  Deputy 1 certifies this module is ready for QA.
  All self-regulation checks (§2.6) have been applied.
  Unit tests passing locally.
```

### 3.3 QA Findings Report (QFR) — Deputy 2 → Deputy 1

After testing, Deputy 2 commits a QFR to `docs/handoffs/`:

```yaml
# File: docs/handoffs/QFR_<MODULE>_<DATE>.yaml
report_version: "1.0"
handoff_type: DEPUTY2_TO_DEPUTY1
timestamp: "2026-04-11T09:45:00Z"

# REFERENCE
mrm_reference: "docs/handoffs/MRM_orthophoto_matching_2026-04-10.yaml"
module_under_test: "core/trn/orthophoto_matching_stub.py"

# IDENTIFICATION
deputy2_session_id: "<session-id>"
sentinel_agent: "Agent 4"
tester_agent: "Agent 3"

# EXECUTION TRACE — mandatory for all QFRs
# Agent 3 must generate these hashes on
# micromind-node01 using:
#   pytest <test_files> --junit-xml=\
#     /tmp/results.xml -v \
#     2>&1 | tee /tmp/terminal.txt
#   sha256sum /tmp/results.xml \
#     /tmp/terminal.txt
execution_trace:
  junit_xml_hash: "<sha256 of results.xml>"
  terminal_output_hash: "<sha256 of terminal.txt>"
  executed_on: "micromind-node01"
  execution_date: "<YYYY-MM-DD>"
  jitter_seed: "<seed provided by Deputy 1>"
  node_hostname: "<hostname from Agent 3>"

# A QFR without execution_trace is a Hard Fail.
# Deputy 1 will reject without review.

# OVERALL VERDICT
overall_verdict: CONDITIONAL_PASS  # PASS | CONDITIONAL_PASS | FAIL | BLOCKED

# GATE RESULTS WITH MARGINS
gate_summary:
  total: 8
  passed: 7
  failed: 1
  blocked: 0

gate_details:
  - gate_id: "OM-01"
    status: PASS
    assertion: "Measurement provider interface correct"
    # No margin applicable for interface test
    
  - gate_id: "OM-03"
    status: PASS
    assertion: "OM correction accuracy < 7 m MAE"
    measured_value: 5.8
    threshold: 7.0
    unit: "m"
    margin_absolute: 1.2
    margin_percent: 17.1
    margin_risk: LOW  # LOW (>20%) | MEDIUM (10-20%) | HIGH (<10%) | CRITICAL (<5%)
    
  - gate_id: "OM-05"
    status: PASS
    assertion: "Match latency < 50 ms P95"
    measured_value: 42.3
    threshold: 50.0
    unit: "ms"
    margin_absolute: 7.7
    margin_percent: 15.4
    margin_risk: MEDIUM
    wall_clock_timestamp: "2026-04-12T14:23:07"
    note: "Approaching boundary — monitor on Jetson Orin"

  # Example latency gate with FAIL status — wall_clock_timestamp mandatory:
  - gate_id: "EC01-03"
    status: FAIL
    measured_value: 16.4
    threshold: 20.0
    unit: "Hz"
    margin_absolute: -3.6
    margin_percent: -18.0
    margin_risk: CRITICAL
    wall_clock_timestamp: "2026-04-12T14:23:07"
    
  - gate_id: "OM-08"
    status: FAIL
    assertion: "Featureless terrain produces zero corrections"
    measured_value: 1
    threshold: 0
    unit: "corrections"
    root_cause: "match_confidence threshold (0.6) too low for synthetic featureless tile"
    recommendation: "Raise threshold to 0.7 or add secondary texture check"
    severity: MEDIUM
    blocks_merge: false

# MARGIN SUMMARY (Critical for Programme Director visibility)
margin_summary:
  total_quantifiable_gates: 6
  low_risk: 4
  medium_risk: 1
  high_risk: 0
  critical_risk: 0
  gates_at_risk:
    - gate_id: "OM-05"
      margin_percent: 15.4
      concern: "Latency may exceed threshold on embedded hardware"

# FAULT INJECTION RESULTS
fault_injection:
  - test_id: "FI-01"
    scenario: "GNSS Spoofing + VIO Outage"
    status: PASS
    measured_value: 23.0
    threshold: 400.0
    unit: "m drift at km 100"
    margin_percent: 94.3
    margin_risk: LOW
    observation: "OM became sole correction source"

# ADVERSARIAL TESTING (Agent 3 specialty)
adversarial_findings:
  - scenario: "Synthetic tile with 50% cloud occlusion"
    status: NOT_TESTED
    reason: "No cloud occlusion model in synthetic tile generator"
    recommendation: "Add to Phase-2 test infrastructure"
    
  - scenario: "Tile edge boundary crossing mid-match"
    status: TESTED
    result: PASS
    observation: "Match correctly spans tiles, no discontinuity"

# SOVEREIGNTY AUDIT (Agent 4 specialty)
sovereignty_audit:
  banned_imports_found: []
  dd01_compliant: true
  pqc_integrity: N/A

logic_bleed_scan:
  violations_found: 0
  notes: "None detected"

# PERFORMANCE OBSERVATIONS
performance:
  - metric: "OM match latency"
    measured: 12.3
    unit: "ms P95"
    budget: 50.0
    margin_percent: 75.4
    status: PASS
    
  - metric: "Memory during 150 km run"
    measured: 3.2
    unit: "MB growth"
    budget: 50.0
    margin_percent: 93.6
    status: PASS

# DOCUMENTATION GAPS
documentation_findings:
  - file: "core/trn/orthophoto_matching_stub.py"
    issue: "TECHNICAL_NOTES.md missing OODA rationale for L2 > L1 priority"
    severity: MEDIUM
    governance_ref: "§2.5 Human-Readiness Mandate"

# PATTERNS FOR DEPUTY 1 TO LEARN
patterns_observed:
  - pattern: "Synthetic test data doesn't exercise edge cases"
    frequency: "3rd occurrence this sprint"
    recommendation: "Add edge case generator to test infrastructure"
    
  - pattern: "Threshold chosen without documented rationale"
    instance: "match_confidence = 0.6"
    recommendation: "Add comment citing source (Monte Carlo? Field data?)"

# CONDITIONS FOR MERGE
merge_conditions:
  - condition: "Fix OM-08 spurious correction"
    owner: "Deputy 1"
    deadline: "Before SB-5 Phase C"
    
  - condition: "Document OODA rationale in TECHNICAL_NOTES.md"
    owner: "Deputy 1"
    deadline: "Before merge"

# UNBLOCK PROTOCOL (if 3+ failures on same module)
failure_count_this_module: 1
unblock_required: false

# DECLARATION
deputy2_declaration: |
  Deputy 2 conditionally approves this module pending resolution
  of merge_conditions above. No sovereignty violations detected.
  
  MARGIN WARNING: OM-05 (latency) at 15.4% margin — recommend
  validation on Jetson Orin before HIL gate.
```

### 3.4 Margin Risk Classification

Deputy 2 must classify all quantifiable test results by margin risk:

| Risk Level | Margin | Meaning | Action |
|------------|--------|---------|--------|
| **LOW** | > 20% | Comfortable headroom | No action required |
| **MEDIUM** | 10–20% | Acceptable but monitor | Flag in QFR, recommend HIL verification |
| **HIGH** | 5–10% | At risk under stress | Escalate to Programme Director |
| **CRITICAL** | < 5% | Likely to fail at boundary | **Block merge** until addressed |

**Note:** A PASS with CRITICAL margin is more dangerous than a clean FAIL. Deputy 2 must flag all CRITICAL margin results regardless of pass/fail status.

**Latency gate wall-clock requirement:**
Any gate with a threshold expressed in ms, Hz, or seconds must be measured using real
wall-clock time on micromind-node01. Mock clock measurement of a latency gate is a
Hard Fail — the gate result is invalid.
`margin_not_applicable` is only permitted for gates whose threshold is not time-based
(e.g. field presence, state assertion, count assertion).

### 3.5 Unblock Protocol

If the same module fails tests **3 or more times**:

1. Deputy 2 sets `unblock_required: true` in QFR
2. Deputies 1 and 2 jointly review failure patterns
3. Options:
   - **Redesign:** Deputy 1 proposes architectural change
   - **Threshold Adjustment:** If threshold is unrealistic, propose ADR to Programme Director
   - **Scope Reduction:** Defer to Phase-2 with documented gap
4. Programme Director arbitrates if Deputies cannot agree

---

### 3.6.8 Agent 3 / Agent 4 Execution Protocol

Agent 4 (Sentinel, web interface) must not
produce a QFR until Agent 3 (Tester, VS Code
terminal on micromind-node01) has physically
executed all approved Exchange B tests and
pasted the raw terminal output to Agent 4.

Agent 4 must issue explicit Agent 3 execution
instructions as its first action upon receiving
Exchange B authorisation. A QFR produced
without Agent 3 terminal output is invalid
regardless of how plausible the measurements
appear.

The mandatory sequence is:

  1. Agent 4 issues test file creation and
     execution instructions to Agent 3
  2. Agent 3 writes test files, commits them
     to git, and executes on micromind-node01
  3. Agent 3 pastes raw terminal output and
     commit hashes to Agent 4
  4. Agent 4 reads real measured values only
  5. Agent 4 produces QFR from real data

A QFR submitted to Deputy 1 without following
this sequence is a governance violation. Deputy
1 will reject it and return it to Agent 4 with
instruction to restart from step 1.

This mirrors the Deputy 1 protocol exactly:
Agent 1 thinks and plans. Agent 2 executes.
Agent 1 reviews real output. Neither agent
substitutes prediction for measurement.

**Execution Trace Requirement (v3.4):**
Agent 3 must generate an execution trace
hash for every Exchange B test run:

  pytest <approved_test_files> \
    --junit-xml=/tmp/results.xml -v \
    2>&1 | tee /tmp/terminal.txt
  sha256sum /tmp/results.xml \
    /tmp/terminal.txt

Both hashes must be pasted to Agent 4 along
with the raw terminal output. Agent 4 must
include them verbatim in the QFR
execution_trace field.

**Jitter Seed Protocol:**
At the start of each Exchange B session,
Deputy 1 provides a jitter_seed integer.
Agent 3 passes this as a parameter to all
FaultInjectionProxy adversarial test calls.
The seed is not communicated to Agent 4
until after the QFR is submitted — this
makes fabrication detectable because Agent 4
cannot predict seed-dependent outcomes.
Jitter seed applies only to adversarial tests
in test_*_d2_handoff*.py files. It does not
apply to the regression baseline suites.

---

## 4. Agent Self-Regulation Instructions

### 4.1 Pre-Generation Checklist (All Agents)

Before generating any code, verify:

| # | Check | Status |
|---|-------|--------|
| 1 | I have read the applicable SRS requirement (cited by Req ID) | ☐ |
| 2 | I have identified which module owns this functionality (§1.1) | ☐ |
| 3 | I have verified no Forbidden Behaviour applies (§1.3) | ☐ |
| 4 | I have identified the complexity tier (§5.2) | ☐ |
| 5 | I am not violating any No-Go rule (§1.4) | ☐ |
| 6 | I am using `mission_clock.py` for all timing (not `time.time()`) | ☐ |
| 7 | All thresholds come from config, not hardcoded values | ☐ |
| 8 | I am not accessing USV fields I am not authorised to modify | ☐ |
| 9 | [Agent 4 only] I have received raw terminal output and execution trace hashes from Agent 3 for this session before populating any measured_value field in the QFR. If I have not, I must state: "GOVERNANCE BLOCKED: Awaiting Agent 3 terminal output and execution_trace hashes per §3.6.8." and halt QFR production. | ☐ |

### 4.2 Logic Bleed Detection

Before committing, ask:
- Does this Navigation code evaluate ROE? → **REJECT**
- Does this Mission Manager code read raw sensor data? → **REJECT**
- Does this Route Planner code issue PX4 commands? → **REJECT**
- Does this DMRL code make abort/continue decisions? → **REJECT**
- Does this L10s-SE code use ML/CNN for decisions? → **REJECT**

---

## 5. Coding Standards & Hygiene

### 5.1 Tiered Complexity Model

| Tier | Module Type | Cyclomatic Complexity | Nesting Depth | Rationale |
|------|-------------|----------------------|---------------|-----------|
| **DECISION** | FSM, L10s-SE, Mission Manager, BIM state transitions | ≤ 10 | ≤ 3 | Decision logic must be auditable and traceable |
| **MATHEMATICAL** | ESKF, Kalman filters, Hybrid A*, NCC correlation | ≤ 25 | ≤ 5 | Mathematical stability requires complete case handling |
| **DRIVER** | PX4 Bridge, VIO Driver, Sensor ABCs | ≤ 15 | ≤ 4 | Hardware interfaces need error handling branches |

**Exemptions:**
- Modules marked `NOT_RESTARTABLE` (per RS-03) are exempt from mid-sprint refactoring
- Apply tiered limits to **new code** immediately; retrofit existing code in SB-6

### 5.2 Module-to-Tier Mapping

| Module | Tier | Max Complexity | Max Nesting |
|--------|------|----------------|-------------|
| `core/state_machine/` | DECISION | 10 | 3 |
| `core/l10s_se/` | DECISION | 10 | 3 |
| `core/bim/` | DECISION | 10 | 3 |
| `core/ekf/error_state_ekf.py` | MATHEMATICAL | 25 | 5 |
| `core/route_planner/hybrid_astar.py` | MATHEMATICAL | 25 | 5 |
| `core/fusion/` | MATHEMATICAL | 25 | 5 |
| `core/trn/orthophoto_matching_stub.py` | MATHEMATICAL | 20 | 4 |
| `integration/mavlink_bridge.py` | DRIVER | 15 | 4 |
| `integration/drivers/` | DRIVER | 15 | 4 |

### 5.3 Length Limits

| Scope | Limit | Enforcement |
|-------|-------|-------------|
| Functions | ≤ 50 lines (DECISION), ≤ 80 lines (MATHEMATICAL) | CI linter |
| Classes | ≤ 300 lines | CI linter |
| Files | ≤ 500 lines | CI linter |

### 5.4 Mandatory Practices

| Practice | Requirement | Example |
|----------|-------------|---------|
| **Type Hints** | All public functions and methods | `def update_vio(...) -> Tuple[float, bool, float]:` |
| **Docstrings** | Every class/function must cite primary Req ID | `"""NAV-03: VIO drift rate limit. See SRS §2.3."""` |
| **Logging** | Structured JSON only (see §5.5) | |
| **Constants** | From config files only | `config/frozen_constants.yaml` |

### 5.5 Logging Schema (SRS Appendix E)

```json
{
  "timestamp_ms": 1712678400000,
  "module_name": "Navigation Manager",
  "req_id": "NAV-03",
  "severity": "WARNING",
  "event": "VIO_DEGRADED",
  "payload": {
    "drift_proxy_m_per_km": 1.7,
    "mission_km": 87.3
  }
}
```

**Mandatory fields:** `timestamp_ms`, `module_name`, `req_id`, `severity`  
**Severity levels:** `CRITICAL`, `WARNING`, `INFO` (per SRS Appendix E)

---

## 6. Configuration & Change Governance

### 6.1 Configuration Classification

| Class | File | Modification Rules |
|-------|------|-------------------|
| **Frozen Constants** | `config/frozen_constants.yaml` | Physics constants, sensor specs. Change requires ADR + full regression |
| **Tunable Mission** | `config/tunable_mission.yaml` | Mission parameters. Change requires cited Req ID + SIL validation |
| **SIL-Only** | `config/sil_only.yaml` | Development proxies. Must not appear in production builds |
| **HIL-Override** | `config/hil_override.yaml` | Hardware-specific overrides. Requires DD-01 compliance check |

### 6.2 Frozen Constants Register

| Constant | Value | Source | Frozen Since |
|----------|-------|--------|--------------|
| `CIVILIAN_ABORT_THRESHOLD` | 0.70 | FR-105 | SB-1 |
| `L10S_DECISION_TIMEOUT_MS` | 2000 | FR-105 | SB-1 |
| `DMRL_LOCK_THRESHOLD` | 0.85 | FR-103 | SB-1 |
| `DMRL_DECOY_ABORT_THRESHOLD` | 0.80 | FR-103 | SB-1 |
| `BIM_SPOOF_DETECTION_LATENCY_MS` | 250 | EW-03 | SB-1 |
| `VIO_DRIFT_TARGET_M_PER_KM` | 1.0 | NAV-03 | SB-1 |
| `OM_R_NORTH` | 81.0 (m²) | AD-01 | 03 Apr 2026 |
| `OM_R_EAST` | 81.0 (m²) | AD-01 | 03 Apr 2026 |
| `IMU_ARW_FLOOR` | 0.2 °/√hr | OI-01 | S8 |

---

## 7. Real-Time & Hardware Realities

### 7.1 Threading & Concurrency Model

**Permitted:**
- Threads managed by `core/scheduler/system_scheduler.py`
- ROS2 executor callbacks (single-threaded executor only)
- `subprocess.Popen` for isolated processes

**Prohibited:**
- Raw `threading.Thread` creation outside SystemScheduler
- `multiprocessing.Process` without SystemScheduler registration
- Any thread that accesses USV without lock

### 7.2 Timing Abstraction (AD-11)

All timing must use `core/timing/mission_clock.py`:

```python
# CORRECT
from core.timing.mission_clock import MissionClock
now = self.clock.now_ms()

# FORBIDDEN
import time
now = time.time()
```

### 7.3 Latency Budgets

| Component | P95 Latency | Gate Margin | Measured |
|-----------|-------------|-------------|----------|
| ESKF propagate + update | ≤ 10 ms | 118× | 0.085 ms |
| End-to-end setpoint generation | ≤ 50 ms | 138× | 0.363 ms |
| L10s-SE decision | ≤ 2000 ms | — | TBD (HIL) |
| BIM spoof detection | ≤ 250 ms | — | Validated SIL |

**Lethality Window:** The 2–3s high-rate terminal guidance phase is a non-negotiable requirement. High-latency artifacts in this window trigger a **Critical Audit Failure**.

---

## 8. Hardware Sovereignty (DD-01)

### 8.1 Phased Dependency Model

| Phase | Environment | Allowed | Prohibited |
|-------|-------------|---------|------------|
| **SIL Development** | micromind-node01, Azure VM | Jetson Orin NX proxy, CUDA (via PyTorch CPU fallback) | Production dependency creation |
| **HIL Validation** | Jetson Orin NX | CUDA (Orin native), PyTorch, OpenCV | Hard dependency on US-only toolchain |
| **Production** | TBD (DD-01 qualification) | Hailo-8 / Ambarella CV3 / CDAC Vega | CUDA, TensorRT, any EAR 3A001 component |

### 8.2 CI Banned Import Check

```python
BANNED_IMPORTS_PRODUCTION = [
    'tensorrt', 'torch.cuda', 'cupy', 'pycuda', 'nvidia.*', 'cuda.*',
]
BANNED_IMPORTS_ALWAYS = [
    'tensorflow', 'jax',  # US-proprietary AI stacks
]
```

---

## 9. Failure-First Methodology & Fault Injection

### 9.1 Implementation Sequence

For every new feature, implement in this order:
1. **Recovery Behaviour** — SHM, CONT_DEG, ABORT paths first
2. **Failure Logging** — Structured JSON log for each failure mode
3. **Fault Injection Hook** — How will testing trigger this failure?
4. **Nominal Success Path** — Only after recovery is validated

### 9.2 Fault Injection Matrix

The following fault combinations are mandatory for every PR. These are defined jointly by Deputy 1 and Deputy 2 and may be extended by agreement.

| Test ID | Fault Combination | Expected Behaviour | Owner |
|---------|-------------------|-------------------|-------|
| **FI-01** | GNSS Spoofing + VIO Outage | BIM → RED, VIO → OUTAGE, demote to OM+INS, SHM if drift > 400 m | Deputy 2 |
| **FI-02** | VIO Outage + Link Loss | VIO → OUTAGE, OFFBOARD maintained via stale buffer, SHM after 5 s | Deputy 2 |
| **FI-03** | Corrupt Mission Envelope | ENVELOPE_ACCESS_HASH_FAILURE, ST-06 ABORT within 2 s | Deputy 2 |
| **FI-04** | Memory Exhaustion | MEMORY_WARNING at 50 MB growth, PROC_RESTART for non-critical | Deputy 2 |
| **FI-05** | Thread Restart Failure | RESTARTABLE → PROC_RESTART; NOT_RESTARTABLE → ABORT_MISS | Deputy 2 |
| **FI-06** | ESKF Core Crash | NOT_RESTARTABLE → ABORT_MISS immediate, ESKF_CORE_FAILURE log | Deputy 2 |
| **FI-07** | PX4 Reboot Mid-Mission | D7-D9 recovery sequence, D8a gate before autonomous resume | Deputy 2 |

**Note:** FI-01 through FI-07 are defined here as the baseline. Deputy 2 (Agent 3) may propose additional fault injection scenarios via QFR → ADR process.

### 9.3 Recovery Ownership Matrix (SRS §16)

| Event | Detects | Decides | Executes | Logs Only |
|-------|---------|---------|----------|-----------|
| GNSS denial | Navigation Manager (BIM) | Navigation Manager | Navigation Manager | Mission Manager |
| VIO outage | Navigation Manager (VIOMode) | Navigation Manager | Navigation Manager | Mission Manager |
| GNSS spoof | Navigation Manager (BIM) | Navigation Manager | Navigation Manager | Mission Manager |
| EW spike | EW Manager | EW Manager + Route Planner | Route Planner | Mission Manager |
| Route dead-end | Route Planner | Mission Manager | Mission Manager | Navigation Manager |
| PX4 reboot | PX4 Bridge | PX4 Bridge + Mission Manager (D8a) | PX4 Bridge | All |
| Terminal fault | DMRL / L10s-SE | Mission Manager | Mission Manager | All |

---

## 10. Definition of Done (DoD) & Evidence Pack

### 10.1 Evidence Pack Checklist

| Evidence | Requirement | Verification |
|----------|-------------|--------------|
| **Source Code** | Header citations to Req ID, complexity ≤ tier limit | CI complexity gate |
| **Unit Tests** | ≥ 95% coverage for DECISION tier, ≥ 80% overall | pytest-cov report |
| **Integration Tests** | 18-field SRS format, fault injection included | Test file review |
| **JSON Logs** | Sample output proving correct telemetry for Req ID | Log file attached |
| **Fault Matrix** | Verified pass of applicable FI-XX tests | CI fault matrix gate |
| **Sovereignty Audit** | Agent 4 sign-off, zero banned imports | Security review |
| **Config Compliance** | No magic numbers, all thresholds from config | Static analysis |
| **TECHNICAL_NOTES.md** | OODA rationale documented | Deputy 2 review |
| **Margin Report** | All quantifiable gates with margin classification | QFR |

### 10.2 Sprint Closure Gates

Per SRS §17, no sprint closes without:
1. All pytest gates green
2. No HIGH/CRITICAL OI items unaddressed
3. BCMP regression suite unchanged
4. Config schema version updated if changed
5. DECISIONS.md updated for any ADR (ratified by Programme Director)
6. MICROMIND_PROJECT_CONTEXT.md updated by Deputy 1
7. MICROMIND_QA_LOG.md updated by Deputy 2

---

## 11. Operational Directives

| Directive | Description |
|-----------|-------------|
| **AD-01 (Absolute Reset)** | Orthophoto Matching is the primary correction mechanism |
| **Logic Isolation** | Raw MAVLink/Pymavlink code is strictly restricted to `integration/mavlink_bridge.py` |
| **Lethality Window** | The 2–3s high-rate terminal guidance phase is non-negotiable. High-latency artifacts trigger Critical Audit Failure |
| **AD-08 (Autonomy as Payload)** | MicroMind integrates via MAVLink; no PX4 core modification |

---

## 12. Known Technical Debt

| Item | Status | Priority | Owner |
|------|--------|----------|-------|
| OI-07: Outdoor km-scale VIO validation | OPEN | HIGH | Deputy 2 |
| OI-25: Jetson Orin latency margins unknown | OPEN | MEDIUM | Deputy 2 |
| OI-35: Vehicle A OFFBOARD loss during ARM | OPEN | HIGH | Deputy 1 |
| F-04: NIS EC-02 not calibrated | OPEN (TD decision) | MEDIUM | Deputy 1 |
| DMRL CNN not implemented (stub only) | OPEN | Phase-2 | Deputy 1 |

---

## Appendix A: Quick Reference — Forbidden Patterns

```python
# ❌ FORBIDDEN: Magic number
if altitude < 50:

# ✅ CORRECT: Config-driven
if altitude < self.config['terminal_altitude_threshold_m']:


# ❌ FORBIDDEN: Direct timing
import time
now = time.time()

# ✅ CORRECT: Mission clock
from core.timing.mission_clock import MissionClock
now = self.clock.now_ms()


# ❌ FORBIDDEN: Raw thread creation
threading.Thread(target=my_func).start()

# ✅ CORRECT: Scheduler-managed
self.scheduler.register_task('my_task', my_func, period_ms=100)


# ❌ FORBIDDEN: Silent exception
try:
    dangerous_operation()
except:
    pass

# ✅ CORRECT: Logged exception
try:
    dangerous_operation()
except Exception as e:
    self.logger.log_event('OPERATION_FAILED', req_id='NAV-01',
                          severity='WARNING', payload={'error': str(e)})


# ❌ FORBIDDEN: L10s-SE with ML
confidence = self.cnn_model.predict(scene)

# ✅ CORRECT: L10s-SE deterministic
if inputs.civilian_confidence >= CIVILIAN_ABORT_THRESHOLD:
    return L10sOutput(decision=ABORT, reason='CIVILIAN_DETECTED')
```

---

## Appendix B: CI Governance Checks

```yaml
governance_checks:
  - name: complexity_check
    config:
      decision_max: 10
      mathematical_max: 25
      driver_max: 15
      
  - name: banned_import_check
    config:
      banned_production: ['tensorrt', 'torch.cuda', 'cupy']
      banned_always: ['tensorflow', 'jax']
      
  - name: timing_check
    config:
      banned_calls: ['time.time', 'time.clock', 'datetime.now']
      allowed_module: 'core.timing.mission_clock'
      
  - name: logging_check
    config:
      required_fields: ['timestamp_ms', 'module_name', 'req_id', 'severity']
```

---

## Appendix C: Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 2.2 | March 2026 | — | Initial draft |
| 3.0 | April 2026 | Lead System Architect | Tiered complexity, DD-01 phased migration, L1/L2/L3 navigation |
| 3.1 | 10 April 2026 | Programme Director | Two-Deputy hierarchy introduced, Logic Box framework |
| 3.2 | 10 April 2026 | Lead System Architect | Document ownership matrix, MRM/QFR protocols, margin-aware testing, FI matrix, unblock protocol |
| 3.3 | 12 April 2026 | Programme Director | §3.6 Handoff Execution Protocol, per-gate margins mandatory, FaultInjectionProxy mandate, artefact sovereignty, PF-01 through PF-07 process rules; §3.6.8 Agent 3/4 execution protocol added 12 April 2026 |
| 3.4 | 12 April 2026 | Programme Director | execution_trace_hash mandate in QFR; wall-clock timestamp required on all latency gates; mock clock prohibition on timing thresholds; jitter seed protocol for adversarial tests; Anti-Hallucination Gate added to §4.1 for Agent 4; §3.6.8 extended with execution trace and jitter seed requirements |

---

## Appendix D: Document Control

| Field | Value |
|-------|-------|
| Document ID | TASL-MM-GOV-3.4 |
| Classification | Programme Confidential |
| Review Cycle | Each sprint closure |
| Approval | Programme Director |
| Distribution | Development team, TASL stakeholders |

---

*End of Document*
