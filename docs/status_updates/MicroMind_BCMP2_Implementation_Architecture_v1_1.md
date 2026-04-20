**MicroMind**

**BCMP-2 Implementation Architecture**

*Comparative Demonstration Framework*

| **Document Status** | Implementation Plan — Pre-Sprint Baseline |
| --- | --- |
| **Version** | 1.1 |
| **Date** | 29 March 2026 |
| **Classification** | RESTRICTED |
| **Programme** | MicroMind / NanoCorteX — DISC 14 / TASL |
| **Preceded by** | Phase 1 Closure Report · Pre-HIL DR-6 |
| **Constraints** | C-1 IMU Parity · C-2 Drift Envelope · C-3 Synthetic Terrain · C-4 Disturbance Parity |

# **1  Purpose and Governing Philosophy**

BCMP-2 is the formal internal acceptance regime for MicroMind following Phase 1 completion. It is simultaneously the primary comparative demonstration framework for TASL leadership, OEM reviewers, DISC evaluators, and future partner discussions.

**BCMP-1 proved that MicroMind works. BCMP-2 must prove that the baseline fails without it.**

## **1.1  The Central Question**

Every BCMP-2 scenario is structured to answer one question:

***“What mission succeeds with MicroMind that fails without it?”***

The objective is not to explain correction algorithms. The objective is to make the operational consequence of removing MicroMind visible and unambiguous to a non-technical leadership audience.

## **1.2  Dual-Track Comparison Model**

Every scenario runs two vehicles simultaneously from the same seed, same environment, and same mission geometry. They diverge only in their navigation and enforcement logic.

|  | **Vehicle A — Baseline** | **Vehicle B — MicroMind** |
| --- | --- | --- |
| **IMU** | STIM300 / ADIS class — identical | STIM300 / ADIS class — identical |
| **Navigation** | INS + GNSS until denial, then INS-only. No TRN, no VIO, no correction. | INS + GNSS + TRN + VIO + BIM + mission enforcement. |
| **Environment** | Identical — shared seed, wind, terrain, GNSS denial schedule, disturbance profile. | Identical — shared seed, wind, terrain, GNSS denial schedule, disturbance profile. |
| **Enforcement** | None. Terminal action proceeds regardless of navigation state. | Full. Terminal action gated by confidence, drift threshold, and NOMINAL state requirement. |
| **Expected outcome** | Mission fails. Drift accumulates. Corridor violated. Terminal action unsafe or aborted. | Mission succeeds. Drift bounded. Corridor maintained. Terminal action deferred until NOMINAL, then authorised. |

# **2  Fixed Constraints**

These four constraints are locked before any sprint begins. They govern the entire BCMP-2 programme and may not be modified without TD approval and a version increment to this document.

## **C-1 — IMU Parity**

Both vehicles use identical IMU class, noise profile, seed, wind model, terrain, and mission geometry. The only removed element in Vehicle A is the MicroMind correction and enforcement stack. This ensures the comparison delta is attributable solely to MicroMind.

*Implementation note: baseline_nav_sim.py must import and use the same IMU noise model and parameters as the MicroMind nav loop. No separate degraded IMU model. No additional noise. Same class, same seed, same profile.*

## **C-2 — Vehicle A Drift Envelope Pre-Calculation**

Vehicle A expected drift envelopes must be derived analytically from the existing ALS-250 STIM300/ADIS characterisation data before baseline_nav_sim.py is written. These envelopes become the acceptance floor and ceiling for AT-2 and later gates.

A BCMP-2 run is rejected if Vehicle A drifts materially below the expected floor (suggesting the baseline is incorrectly gentle, weakening the comparison) or materially above the expected ceiling (suggesting an unrealistic catastrophic failure). Both directions invalidate the demonstration.

**Provisional phase-boundary reference ranges (to be confirmed by ALS-250 STIM300/ADIS analytical derivation in SB-1):**

| **Phase Boundary** | **Expected Vehicle A Drift (Provisional)** | **Note** |
| --- | --- | --- |
| **km 60 — end P2** | 50–150 m lateral | *After 30 km INS-only in valley corridor. TRN suppressed. VIO unavailable to Vehicle A.* |
| **km 100 — end P3** | 300–800 m lateral | *70 km cumulative INS-only. Flat terrain removes TRN opportunity. Drift compounds at STIM300/ADIS class rate.* |
| **km 120 — end P4** | 800–1800 m lateral | *Corridor width typically 500–1000 m. Vehicle A expected to breach corridor before this point. Terminal approach geometry unsafe.* |

*These ranges are provisional. bcmp2_drift_envelopes.py must derive exact floor and ceiling from ALS-250 characterisation data before baseline_nav_sim.py is written. The table provides order-of-magnitude anchoring for SB-1 planning and reviewer expectations.*

*The first deliverable of SB-1 is a committed reference constants file (bcmp2_drift_envelopes.py) and a supporting analytical note. The runner cannot be acceptance-tested until those values exist.*

## **C-3 — Committed Synthetic Terrain**

All BCMP-2 terrain is committed synthetic, seeded from SRTM-style mountain profiles for credible ridge spacing, slope, and elevation variance. No network dependencies. No external data failures. Reproducible on any machine without downloads.

The terrain generator must expose and document the following parameters so that a reviewer can verify the terrain is not cherry-picked:

- elevation_range_m — min/max elevation band

- ridge_spacing_m — mean spacing between ridge peaks

- ridge_orientation_deg — dominant ridge bearing

- valley_width_m — characteristic valley floor width

- local_slope_variance — RMS slope variation

- terrain_seed — deterministic seed for exact reproduction

- terrain_roughness_factor — high-frequency texture amplitude

*Optional real DEM tile loading is deferred as an enhancement layer, to be added only if a specific TASL corridor or customer requirement makes it necessary. It must not appear in the default execution path.*

## **C-4 — Disturbance Parity**

Both vehicles must experience identical external disturbances. The disturbance schedule is generated once from the shared seed before the run begins and passed as a shared input to both vehicle tracks.

**Covered by C-4:**

- Wind profile and gust events

- Turbulence injection schedule

- GNSS denial timing and zone geometry

- VIO outage timing and duration

- RADALT degradation timing

- EO clutter and false target timing

- Mission corridor geometry and target position

The disturbance schedule is serialised into each run's KPI JSON at the top level (not inside vehicle_a or vehicle_b) so that a reviewer can independently verify both vehicles received identical inputs.

*Without C-4, a reviewer can argue that Vehicle B succeeded because it had an easier environment. With C-4, the comparison is a controlled ablation. The only independent variable is MicroMind.*

# **3  Mission Definition**

## **3.1  Overview**

BCMP-2 is a single 150 km equivalent mission divided into five operational phases. The mission is deterministic and repeatable from a fixed seed. Three canonical seeds are defined:

- Seed 42 — nominal reference run

- Seed 101 — alternate weather and sensor noise profile

- Seed 303 — degraded and stress profile

## **3.2  Five-Phase Mission Profile**

The phases are designed so that the first phase (P1) looks identical for both vehicles, and the divergence becomes progressively more dramatic from P2 onward. By P5 the outcome separation is decisive.

| **Phase** | **Distance** | **Terrain** | **Vehicle B Nav** | **Vehicle A Fate** | **Story Beat** |
| --- | --- | --- | --- | --- | --- |
| **P1** | 0–30 km | Mountain ingress—synthetic DEM | TRN-primary | GNSS available. Both vehicles identical. | *Calm. No separation yet.* |
| **P2** | 30–60 km | Valley corridor—simplified synthetic | VIO-primary; TRN suppressed | GNSS denied. INS-only. Drift begins. | *Divergence visible on route map.* |
| **P3** | 60–100 km | Plains—flat synthetic | VIO-primary; long-range drift stability | Drift compounds. Corridor breach risk. | *Dramatic visual separation.* |
| **P4** | 100–120 km | Industrial clutter—simplified synthetic | VIO+EO assist; sensor ambiguity handling | Large lateral error. Corridor violation. | *Vehicle A loses corridor.* |
| **P5** | 120–150 km | Terminal—high-detail synthetic zone | Mission enforcement active; L10s-SE + DMRL | Unsafe terminal attempt or forced abort. | *Decisive outcome separation.* |

## **3.3  Execution Modes**

| **Mode** | **Speed** | **Duration** | **Target Audience** |
| --- | --- | --- | --- |
| **Executive replay** | 30–60× | 2–3 min | TASL / DISC leadership. Starts at P2 GNSS denial. P1 skipped (no divergence). |
| **Technical replay** | 8–12× | 8–10 min | OEM technical leads. Full mission. All panels active. |
| **Side-by-side synchronized** | Any | Any | Dual-window mode: Vehicle A only (left) and Vehicle B only (right), time-locked. Preferred for P3/P4 where overlaid paths become cluttered. Available in executive and technical speeds. |
| **High-fidelity** | 1–2× | Near real-time | Internal review. Live SITL mode. Operator fault injection enabled. |
| **Overnight stress** | 1× + logging | Unlimited | AT-6 endurance. Memory and queue growth monitoring. |

*The overlaid comparison (single route panel, both paths) remains the default. Side-by-side mode is an option flag in bcmp2_replay.py: --layout overlay | sidebyside. In side-by-side mode both windows share the same timeline cursor and event log, so fault injection events remain synchronised.*

# **4  Architecture Overview**

## **4.1  Frozen Baseline**

The following modules are frozen. Zero modifications permitted. Any change requires TD approval and S-NEP-09 regression re-run.

- core/ekf/error_state_ekf.py

- core/fusion/vio_mode.py

- core/fusion/frame_utils.py

- core/bim/bim.py

- scenarios/bcmp1/bcmp1_runner.py — enforcement blocks E-1..E-5

- All 332 SIL gates — must remain green after every BCMP-2 addition

## **4.2  New File Ownership Plan**

All new BCMP-2 files are additive. No existing file is modified.

| **File** | **Sprint** | **Responsibility** |
| --- | --- | --- |
| scenarios/bcmp2/bcmp2_drift_envelopes.py | **SB-1** | Pre-calculated Vehicle A drift floor/ceiling per phase boundary. First SB-1 deliverable. |
| scenarios/bcmp2/bcmp2_terrain_gen.py | **SB-1** | Synthetic terrain generator. Exposes all 7 C-3 parameters. Deterministic from seed. |
| scenarios/bcmp2/bcmp2_scenario.py | **SB-1** | Five-phase mission definition. Disturbance schedule generator. Shared by both tracks. |
| scenarios/bcmp2/baseline_nav_sim.py | **SB-1** | Vehicle A INS+GNSS→INS-only simulator. Uses same IMU class as Vehicle B. No correction stack. |
| scenarios/bcmp2/bcmp2_runner.py | **SB-1** | Dual-track orchestrator. Runs both vehicles per tick. Produces dual-track KPI JSON. |
| fault_injection/fault_manager.py | **SB-2** | Thread-safe fault state registry. Shared by GUI callbacks and scripted schedule. |
| fault_injection/sensor_fault_proxy.py | **SB-2** | Wraps sensor outputs. Intercepts on fault signal. Transparent when no fault active. |
| fault_injection/nav_source_proxy.py | **SB-2** | Wraps nav source selection. Forces degradation modes. No core modifications. |
| scenarios/bcmp2/bcmp2_report.py | **SB-3** | JSON + HTML report generator. Business comparison block first. Extends bcmp1_report style. |
| dashboard/bcmp2_dashboard.py | **SB-4** | Plotly Dash 7-panel GUI. Live and replay modes. Fault injection panel. |
| dashboard/bcmp2_replay.py | **SB-4** | Replay driver. Feeds pre-recorded JSON to dashboard at target speed. Mode argument. |
| tests/test_bcmp2_at1.py | **SB-1** | AT-1: 5 km boot and regression check. |
| tests/test_bcmp2_at2.py | **SB-3** | AT-2: 150 km nominal dual-track pass criteria. |
| tests/test_bcmp2_at3_5.py | **SB-3** | AT-3 through AT-5: failure mission gates. |
| tests/test_bcmp2_at6.py | **SB-5** | AT-6: 3× repeatability on seeds 42/101/303. |
| run_bcmp2_tests.py | **SB-1** | Repo root test runner. Mirrors run_s5_tests.py pattern. |

## **4.3  Runner Architecture — Dual-Track Loop**

bcmp2_runner.py owns a single simulation loop. At each tick it advances both vehicles. The disturbance schedule and shared environment are generated once before the loop begins.

*A hardware_source field is included in the runner configuration to support future evolution without restructuring. It is not required for BCMP-2 SIL/SITL execution today.*

| **hardware_source value** | **Meaning** |
| --- | --- |
| **simulated** | Default. All sensors synthetic. IMU noise model from ALS-250 characterisation. Runs on any machine. |
| **SITL** | PX4 SITL + Gazebo active. Setpoints sent via MAVLink bridge. Used for AT-1 through AT-5. |
| **Jetson replay** | Pre-recorded sensor logs from Jetson Orin hardware replayed through the driver abstraction layer. Post-TASL use case. |
| **live sensor** | Real IMU, RADALT, and EO feeds via RealDriver implementations. HIL phase only. Post-TASL. |

| **bcmp2_runner.run_bcmp2(seed, mode, kpi_log_path)**   ├─ generate disturbance_schedule(seed)          # C-4: single schedule, shared   ├─ terrain = bcmp2_terrain_gen.generate(seed)  # C-3: committed synthetic   ├─ imu_model = IMUNoiseModel(seed)             # C-1: same class both vehicles   │   ├─ for tick in mission_ticks:   │     env = disturbance_schedule[tick]   │   │     # Vehicle A — INS+GNSS then INS-only (no correction)   │     state_a = baseline_nav_sim.step(state_a, imu_model, env)   │   │     # Vehicle B — Full MicroMind stack via proxies   │     gnss_obs  = sensor_fault_proxy.gnss(env.gnss_raw)   │     vio_frame = sensor_fault_proxy.vio(env.vio_raw)   │     state_b   = micromind_nav_loop.step(state_b, imu_model, gnss_obs, vio_frame, env)   │   │     kpi_log.record(tick, state_a, state_b, env, fault_manager.active_faults())   │ **  └─ output: { disturbance_schedule, vehicle_a, vehicle_b, comparison }** |
| --- |

# **5  Fault Injection Architecture**

## **5.1  Design Principle**

Fault injection uses wrapper/proxy layers only. No frozen core module is modified. The three proxy classes sit between the runner and the frozen core. When no fault is active they are transparent pass-throughs. When a fault is active they intercept the relevant data stream.

fault_manager.py is the single source of truth for all active fault states. It is a threading.Lock-protected singleton. The Dash GUI and the scripted injection schedule both write to it. The proxies read from it on every call. It owns the timestamped fault event log.

*Vehicle A does not use the proxy system. Vehicle A has no correction logic to suppress, so there is nothing to inject. It naturally degrades to INS-only at GNSS denial and never recovers.*

## **5.2  Proxy Responsibilities**

| **Proxy** | **Intercepts** | **Simulates** |
| --- | --- | --- |
| **sensor_fault_proxy.py** | GNSS output, VIO frame, RADALT reading, EO frame | GPS_LOSS, VIO_LOSS, RADALT_LOSS, EO_FREEZE, EO stale-frame injection |
| **nav_source_proxy.py** | TRN correction call, VIO update injection, RADALT altitude gate | TERRAIN_CONF_DROP, VIO suppression, IMU_JITTER timing perturbation |
| **fault_manager.py** | All fault state registration, GUI callbacks, scripted schedule | Fault activation, duration tracking, event log, multi-fault preset registry |

## **5.3  Fault Injection Catalogue**

| **ID** | **Injection** | **Phase** | **Proxy** | **Expected Vehicle B Behaviour** |
| --- | --- | --- | --- | --- |
| FI-01 | GNSS denied entire mission | All | sensor_fault_proxy | No mission abort. TRN+VIO maintain corridor. |
| FI-02 | 10s VIO outage during forward motion | P2 | sensor_fault_proxy | Drift grows. OUTAGE state. Precision suppressed. |
| FI-03 | 30s VIO outage during plains | P3 | sensor_fault_proxy | Larger drift. Spike alert on recovery. RESUMPTION chain. |
| FI-04 | RADALT lost below terminal altitude | P5 | sensor_fault_proxy | Conservative altitude handling. Terminal deferred. |
| FI-05 | EO feed frozen | P4/P5 | sensor_fault_proxy | Stale-frame detection. No crash. No lock claimed. |
| FI-06 | Non-monotonic IMU timestamp | Any | nav_source_proxy | IFM-01 fires. Loop continues. Event logged. |
| FI-07 | MAVLink disconnect 5s | Any | fault_manager | OFFBOARD retained or recovered. Heartbeat maintained. |
| FI-08 | Terrain confidence below threshold | P1/P2 | nav_source_proxy | TRN suppressed correctly. VIO takes over. |
| FI-09 | Combined VIO + GNSS outage | P3 | sensor_fault_proxy | INS-only propagation. Drift grows. No crash. |
| FI-10 | Combined VIO + RADALT in terminal | P5 | sensor_fault_proxy | Terminal precision blocked. Mission-layer suppression. |
| FI-11 | CPU overload / logger stress | Any | fault_manager | Setpoint loop remains ≥20 Hz. No queue growth. |
| FI-12 | Memory pressure | Any | fault_manager | No runaway RSS growth. |
| FI-13 | Module time skew | Any | nav_source_proxy | Deterministic ordering preserved. |
| FI-14 | EO false target / decoy | P5 | sensor_fault_proxy | L10s-SE suppressed. DMRL decoy reject fires. |
| FI-15 | Sensor reconnect after outage | Any | sensor_fault_proxy | Clean recovery chain. RESUMPTION→NOMINAL. |

## **5.4  Operator-Triggered Faults — GUI Panel**

The Dash GUI fault injection panel writes to fault_manager.py via Dash callbacks. Operator-triggered faults must produce identical Vehicle B behaviour to scripted faults. This is an AT-5 exit criterion.

**Panel controls:**

- Individual toggle buttons per sensor/subsystem with active state indicator

- Duration field in seconds

- Severity/intensity field where applicable

- Single-failure and multi-failure presets (see FI-09, FI-10 as default multi-fault presets)

- Live event log — timestamped, visible during run

- Numerical inputs: outage duration, noise multiplier, drift multiplier, telemetry delay ms, frame drop %, IMU jitter magnitude, terrain confidence level

# **6  GUI Structure — Plotly Dash**

Framework: Plotly Dash, browser-based, localhost. No installation beyond pip. Same dashboard used for live SITL and log replay — data source changes, layout does not.

Colour convention: Green = NOMINAL / MicroMind success. Amber = RESUMPTION / degraded. Red = OUTAGE / suppression / baseline failure. Blue = planned path. Orange = Vehicle A (baseline) path. White = Vehicle B (MicroMind) actual corrected path. Grey fill = INS-only prediction cone.

## **Panel 1 — Route Comparison (Primary Panel)**

The main visual. Largest panel. This is what leadership looks at. Everything else is supporting context.

- Blue line: planned corridor

- Orange line: Vehicle A (baseline) actual path

- White line: Vehicle B (MicroMind) actual path

- Grey shaded cone: drift uncertainty growing on Vehicle A

- Red zone: target area

- Dashed boundary: mission corridor walls

- Event markers: GNSS denial point, VIO outage events, terminal gate, phase boundaries

## **Panel 2 — 3D Flight View**

Vehicle B trajectory in 3D. Terrain surface for P1 mountain segment. Altitude gate markers for terminal phase. Rendered as static 3D Plotly figure from pre-computed geometry for executive demo mode. Gazebo live feed available in high-fidelity mode.

## **Panel 3 — Navigation Source and Sensor Health**

- Active nav source for Vehicle B: TRN / VIO / INS-only

- Sensor health indicators per sensor: green / amber / red

- BIM trust score live gauge

- VIO mode state: NOMINAL / OUTAGE / RESUMPTION

## **Panel 4 — Drift and Uncertainty**

- Vehicle A lateral drift vs corridor boundary — the key failure metric

- Vehicle B drift envelope (bounded)

- Side-by-side drift at phase transition boundaries

- INS-only prediction cone during OUTAGE

## **Panel 5 — Mission Integrity**

- Terminal action state: SUPPRESSED / DEFERRED / AUTHORISED

- Precision action count: blocked vs executed (both vehicles)

- L10s-SE gate status

- EO confidence gauge

- Decoy detection events

## **Panel 6 — Event Log**

Timestamped chronological list of all events: GNSS denial, VIO outage, fault injections, mode transitions, terminal gates. Colour-coded by severity. Vehicle label prefix for each entry. Shared log, both vehicles.

## **Panel 7 — Outcome Summary (Always Visible)**

Bottom bar. Always on screen. Updates live. Designed to be photographable and self-explanatory without narration.

| **BASELINE (Vehicle A)** | **MicroMind (Vehicle B)** |
| --- | --- |
| Terminal error: 1.8 km Corridor violation: km 112 Terminal action: UNSAFE Time in degraded state: N/A (no state awareness) **Mission result: FAILED** | Drift bounded: 80 m Corridor: maintained throughout Terminal action: DEFERRED → AUTHORISED Time in degraded state: X min Y sec (OUTAGE + RESUMPTION) **Mission result: SUCCEEDED** |

# **7  Acceptance Tests**

All acceptance tests run via run_bcmp2_tests.py at the repo root. The 332 existing SIL gates must remain green after every test addition. No gate may be modified to pass.

## **AT-1 — Boot and Regression Check (5 km)**

*Primary mode: SITL + Gazebo.*

- All modules load. No import errors.

- PX4 SITL arms. OFFBOARD maintained.

- Setpoint rate ≥20 Hz on Vehicle B.

- Dual-track JSON log produced. Both vehicle tracks populated.

- No NaN in either track.

- 332 SIL gates unchanged.

## **AT-2 — Nominal 150 km Dual-Track**

*Primary mode: SITL + Gazebo. Seed 42.*

- All five phases complete on Vehicle B. No mission abort.

- Vehicle A drift falls within C-2 envelope at each phase boundary. Both floor and ceiling enforced.

- Vehicle B drift remains within expected corrected envelope.

- Outcome summary: Vehicle B succeeded, Vehicle A failed.

- HTML report generated. Business comparison block present and correct.

- JSON KPI log complete. disturbance_schedule field present at top level.

- All mode transitions deterministic. OFFBOARD maintained.

## **AT-3 — Single-Failure Mission**

*Primary mode: SITL + Gazebo. One fault from FI catalogue per run.*

- Vehicle B continues mission without crash.

- Correct mode transition chain: NOMINAL → OUTAGE → RESUMPTION → NOMINAL.

- Precision actions suppressed where required by nav state.

- Recovery logic executes correctly. Event logged with timestamp.

## **AT-4 — Multi-Failure Mission**

*Primary mode: SITL + Gazebo. FI-09 and FI-10 as required combinations.*

- No invalid setpoints. No NaN propagation.

- No uncontrolled OFFBOARD drop.

- Safe degradation: mission-layer suppression fires correctly.

- Vehicle B survives. Vehicle A degraded state shown in comparison.

## **AT-5 — Terminal Integrity Mission**

*Primary mode: SITL + Gazebo. FI-10 and FI-14.*

- Terminal precision blocked during OUTAGE and RESUMPTION on Vehicle B.

- EO confidence gates enforced.

- Drift threshold gates enforced.

- Terminal action resumes only after NOMINAL state and confidence recovery.

- Vehicle A attempts unsafe terminal approach OR is forced to abort — outcome recorded in comparison block.

- Operator-triggered fault via Dash panel produces identical Vehicle B behaviour to scripted fault (same mode chain, same suppression).

## **AT-6 — Repeatability and Endurance**

*Primary mode: accelerated SIL. Seeds 42, 101, 303.*

- Three consecutive runs produce identical phase transition chains.

- Vehicle A drift variance within tolerance across seeds.

- Vehicle B drift variance within tolerance across seeds.

- No memory leak over 4-hour stress run.

- No queue growth over sustained run.

- No change in deterministic mode transition chain between runs.

# **8  Report Outputs**

## **8.1  Machine-Readable JSON**

Per-run JSON with three top-level blocks:

- disturbance_schedule — the full shared disturbance schedule (C-4 verification record)

- vehicle_a — INS-only navigation metrics, drift per phase, terminal outcome

- vehicle_b — MicroMind navigation metrics, mode transitions, enforcement events, terminal outcome

- comparison — the seven business comparison metrics: terminal error delta, corridor violations, precision actions blocked/executed, mission outcome, max drift, phase completion status

## **8.2  HTML Summary Report**

Extends bcmp1_report.py pattern. The first visible element when the report opens is the outcome summary block. No technical metric appears before the business comparison.

**Report structure:**

- Business comparison block (human-readable, large font, colour-coded)

- Route map — static Plotly figure with dual-track paths embedded

- Drift chart per phase — dual-track on shared axis

- Navigation source timeline (Vehicle B)

- Event log — full timestamped list with fault injection events

- Technical metrics tables — all AT criteria evidence

## **8.3  Business Comparison Block Format**

This block heads every HTML report and is the top panel of the Dash GUI. Exact format:

| **BCMP-2 Run — Seed 42 — 29 March 2026** **WITHOUT MicroMind (Baseline Vehicle):**   GNSS loss at km 30 caused INS-only propagation   No terrain or visual correction available — drift accumulated unchecked   Drift exceeded corridor threshold by km 112 (estimated 900 m lateral error)   No mission-layer awareness of navigation state   Unsafe terminal approach attempted despite degraded position confidence **  Mission: FAILED** **WITH MicroMind:**   GNSS loss at km 30 triggered TRN→VIO transition (automatic)   VIO outage at km 48 detected within 250 ms — OUTAGE state entered   INS-only propagation held for X min Y sec — drift bounded to 82 m   VIO recovery at km 61 — RESUMPTION chain, NOMINAL restored   Terminal action deferred during OUTAGE, authorised only after NOMINAL recovery **  Mission: SUCCEEDED** |
| --- |

# **9  Sprint Breakdown**

## **SB-1 — Dual-Track Foundation**

Scope: derive Vehicle A envelopes, build terrain generator, build scenario, build baseline sim, build runner, pass AT-1.

**Deliverables in required sequence:**

- bcmp2_drift_envelopes.py — Vehicle A floor/ceiling from ALS-250 STIM300/ADIS characterisation. Committed before any runner code is written.

- bcmp2_terrain_gen.py — synthetic mountain terrain. All 7 C-3 parameters documented and exposed.

- bcmp2_scenario.py — five-phase definition plus shared disturbance schedule generator.

- baseline_nav_sim.py — Vehicle A INS→INS-only. Validated against C-2 envelope before proceeding.

- bcmp2_runner.py — dual-track loop. Both vehicles produce logs.

- test_bcmp2_at1.py and run_bcmp2_tests.py — AT-1 gates pass.

*Exit condition: run_bcmp2_tests.py AT-1 passes. Dual-track JSON produced with both tracks populated. Vehicle A drift matches C-2 envelope. 332 SIL gates green.*

## **SB-2 — Fault Injection Infrastructure**

Scope: build the three proxy layers, wire into Vehicle B track, validate scripted single-fault injection.

**Deliverables:**

- fault_manager.py — thread-safe singleton. threading.Lock on all reads/writes.

- sensor_fault_proxy.py — GNSS, VIO, RADALT, EO intercept. Transparent when no fault active.

- nav_source_proxy.py — TRN suppression, VIO suppression, IMU jitter.

- Scripted injection of FI-01, FI-02, FI-05 as initial validation set.

*Exit condition: three scripted fault runs produce correct Vehicle B mode transitions. Fault event log populated. No proxy call modifies any frozen core file.*

## **SB-3 — Full Mission and Reports**

Scope: complete five-phase terrain, full FI catalogue, AT-2 through AT-5 tests, report generation.

**Deliverables:**

- Full five-phase terrain profile integrated into bcmp2_scenario.py.

- Remaining FI-01 through FI-15 scripted in test fixtures.

- test_bcmp2_at2.py and test_bcmp2_at3_5.py.

- bcmp2_report.py — JSON + HTML. Business comparison block first.

*Exit condition: AT-2 nominal run produces expected comparative outcome. Vehicle A exceeds corridor by P4. Vehicle B corridor maintained. HTML report business block correct.*

## **SB-4 — Dash GUI and Replay**

Scope: seven-panel Dash GUI, four replay modes, operator fault injection panel.

**Deliverables:**

- bcmp2_dashboard.py — seven-panel layout. Panel 7 outcome summary always visible.

- bcmp2_replay.py — executive / technical / high-fidelity / overnight modes.

- Fault injection panel wired to fault_manager.py via Dash callbacks.

- Executive demo sequence validated: 2–3 min, starts at P2 denial event.

*Exit condition: executive replay runs 2–3 min correctly. Panel 7 updates live. Operator-triggered fault produces same Vehicle B behaviour as scripted fault (AT-5 operator gate).*

## **SB-5 — Repeatability and Closure**

Scope: AT-6 three-seed repeatability, overnight stress, closure document.

**Deliverables:**

- test_bcmp2_at6.py — seeds 42/101/303.

- Overnight stress mode validation — 4-hour run, memory and queue monitoring.

- Final HTML report with all AT criteria evidence.

- BCMP-2 Closure Report.

*Exit condition: three consecutive seed runs produce identical phase transition chains. No memory leak. All AT-1 through AT-6 gates documented with evidence.*

# **10  Technical Risks**

| **ID** | **Risk** | **Severity** | **Mitigation** |
| --- | --- | --- | --- |
| **R-1** | Vehicle A drift calibration — baseline drifts too gently, weakening comparison | **HIGH** | C-2 envelope pre-calculated from ALS-250 before runner code is written. AT-2 enforces both floor and ceiling. SB-1 is blocked until envelope is committed. This is the highest-priority item in the entire programme. |
| **R-2** | Terrain generator produces terrain that TRN finds trivially easy, overstating Vehicle B advantage | **MEDIUM** | C-3 parameters committed alongside scenario. Ridge spacing and slope variance seeded to realistic SRTM-class profiles. Reviewer can inspect and challenge parameters before any demo. |
| **R-3** | fault_manager threading race under Dash callbacks | **MEDIUM** | threading.Lock on all fault_manager reads and writes. Pattern precedent from Pre-HIL B-2 and B-3 threading fixes. Design the lock pattern in SB-2 before any proxy code is written. |
| **R-4** | Dash + SITL + Gazebo CPU contention on micromind-node01 | **LOW-MEDIUM** | Panel 2 uses static 3D Plotly geometry in executive mode. Live Gazebo feed reserved for high-fidelity mode. CP-2 data shows 3.2% CPU at baseline — large headroom. Monitor during SB-4 integration test. |
| **R-5** | Vehicle A accidentally succeeds in P5 due to overly forgiving terminal geometry | **MEDIUM** | Terminal corridor must be sized so that the expected Vehicle A drift at km 120 (from C-2 envelope) guarantees a corridor overshoot or unsafe approach geometry. Verify during SB-3 scenario geometry review. |
| **R-6** | Vehicle B appears too perfect relative to Vehicle A, causing reviewers to distrust the comparison | **MEDIUM** | Vehicle B must show visible bounded degradation during outages: non-zero drift growth in OUTAGE state, explicit RESUMPTION period before NOMINAL recovery, and temporary suppression of terminal actions with logged suppression events. If Vehicle B trajectory is a smooth perfect line throughout, it will appear unrealistic. The GUI should make OUTAGE and RESUMPTION periods visually prominent (amber segments on the Vehicle B path, not just in the mode panel). This is a presentation discipline requirement as much as an engineering one. |

# **11  Development Order**

The sequence that minimises risk and maximises early demonstrability. The first item is not code. Nothing else is started until the drift envelopes are committed.

| **#** | **Item** | **Sprint** | **Why this order** |
| --- | --- | --- | --- |
| **1** | **Derive Vehicle A drift envelopes analytically** | SB-1 | Nothing else is trustworthy until this exists. Anchors every downstream gate. |
| **2** | **Build bcmp2_terrain_gen.py with all 7 C-3 parameters** | SB-1 | Terrain is needed by scenario before the runner can execute any tick. |
| **3** | **Build bcmp2_scenario.py with shared disturbance schedule** | SB-1 | Disturbance schedule must exist before either vehicle track can run a step. |
| **4** | **Build baseline_nav_sim.py and validate against C-2 envelope** | SB-1 | Vehicle A must drift correctly before the comparison means anything. |
| **5** | **Build bcmp2_runner.py dual-track loop — AT-1 pass** | SB-1 | First integrated execution. Confirms both tracks produce logs. |
| **6** | **Build fault_manager.py + proxies — AT-3 scripted faults** | SB-2 | Fault injection infrastructure before any failure-mode acceptance test. |
| **7** | **Full five-phase terrain + AT-2/AT-3/AT-4/AT-5 + reports** | SB-3 | Full comparative demonstration only after fault injection is stable. |
| **8** | **Dash GUI + replay modes + operator fault panel** | SB-4 | GUI is a presentation layer. Build on stable foundation. |
| **9** | **AT-6 repeatability + overnight stress + closure** | SB-5 | Final gate. Run last when all earlier layers are proven. |
