**MicroMind / NanoCorteX**

**Part Two --- Demo Edition**

*DRDO / TASL Phase-1 SIL Demonstration*

Version 1.0 · 28 February 2026

Forked from: Part Two V7 + TechReview v1.1

Programme Director: Amit (amitkr-anand)

Repository: github.com/amitkr-anand/micromind-autonomy

Sprint state at fork: S8 COMPLETE \| 215/215 tests passing \| commit
f91180d

**Scope Statement**

This document is the engineering contract for MicroMind / NanoCorteX
Phase-1 SIL demonstration to DRDO and TASL leadership. It is a
controlled fork of Part Two V7 and TechReview v1.1, scoped to the
Phase-1 autonomy core.

The fork isolates the eight Phase-1 Functional Requirements that
constitute a complete, verifiable, and defensible demonstration. All
other capabilities are explicitly deferred to Phase-2 and Phase-3
roadmap and are catalogued in Appendix A and Appendix B.

This is not a retreat in ambition. It is sequencing. The programme goal
is 100% of a bounded, defensible capability set --- not 60% of an
expansive digital warfare stack.

**Programme Guardrails**

-   This document does not re-evaluate Sprint S0--S8. All completed work
    remains fully valid.

-   This document does not reopen architectural decisions (state
    machine, authority chain, compute sovereignty, EKF/UKF selection).

-   This document does not reintroduce deferred modules into the main
    implementation scope.

-   CEMS is architecturally present in the repository (S6, commit
    a7633ab) and is deliberately classified as Phase-2. No code was
    deleted.

-   Predictive EW modelling is present in the EW Engine codebase and is
    deliberately not invoked in Phase-1 execution. No code was deleted.

**Section 1**

**1. Operational Context --- BCMP-1 Normative Scenario**

*The Phase-1 demonstration is defined against a single, quantitatively
specified normative scenario. Every capability claim in this document is
measurable against it.*

**1.1 Scenario Definition**

BCMP-1 (Battlefield Corridor Mission Profile --- 1) is a 100 km
contested corridor navigation and terminal engagement scenario. It is
the normative test case for the Phase-1 SIL demonstration.

Platform: ALS-250 class tactical UAV / loitering munition. The 250 km
corridor simulation (ALS-250, Sprint S8) extends the baseline scenario
for IMU characterisation and TASL platform credibility.

  -----------------------------------------------------------------------
  **Parameter**   **Value**              **Notes**
  --------------- ---------------------- --------------------------------
  Corridor length 100 km (BCMP-1) / 250  BCMP-1 is the demo acceptance
                  km (ALS-250)           scenario; ALS-250 extends for
                                         IMU characterisation

  GNSS denial     Starts at km 15;       70 km of fully GNSS-denied
  event           continuous through km  inertial + TRN navigation
                  85                     

  Adaptive        2 nodes; mobile;       Triggers EW cost-map update and
  jamming nodes   position updated every mandatory route replan events
                  500 ms                 

  Satellite       1 event; mid-corridor  Tests terrain masking protocol
  overpass                               --- Phase-2 FR-108; not
                                         activated in Phase-1 demo

  Terminal target Thermal target with 1  Decoy must be rejected across ≥
                  active decoy           3 consecutive frames at ≥ 0.80
                                         confidence

  RF environment  Jamming + spoofing     BIM must detect and classify;
                  simultaneous           trust score must drop to Red
                                         within ≤ 250 ms

  Acceptance runs 5 independent seeded   All 5 must pass all 11 BCMP-1
                  runs                   criteria

  IMU (Phase-1    Safran STIM300         ARW 0.15°/√hr (typical); Gyro
  reference)      (tactical grade)       bias 0.5°/hr --- see Section 4.3
                                         for spec correction
  -----------------------------------------------------------------------

**1.2 BCMP-1 Acceptance Criteria (11 Gates)**

All 5 runs must pass all 11 criteria simultaneously. This is the sole
acceptance gate for the Phase-1 SIL demonstration.

  ----------------------------------------------------------------------------------
  **Criterion**   **Description**             **Threshold**
  --------------- --------------------------- --------------------------------------
  C-01            GNSS spoof detected and     BIM trust → Red within ≤ 250 ms
                  classified                  

  C-02            GNSS excluded from          Zero GNSS aiding while trust \< 0.4
                  navigation when trust = Red 

  C-03            Navigation drift under GNSS \< 2% of path length per 5 km segment
                  denial                      

  C-04            TRN correction applied      Every 2 km; NCC correlation ≥ 0.45

  C-05            EW cost-map updated on      Update latency ≤ 500 ms
                  jammer detection            

  C-06            Route replanned under EW    ≥ 2 mandatory replans; each ≤ 1 s
                  pressure                    

  C-07            ZPI zero-RF enforced in     Zero RF in SHM (ST-05); pre-terminal
                  terminal phase              burst transmitted

  C-08            DMRL target lock acquired   Lock confidence ≥ 0.85; temporal
                                              association ≥ 5 frames

  C-09            Thermal decoy rejected      Rejection confidence ≥ 0.80 over ≥ 3
                                              consecutive frames

  C-10            L10s-SE decision completed  Abort/continue decision ≤ 2 s from
                  within window               activation

  C-11            Mission log completeness    ≥ 99% of expected log fields captured
  ----------------------------------------------------------------------------------

**1.3 SIL Acceptance Status (Commit f91180d)**

> **CURRENT STATUS:** *BCMP-1: 5/5 runs × 11/11 criteria --- PASS.
> 215/215 tests across S0--S8 passing. This is the baseline against
> which the Phase-1 demo is presented.*

**Section 2**

**2. System Architecture --- Autonomy Core**

*MicroMind is a bounded, deterministic autonomy payload. It provides
survivable navigation and terminal guidance to existing autopilots ---
it does not replace them.*

**2.1 Three-Subsystem Architecture**

The MicroMind family is structured across three subsystems with a strict
authority hierarchy. Each subsystem has a defined role and cannot
override a higher layer.

  ----------------------------------------------------------------------------
  **Subsystem**   **Role**                **Runs On**      **Key Output**
  --------------- ----------------------- ---------------- -------------------
  MicroMind-OS    Mission planning, route Ground station / Cryptographically
                  definition, signed      commander        signed Mission
                  mission envelope        terminal         Package (route,
                  generation. Operator                     ROE, target, ZPI
                  interface.                               seed, DEM tiles)

  MicroMind-X     Onboard sensor fusion,  Onboard mission  Unified State
                  EW observation, unified computer         Vector (100 Hz
                  state vector. Runs the  (platform TBD    internal; 30 Hz to
                  Sensor Fusion Layer and per DD-01)       autopilot); GNSS
                  BIM.                                     Trust Score

  NanoCorteX      Autonomy engine:        Co-located with  Navigation commands
                  7-state FSM, navigation MicroMind-X (or  to autopilot;
                  mode management, route  dedicated        L10s-SE decisions;
                  replan, terminal        processor)       mission log entries
                  guidance orchestration,                  
                  L10s-SE enforcement.                     

  Autopilot (e.g. Vehicle stabilisation   Flight           Vehicle actuation
  PX4)            and attitude control    controller       
                  only. Not modified.     (existing        
                                          hardware)        
  ----------------------------------------------------------------------------

**2.2 NanoCorteX Deterministic State Machine (FR-106)**

NanoCorteX executes a formally specified deterministic finite automaton.
No free-form AI decision-making occurs in-mission. All state transitions
are logged with timestamp, trigger event, and guard evaluation result.
Transition latency ≤ 2 s (NFR-002).

  ------------------------------------------------------------------------------------
  **State   **State Name**   **Navigation   **RF        **Key Guard Conditions**
  ID**                       Primary**      Posture**   
  --------- ---------------- -------------- ----------- ------------------------------
  ST-01     NOMINAL          GNSS (Trust:   Full RF     GNSS trust = Green; BIM OK; no
                             Green)         active      active EW alert

  ST-02     EW_AWARE         GNSS (Trust:   Full RF     EW cost tile updated; jammer
                             Amber)         active      hypothesis active; GNSS trust
                                                        ≤ Amber

  ST-03     GNSS_DENIED      VIO + TRN      Full RF     GNSS trust = Red; VIO feature
                             primary        active      count ≥ 20; TRN correlation
                                                        valid

  ST-04     SILENT_INGRESS   VIO + TRN +    ZPI bursts  Terminal zone boundary crossed
                             DEM            only        per envelope; RF suppressed
                                                        except ZPI

  ST-05     SHM_ACTIVE       IMU + VO + DEM Zero RF     Final approach active; L10s-SE
                             only                       enforcement engaged; EO
                                                        terminal lock acquiring

  ST-06     ABORT            Loiter /       Per         L10s-SE abort decision;
                             egress         envelope    corridor violation; lock
                                                        confidence below threshold

  ST-07     MISSION_FREEZE   All outputs    Zero RF     Anti-capture triggered; tamper
                             suppressed                 detected; revoked key detected
  ------------------------------------------------------------------------------------

  ------------------------------------------------------------------------------------
  **From**   **To**   **Trigger**              **Guard**                **Max
                                                                        Transition**
  ---------- -------- ------------------------ ------------------------ --------------
  ST-01      ST-02    EW Engine: jammer        GNSS trust ≤ Amber OR EW ≤ 500 ms
                      hypothesis confidence \> cost tile updated        
                      0.6                                               

  ST-02      ST-03    BIM: GNSS trust state =  VIO feature count ≥ 20   ≤ 2 s
                      Red                      or TRN correlation valid 

  ST-02      ST-01    BIM: GNSS trust returns  No active jammer         ≤ 500 ms
                      Green for 3 consecutive  hypothesis above         
                      samples                  threshold                

  ST-03      ST-04    Navigation Engine:       SHM pre-conditions met:  ≤ 2 s
                      terminal zone boundary   VIO lock, DEM loaded, EO 
                      crossing per mission     initialised              
                      envelope                                          

  ST-04      ST-05    L10s-SE: final approach  EO target lock           ≤ 1 s
                      initiated                confidence ≥ 0.85; ≤ 10  
                                               s to impact              

  ST-05      ST-06    L10s-SE: abort decision  Decoy flag; civilian     ≤ 500 ms
                                               detection; corridor      
                                               violation; lock          
                                               confidence below         
                                               threshold                

  ANY        ST-06    Mission envelope: abort  Abort condition encoded  ≤ 2 s
                      command (pre-loaded)     in signed envelope       

  ANY        ST-07    Cybersecurity: key       Hardware tamper sensor   ≤ 1 s
                      revocation OR tamper     OR software key          
                      detection                revocation check         
  ------------------------------------------------------------------------------------

> **IMPLEMENTATION CONSTRAINT:** *The state machine must be implemented
> as a deterministic finite automaton --- not as conditional logic
> embedded in navigation code. A formal UML statechart specification was
> produced in Sprint S1 and is maintained as a controlled document.*

**2.3 Authority Chain**

The following hierarchy governs all decisions made by MicroMind. No
lower layer can override a higher layer. The Mission Envelope (Layer 0)
is the root of all authority and cannot be modified in flight.

  --------------------------------------------------------------------------------
  **Layer**   **Authority     **Can           **Cannot        **Enforcement**
              Holder**        Authorise**     Override**      
  ----------- --------------- --------------- --------------- --------------------
  0           Mission         All             Nothing --- is  PQC signature
              Envelope        mission-phase   the root        verified at load
              (signed,        parameters,     authority       time; cryptographic
              pre-loaded)     corridor, ROE,                  hash checked each
                              target                          access

  1           L10s-SE         Abort or        Mission         Deterministic
              (FR-105)        continue in     envelope        decision tree; no ML
                              terminal phase  corridor and    in this path
                              (ST-05 / ST-06) ROE             

  2           NanoCorteX      Navigation mode L10s-SE         EW cost map + BIM
              Decision Core   transitions;    terminal        trust state; FSM
                              route replan    decision        guards
                              within envelope                 

  3           MicroMind-X     State vector    Any routing or  Fused pose +
              (Sensor Fusion) outputs; GNSS   terminal        covariance; BIM
                              trust score     decision        output

  4           Autopilot (e.g. Vehicle         Any mission     MAVLink or
              PX4)            attitude and    logic           equivalent from
                              stabilisation                   NanoCorteX
  --------------------------------------------------------------------------------

**2.4 Data Flow Summary**

-   MicroMind-OS generates the Mission Package: signed envelope
    containing route, EW heatmap, target signature, ROE parameters, ZPI
    hop-plan seed, and DEM tiles.

-   The Mission Package is encrypted (AES-256) and transferred to
    MicroMind-X at mission load time. No field-level modification is
    possible post-signing.

-   MicroMind-X runs the Sensor Fusion pipeline and EW Observation,
    producing a Unified State Vector and GNSS Trust Score.

-   NanoCorteX consumes the State Vector, BIM output, and EW Engine
    output to drive the state machine and issue navigation commands to
    the autopilot.

-   Secure logs are written continuously to tamper-evident storage. ZPI
    bursts carry EW observations during transit; a mandatory EW summary
    burst is transmitted in the pre-terminal window per DD-02 Phase-1.

**Section 3**

**3. Phase-1 Functional Requirements**

*Eight Functional Requirements constitute the complete Phase-1
implementation scope. All other FRs from Part Two V7 are explicitly
deferred and are catalogued in Appendix B.*

**3.1 Phase-1 FR Set --- Governing Specification**

  --------------------------------------------------------------------------------------
  **FR ID**     **Capability**         **Boundary Constants**  **Sprint**   **Status**
  ------------- ---------------------- ----------------------- ------------ ------------
  FR-101        BIM --- GNSS Integrity Trust → Red \< 0.1;     S2           **✅
                & Trust Scaling        Spoof detect ≤ 250 ms                COMPLETE**
                                       latency; 3-sample                    
                                       hysteresis before state              
                                       transition                           

  FR-103        DMRL --- Deterministic Lock confidence ≥ 0.85; S5           **✅
                Multi-Frame Terminal   Decoy reject ≥ 0.80 / 3              COMPLETE**
                Discriminator          consecutive frames; Min              
                                       dwell ≥ 5 frames @ 25                
                                       FPS; Aimpoint                        
                                       correction ≤ ±15°;                   
                                       Re-acquisition timeout               
                                       1.5 s                                

  FR-104        ZPI --- Zero RF        Burst duration ≤ 10 ms; S6           **✅
                Terminal Phase Burst   Inter-burst interval                 COMPLETE**
                Scheduler              2--30 s randomised;                  
                                       Duty cycle ≤ 0.5%;                   
                                       Pre-terminal mandatory               
                                       burst; Zero RF in SHM                

  FR-105        L10s-SE ---            Decision within ≤ 2 s   S5           **✅
                Deterministic          of activation; Civilian              COMPLETE**
                Abort/Continue Safety  detect ≥ 0.70                        
                Engine                 confidence; Corridor                 
                                       hard enforcement;                    
                                       Decision window ≤ 10 s               
                                       total; Structured log                
                                       entry mandatory                      

  FR-106        FSM --- NanoCorteX     All transitions ≤ 2 s   S1           **✅
                7-State Deterministic  (NFR-002); Formally                  COMPLETE**
                Autonomy State Machine specified UML                        
                                       statechart; All                      
                                       transitions logged                   

  FR-107        TRN ---                Kalman correction every S3           **✅
                Terrain-Referenced     2 km; NCC correlation                COMPLETE**
                Navigation             threshold ≥ 0.45; TRN                
                (RADALT-validated in   position error \< 50 m               
                SIL)                   CEP-95; RADALT                       
                                       simulated in SIL via                 
                                       plugin                               

  FR-NAV-01     GNSS-Denied Navigation Drift \< 2% of path     S0+S8        **✅
                Drift Limit            length per 5 km                      COMPLETE**
                                       segment; Verified over               
                                       100 km (BCMP-1) and 250              
                                       km (ALS-250)                         

  FR-EW-01/02   EW Reactive Cost-Map + Cost-map update ≤ 500   S4           **✅
                Hybrid A\* Route       ms from detection;                   COMPLETE**
                Replan                 Route replan ≤ 1 s;                  
                                       Replan trigger: route                
                                       cost increase \> 20%                 
                                       following EW update                  
  --------------------------------------------------------------------------------------

**3.2 Non-Functional Requirements --- Phase-1**

  --------------------------------------------------------------------------
  **NFR     **Requirement**                     **Value**
  ID**                                          
  --------- ----------------------------------- ----------------------------
  NFR-001   BIM processing latency (end-to-end) ≤ 250 ms

  NFR-002   Autonomy state machine transition   ≤ 2 s
            time (any state)                    

  NFR-003   SDR EW processing: snapshot to      ≤ 500 ms
            cost-map update                     

  NFR-004   TRN update loop rate                ≥ 2 Hz

  NFR-006   EW detection probability            Pd ≥ 0.90

  NFR-007   EW false alarm rate                 Pfa ≤ 0.05

  NFR-008   EO/IR terminal lock latency         ≤ 400 ms (25 FPS × 10
                                                frames)

  NFR-009   Encryption standard                 AES-256 at rest and in
                                                transit; PQC-ready
                                                signatures

  NFR-013   Log completeness per mission        ≥ 99% of expected log fields

  NFR-014   Anti-capture erase time from        ≤ 5 s to mission data
            trigger                             erasure
  --------------------------------------------------------------------------

**Section 4**

**4. Navigation Subsystem**

*The Navigation Subsystem provides GNSS-denied route following, terrain
correlation, and visual-inertial odometry under all operational states.
It is the longest and most critical segment of the Phase-1 scenario.*

**4.1 Sensor Fusion Architecture**

Primary fusion uses a 15-state Error-State Kalman Filter (ESKF V2,
Sprint S0). The filter selects EKF for cruise and UKF when velocity \>
20 m/s or attitude rate \> 45°/s. Visual-Inertial Odometry (VIO) and
Terrain-Referenced Navigation (TRN) provide aiding when GNSS is
excluded.

  -----------------------------------------------------------------------------
  **Input**       **Rate**    **Role in Fusion**
  --------------- ----------- -------------------------------------------------
  IMU (Accel /    200--1000   High-rate inertial propagation; bias estimation;
  Gyro / Baro)    Hz          noise characterised per Allan Variance (Section
                              4.3)

  EO/IR Camera    25--50 FPS  Visual Odometry (feature tracking); thermal ROI
  (gimbal or                  for DMRL
  nadir)                      

  GNSS Raw        1--10 Hz    Position and velocity aiding; integrity checked
  (NMEA + Rx                  by BIM before use; excluded when trust = Red
  metadata)                   

  Visual Odometry 25--50 Hz   Pose delta from consecutive frames; compensates
  delta (VO)                  IMU drift when feature count ≥ 20

  RADALT (height  ≥ 10 Hz     TRN altitude reference; simulated via Gazebo
  AGL)                        plugin in SIL phase

  DEM Tiles       On access   Terrain correlation for TRN; ±20 km pre-load
  (pre-loaded)                window; refreshed at 5 Hz
  -----------------------------------------------------------------------------

  -----------------------------------------------------------------------
  **Output**                **Rate**         **Consumer**
  ------------------------- ---------------- ----------------------------
  Unified State Vector      100 Hz internal; NanoCorteX; Autopilot
  (position, velocity,      30 Hz to         (ODOMETRY message or
  attitude, covariance)     autopilot        equivalent)

  GNSS Trust Score          ≥ 10 Hz          BIM; EKF measurement noise
  (continuous 0.0--1.0)                      scaling (R_GNSS = R_nominal
                                             / trust_score, capped 10×)

  GNSS Integrity Flag /     On event         BIM; NanoCorteX Decision
  Events                                     Core
  -----------------------------------------------------------------------

**4.2 BIM --- Beacon & Integrity Monitor (FR-101)**

BIM evaluates trust in GNSS signals and produces a continuous trust
score and discrete G/A/R trust state used by the EKF and the NanoCorteX
FSM. Total processing latency ≤ 250 ms (NFR-001).

  -------------------------------------------------------------------------
  **Parameter**         **Value**           **Rationale**
  --------------------- ------------------- -------------------------------
  Trust states          Green (G) / Amber   G = full use; A =
                        (A) / Red (R)       trust-weighted; R = exclude
                                            from EKF

  RAIM PDOP threshold → PDOP \> 3.0         Standard aviation RAIM
  Amber                                     threshold; conservative for
                                            tactical use

  RAIM PDOP threshold → PDOP \> 6.0         GNSS solution considered
  Red                                       unreliable above this value

  Doppler deviation →   \|deviation\| \>    High confidence of spoofing or
  Red                   1.5 m/s             severe jamming

  Multi-constellation   GPS vs GLONASS      Spoof signals rarely manipulate
  delta → Red           position delta \>   two independent constellations
                        15 m                consistently

  State transition      3 consecutive       Prevents rapid G→A→G
  hysteresis            samples at new      oscillation in marginal signal
                        state before        conditions
                        transition          

  Continuous trust      0.0--1.0 at ≥ 10 Hz Drives EKF noise scaling
  score output                              continuously; not just a
                                            discrete state flag

  BIM processing        GNSS analyse ≤ 80   Total ≤ 250 ms with margin
  latency budget        ms + score combine  (NFR-001)
                        ≤ 30 ms + output ≤  
                        20 ms = ≤ 130 ms    
                        target              
  -------------------------------------------------------------------------

Score weighting model (starting values --- tuned in Sprint S2 against
recorded GNSS attack data):

  ---------------------------------------------------------------------------------------
  **Score Component**   **Weight**   **Range**   **Notes**
  --------------------- ------------ ----------- ----------------------------------------
  RAIM / PDOP score     0.35         0--1        Highest weight; most reliable integrity
                                                 indicator

  Doppler deviation     0.25         0--1        Strong spoof discriminator
  score                                          

  Multi-constellation   0.20         0--1        Requires ≥ 2 constellations tracked
  consistency                                    

  Pose innovation       0.15         0--1        Detects position jumps vs inertial
  residual                                       prediction

  EW impact score       0.05         0--1        Contextual; increases sensitivity near
                                                 jammer zones
  ---------------------------------------------------------------------------------------

Thresholds: G if weighted score ≥ 0.7; A if 0.4 ≤ score \< 0.7; R if
score \< 0.4.

**4.3 IMU Characterisation --- Allan Variance Parameterisation (Sprint
S8)**

Sprint S8 characterised three IMU sensor profiles from published
datasheets and propagated their noise through the full SIL stack. The
ALS-250 250 km GNSS-denied corridor simulation provides quantitative
per-sensor navigation performance.

  -------------------------------------------------------------------------------------------
  **Profile     **Sensor**     **Grade**   **ARW       **Gyro Bias **Accel VRW   **Demo
  Key**                                    (°/√hr)**   (°/hr)**    (m/s/√hr)**   Role**
  ------------- -------------- ----------- ----------- ----------- ------------- ------------
  STIM300       Safran STIM300 Tactical    0.15        0.5         0.05          Phase-1
                                           (typical)                             reference
                                                                                 IMU

  ADIS16505_3   Analog Devices MEMS        0.22        8.0         0.12          Benchmark
                ADIS16505-3                                                      lower bound

  BASELINE      Simplified     ---         0.05        0.3         0.02          Algorithm
                (S0--S7                                                          validation
                equivalent)                                                      baseline
  -------------------------------------------------------------------------------------------

> **SPEC CORRECTION (S8 FINDING):** *Part Two V7 specified IMU ARW floor
> as ≤ 0.1°/√hr. The Safran STIM300 (the targeted tactical-grade sensor)
> has a typical ARW of 0.15°/√hr per published datasheet. The corrected
> specification is ≤ 0.2°/√hr. This is a spec correction, not a design
> regression --- STIM300 meets all navigation accuracy targets. Action
> required before TASL meeting.*
>
> **CHART PLACEHOLDER:** *ALS-250 250 km Drift Characterisation Chart
> --- Three-curve comparison: STIM300 vs ADIS16505-3 vs BASELINE. To be
> inserted from Sprint S8-D output (dashboard/als250_drift_chart.py).
> Overnight run results: sim/als250_results/. Do not fabricate values
> --- insert from generated chart file.*

**4.4 Terrain-Referenced Navigation --- TRN (FR-107)**

TRN fuses radar altimeter data with pre-loaded Digital Elevation Maps
(DEM) to correct inertial drift without GNSS. The NCC-based DEM
correlation is the primary drift correction mechanism in GNSS-denied
phases.

  -----------------------------------------------------------------------
  **TRN Parameter**   **Value**              **Rationale**
  ------------------- ---------------------- ----------------------------
  DEM correlation     Normalised             TERCOM heritage;
  algorithm           Cross-Correlation      computationally tractable;
                      (NCC) of altimeter     proven accuracy on 30 m DEMs
                      profile vs DEM strip   

  Kalman correction   Every 2 km of ground   Balances drift accumulation
  interval            track                  vs correction frequency

  NCC correlation     ≥ 0.45                 Below this threshold,
  threshold                                  correction is rejected; log
                                             event and continue inertial

  TRN position        \< 50 m CEP-95         TERCOM-class on 30 m DEM;
  correction error                           conservative estimate for
  (target)                                   varied terrain

  DEM resolution      SRTM 1-arc-sec (≈ 30   Freely available; adequate
                      m)                     for TRN in Indian
                                             operational terrain

  RADALT --- SIL      Simulated via Gazebo   No hardware procurement
  phase               plugin                 required to gate SIL
                                             progress

  RADALT ---          Physical RADALT        Real sensor noise
  HIL/Production      required; 0.5--50 m    characteristics validated in
  phase               AGL; ≥ 10 Hz; ≤ 40 g;  HIL phase
                      ≤ 0.5 W                
  -----------------------------------------------------------------------

**4.5 Navigation Modes (State-Linked)**

  -----------------------------------------------------------------------------
  **FSM     **Navigation   **Primary        **Aiding Sources**
  State**   Mode**         Algorithm**      
  --------- -------------- ---------------- -----------------------------------
  ST-01     Full GNSS      EKF with GNSS    IMU, GNSS, VO, DEM
                           measurement      

  ST-02     EW-Aware GNSS  EKF with         IMU, GNSS (trust-scaled), VO, EW
                           trust-weighted   cost map
                           GNSS             

  ST-03     GNSS-Denied    EKF/UKF; VIO     IMU, VO, TRN, DEM
            (VIO primary)  primary          

  ST-04     Silent Ingress EKF/UKF; TRN     IMU, VO, TRN (RADALT + DEM); no
                           primary          GNSS

  ST-05     Silent Homing  Inertial + DEM   IMU, VO, DEM; Zero RF; EO terminal
            (SHM)          only             lock
  -----------------------------------------------------------------------------

**Section 5**

**5. EW Engine + Route Planner (FR-EW-01/02)**

*The Phase-1 EW Engine provides reactive electronic warfare awareness
and route replanning. The scope is deliberately bounded to reactive
behaviour. Predictive EW modelling is architecturally present in the
codebase and is explicitly deferred to Phase-2.*

**5.1 Phase-1 EW Engine Functions**

Three functions constitute the complete Phase-1 EW Engine scope:

-   Jam / spoof detection --- wideband SDR sweep with spectral analysis;
    classification of RF anomalies as jamming or spoofing events. Spoof
    trigger: multi-constellation position delta \> 15 m AND C/N0 drop \>
    6 dB simultaneously.

-   Reactive cost-map update --- tile-based risk surface generated from
    jammer node hypotheses and updated within ≤ 500 ms of detection
    (NFR-003/005).

-   Hybrid A\* route replan --- EW cost overlay fed directly into A\*
    heuristic; replan triggered when route cost increases \> 20%
    following an EW update; replan completed within ≤ 1 s.

**5.2 Boundary Conditions**

  ---------------------------------------------------------------------------
  **Parameter**         **Value**             **Notes**
  --------------------- --------------------- -------------------------------
  EW cost-map update    ≤ 500 ms              From SDR snapshot to cost-map
  latency               (NFR-003/005)         update

  EW cost-map tile      ≤ 50 m                Required for meaningful route
  resolution                                  planner avoidance corridor

  Route replan time     ≤ 1 s (FR-EW-02)      Hybrid A\* with pre-computed
  budget                                      DEM achieves this

  Route replan trigger  Route cost increase   Balances stability vs
  threshold             \> 20% following EW   responsiveness
                        update                

  Spoof detection ---   Multi-constellation   Dual-condition prevents
  primary trigger       position delta \> 15  single-fault false positives
                        m AND C/N0 drop \> 6  
                        dB simultaneously     

  Signature clustering  DBSCAN (eps = 3 dB    Starting values; tuned against
  algorithm             RSSI domain;          BCMP-1 EW simulator in Sprint
                        min_samples = 3)      S4

  EW detection          Pd ≥ 0.90             Neyman-Pearson criterion; SIL
  probability (NFR-006)                       starting value

  EW false alarm rate   Pfa ≤ 0.05            SIL starting value; tuned in
  (NFR-007)                                   Sprint S4
  ---------------------------------------------------------------------------

**5.3 Phase-1 Scope Boundary**

> **SCOPE BOUNDARY:** *The predictive EW branch (Kalman-tracked jammer
> velocity; 30 s lookahead horizon; CEP quantification) is
> architecturally present in core/ew_engine/ew_engine.py and is not
> invoked in Phase-1 execution. No code changes are required. This
> capability is explicitly roadmapped for Phase-2. Reactive cost-map
> updates and Hybrid A\* replan are sufficient to demonstrate EW
> survivability in the BCMP-1 scenario.*

**5.4 BCMP-1 EW Performance (Sprint S4 --- 8/8 Gates PASS)**

  -----------------------------------------------------------------------
  **Validation**                 **Result**
  ------------------------------ ----------------------------------------
  Cost-map update latency on     ≤ 500 ms --- PASS
  jammer event                   

  Route replan on EW pressure    ≥ 2 mandatory replans, each ≤ 1 s ---
                                 PASS

  Jammer detection probability   Pd ≥ 0.90 --- PASS
  in scenario                    

  Regression (S0--S3 + S4)       All 8/8 S4 acceptance gates passed; no
                                 regressions
  -----------------------------------------------------------------------

**Section 6**

**6. Terminal Guidance --- DMRL (FR-103)**

*The Deep Multi-frame Recognition & Lock system is a deterministic,
bounded-inference terminal discriminator. It is not a free-form
classifier. Every decision is governed by explicit thresholds, temporal
constraints, and envelope-bound model versions.*

**6.1 Design Principles**

The DMRL is specified and implemented as a deterministic multi-frame
discriminator, not an adaptive AI system. The following principles are
non-negotiable for Phase-1:

-   Preloaded target models only --- 3 to 4 representative classes (e.g.
    artillery, logistics vehicle, radar platform, decoy class). No live
    ingestion.

-   Model binding --- each model version carries a version ID and
    SHA-256 hash. The hash is verified against the signed mission
    envelope at load time. Model changes are pre-mission only.

-   Temporal association --- lock confidence accumulates across ≥ 5
    consecutive frames. Single-frame decisions are not permitted.

-   Deterministic decision window --- all lock and abort/continue
    decisions must complete within ≤ 2 s of L10s-SE activation.

-   No onboard retraining. No adaptive mid-mission behaviour. No live
    model uploads.

-   Dataset: synthetic data and commercially available datasets. No
    classified ingestion infrastructure in Phase-1.

**6.2 Processing Pipeline**

  ------------------------------------------------------------------------
  **Stage**      **Description**              **Output**
  -------------- ---------------------------- ----------------------------
  Frame          Rectify; stabilise via IMU   Calibrated thermal frame
  preparation    delta; apply radiometric     
                 calibration (16-bit LWIR)    

  Feature        Edges; optical flow; thermal Target region of interest
  extraction     ROI identification           (ROI)

  Temporal       Multi-frame target tracking  Temporal track with
  association    across ≥ 5 consecutive       confidence accumulation
                 frames at 25 FPS (200 ms     
                 minimum dwell)               

  Decoy          Multi-frame thermal          Decoy flag + rejection
  rejection      dissipation model; CNN       confidence (0.0--1.0)
                 classifier with confidence   
                 accumulation                 

  Lock           Probability + covariance     Lock confidence score
  confidence     across temporal track; gate  (0.0--1.0)
  computation    for L10s-SE                  

  Aimpoint       Ballistic offset computation Corrected aimpoint
  correction     within L10s-SE corridor      
                 limits (±15° bearing)        
  ------------------------------------------------------------------------

**6.3 Boundary Conditions (FR-103)**

  ------------------------------------------------------------------------
  **Parameter**         **Value**       **Rationale**
  --------------------- --------------- ----------------------------------
  Lock confidence       ≥ 0.85          High confidence required before
  threshold to proceed  probability     committing to terminal run

  Decoy rejection       ≥ 0.80 over 3   Multi-frame requirement prevents
  confidence to abort   consecutive     single-frame misclassification
                        frames          from triggering abort

  Minimum temporal      ≥ 5 frames at   Minimum dwell for thermal
  association window    25 FPS (200 ms  dissipation signature to manifest
                        dwell)          

  Minimum thermal ROI   8 × 8 pixels at Below this threshold, thermal
  size                  maximum         discrimination is unreliable
                        engagement      
                        range           

  Aimpoint correction   ± 15° bearing   Prevents grossly off-axis
  limit                 within L10s-SE  corrections exceeding airframe
                        window          manoeuvre limits

  Re-acquisition        If EO lock lost Prevents blind terminal
  timeout               \> 1.5 s during continuation without confirmed
                        terminal run →  target lock
                        abort           

  L10s-SE decision      Lock/abort      Leaves ≤ 8 s for terminal run;
  window                decision within validated in SIL Sprint S5
                        ≤ 2 s of        
                        L10s-SE         
                        activation      
  ------------------------------------------------------------------------

**6.4 Model Governance**

  -----------------------------------------------------------------------
  **Parameter**             **Specification**
  ------------------------- ---------------------------------------------
  Number of preloaded       3--4 representative target classes
  models (Phase-1)          

  Model binding mechanism   Version ID + SHA-256 hash; hash verified
                            against signed mission envelope at load time

  Dataset (Phase-1)         Synthetic data matching Indian operational
                            terrain and lighting conditions; commercially
                            available datasets

  Mid-mission model change  Not permitted. Model changes are pre-mission
                            only.

  Onboard retraining        Not implemented in Phase-1. Not permitted in
                            Phase-1 envelope.

  Classified data pipeline  Not built in Phase-1. Architecture is
                            sovereign-ready for customer-provided threat
                            libraries in Phase-2.

  Thermal sensor            LWIR 8--14 μm; radiometric 16-bit; ≥ 320×240
  requirement (SIL)         px; ≥ 25 FPS; GigE/USB3. Simulated in SIL.
  -----------------------------------------------------------------------

**6.5 BCMP-1 Terminal Guidance Performance (Sprint S5 --- 111/111
PASS)**

  -----------------------------------------------------------------------
  **Criterion**                  **Result**
  ------------------------------ ----------------------------------------
  C-08: DMRL target lock         PASS --- all 5 runs
  acquired (≥ 0.85 confidence)   

  C-09: Thermal decoy rejected   PASS --- all 5 runs
  (≥ 0.80 / 3 consecutive        
  frames)                        

  C-10: L10s-SE decision within  PASS --- all 5 runs
  ≤ 2 s                          

  Regression (S0--S4 + S5)       No regressions --- 111/111 S5 tests
                                 passing
  -----------------------------------------------------------------------

**Section 7**

**7. Safety & Authority Chain --- L10s-SE (FR-105)**

*The Last-10-Second Safety Envelope is the final-phase enforced
authority layer. It is deterministic, auditable, and contains no machine
learning. It is the system\'s proof of ethical engagement constraint
enforcement.*

**7.1 Purpose and Design Constraint**

L10s-SE operates in ST-05 (SHM_ACTIVE) and ST-06 (ABORT). The decision
tree encoded here reflects the Rules of Engagement pre-loaded in the
signed mission envelope. No machine learning is used in this decision
path. Every decision is logged with timestamp, trigger, sensor state,
and decision outcome.

> **DESIGN CONSTRAINT:** *L10s-SE is the one decision path in NanoCorteX
> where determinism is not merely preferred --- it is mandatory. The
> absence of ML in this path is an architectural guarantee, not an
> implementation convenience.*

**7.2 L10s-SE Functions and Boundary Conditions**

  -------------------------------------------------------------------------
  **L10s-SE        **Description**        **Boundary Condition**
  Function**                              
  ---------------- ---------------------- ---------------------------------
  Civilian         EO/IR classifier       Detection confidence ≥ 0.70 in
  detection        checks terminal scene  any frame → immediate abort to
                   for non-combatant      ST-06
                   signatures in each     
                   frame within the L10s  
                   window                 

  Abort / continue Deterministic tree:    Decision within ≤ 2 s of
  decision         evaluates lock         activation (Authority Chain Layer
                   confidence, decoy      1)
                   flag, civilian flag,   
                   corridor compliance    
                   simultaneously         

  Corridor hard    Terminal trajectory    Any predicted corridor violation
  enforcement      prediction checked     → immediate abort to ST-06
                   against                
                   envelope-defined       
                   corridor               

  Temporal         L10s-SE window = ≤ 10  Hard timer enforced by state
  constraint       s from activation to   machine; cannot be extended
                   impact or abort        in-mission

  EO lock          Single re-acquisition  Re-acquisition timeout = 1.5 s;
  re-acquisition   attempt if lock lost   if failed → abort to ST-06
                   during terminal run    

  Secure log entry All L10s-SE events     Required for post-mission audit
                   logged: timestamp,     and ROE compliance review
                   decision, sensor       (FR-NFR-013)
                   state, trigger         
  -------------------------------------------------------------------------

**7.3 L10s-SE in the Authority Chain**

L10s-SE sits at Layer 1 of the Authority Chain. It can authorise abort
or continue in the terminal phase. It cannot override the Mission
Envelope (Layer 0). NanoCorteX Decision Core (Layer 2) cannot override
L10s-SE decisions.

**7.4 BCMP-1 L10s-SE Performance (Sprint S5 --- PASS)**

  -----------------------------------------------------------------------
  **Validation**                 **Result**
  ------------------------------ ----------------------------------------
  Decision latency within ≤ 2 s  PASS --- all 5 runs

  Civilian detect threshold      PASS --- tested against synthetic
  functional (≥ 0.70)            civilian target injection

  Corridor hard enforcement      PASS --- corridor violation injection
  (abort on violation)           test

  Secure log entry written per   PASS --- log completeness ≥ 99% (C-11)
  event                          
  -----------------------------------------------------------------------

**Section 8**

**8. Zero RF Terminal Phase --- ZPI (FR-104)**

*ZPI (Zero Probability of Intercept burst scheduler) provides
low-probability-of-intercept signalling for telemetry and EW flash
updates during transit, and enforces complete RF silence in the Silent
Homing phase.*

**8.1 ZPI Phase-1 Functions**

-   Burst scheduling --- randomised burst durations and inter-burst
    intervals that defeat wideband direction-finding.

-   Hop plan derivation --- frequency hop plan seeded from the mission
    key using HKDF-SHA256, providing implicit time-synchronisation with
    peers sharing the same mission package.

-   SHM suppression --- all RF transmissions are suppressed upon entry
    to ST-05 (SHM_ACTIVE). The pre-terminal mandatory burst must be
    completed before SHM activation.

-   Pre-terminal mandatory burst --- an EW observation summary packet is
    transmitted in the pre-terminal window, before SHM activation, per
    DD-02 Phase-1 minimum.

**8.2 Boundary Conditions (FR-104)**

  ------------------------------------------------------------------------
  **Parameter**       **Value**             **Rationale**
  ------------------- --------------------- ------------------------------
  Burst duration      ≤ 10 ms               Sub-frame duration makes
                                            wideband direction-finding
                                            very difficult

  Inter-burst         Randomised 2--30 s    Exceeds typical DF integration
  interval                                  time of 1--3 s

  Frequency hop range ± 5 MHz around        Forces DF to cover wide
                      mission-defined       aperture per burst
                      centre                

  Maximum duty cycle  ≤ 0.5%                Limits detectability to near
                                            noise floor at 5 km stand-off

  Power variation     −10 to 0 dB relative  Variable power prevents
  range               to maximum            RSSI-based stand-off ranging

  Hop plan seed       Derived from mission  Deterministic but
                      key using HKDF-SHA256 unpredictable to adversary
                                            without mission key

  Pre-terminal        EW observation        Required per DD-02 Phase-1
  mandatory burst     summary packet        minimum
                      transmitted before    
                      SHM activation        

  ZPI in SHM (ST-05)  Zero RF --- all       Absolute requirement; enforced
                      transmitters          by state machine guard
                      suppressed            
  ------------------------------------------------------------------------

**8.3 Phase-1 Scope Boundary**

> **SCOPE BOUNDARY:** *Adaptive anti-DF logic (burst rate reduction on
> DF bearing detection) and power shaping intelligence are
> architecturally present in core/zpi/zpi.py and are deliberately
> deferred to Phase-2. Phase-1 ZPI provides the core LPI burst
> scheduling and SHM suppression --- which are sufficient to demonstrate
> zero-RF terminal phase enforcement in BCMP-1.*

**8.4 BCMP-1 ZPI Performance (Sprint S6 --- 36/36 PASS)**

  -----------------------------------------------------------------------
  **Validation**                 **Result**
  ------------------------------ ----------------------------------------
  Burst duration ≤ 10 ms         PASS

  Inter-burst interval 2--30 s   PASS
  randomised                     

  Pre-terminal mandatory burst   PASS --- all 5 BCMP-1 runs
  transmitted                    

  Zero RF enforced in ST-05      PASS --- C-07 criterion; all 5 runs
  (SHM)                          

  Regression (S0--S5 + ZPI)      36/36 S6 tests passing; 111/111 S5
                                 regression clean
  -----------------------------------------------------------------------

**Section 9**

**9. Mission Logging**

*Structured, tamper-evident mission logging is a Phase-1 requirement.
Complete logs are a prerequisite for post-mission ROE compliance review
and for DD-02 Phase-1 minimum data capture.*

**9.1 Log Schema (DD-02 Phase-1)**

The Mission Log Schema (Sprint S1, logs/mission_log_schema.py) captures
all state transitions, sensor events, BIM outputs, EW events, DMRL
decisions, L10s-SE decisions, and ZPI burst events. The schema is
learning-field-aware per DD-02 Phase-1 minimum --- cross-mission
learning logic is deferred to Phase-2.

  ------------------------------------------------------------------------
  **Log          **Fields Captured**          **FR / NFR Traceability**
  Category**                                  
  -------------- ---------------------------- ----------------------------
  FSM State      Timestamp; from-state;       FR-106; NFR-002
  Transitions    to-state; trigger; guard     
                 evaluation result            

  BIM Events     Timestamp; trust score;      FR-101; NFR-001
                 trust state (G/A/R); trigger 
                 type; GNSS exclusion flag    

  EW Events      Timestamp; jammer            FR-EW-01/02; NFR-003
                 hypothesis; bearing; RSSI;   
                 cost-map update timestamp;   
                 replan trigger               

  TRN            Timestamp; correction        FR-107
  Corrections    applied (m); NCC correlation 
                 score; position before/after 

  DMRL Events    Timestamp; lock confidence;  FR-103
                 decoy flag; decoy            
                 confidence; frame count;     
                 model version + hash         

  L10s-SE        Timestamp; decision          FR-105
  Decisions      (abort/continue); trigger;   
                 civilian flag; corridor      
                 flag; sensor state           

  ZPI Events     Timestamp; burst type; burst FR-104
                 duration; hop frequency; SHM 
                 suppression flag             

  Navigation     Position; velocity;          FR-NAV-01; FR-107
  State          attitude; covariance;        
                 navigation mode; active      
                 aiding sources               
  ------------------------------------------------------------------------

**9.2 Log Completeness and Presentation**

-   Log completeness requirement: ≥ 99% of expected log fields captured
    per mission (NFR-013 / BCMP-1 C-11).

-   Logs are written to tamper-evident storage continuously throughout
    the mission.

-   Post-mission HTML debrief report: self-contained, no external
    dependencies. Generated by dashboard/bcmp1_report.py (Sprint S7).

-   Mission dashboard: 9-panel Plotly Dash display covering all S0--S6
    subsystems. Generated by dashboard/bcmp1_dashboard.py (Sprint S7).

**Section 10**

**10. SIL Sprint Plan --- Demo Milestone Alignment**

*All sprints S0--S8 are complete. The following maps each sprint to the
Phase-1 demo milestone structure. No re-planning is required under the
Demo Edition scope.*

**10.1 Completed Sprints --- Demo Milestone Map**

  --------------------------------------------------------------------------------------------
  **Sprint**   **Commit**   **Demo Milestone** **Capability Proven**               **Tests**
  ------------ ------------ ------------------ ----------------------------------- -----------
  S0           6e1c70a      M1 --- Nav         ESKF V2; quaternion math; 15-state  ---
                            Foundation         INS mechanisation; ENU frame        
                                               constants                           

  S1           5005a5d      M1 --- Autonomy    7-state FSM (FR-106); SimClock;     9/9
                            Shell              Mission Log Schema (DD-02 P1);      
                                               BCMP-1 scenario definition          

  S2           e86140f      M2 --- GNSS        BIM trust scorer ≤ 250 ms (FR-101); 9/9
                            Integrity          GNSS spoof injector test harness    

  S3           284acb4      M2 --- GNSS-Denied TRN NCC stub (FR-107); 50 km nav    8/8
                            Nav                scenario; Plotly nav dashboard      

  S4           366f963      M3 --- EW          EW reactive cost-map ≤ 500 ms;      8/8
                            Survivability      Hybrid A\* replan ≤ 1 s             
                                               (FR-EW-01/02); BCMP-1 EW sim        

  S5           7ad5db5      M4 --- Terminal    DMRL multi-frame discriminator      111/111
                            Autonomy           (FR-103); L10s-SE (FR-105); BCMP-1  
                                               full runner (11 criteria / 5 runs)  

  S6           a7633ab      M3 (ZPI) + Phase-2 ZPI zero-RF burst scheduler         36/36
                            (CEMS)             (FR-104) \[Phase-1\]. CEMS          
                                               multi-UAV engine \[Phase-2, frozen  
                                               in repo\].                          

  S7           aa3302a      M5 --- Demo        BCMP-1 9-panel dashboard;           ---
                            Presentation       self-contained HTML debrief report  

  S8           f91180d      M2 --- IMU         Safran STIM300 / ADIS16505-3 /      68/68
                            Characterisation   BASELINE noise models; ALS-250 250  
                                               km corridor sim; BCMP-1 IMU         
                                               integration                         
  --------------------------------------------------------------------------------------------

**10.2 Demo Milestone Summary**

  ----------------------------------------------------------------------------
  **Milestone**   **Gate Criterion**     **Evidence**           **Status**
  --------------- ---------------------- ---------------------- --------------
  M1 --- Autonomy FSM transitions        S0 + S1 passing; ESKF  **✅
  Core            deterministic; INS     V2 verified            COMPLETE**
                  propagates without                            
                  drift                                         

  M2 ---          Drift \< 2%/5 km; BIM  S2, S3, S8 passing;    **✅ (chart
  GNSS-Denied     ≤ 250 ms; TRN every 2  ARW spec corrected to  pending
  Navigation      km; ALS-250 250 km     ≤ 0.2°/√hr             S8-D)**
                  corridor characterised                        

  M3 --- EW       Cost-map ≤ 500 ms;     S4 + S6 ZPI passing;   **✅
  Survivability   Replan ≤ 1 s; ZPI      BCMP-1 EW sim PASS     COMPLETE**
                  zero-RF enforced in                           
                  SHM                                           

  M4 --- Terminal DMRL lock ≥ 0.85;      S5 --- 111/111; BCMP-1 **✅
  Autonomy        Decoy reject ≥         5/5 × 11/11            COMPLETE**
                  0.80/3f; L10s decision                        
                  ≤ 2 s                                         

  M5 --- Demo     HTML debrief; 9-panel  S7 passing; drift      **✅ (chart
  Presentation    dashboard; TASL-ready  chart pending S8-D     pending
                  ALS-250 drift chart    overnight run          S8-D)**
  ----------------------------------------------------------------------------

**10.3 Sprint S9 --- Scope Options (Pending TASL Outcome)**

S9 is not started. Scope is pending TASL meeting outcome. All options
are Phase-1 aligned or hardware-blocked.

  ----------------------------------------------------------------------------------
  **Option**   **Fork**                      **Readiness**      **Demo Relevance**
  ------------ ----------------------------- ------------------ --------------------
  A            Cybersecurity Hardening ---   No blockers        High --- mission
               FR-109--112: key loading,                        envelope integrity
               envelope verification,                           is a TASL
               PQC-ready stack                                  credibility
                                                                requirement

  B            DMRL CNN Upgrade --- replace  Blocked: GPU +     High --- upgrades
               rule-based stub with trained  dataset +          FR-103 from
               CNN (Hailo-8 target)          Indigenous Threat  rule-based to
                                             Library clearance  learned classifier

  C            HIL Integration Prep --- ROS2 Blocked: TASL      Critical for
               node wrappers, PX4 SITL       hardware platform  post-SIL hardware
               skeleton                      decision required  bridge

  D            S8-D Drift Chart + S6 5×      Ready once         Immediate ---
               Clean Sweep ---               overnight run      completes S8
               als250_drift_chart.py         completes          deferred item; TASL
               generation + CEMS diagnostic                     presentation chart
  ----------------------------------------------------------------------------------

**Section 11**

**11. KPI Dashboard --- Phase-1 Demo Metrics**

*The following KPIs constitute the complete Phase-1 demonstration
scorecard. All are verifiable against BCMP-1 SIL run outputs. Values
reported are from Sprint S8 baseline (5 runs × 11 criteria).*

**11.1 BCMP-1 Acceptance Criteria (Primary Scorecard)**

  -----------------------------------------------------------------------------------
  **Criterion**   **Metric**            **Threshold**    **SIL Result**
  --------------- --------------------- ---------------- ----------------------------
  C-01            BIM spoof detection   ≤ 250 ms         PASS --- all 5 runs
                  latency                                

  C-02            GNSS exclusion when   Zero GNSS aiding PASS --- all 5 runs
                  trust = Red                            

  C-03            Navigation drift      \< 2% / 5 km     PASS --- all 5 runs
                  under GNSS denial                      

  C-04            TRN correction        Every 2 km; NCC  PASS --- all 5 runs
                  applied               ≥ 0.45           

  C-05            EW cost-map update    ≤ 500 ms         PASS --- all 5 runs
                  latency                                

  C-06            Route replans on EW   ≥ 2 replans;     PASS --- all 5 runs
                  pressure              each ≤ 1 s       

  C-07            ZPI zero-RF in SHM    Zero RF in ST-05 PASS --- all 5 runs

  C-08            DMRL target lock      Confidence ≥     PASS --- all 5 runs
                                        0.85; ≥ 5 frames 

  C-09            Thermal decoy         ≥ 0.80 / 3       PASS --- all 5 runs
                  rejected              consecutive      
                                        frames           

  C-10            L10s-SE decision      ≤ 2 s from       PASS --- all 5 runs
                  latency               activation       

  C-11            Mission log           ≥ 99% fields     PASS --- all 5 runs
                  completeness          captured         
  -----------------------------------------------------------------------------------

**11.2 Navigation KPIs**

  ------------------------------------------------------------------------------
  **KPI**                   **Threshold**   **Sensor    **Note**
                                            Profile**   
  ------------------------- --------------- ----------- ------------------------
  Maximum drift / 5 km      \< 2% of path   STIM300     BCMP-1 acceptance
  segment (BCMP-1, 100 km)  length                      criterion C-03

  Maximum drift / 5 km      \< 2% of path   STIM300     Extended corridor ---
  segment (ALS-250, 250 km) length                      ALS-250 chart pending
                                                        S8-D

  BIM processing latency    ≤ 250 ms        All sensors NFR-001

  TRN correction interval   Every 2 km      All sensors FR-107

  NCC correlation threshold ≥ 0.45          All sensors Corrections below
  (accepted corrections)                                threshold are rejected
                                                        and logged
  ------------------------------------------------------------------------------

**11.3 EW KPIs**

  -----------------------------------------------------------------------
  **KPI**                   **Threshold**    **Result**
  ------------------------- ---------------- ----------------------------
  EW cost-map update        ≤ 500 ms         PASS
  latency                                    

  Route replan latency per  ≤ 1 s            PASS
  event                                      

  Detection probability     ≥ 0.90           PASS --- NFR-006
  (Pd)                                       

  False alarm rate (Pfa)    ≤ 0.05           PASS --- NFR-007
  -----------------------------------------------------------------------

**11.4 Terminal Guidance KPIs**

  -----------------------------------------------------------------------
  **KPI**                **Threshold**    **Result**
  ---------------------- ---------------- -------------------------------
  DMRL target lock       ≥ 0.85           PASS --- all 5 runs
  confidence                              

  Temporal association   ≥ 5 frames @ 25  PASS --- all 5 runs
  frames                 FPS              

  Decoy rejection        ≥ 0.80 / 3       PASS --- all 5 runs
  confidence             consecutive      
                         frames           

  L10s-SE decision       ≤ 2 s from       PASS --- all 5 runs
  latency                activation       

  EO lock latency        ≤ 400 ms         PASS --- all 5 runs
                         (NFR-008)        

  Civilian detect        ≥ 0.70           PASS --- synthetic injection
  threshold (L10s-SE)    confidence       test
  -----------------------------------------------------------------------

**11.5 Test Coverage Summary**

  --------------------------------------------------------------------------------------
  **Sprint**   **Test Suite**                   **Tests**   **Result**
  ------------ -------------------------------- ----------- ----------------------------
  S1           test_sprint_s1_acceptance.py     9           ✅ PASS

  S2           test_sprint_s2_acceptance.py     9           ✅ PASS

  S3           test_sprint_s3_acceptance.py     8           ✅ PASS

  S4           test_sprint_s4_acceptance.py     8           ✅ PASS

  S5           test_s5_dmrl.py +                111         ✅ PASS
               test_s5_l10s_se.py +                         
               test_s5_bcmp1_runner.py                      

  S6           test_s6_zpi_cems.py              36          ✅ PASS

  S8           test_s8a--e (4 suites)           68          ✅ PASS

  Total        All suites --- master runner     215         **✅ 215/215 PASS**
  --------------------------------------------------------------------------------------

**Section 12**

**12. Design Decisions Retained**

*Two design decisions from Part Two V7 govern Phase-1 architecture. Both
are retained without change. Both are closed decisions --- they are not
subject to re-evaluation in this scope.*

**12.1 DD-01 --- Hardware Sovereignty & Compute Platform Selection**

No specific compute platform is named in the Phase-1 architecture. The
system is specified against a Compute Performance Specification.
Platform selection occurs post-SIL, following the TASL hardware
qualification phase.

  -----------------------------------------------------------------------
  **DD-01 Element**     **Decision**
  --------------------- -------------------------------------------------
  Compute platform      No. Hardware-agnostic architecture.
  named in Phase-1 SIL  

  Platform selection    Post-SIL; following TASL hardware qualification
  trigger               (HIL phase)

  Indigenous compute    CDAC Vega --- roadmapped; not a Phase-1
  target (Phase-3)      constraint

  SWaP specification    Compute Performance Specification (Section 4.3 of
  approach              V7). Platform-agnostic.

  Secure boot mechanism Hardware root of trust mandatory; specific
                        mechanism determined at hardware qualification
                        per DD-01

  Rationale             Avoiding platform lock-in before TASL hardware
                        decision preserves negotiation flexibility and
                        prevents architecture-breaking rework
  -----------------------------------------------------------------------

**12.2 DD-02 --- Cross-Mission Learning Architecture**

DD-02 is a two-phase design decision. Phase-1 minimum is active in the
current implementation. Phase-2 learning logic is deferred.

  ------------------------------------------------------------------------
  **DD-02      **Scope**                    **Status**
  Phase**                                   
  ------------ ---------------------------- ------------------------------
  Phase-1      Log schema captures          ACTIVE ---
  (Active)     learning-relevant fields;    logs/mission_log_schema.py
               mandatory pre-terminal ZPI   (S1)
               burst transmits EW           
               observation summary;         
               structure supports future    
               learning ingestion           

  Phase-2      Fleet-level cross-mission    DEFERRED --- post-HIL
  (Deferred)   learning pipeline; EW        
               observation aggregation;     
               model update workflows       
  ------------------------------------------------------------------------

> **DECISION STATUS:** *Both DD-01 and DD-02 are closed decisions. They
> were established in Part Two V7 after technical review and are carried
> forward without change into this Demo Edition. Neither decision
> requires re-evaluation for Phase-1 SIL completion.*

**Programme Statement**

**Programme Control**

***This is not a retreat in ambition. It is sequencing.***

MicroMind is a bounded, deterministic autonomy payload for tactical UAVs
and loitering munitions. Phase-1 proves that the autonomy core works ---
robustly, repeatably, and verifiably --- in a fully contested
environment. That is the correct objective for a first demonstration to
DRDO and TASL.

The Phase-1 demo must prove, quantitatively, seven statements:

-   The autonomy core survives 100 km of GNSS denial with drift \< 2%/5
    km --- verified on ALS-250 class platform with Safran STIM300
    tactical IMU characterisation.

-   It detects and classifies GNSS spoofing within 250 ms and scales
    ESKF trust accordingly.

-   It re-routes under electronic attack within 1 second of jammer
    detection.

-   It completes terminal engagement autonomously, rejects thermal
    decoys across ≥ 3 consecutive frames at ≥ 0.80 confidence, and locks
    the intended target at ≥ 0.85 confidence.

-   It enforces the L10s-SE deterministic abort/continue decision within
    2 seconds, with civilian detection at ≥ 0.70 confidence, and logs
    the full structured reasoning chain.

-   It enforces zero RF in the terminal phase using ZPI burst
    scheduling.

-   It logs everything --- ≥ 99% completeness --- with a self-contained
    HTML debrief report.

If these seven statements hold across 5 independent BCMP-1 runs, the
Phase-1 demonstration succeeds. The programme has already demonstrated
this in SIL at commit f91180d: 5/5 runs × 11/11 criteria.

Every deferred module is architecturally present, explicitly roadmapped,
and clearly visible to DRDO and TASL in Appendix A. The sprint work is
not lost --- it is correctly sequenced.

The programme is founder-led and resource-constrained. Cognitive load
and integration complexity are real risks. A complete demonstration of a
bounded, defensible capability set --- executed to 100% --- is more
credible than a partial demonstration of a broader digital warfare
stack. That is programme control.

**Appendix A**

**Appendix A --- Phase-2 / Phase-3 Capability Roadmap**

*The following capabilities are explicitly roadmapped. They are not
abandoned. Architecture and --- in several cases --- implementation are
retained in the repository. Reintroduction follows Phase-1 demonstration
acceptance.*

> **NOTE TO DRDO / TASL:** *This appendix demonstrates that MicroMind
> has long-term architectural depth. Every capability listed below is
> designed into the system. Phase-1 demonstrates the autonomy spine.
> Phase-2 and Phase-3 complete the digital warfare stack.*

  ------------------------------------------------------------------------------------------------
  **Phase**   **Capability**      **FR / Module**               **Current Status**   **Trigger**
  ----------- ------------------- ----------------------------- -------------------- -------------
  Phase-2     CEMS ---            FR-102 / core/cems/cems.py    IMPLEMENTED (S6,     After Phase-1
              Cooperative EW                                    commit a7633ab) ---  demo
              Sharing                                           frozen in repo       acceptance

  Phase-2     Predictive EW       core/ew_engine/ew_engine.py   Present in codebase  After Phase-1
              Modelling ---       (branch not invoked)          --- not invoked in   demo
              Kalman jammer                                     Phase-1              acceptance
              velocity + 30 s                                                        
              horizon                                                                

  Phase-2     Cybersecurity       FR-109--112 / S9 candidate A  Architecture ready   S9 ---
              Hardening ---                                     --- no blockers      pending TASL
              PQC-ready stack,                                                       meeting
              envelope                                                               
              verification, key                                                      
              management                                                             

  Phase-2     Satellite Avoidance FR-108                        FR specified in V7   After Phase-1
              / Terrain Masking                                 --- not yet          demo
              --- SGP4                                          implemented          acceptance
              propagation +                                                          
              depression logic                                                       

  Phase-2     DMRL CNN Upgrade    FR-103 upgrade / S9 candidate Blocked: GPU +       After
              --- trained CNN on  B                             dataset + threat     clearance
              Hailo-8 target                                    library clearance    obtained

  Phase-2     Advanced            Internal testing hooks        Internal hooks       After DMRL
              Adversarial                                       present --- not      CNN is live
              Robustness ---                                    surfaced in demo     
              FGSM/PGD red-team                                                      
              framework                                                              

  Phase-3     HIL Integration --- S9 candidate C                Blocked: TASL        After TASL
              ROS2 node wrappers,                               hardware platform    platform
              PX4 SITL skeleton                                 decision required    decision

  Phase-3     Cross-Mission       DD-02 Phase-2                 Log schema active    Post-HIL
              Learning Pipeline                                 (DD-02 Phase-1);     
              --- fleet-level EW                                learning logic       
              observation                                       deferred             
              aggregation                                                            

  Phase-3     Swarm Operations    FR-102 extension + new module CEMS foundation      Post-HIL
              --- multi-UAV CEMS                                present (S6)         
              coordination +                                                         
              orchestration layer                                                    

  Phase-3     Indigenous Compute  DD-01 post-SIL                Architecture         Post-HIL +
              Transition --- CDAC                               hardware-agnostic;   partnership
              Vega platform                                     platform TBD         formalised
              migration                                         post-TASL            
  ------------------------------------------------------------------------------------------------

**Appendix B**

**Appendix B --- Deferred FR Registry**

*Complete classification of all Functional Requirements from Part Two
V7. Phase-1 FRs are marked complete. All other FRs are classified with
their target phase and current implementation status.*

  -------------------------------------------------------------------------------------------
  **FR ID**     **Capability**       **Phase**     **Implementation     **Rationale for
                                                   Status**             Deferral**
  ------------- -------------------- ------------- -------------------- ---------------------
  FR-101        BIM --- GNSS         **Phase-1**   ✅ COMPLETE (S2)     Core Phase-1
                Integrity & Trust                                       requirement
                Scaling                                                 

  FR-103        DMRL --- Terminal    **Phase-1**   ✅ COMPLETE (S5)     Core Phase-1
                Multi-Frame                                             requirement
                Discriminator                                           

  FR-104        ZPI --- Zero RF      **Phase-1**   ✅ COMPLETE (S6)     Core Phase-1
                Burst Scheduler                                         requirement

  FR-105        L10s-SE ---          **Phase-1**   ✅ COMPLETE (S5)     Core Phase-1
                Deterministic Safety                                    requirement
                Engine                                                  

  FR-106        FSM --- NanoCorteX   **Phase-1**   ✅ COMPLETE (S1)     Core Phase-1
                7-State Autonomy                                        requirement
                Machine                                                 

  FR-107        TRN ---              **Phase-1**   ✅ COMPLETE (S3)     Core Phase-1
                Terrain-Referenced                                      requirement
                Navigation                                              

  FR-NAV-01     GNSS-Denied          **Phase-1**   ✅ COMPLETE (S0+S8)  Core Phase-1
                Navigation Drift                                        requirement
                Limit                                                   

  FR-EW-01/02   EW Reactive          **Phase-1**   ✅ COMPLETE (S4)     Core Phase-1
                Cost-Map + Hybrid                                       requirement
                A\* Replan                                              

  FR-102        CEMS --- Cooperative Phase-2       IMPLEMENTED (S6) --- Swarm intelligence
                EW Sharing                         frozen               not required to prove
                                                                        single-UAV autonomy
                                                                        core

  FR-108        Satellite Avoidance  Phase-2       FR specified --- not Not central to
                / SGP4 Terrain                     implemented          GNSS-denied nav demo;
                Masking                                                 high simulation
                                                                        overhead

  FR-109        Mission Envelope     Phase-2       Architecture         S9 candidate A; no
                Cryptographic        (S9-A)        specified --- not    blockers
                Validation                         implemented          

  FR-110        Adversarial          Phase-2       Internal hooks       Valuable internally;
                Robustness (CNN                    present --- not      not required for demo
                white-box; BIM                     surfaced             credibility
                black-box)                                              

  FR-111        Secure Boot and      Phase-2 (HIL) Abstracted per DD-01 Specific mechanism
                Firmware Integrity                 ---                  determined at
                                                   hardware-dependent   hardware
                                                                        qualification

  FR-112        Key Management ---   Phase-2       Algorithm specified  S9 candidate A; no
                provision, rotation, (S9-A)        in V7 --- not        blockers
                revocation                         implemented          

  DD-02 Ph.2    Cross-Mission        Phase-3       Log schema active;   Learning does not
                Learning Pipeline                  learning logic       change in-mission
                                                   deferred             behaviour; post-HIL
  -------------------------------------------------------------------------------------------

**Appendix C**

**Appendix C --- Diagram Inventory Classification**

*Classification of all diagrams from Part Two V7 and TechReview v1.1
under Demo Edition scope. Governs what is retained, what is simplified,
and what is moved to Appendix A.*

**C.1 Diagrams to Retain (Unchanged)**

  ---------------------------------------------------------------------------
  **Diagram**           **Source**   **Disposition**
  --------------------- ------------ ----------------------------------------
  NanoCorteX 7-State    V7 §1.2      RETAIN --- core demo asset; formalise
  FSM (UML statechart)               with UML notation for TASL presentation

  System Architecture   V7 §1.1      RETAIN --- essential context; no changes
  (3-subsystem block                 needed
  diagram: OS / X /                  
  NanoCorteX /                       
  Autopilot)                         

  BCMP-1 Scenario Map   V7 §1        RETAIN --- the normative test scenario;
  (100 km corridor,                  central to demo narrative
  jammer events,                     
  terminal target)                   

  BIM Trust State       V7 §1.8      RETAIN --- demonstrates GNSS integrity
  Diagram (G/A/R +                   logic concisely
  hysteresis + score                 
  weighting)                         

  L10s-SE Decision Tree V7 §1.12     RETAIN --- critical for TASL / DRDO
  (deterministic                     credibility; shows ethical constraint
  abort/continue)                    enforcement

  Authority Chain       V7 §1.3      RETAIN --- demonstrates chain of command
  Hierarchy Table                    architecture

  BCMP-1 9-Panel        S7 output    RETAIN --- demo presentation asset;
  Dashboard Screenshot               generated by
                                     dashboard/bcmp1_dashboard.py

  BCMP-1 HTML Debrief   S7 output    RETAIN --- demo evidence artefact;
  Report                             self-contained; generated by
                                     dashboard/bcmp1_report.py

  ALS-250 Three-Curve   S8-D pending RETAIN --- TASL-targeted quantitative
  Drift Chart (STIM300               navigation claim. To be inserted from
  / ADIS / BASELINE)                 overnight run output.
  ---------------------------------------------------------------------------

**C.2 Diagrams to Simplify for Demo Edition**

  ------------------------------------------------------------------------
  **Diagram**     **Current State       **Simplified Scope**
                  (V7)**                
  --------------- --------------------- ----------------------------------
  EW Engine       Shows reactive +      Retain reactive branch only:
  Architecture    predictive branches + detect → tile update → replan.
                  CEP quantification +  Remove predictive horizon and CEP
                  multi-node clustering quantification. Single linear flow
                                        diagram.

  DMRL            Full ML pipeline:     Show: preloaded models → NCC frame
  Architecture    adversarial testing,  association → multi-frame lock /
                  live model ingestion, decoy decision → L10s-SE handoff.
                  onboard retraining,   Remove: adversarial ML, retraining
                  adaptive behaviour    pipeline, live ingestion.

  ZPI Burst       Full diagram          Retain: burst timing, SHM
  Scheduler       including adaptive    suppression, mandatory
                  anti-DF response and  pre-terminal burst. Remove:
                  power shaping         adaptive anti-DF logic branch.
                  intelligence          

  Sensor Fusion   All inputs including  Retain all Phase-1 inputs. Remove
  Pipeline        SDR signature and DEM DEM-for-satellite-masking
                  masking for satellite annotation (FR-108 is deferred).
                  avoidance             
  ------------------------------------------------------------------------

**C.3 Diagrams Moved to Appendix A (Deferred)**

  ---------------------------------------------------------------------------
  **Diagram**           **Destination**   **Reason**
  --------------------- ----------------- -----------------------------------
  CEMS Multi-UAV Mesh   Appendix A ---    CEMS is Phase-2. Architecture is
  Topology (packet      Phase-2           retained in repo.
  flow, merge engine)                     

  Satellite Avoidance / Appendix A ---    FR-108 is Phase-2.
  SGP4 Propagation      Phase-2           
  Logic                                   

  Cross-Mission         Appendix A ---    Learning logic is Phase-3,
  Learning Data         Phase-3           post-HIL.
  Pipeline (DD-02                         
  Phase-2)                                

  Red-Team Adversarial  Internal          Not surfaced in demo document.
  Attack Framework      documentation     Internal use.
  (FGSM/PGD attack      only              
  flows)                                  

  Fleet-Level Swarm     Appendix A ---    Multi-UAV swarm is Phase-3.
  Coordination          Phase-3           
  Architecture                            

  CDAC Vega Indigenous  Appendix A ---    Platform migration is Phase-3,
  Compute Migration     Phase-3           post-HIL.
  Architecture                            
  ---------------------------------------------------------------------------

**Glossary**

**Glossary of Terms**

  ---------------------------------------------------------------------------
  **Term**      **Definition**
  ------------- -------------------------------------------------------------
  ALS-250       ALS-250 class tactical UAV / loitering munition --- the
                Phase-1 target platform for the SIL demonstration

  BCMP-1        Battlefield Corridor Mission Profile --- 1. The normative 100
                km contested corridor navigation and terminal engagement
                scenario.

  BIM           Beacon & Integrity Monitor (FR-101). GNSS trust evaluation
                and scoring module.

  CEMS          Cooperative EW Sharing (FR-102). Multi-UAV EW intelligence
                mesh --- Phase-2. Implemented in S6; frozen.

  DEM           Digital Elevation Map. Pre-loaded terrain data used for TRN
                correlation and route planning.

  DMRL          Deep Multi-frame Recognition & Lock (FR-103). Deterministic
                multi-frame terminal discriminator.

  ESKF          Error-State Kalman Filter. 15-state inertial navigation
                filter (V2, Sprint S0).

  FSM           Finite State Machine. NanoCorteX 7-state deterministic
                autonomy state machine (FR-106).

  GNSS          Global Navigation Satellite System.

  HIL           Hardware-in-the-Loop. Post-SIL phase requiring physical
                hardware integration.

  HKDF-SHA256   HMAC-based Key Derivation Function using SHA-256. Used for
                ZPI hop plan seed derivation from mission key.

  L10s-SE       Last-10-Second Safety Envelope (FR-105). Deterministic
                final-phase authority layer.

  LWIR          Long-Wave Infrared (8--14 μm). Mandatory thermal sensor band
                for DMRL operation.

  NCC           Normalised Cross-Correlation. DEM matching algorithm for TRN.

  NanoCorteX    The autonomy engine subsystem of MicroMind. Runs the FSM,
                navigation mode management, route replan, and L10s-SE
                enforcement.

  PQC           Post-Quantum Cryptography. SPHINCS+ / Dilithium signature
                schemes.

  RADALT        Radar Altimeter. Provides height AGL for TRN. Simulated in
                SIL; physical in HIL/Production.

  ROE           Rules of Engagement. Encoded in the signed mission envelope
                (Layer 0 authority).

  SDR           Software-Defined Radio. Used for EW wideband monitoring.

  SHM           Silent Homing Mode. Zero-RF terminal navigation mode (ST-05).

  SIL           Software-in-the-Loop. The current development and validation
                phase.

  TASL          Tata Advanced Systems Limited. Target hardware partner
                post-SIL proof.

  TRN           Terrain-Referenced Navigation (FR-107). Inertial drift
                correction using RADALT + DEM correlation.

  VIO           Visual-Inertial Odometry. Pose estimation from camera
                features + IMU integration.

  ZPI           Zero Probability of Intercept burst scheduler (FR-104). LPI
                communications for transit telemetry and EW updates.
  ---------------------------------------------------------------------------
