**MicroMind / NanoCorteX**

**Part Two --- Demo Edition**

*DRDO / TASL Phase-1 SIL Demonstration*

Scope Realignment --- Controlled Fork from V7 + TechReview v1.1

Programme Director: Amit (amitkr-anand)

Repository: github.com/amitkr-anand/micromind-autonomy

Status at fork: S8 COMPLETE \| 215/215 tests passing \| commit f91180d

Date: 28 February 2026

*This document is a scope isolation exercise --- not an architectural
redesign. All completed sprint work remains valid. The goal is 100% of a
bounded, defensible Phase-1 capability set rather than 60% of an
expansive digital warfare stack.*

**1. Impact Analysis --- Pre-Restructuring**

Before any documentation changes, a structured impact assessment was
conducted across four questions raised in the prompt. The objective is
to preserve development momentum and minimise disruption.

**1.1 Sprint Deliverables Fully Valid Under Narrowed Demo Scope**

  --------------------------------------------------------------------------------
  **Sprint**   **Module(s)**         **Verdict**      **Rationale**
  ------------ --------------------- ---------------- ----------------------------
  S0           ESKF V2, quaternion,  ✅ Fully Valid   Core nav math ---
               INS mechanisation,                     indispensable. Zero changes
               constants                              required.

  S1           FSM (7-state),        ✅ Fully Valid   FSM IS the autonomy spine.
               SimClock,                              Log schema stays (DD-02
               MissionLogSchema,                      Phase-1). BCMP-1 scenario is
               BCMP-1 scenario                        the normative test.

  S2           BIM trust scorer,     ✅ Fully Valid   BIM ≤250 ms is a Phase-1
               GNSS spoof injector                    KPI. Spoof injector is the
                                                      test harness.

  S3           TRN stub, nav         ✅ Fully Valid   TRN + VIO is Phase-1 core.
               scenario (50 km),                      Dashboard is a demo asset.
               Plotly dashboard                       No changes.

  S4           EW engine, Hybrid     ✅ Fully Valid   Reactive cost-map + replan
               A\*, BCMP-1 EW sim    --- with         are Phase-1 core. Predictive
                                     simplification   EW deferred (see Section 3).
                                     note             No code changes needed ---
                                                      scope boundary is
                                                      documentation only.

  S5           DMRL stub, L10s-SE,   ✅ Fully Valid   These are Phase-1
               BCMP-1 full runner                     non-negotiable. BCMP-1
                                                      runner is the acceptance
                                                      harness.

  S6           ZPI burst scheduler,  ⚡ Partial ---   ZPI (zero-RF terminal phase)
               CEMS engine,          ZPI valid; CEMS  is Phase-1 core. CEMS
               multi-UAV sim         deferred         (multi-UAV EW sharing) is
                                                      deferred to Phase-2. No code
                                                      deletion --- CEMS is
                                                      reclassified.

  S7           bcmp1_dashboard.py,   ✅ Fully Valid   Mission debrief report and
               bcmp1_report.py                        dashboard are demo
                                                      presentation assets.

  S8           IMU model,            ✅ Fully Valid   IMU characterisation
               mechanisation                          supports TASL credibility.
               extension, ALS-250                     ALS-250 250 km corridor
               sim, BCMP-1 IMU ext.                   directly substantiates
                                                      Phase-1 navigation claims.
  --------------------------------------------------------------------------------

**1.2 Modules Requiring Only Documentation Reclassification (No Code
Changes)**

The following modules are fully implemented and passing tests. They
require zero code modification --- only their documentation category
changes from \'Phase-1\' to \'Phase-2 Roadmap\':

  ---------------------------------------------------------------------------------------
  **Module**            **Current          **Demo Edition     **Action**
                        Classification**   Classification**   
  --------------------- ------------------ ------------------ ---------------------------
  CEMS --- Cooperative  S6 delivery,       Phase-2 Roadmap    Documentation
  EW Sharing            FR-102                                reclassification only. Code
  (core/cems/cems.py)                                         frozen in place.

  bcmp1_cems_sim.py     S6 acceptance sim  Phase-2 test asset Retained in repo, removed
                                                              from demo narrative.

  Predictive EW         S4 feature         Phase-2 Roadmap    EW engine simplified to
  modelling (EW engine                                        reactive cost-map. No code
  --- Kalman jammer                                           change needed ---
  velocity)                                                   predictive branch simply
                                                              not exercised in demo.

  Advanced adversarial  Internal testing   Phase-2 internal   Not surfaced in demo
  ML (FGSM/PGD hooks)                                         documentation. No change to
                                                              code.

  Cross-mission         Log schema         Phase-2 Roadmap    Log schema retained as-is.
  learning pipeline     captures fields                       Learning logic never
  (DD-02 Phase-2)                                             implemented in Phase-1.

  Satellite masking /   V7 FR, not yet     Phase-2 Roadmap    FR removed from Phase-1
  SGP4 (FR-108)         implemented                           scope list. No code exists
                                                              to remove.
  ---------------------------------------------------------------------------------------

**1.3 Modules Requiring Minor Feature Isolation (Feature Flags)**

No module requires structural modification. One module benefits from a
documentation boundary flag to prevent scope creep:

  -----------------------------------------------------------------------------------
  **Module**                    **Isolation      **Detail**
                                Mechanism**      
  ----------------------------- ---------------- ------------------------------------
  core/ew_engine/ew_engine.py   Documentation    Reactive cost-map behaviour (jam
                                boundary only    detection → tile update → replan) is
                                                 already the primary code path.
                                                 Predictive jammer velocity
                                                 sub-feature is present in code but
                                                 not exercised by Phase-1 test suite.
                                                 No feature flag needed --- the demo
                                                 simply does not invoke it.

  core/cems/cems.py             Import guard     CEMS import in bcmp1_cems_sim.py is
                                (optional)       isolated from bcmp1_runner.py. The
                                                 runner does not import CEMS. No
                                                 changes required to run BCMP-1
                                                 without CEMS.

  core/ins/imu_model.py ---     No isolation     Multiple profiles exist. Demo
  ADIS16505_3 / BASELINE        needed           selects STIM300 (tactical grade).
  profiles                                       Other profiles remain as-is for
                                                 internal benchmarking.
  -----------------------------------------------------------------------------------

**1.4 Sprint Trajectory --- Can S1--S8 Continue Without Re-planning?**

**Verdict:** Yes --- the Sprint trajectory requires no re-planning. S9
scope options are unaffected.

Reasoning:

-   Sprints S0--S8 are complete and passing (215/215 tests). This
    realignment has no retroactive impact on any sprint.

-   CEMS (S6) is the only module moving to Phase-2. It is already
    implemented, tested, and merged. The reclassification is a
    documentation act only.

-   S9 candidate forks (Cybersecurity / DMRL CNN / HIL / S8-D chart) are
    all Phase-1 aligned or hardware-blocked. None are affected by this
    scope isolation.

-   The STIM300 ARW spec update (V7 floor: 0.1→0.2°/√hr) is the only
    required document change carried forward from S8. This Demo Edition
    incorporates the correction.

> *⚠ This fork reduces integration complexity. It introduces zero new
> rework. Development momentum is fully preserved.*

**2. Demo Edition --- Document Structure**

The following is the complete outline for MicroMind Part Two --- Demo
Edition (DRDO/TASL Phase-1). It consolidates V7 and TechReview v1.1,
removes deferred scope, and adds the Phase-2 roadmap as a visible
appendix.

  ------------------------------------------------------------------------------
  **\#**   **Section**           **Content**
  -------- --------------------- -----------------------------------------------
  0        Cover + Programme     Purpose statement, fork justification, scope
           Guardrails            boundary declaration

  1        Operational Context   100 km contested corridor, GNSS denial,
           --- BCMP-1 Normative  adaptive jamming, thermal target + decoy.
           Scenario              ALS-250 platform.

  2        System Architecture   Three-subsystem view: MicroMind-X (sensor
           --- Autonomy Core     fusion), NanoCorteX (autonomy engine),
                                 MicroMind-OS (mission envelope). UML FSM.

  3        Phase-1 Functional    FR-101 through FR-108 --- Phase-1 only. Each FR
           Requirements (FR      with boundary constant, acceptance criterion,
           List)                 and sprint traceability. (See Section 3 of this
                                 document.)

  4        Navigation Subsystem  ESKF V2, BIM trust scoring, VIO + TRN
                                 (RADALT-validated in SIL), GNSS-denied
                                 propagation. IMU characterisation: STIM300 /
                                 ADIS16505-3 / BASELINE. ALS-250 250 km corridor
                                 results.

  5        EW Engine + Route     Reactive cost-map (jam/spoof detection → tile
           Planner               update ≤500 ms → Hybrid A\* replan ≤1 s).
                                 Scope: reactive only. Predictive EW deferred to
                                 Phase-2.

  6        Terminal Guidance --- Multi-frame EO lock ≥0.85, decoy rejection
           DMRL-Lite             ≥0.80/3 frames, 3--4 preloaded target models,
                                 SHA-256 envelope binding. Synthetic +
                                 commercial datasets only.

  7        Safety & Authority    Deterministic abort/continue tree, decision
           Chain --- L10s-SE     window ≤2 s, civilian detect ≥0.70, structured
                                 reasoning log.

  8        Zero-RF Terminal      Burst scheduler: ≤10 ms bursts, 2--30 s random
           Phase --- ZPI         intervals, mandatory pre-terminal burst, zero
                                 RF in SHM.

  9        Mission Logging       Log completeness ≥99%, schema v1 (DD-02
                                 Phase-1), HTML debrief report.

  10       SIL Sprint Plan ---   Sprint S0--S9 mapped to demo milestones. (See
           Demo Milestone        Section 4 of this document.)
           Alignment             

  11       KPI Dashboard ---     11 BCMP-1 criteria, 5 navigation KPIs, 3 EW
           Demo-Critical Metrics KPIs, 4 terminal guidance KPIs.

  12       Design Decisions      DD-01: bounded determinism / authority chain.
           Retained              DD-02 Phase-1: log schema.

  A        Appendix A ---        CEMS, predictive EW, satellite masking,
           Phase-2 / Phase-3     cross-mission learning, adversarial robustness,
           Capability Roadmap    swarm ops, CDAC Vega transition.

  B        Appendix B ---        Full FR list with Phase-1 / Phase-2 / Phase-3
           Deferred FR Registry  classification.

  C        Appendix C ---        Which diagrams remain / simplified / deferred.
           Diagram Inventory     (See Section 5 of this document.)
  ------------------------------------------------------------------------------

**3. Phase-1 Functional Requirements --- Revised FR List**

The following 8 FRs constitute the complete Phase-1 scope. All other FRs
from Part Two V7 are reclassified to Phase-2 or Phase-3 and appear in
Appendix B of the Demo Edition.

**3.1 Phase-1 FR Set**

  -------------------------------------------------------------------------------------
  **FR**        **Capability**       **Boundary              **Sprint**   **Status**
                                     Constant(s)**                        
  ------------- -------------------- ----------------------- ------------ -------------
  FR-101        BIM --- GNSS         Trust→Red \< 0.1; Spoof S2           ✅ COMPLETE
                Integrity + Trust    detect ≤250 ms;                      
                Scaling              3-sample hysteresis                  

  FR-103        DMRL-Lite ---        Lock conf ≥0.85; Decoy  S5           ✅ COMPLETE
                Terminal Decoy       abort ≥0.80/3 frames;                
                Rejection            Min dwell 5×@25 FPS;                 
                                     Aimpoint ±15°; Reacq                 
                                     timeout 1.5 s                        

  FR-104        ZPI --- Zero-RF      Burst ≤10 ms; Interval  S6           ✅ COMPLETE
                Terminal Phase Burst 2--30 s random;                      
                Scheduler            Pre-terminal mandatory               
                                     burst; Zero RF in SHM                

  FR-105        L10s-SE ---          Decision timeout ≤2 s;  S5           ✅ COMPLETE
                Deterministic        Civilian detect ≥0.70;               
                Abort/Continue       Structured reasoning                 
                Safety Engine        log                                  

  FR-106        FSM --- NanoCorteX   Transition latency ≤2   S1           ✅ COMPLETE
                7-State Autonomy     s; Deterministic                     
                State Machine        UML-formalised                       

  FR-107        TRN ---              Correction every 2 km;  S3           ✅ COMPLETE
                Terrain-Referenced   NCC threshold ≥0.45;                 
                Navigation           RADALT mandatory for                 
                (RADALT-validated in TRN validation                       
                SIL)                                                      

  FR-NAV-01     GNSS-Denied          Drift \< 2%/5 km; 100   S0+S8        ✅ COMPLETE
                Navigation Drift     km corridor; 250 km                  
                Limit                corridor (ALS-250)                   

  FR-EW-01/02   EW Reactive          Cost-map update ≤500    S4           ✅ COMPLETE
                Cost-Map + Hybrid    ms; Replan ≤1 s                      
                A\* Replan                                                
  -------------------------------------------------------------------------------------

**3.2 Deferred FRs --- Phase-2 Roadmap**

  ----------------------------------------------------------------------------------
  **FR**        **Capability**        **Phase**    **Rationale for Deferral**
  ------------- --------------------- ------------ ---------------------------------
  FR-102        CEMS --- Cooperative  Phase-2      Swarm logic not required to prove
                EW Sharing                         single-UAV autonomy core. S6
                (multi-UAV)                        implementation retained, frozen.

  FR-108        Satellite Avoidance / Phase-2      Not central to GNSS-denied nav
                Terrain Masking                    demo. High simulation overhead.
                (SGP4)                             Not required for artillery
                                                   scenario.

  FR-109--112   Cybersecurity         Phase-2 (S9  Architecture ready. No blockers.
                Hardening (PQC stack, candidate)   Deferred pending TASL meeting
                envelope                           outcome.
                verification)                      

  DD-02 Phase-2 Cross-Mission         Phase-3      Log schema captures fields
                Learning Pipeline                  (Phase-1 active). Learning logic
                                                   is post-HIL.

  Internal      Adversarial ML        Phase-2      Internal testing hooks retained.
                Red-Team (FGSM/PGD)   internal     Not surfaced as demo feature.

  Internal      Predictive EW         Phase-2      Reactive cost-map sufficient for
                Modelling (Kalman                  demo. Predictive branch present
                jammer velocity)                   in code, not exercised.
  ----------------------------------------------------------------------------------

**4. SIL Sprint Plan --- Demo Milestone Alignment**

The following maps all completed sprints to the Phase-1 demo milestone
structure. S9 scope options are listed below. No sprint re-planning is
required.

**4.1 Completed Sprints --- Demo Milestone Map**

  -----------------------------------------------------------------------------------------
  **Sprint**   **Commit**   **Demo Milestone** **Capability Proven**            **Tests**
  ------------ ------------ ------------------ -------------------------------- -----------
  S0           6e1c70a      M1 --- Nav         ESKF V2, quaternion math, INS    ---
                            Foundation         mechanisation, ENU frame         

  S1           5005a5d      M1 --- Autonomy    7-state FSM, SimClock, mission   9/9
                            Shell              log schema (DD-02 P1), BCMP-1    
                                               scenario                         

  S2           e86140f      M2 --- GNSS        BIM trust scorer ≤250 ms, GNSS   9/9
                            Integrity          spoof injector                   

  S3           284acb4      M2 --- GNSS-Denied TRN NCC stub, 50 km nav          8/8
                            Nav                scenario, Plotly dashboard       

  S4           366f963      M3 --- EW          EW reactive cost-map ≤500 ms,    8/8
                            Survivability      Hybrid A\* replan ≤1 s, BCMP-1   
                                               EW sim                           

  S5           7ad5db5      M4 --- Terminal    DMRL-Lite (lock/decoy            111/111
                            Autonomy           rejection), L10s-SE              
                                               abort/continue, BCMP-1 full      
                                               runner (11 criteria)             

  S6           a7633ab      M3 (ZPI) / Phase-2 ZPI zero-RF burst scheduler      36/36
                            (CEMS)             \[Phase-1\]. CEMS multi-UAV      
                                               \[Phase-2 frozen\].              

  S7           aa3302a      M5 --- Demo        BCMP-1 9-panel dashboard,        ---
                            Presentation       self-contained HTML debrief      
                                               report                           

  S8           f91180d      M2 --- IMU         STIM300/ADIS16505-3 noise        68/68
                            Characterisation   models, 250 km ALS-250 corridor  
                                               sim, BCMP-1 IMU integration      
  -----------------------------------------------------------------------------------------

**4.2 Sprint S9 --- Scope Options (Pending TASL Meeting)**

S9 is not started. The following options remain valid under the Demo
Edition scope:

  --------------------------------------------------------------------------------------
  **Option**   **Fork**         **Modules**             **Demo           **Readiness**
                                                        Relevance**      
  ------------ ---------------- ----------------------- ---------------- ---------------
  A            Cybersecurity    core/cybersec/ --- key  High --- mission No blockers
               Hardening        loading, envelope       envelope         
                                verification, PQC-ready integrity is a   
                                stack (FR-109--112)     TASL credibility 
                                                        requirement      

  B            DMRL CNN Upgrade Replace rule-based DMRL High ---         Blocked: GPU +
                                stub with trained CNN   upgrades FR-103  dataset +
                                (Hailu-8 target)        from rule-based  Indigenous
                                                        to learned       Threat Library
                                                        classifier       clearance

  C            HIL Integration  ROS2 node wrappers, PX4 Critical         Blocked:
               Prep             SITL skeleton           post-SIL ---     hardware
                                                        enables hardware platform
                                                        bridge           decision from
                                                                         TASL

  D            S8-D Chart + S6  als250_drift_chart.py   Immediate ---    Ready once
               5× Clean Sweep   (3-curve TASL chart) +  completes S8     overnight run
                                CEMS diagnostic         deferred item    completes
                                                        and TASL         
                                                        presentation     
                                                        chart            
  --------------------------------------------------------------------------------------

**4.3 Demo Milestone Summary**

  -------------------------------------------------------------------------
  **Milestone**   **Gate Criterion**    **Evidence**          **Status**
  --------------- --------------------- --------------------- -------------
  M1 --- Autonomy FSM transitions       S0 + S1 passing; ESKF ✅
  Core            deterministic; INS    V2 verified           
                  propagates without                          
                  drift                                       

  M2 ---          Drift \<2%/5 km over  S2, S3, S8 passing;   ✅ (S8-D
  GNSS-Denied     100 km and 250 km;    ALS-250 overnight run chart
  Navigation      BIM latency ≤250 ms;                        pending)
                  TRN every 2 km                              

  M3 --- EW       Cost-map ≤500 ms;     S4 + S6 ZPI passing;  ✅
  Survivability   Replan ≤1 s; ZPI      BCMP-1 EW sim passing 
                  zero-RF enforced                            

  M4 --- Terminal DMRL lock ≥0.85;      S5 111/111; BCMP-1    ✅
  Autonomy        Decoy reject          5/5 runs × 11/11      
                  ≥0.80/3f; L10s        criteria              
                  decision ≤2 s                               

  M5 --- Demo     HTML debrief report;  S7 dashboard/report   ✅ (chart
  Presentation    9-panel dashboard;    passing; drift chart  pending)
                  TASL-ready ALS-250    pending S8-D          
                  drift chart                                 
  -------------------------------------------------------------------------

**5. Diagram Inventory --- Retain / Simplify / Defer**

The following classifies all diagrams from V7 and TechReview v1.1 under
Demo Edition scope:

**5.1 Diagrams to Retain (unchanged)**

  --------------------------------------------------------------------------
  **Diagram**           **Source**   **Rationale**
  --------------------- ------------ ---------------------------------------
  NanoCorteX 7-State    V7 §3        Core demo asset --- proves
  FSM (UML)                          deterministic autonomy. Formalise with
                                     UML notation for TASL.

  System Architecture   V7 §2        Essential context. Retain as-is.
  (3-subsystem block                 
  diagram)                           

  BCMP-1 Scenario Map   V7 §1        The normative test scenario. Central to
  (100 km corridor)                  demo narrative.

  BIM Trust State       V7 §4        Demonstrates GNSS integrity logic.
  Diagram (G/A/R +                   Keep.
  hysteresis)                        

  L10s-SE Decision Tree V7 §7        Deterministic safety logic --- critical
                                     for TASL / DRDO credibility.

  BCMP-1 9-Panel        S7 output    Demo presentation asset.
  Dashboard Screenshot               

  BCMP-1 HTML Debrief   S7 output    Demo evidence artefact.
  Report                             

  ALS-250 Three-Curve   S8-D pending TASL-targeted quantitative navigation
  Drift Chart (STIM300               claim. Must be generated from overnight
  / ADIS / BASELINE)                 run before demo.
  --------------------------------------------------------------------------

**5.2 Diagrams to Simplify**

  ------------------------------------------------------------------------
  **Diagram**      **Current State**     **Simplification**
  ---------------- --------------------- ---------------------------------
  EW Engine        Shows reactive +      Retain reactive branch only
  Architecture     predictive branches + (detect → tile update → replan).
                   CEP quantification +  Remove predictive horizon and
                   multi-node clustering clustering sophistication. Single
                                         flow diagram.

  DMRL             Full ML pipeline with Show: preloaded models → NCC
  Architecture     adversarial testing,  frame association → lock/decoy
                   live model ingestion, decision → L10s handoff. Remove:
                   onboard retraining    adversarial ML, retraining
                                         pipeline, live ingestion.

  ZPI Burst        Full diagram          Retain: burst timing, SHM
  Scheduler        including adaptive    suppression, pre-terminal burst.
                   anti-DF and power     Remove: adaptive anti-DF
                   shaping               intelligence branch.
  ------------------------------------------------------------------------

**5.3 Diagrams to Remove or Defer to Appendix A**

  -----------------------------------------------------------------------
  **Diagram**           **Reason**
  --------------------- -------------------------------------------------
  CEMS Multi-UAV Mesh   Phase-2. Move to Appendix A roadmap section.
  Topology              

  Satellite Avoidance / Phase-2. Remove from main body. Note in Appendix
  SGP4 Propagation      A.

  Cross-Mission         Phase-3. Move to Appendix A.
  Learning Data         
  Pipeline              

  Red-Team Adversarial  Internal only. Not surfaced in demo document.
  Attack Framework      
  (FGSM/PGD)            

  Fleet-Level Swarm     Phase-3. Move to Appendix A.
  Coordination          
  Architecture          

  CDAC Vega Indigenous  Phase-3. Move to Appendix A.
  Compute Migration     
  Architecture          
  -----------------------------------------------------------------------

**6. Programme Positioning --- Clean Narrative for DRDO / TASL**

**6.1 What MicroMind Is**

**MicroMind is a bounded, deterministic autonomy payload for tactical
UAVs and loitering munitions. It provides survivable navigation and
terminal guidance in fully contested environments --- without GNSS,
without RF link, and with decoy rejection --- through a formalised
authority chain that enforces ethical engagement constraints at every
decision node.**

**6.2 What MicroMind Is Not**

  -----------------------------------------------------------------------
  **MicroMind is NOT**  **Why this matters for DRDO / TASL**
  --------------------- -------------------------------------------------
  A free-form AI weapon Every decision is bounded by a deterministic FSM.
  system                No unbounded inference at engagement time.
                        Authority chain is formalised and auditable.

  A full digital        Phase-1 demonstrates the autonomy core. Swarm, EW
  warfare stack         prediction, cross-mission learning, and satellite
                        masking are explicit Phase-2/3 roadmap items ---
                        not demo scope.

  A black-box ML system DMRL-Lite uses preloaded, versioned, SHA-256
                        envelope-bound target models. Model changes are
                        pre-mission only. No onboard retraining. No live
                        ingestion.

  A classified data     Phase-1 uses synthetic and commercially available
  handler               datasets. The architecture is sovereign-ready for
                        customer-provided threat libraries in Phase-2 ---
                        but the infrastructure is not built in Phase-1.

  A replacement for the MicroMind is an autonomy payload that provides
  autopilot             intelligent guidance TO existing autopilots ---
                        not a full autopilot replacement.
  -----------------------------------------------------------------------

**6.3 The Phase-1 Demo Proof Statement**

A successful Phase-1 SIL demonstration proves, quantitatively:

-   The autonomy core survives 100 km of GNSS denial with drift \< 2%/5
    km --- verified on ALS-250 platform with STIM300 tactical IMU.

-   It detects and classifies GNSS spoofing within 250 ms and scales
    ESKF trust accordingly.

-   It re-routes under electronic attack within 1 second of jammer
    detection --- reactive, not reactive-and-predictive.

-   It completes terminal engagement autonomously, rejects thermal
    decoys over ≥3 consecutive frames at ≥0.80 confidence, and locks the
    intended target at ≥0.85 confidence.

-   It enforces the L10s-SE deterministic abort/continue decision within
    2 seconds, with civilian detection at ≥0.70 confidence, and logs the
    full structured reasoning chain.

-   It enforces zero RF in the terminal phase (SHM) using ZPI burst
    scheduling.

-   It logs everything --- ≥99% completeness --- with a self-contained
    HTML debrief report.

If these seven statements hold across 5 independent BCMP-1 runs, the
demo succeeds. Everything else is Phase-2.

**6.4 Phase-2 / Phase-3 Capability Roadmap (Investor and DRDO
Visibility)**

The Demo Edition document will contain Appendix A with the following
roadmap --- ensuring DRDO and TASL see long-term depth:

  ------------------------------------------------------------------------------
  **Phase**   **Capability**        **FR / Module**     **Trigger**
  ----------- --------------------- ------------------- ------------------------
  Phase-2     CEMS --- Cooperative  FR-102 /            After Phase-1 demo
              EW Sharing            core/cems/cems.py   acceptance
                                    (S6, frozen)        

  Phase-2     Predictive EW         EW engine extension After Phase-1 demo
              Modelling (Kalman                         acceptance
              jammer velocity + 30                      
              s horizon)                                

  Phase-2     Cybersecurity Stack   FR-109--112 / S9    S9 --- no blockers
              (PQC-ready, envelope  candidate A         
              verification)                             

  Phase-2     Satellite Masking /   FR-108              After Phase-1 demo
              Terrain Avoidance                         acceptance
              (SGP4)                                    

  Phase-2     DMRL CNN Upgrade      FR-103 upgrade / S9 After GPU + dataset
              (Hailo-8 target)      candidate B         clearance

  Phase-2     Advanced Adversarial  Internal module     After DMRL CNN is live
              Robustness (FGSM/PGD)                     

  Phase-3     HIL Integration ---   S9 candidate C      After TASL platform
              ROS2 wrappers, PX4                        decision
              SITL                                      

  Phase-3     Cross-Mission         DD-02 Phase-2       Post-HIL
              Learning Pipeline                         

  Phase-3     Swarm Operations      CEMS + new          Post-HIL
              (multi-UAV            orchestration layer 
              coordination)                             

  Phase-3     Indigenous Compute    Platform migration  Post-HIL + partnership
              Transition (CDAC                          
              Vega)                                     
  ------------------------------------------------------------------------------

**6.5 Programme Control Statement**

***This is not a retreat in ambition. It is programme control.***

The programme is founder-led and resource-constrained. Cognitive load
and integration complexity are real programme risks. A bounded,
defensible demonstration of the autonomy core --- executed to 100% ---
is more credible to DRDO and TASL than a partial demonstration of a
broader digital warfare stack.

Every deferred module is architecturally present and explicitly
roadmapped. The sprint work is not lost --- it is correctly sequenced.

**Immediate Next Actions**

  ----------------------------------------------------------------------------------
  **Priority**   **Action**                      **Owner**    **Deadline**
  -------------- ------------------------------- ------------ ----------------------
  1              Generate ALS-250 drift chart    Amit +       Morning after
                 (S8-D) once overnight run       Claude       overnight run
                 completes                                    

  2              Update Part Two V7 spec:        Amit         Before TASL meeting
                 STIM300 ARW floor 0.1→0.2°/√hr               

  3              Confirm S9 scope with TASL      Amit         Post-TASL meeting
                 meeting outcome (Option A                    
                 Cybersec recommended as                      
                 no-blocker)                                  

  4              Generate Demo Edition document  Claude       This session or next
                 from this outline (Demo Edition              
                 v1.0)                                        

  5              Archive this realignment        Amit         End of session
                 document in Daily Logs/                      
  ----------------------------------------------------------------------------------
