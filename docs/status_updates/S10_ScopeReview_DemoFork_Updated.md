**MicroMind / NanoCorteX**

S10 Scope Review --- Demonstration Fork Alignment

Programme Director: Amit (amitkr-anand) \| Date: 4 March 2026 \|
Updated: 8 March 2026

Reference documents: SPRINT_STATUS.md · HANDOFF_S9_to_S10.md · Demo
Edition v1.0 · Realignment Analysis · TRN Sandbox Closure Report

**Governing constraint: Deliver 100% of a bounded, defensible autonomy
core --- not 60% of a broader digital warfare stack.**

A S10 Scope Review Under Demonstration Fork

Sprint S9 closed the NAV-01 requirement with a 7× performance margin
(14.5 m / 5 km against a 100 m limit). The pre-S10 TRN sandbox
additionally confirmed NAV-01 compliance on a 150 km manoeuvring
Himalayan corridor with heading changes up to 60° (NAV-01 = 33.3 m, 3.0×
margin). The navigation architecture is validated on both linear and
manoeuvring flight profiles. The question for S10 is not what can be
built --- it is what the demonstration requires.

A.1 Candidate Assessment

  ----------------- ----------------------- -------------- -------------------
  **S10 Candidate** **What it does**        **Demo         **Verdict**
                                            relevance**    

  **S8-D Drift      Three-curve ALS-250     **CRITICAL**   IN S10. Milestone
  Chart**           chart --- STIM300 vs                   M2/M5 gate. Primary
                    ADIS16505-3 vs BASELINE                TASL artefact.
                    drift over 250 km.                     Completes M5 demo
                    Quantitative TASL                      presentation.
                    presentation artefact.                 

  NCC Vectorisation Replaces pure-Python    **ENABLER**    IN S10.
                    NCC loop with numpy                    Infrastructure
                    sliding_window_view.                   prerequisite. Low
                    Prerequisite for S8-D                  regression risk.
                    250 km three-model run.                

  **S9 Regression   test_s9_nav01_pass.py   **REQUIRED**   IN S10. Converts
  Test**            --- 7 automated gates                  simulation evidence
                    protecting the S9                      into automated
                    TRN+ESKF architecture.                 gate. Grows suite
                                                           to 222/222.

  Terrain           Log σ_terrain, NCC      **ADDED**      IN S10 (Task 5).
  Observability     innovations, and                       Implement alongside
  Instrumentation   suppression events to                  S10-1. No
                    mission log. Enables                   regression risk.
                    offline envelope                       Required to support
                    analysis.                              TASL technical
                                                           questions on
                                                           terrain coverage.

  Manoeuvring       Automated test: 150 km  **ADDED**      IN S10 (Task 6).
  Corridor          five-segment corridor                  Packages sandbox
  Regression        with heading changes up                result as permanent
                    to 60°. Validated by                   regression
                    sandbox --- formalises                 protection.
                    the result as a                        Demonstrates
                    regression gate.                       manoeuvring
                                                           capability to TASL.

  Cybersecurity     core/cybersec/ ---      **DEFERRED**   Phase-2 candidate.
  (FR-109--112)     signed mission                         No BCMP-1 KPI
                    envelope, PQC-ready                    impact. Add after
                    adapter stubs.                         TASL meeting
                                                           confirms programme
                                                           continuation.

  HIL Integration   ROS2 node wrappers, PX4 **BLOCKED**    Phase-3.
  Prep              SITL skeleton.                         Hard-blocked on
                                                           TASL platform
                                                           decision. Out of
                                                           scope for S10.
  ----------------- ----------------------- -------------- -------------------

A.2 Verdict Summary

  --------------------- ------------- ------------------------------------
  **Item**              **S10         **Rationale**
                        Decision**    

  **NCC Vectorisation** **IN ---      Infrastructure prerequisite for
                        first**       S8-D. Zero demo-visible risk.

  **S9 Regression       **IN ---      Converts simulation-validated NAV-01
  Test**                second**      into automated gate. Grows suite to
                                      222/222.

  **S8-D Drift Chart**  **IN ---      Critical TASL presentation artefact.
                        third**       Completes milestone M5. Requires NCC
                                      speed.

  Terrain Observability **IN --- Task Implement during S10-1. Logs
  Instrumentation       5**           σ_terrain, innovations, suppression
                                      events. Supports TASL technical
                                      review.

  Manoeuvring Corridor  **IN --- Task Formalises sandbox manoeuvring
  Regression            6**           result as permanent regression gate.
                                      Demonstrates 60° heading change
                                      capability.

  Cybersecurity         DEFERRED to   No BCMP-1 KPI impact. Realignment
  FR-109--112           Phase-2       Analysis already classifies as
                                      Phase-2.

  HIL Integration       DEFERRED to   Hard-blocked on TASL hardware
                        Phase-3       decision. Cannot start.
  --------------------- ------------- ------------------------------------

B Proposed S10 Sprint --- Updated

B.1 Sprint Goal Statement

S10 objective: Complete Milestone M5 (Demo Presentation). Generate the
ALS-250 three-curve drift chart with automated architecture protection
and terrain observability instrumentation. No new functional modules.

B.2 Deliverables and Sequence

  ----------- ----------------- ---------------------------------- ----------------- ----------
  **\#**      **Deliverable**   **Description**                    **Acceptance      **Risk**
                                                                   gate**            

  **S10-1**   **NCC             Replace                            68/68 S8 tests    LOW
              Vectorisation**   \_normalised_cross_correlation()   pass. 50 km       
                                pure-Python loop with numpy        STIM300 NAV-01    
                                sliding_window_view. Drop-in:      pass.             
                                identical inputs/outputs. Backup                     
                                original. Include terrain                            
                                observability logging (Task 5) in                    
                                same deployment.                                     

  **S10-2**   **S9 Regression   tests/test_s9_nav01_pass.py --- 7  7/7 gates pass.   LOW
              Test**            automated gates (S9-A through      Regression →      
                                S9-G). 215/215 regression must     222/222.          
                                remain clean after patch.                            

  **S10-3**   **S8-D Drift      dashboard/als250_drift_chart.py    Chart PNG + PDF   LOW
              Chart**           --- run 250 km for STIM300,        generated.        
                                ADIS16505-3, BASELINE in tmux.     STIM300 NAV-01    
                                Three-curve PNG + PDF. Both        PASS at 250 km.   
                                STIM300 curves must show NAV-01    Milestone M5      
                                PASS.                              closed.           

  **S10-4**   **Parallel IMU    run_als250_parallel.py ---         50 km smoke: all  LOW
              Runner**          subprocess.Popen per IMU. Required 3 IMUs PASS       
                                for S8-D viability. Supersedes     simultaneously,   
                                broken run_als250_parallel_v2.py.  exit code 0.      

  **S10-5**   **Terrain         Add logging of σ_terrain, NCC      Logged fields in  LOW
              Observability**   innovation magnitude, and          .npy output. Zero 
                                suppression events at each update  impact on         
                                attempt. Bundled with S10-1        existing          
                                deployment.                        regression gates. 

  **S10-6**   **Manoeuvring     Automated regression test: 150 km  4 gates pass.     LOW
              Corridor          Himalayan manoeuvring corridor (5  Optional add to   
              Regression**      segments, heading changes up to    suite post-S10    
                                60°). 4 gates: NAV-01 \< 100 m,    close.            
                                accepted ≥ 60/75, max gap ≤ 20 km,                   
                                innov-rej = 0.                                       
  ----------- ----------------- ---------------------------------- ----------------- ----------

B.3 What S10 Explicitly Does Not Include

  ---------------------- -------------------- ------------------------------
  **Item**               **Classification**   **When it opens**

  Cybersecurity          Phase-2 --- no       After TASL meeting confirms
  (FR-109--112)          blockers             continuation

  DMRL CNN upgrade       Phase-2 --- blocked  After GPU + dataset +
                                              clearance

  ROS2 / PX4 SITL (HIL)  Phase-3 --- blocked  After TASL platform decision

  VIO fusion (flat       Phase-2              Addresses σ_terrain \< 10 m
  terrain)                                    boundary condition. Post-TASL.

  Predictive EW          Phase-2 --- code     After Phase-1 demo acceptance
  modelling              present              

  CEMS multi-UAV EW      Phase-2 --- frozen   After Phase-1 demo acceptance
  sharing                at S6                

  Any architectural      Not in scope         Demo Fork: hardening only, no
  change                                      re-engineering
  ---------------------- -------------------- ------------------------------

B.4 Milestone State After S10

  ------------------ ---------------------- --------------- ----------------
  **Milestone**      **Gate criterion**     **Evidence**    **Status after
                                                            S10**

  **M1 --- Autonomy  FSM deterministic; INS S0+S1 passing   **CLOSED**
  Core**             propagates without                     
                     drift                                  

  **M2 ---           Drift \< 100 m/5 km    S2/S3/S8/S9 +   **CLOSES S10**
  GNSS-Denied        over 250 km; BIM ≤ 250 S8-D chart      
  Navigation**       ms; TRN every 2 km                     

  **M3 --- EW        Cost-map ≤ 500 ms;     S4+S6 passing   **CLOSED**
  Survivability**    Replan ≤ 1 s; ZPI                      
                     zero-RF enforced                       

  **M4 --- Terminal  DMRL lock ≥ 0.85;      S5 111/111      **CLOSED**
  Autonomy**         Decoy reject ≥                         
                     0.80/3f; L10s ≤ 2 s                    

  **M5 --- Demo      HTML debrief; 9-panel  S7 done; chart  **CLOSES S10**
  Presentation**     dashboard; ALS-250     pending         
                     three-curve drift                      
                     chart                                  
  ------------------ ---------------------- --------------- ----------------

Regression suite after S10: 222/222 (minimum). Milestone M5 status after
S10: CLOSED. Programme status: Phase-1 demonstration package complete.
Ready for TASL meeting.

MicroMind / NanoCorteX \| S10 Scope Review \| Demo Fork \| PROGRAMME
CONFIDENTIAL
