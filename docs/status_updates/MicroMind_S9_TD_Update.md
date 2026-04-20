**MICROMIND PROGRAMME**

**Technical Director Status Update**

Sprint S9 Close --- NAV-01 Closure

4 March 2026 \| Amit (amitkr-anand) \| Ref: HANDOFF_S9_to_S10.md

**1. Executive Summary**

Sprint S9 has been completed. The primary objective --- closing the
NAV-01 navigation accuracy requirement across 150km GNSS-denied
corridors --- has been achieved with the Safran STIM300 IMU.

NAV-01 mandates INS drift below 100m per 5km segment. The system now
achieves a maximum 5km drift of 14.5m over a full 150km corridor --- a
7x margin against the requirement. This result was not possible under
the previous architecture, which contained five compounding errors
across the TRN+ESKF integration.

The programme regression suite remains at 215/215 tests passing. The
system is ready for the TASL partnership discussion.

  --------------- ------------------ ------------------ -----------------
  **Corridor**    **Max 5km Drift**  **Final Drift**    **NAV-01 Status**

  20 km           \~4 m              \~4 m              **PASS (limit:
                                                        100m)**

  50 km           9.7 m              3.7 m              **PASS**

  150 km          14.5 m             4.2 m              **PASS**
  --------------- ------------------ ------------------ -----------------

**2. What Was Built in S9**

S9 was a corrective sprint --- no new functional modules were added. The
sprint diagnosed and repaired five architectural errors that had been
present but hidden in the TRN+ESKF integration since Sprint S3. The
changes touched three existing files.

  --------------------------------- --------------------------------- ----------------
  **File**                          **Change**                        **Gate**

  core/ins/trn_stub.py              Removed internal Kalman filter.   **S9-1/S9-2
                                    update() now returns raw NCC      PASS**
                                    offset only --- no state          
                                    mutation. New scalar signature.   

  core/ekf/error_state_ekf.py       Q-matrix corrected to STIM300     **S9-4 PASS**
                                    TS1524 rev.31 values. Position    
                                    process noise term added.         

  sim/als250_nav_sim.py             Correct propagation order         **S9-3 PASS**
                                    implemented. corridor_km          
                                    parameter added. NAV-01 metric    
                                    corrected to 2D horizontal norm.  

  scenarios/bcmp1/bcmp1_runner.py   S5 dispatcher routing fixed ---   **S9-0 PASS**
                                    111/111 regression restored       
                                    (regression introduced in S8-E).  
  --------------------------------- --------------------------------- ----------------

**3. Root Causes of Previous NAV-01 Failures**

Five distinct root causes were identified and resolved. Each is
documented in full in HANDOFF_S9_to_S10.md. A summary follows.

**RC-1 --- Duplicate Kalman filter in TRNStub**

The TRN module contained its own internal 2x2 position Kalman filter
that directly mutated INS position state. This ran in parallel with the
15-state ESKF, with inconsistent noise parameters and no covariance
coupling. The two estimators fought each other. The TRN module should be
an observation source only --- not an estimator.

Fix: Removed the internal Kalman entirely. TRNStub.update() now returns
a raw correction record. The caller applies it via the ESKF.

**RC-2 --- ESKF gyro bias random walk 247x too large**

The gyro bias random walk constant in the ESKF Q-matrix was set to 1e-5
rad/s/sqrt(s). The correct STIM300 value derived from the TS1524 rev.31
datasheet is 4.04e-8 rad/s/sqrt(s) --- 247 times smaller. The inflated Q
caused the filter to expect rapid bias variation, destabilising
convergence.

Fix: Updated both bias RW constants to datasheet-derived values (STIM300
TS1524 rev.31, 600s Allan variance).

**RC-3 --- Position block of Q was zero, Kalman gain effectively zero**

After correcting the bias RW values, the position uncertainty block
Q\[0:3,0:3\] remained zero. With P_pos approximately 1e-6 m\^2 and TRN
measurement noise R = 25 m\^2, the Kalman gain was approximately 4e-8
--- effectively zero. TRN fired 33 corrections over 50km but applied
nothing meaningful.

Fix: Added a position process noise term (1.0 m/sqrt(s)) reflecting the
physical INS drift rate. This brings P_pos to approximately 27 m\^2 over
a 1500m correction interval, matching R and producing a healthy Kalman
gain of approximately 0.5.

**RC-4 --- Propagation order violated**

The simulation loop called ins_propagate() before eskf.propagate(),
meaning the ESKF received a stale pre-step state for covariance
propagation. The correct order is: ESKF propagate (with bias-corrected
accel) → INS propagate → TRN → ESKF update → inject.

**RC-5 --- NAV-01 drift metric included altitude**

The segment drift was computed as a 3D Euclidean norm including
altitude. The ALS-250 trajectory includes a sinusoidal altitude profile,
and barometric altitude is not corrected by TRN (a horizontal-only
system). Altitude drifts at approximately 3.4 m/km, giving \~102m of
apparent 3D drift at 30km even when horizontal position was accurate to
under 5m.

Fix: Changed the drift metric to 2D horizontal norm. This is the
physically correct definition --- TRN is a horizontal navigation aid.
Altitude accuracy is a separate channel responsibility.

**4. Programme State**

**Test Suite**

  ---------------------- --------------- --------------- ---------------------------
  **Suite**              **Tests**       **Result**      **Runner**

  S5 --- DMRL, L10s-SE,  111             **111/111       run_s5_tests.py
  BCMP-1                                 PASS**          

  S6 --- ZPI, CEMS       36              **36/36 PASS**  tests/test_s6_zpi_cems.py

  S8 --- IMU, INS,       68              **68/68 PASS**  run_s8_tests.py
  ALS-250, BCMP-1 IMU                                    

  TOTAL                  215             **215/215       ---
                                         PASS**          
  ---------------------- --------------- --------------- ---------------------------

**Latest Commit**

Commit: 7fba53c \| Tag: s9-nav01-pass \| Branch: main

Repository: amitkr-anand/micromind-autonomy

**5. Open Items Before TASL Meeting**

  -------------------------- -------------------------- -----------------
  **Item**                   **Detail**                 **Priority**

  V7 spec update --- STIM300 Spec floor is 0.1          **HIGH --- before
  ARW                        deg/sqrt(hr). STIM300      TASL**
                             typical is 0.15            
                             deg/sqrt(hr). Must update  
                             to 0.2 deg/sqrt(hr) before 
                             TASL.                      

  S8-D drift chart           Three-curve TASL chart     **HIGH --- TASL
                             (STIM300 vs ADIS vs        presentation**
                             BASELINE) not yet          
                             generated. Requires full   
                             250km x 3 IMU run.         

  NCC simulation speed       150km run takes 3.5 hours  MEDIUM
                             (pure Python NCC). Full    
                             250km x 3 IMU overnight    
                             run is impractical without 
                             vectorisation.             

  S9 test suite              No dedicated               LOW
                             test_s9\_\*.py written. S9 
                             changes validated by       
                             simulation results and     
                             215/215 regression only.   
  -------------------------- -------------------------- -----------------

**6. Way Ahead --- Sprint S10 Options**

Scope for S10 depends on TASL meeting outcome. Four candidate forks are
ready to start immediately. One is blocked on the TASL platform
decision.

  ---- ---------------- ------------------------------ -------------------
       **Fork**         **Description**                **Status**

  A    S8-D Chart       Generate ALS-250 three-curve   **Ready now**
                        drift chart for TASL           
                        presentation. Recommended      
                        first action --- high          
                        visibility, one session.       

  B    NCC              Replace pure-Python NCC with   **Ready now**
       vectorisation    scipy.signal.correlate2d.      
                        Target: 150km run in under 10  
                        minutes. Unblocks overnight    
                        multi-model runs.              

  C    Cybersecurity    core/cybersec/ --- key         **Ready now**
       hardening        loading, signed mission        
                        envelope, PQC-ready stack.     
                        FR-109 through FR-112.         

  D    HIL integration  ROS2 node wrappers, PX4 SITL   **Blocked ---
       prep             skeleton. Blocked on TASL      TASL**
                        platform decision.             
  ---- ---------------- ------------------------------ -------------------

*Full technical detail including interface changes, decision log, and
session start checklist: HANDOFF_S9_to_S10.md (Daily Logs/). Sprint
state: SPRINT_STATUS.md.*

