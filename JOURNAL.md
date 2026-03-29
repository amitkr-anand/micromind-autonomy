# MicroMind Development Journal
## IP Traceability Record
**Developer:** Amit (amitkr-anand)
**Location:** micromind-node01, independent of TASL premises
**Timezone:** IST (UTC+0530)

---

## 2026-03-29

### Session 1: ~0130–0510 IST

#### 0130 — Session start (Pre-HIL Phase 0–3)
- Repo baseline: micromind-autonomy @ 5a32c8e
- CP-0 through CP-2 already PASS at session open
- SIL: 332/332, Integration: 228/228
- Objective: Complete Phase 3 (OfflineVIODriver, RC-7, RC-8, RC-11) and declare CP-3

#### 0145 — mark_send native integration (Pre-Phase 3 cleanup)
- Integrated mark_send into mavlink_bridge._setpoint_loop natively
- CP-2 asterisk closed
- Commit: 808c53f

#### 0200 — OfflineVIODriver written (Phase 3 / S-NEP-04 interface)
- ENU→NED rotation, IFM-01 monotonicity gate, LiveVIODriver stub
- 22/22 gates passing
- Commit: 65ddd2c

#### 0230 — RC-7: IFM-01 timestamp injection test
- Injected non-monotonic timestamp into 200Hz loop
- IFM-01 fired: event_id=IFM01-0000, t_prev=99999.000s
- 10/10 post-violation frames valid, no crash
- RC-7: PASS

#### 0300 — RC-8: Logger drop-rate test
- 60s run at 200Hz: drop_rate=0.000%, ESKF=200.0Hz
- Jetson note (renice +10): drop_rate=0.000%, ESKF=200.0Hz
- RC-8: PASS, RC-8 Jetson note: PASS

#### 0330 — RC-11: Control loop independence (live SITL)
- VIO outage injected 10s mid-flight
- RC-11a: SP continuity 20Hz throughout OUTAGE ✅
- RC-11b: No NaN/non-finite ✅
- RC-11c: OFFBOARD retained ✅
- RC-11d: 1000 loops during 5s MAVLink disconnect ✅
- RC-11e: NOMINAL restored in 0.3s after VIO resume ✅
- RC-11: PASS

#### 0400 — Gazebo rendering root cause and fix
- Root cause: RTX 5060 Ti requires NVIDIA EGL; Mesa gallium crashes on this GPU
- Fix: __EGL_VENDOR_LIBRARY_FILENAMES=10_nvidia.json in px4-rc.gzsim
- Drone visible in Gazebo, flight confirmed live
- Snap/libpthread conflict resolved with LD_PRELOAD

#### 0430 — CP-3 declared
- All 6 exit conditions closed
- Readiness report committed to dashboard/
- Tag: cp3-prehil-ready @ 5a32c8e

#### 0510 — Session 1 ends

---

### Session 2: ~1000–1235 IST

#### 1000 — Session restart
- Baseline confirmed: 8dc6509, 332/332, 228/228, baylands world

#### 1010 — Phase 4 begins: inject_outage.py (D-3)
- Full §Part 9 flight sequence implemented
- First run PASS: drift_envelope=8.06m, nan_detected=False
- Mode chain: NOMINAL→OUTAGE→RESUMPTION→NOMINAL confirmed
- Commit: 80bf64f

#### 1027 — run_demo.sh written
- Single-command demo launcher with EKF2 alignment wait
- gnome-terminal overlay auto-launch

#### 1027–1033 — DR-3: 3× consecutive runs
| Run | Time (IST) | E2E P95 | VIO drift | Result |
|---|---|---|---|---|
| 1 | 1027 | 0.324ms | 8.06m | PASS |
| 2 | 1030 | 0.273ms | 8.04m | PASS |
| 3 | 1032 | 0.337ms | 7.94m | PASS |
- Tag: dr3-demo-pass @ 123c17d

#### 1051 — demo_overlay.py written (DR-4)
- 6 mandatory overlays per v1.2 §9.11
- Curses panel: ① VIO mode ② drift ③ trajectory ④ event log ⑤ SP rate ⑥ PX4 mode
- Auto-launched by run_demo.sh via gnome-terminal

#### 1054 — DR-4: overlays confirmed working
- Screenshot: VIO MODE OUTAGE in red, DRIFT 5.13m, VIO_OUTAGE_START/END in event log
- All 6 overlays visible simultaneously during live flight
- Tag: dr4-overlays-pass @ cb7f13b

#### 1105–1119 — DR-5: screen recording attempt
- ffmpeg v6.1.1 installed
- x11grab: black frames — NVIDIA EGL bypasses X11 framebuffer
- kmsgrab: /dev/dri/card1, card2 — DRM device not accessible to ffmpeg
- 25MB MP4 captured but blank content
- DR-5: OPEN — deferred to next attempt via OBS/NVFBC

#### 1130 — Gazebo world enhancement
- Switched to baylands world (road, trees, water, sky backdrop)
- Camera follow delay fix: sleep 5 before gz follow command
- Heartbeat timeout increased to 30s
- Full demo flight confirmed over baylands: DEMO PASS
- Commit: 8dc6509

#### 1231 — Journal created, DR-5 session begins
- JOURNAL.md created for IP traceability
- All development confirmed at micromind-node01, outside TASL premises

---
*All timestamps in IST (UTC+0530). System clock on micromind-node01 runs UTC.*
*This journal constitutes timestamped evidence of independent development.*
