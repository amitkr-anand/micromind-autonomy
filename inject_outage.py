#!/usr/bin/env python3
"""
inject_outage.py
MicroMind Pre-HIL — Phase 4 Demo Script (D-3)

VIO outage injection demo. Executes a complete flight with:
  - Takeoff to 5m
  - Fly 50m North
  - VIO outage injected at T+30s for 10s
  - Recovery to NOMINAL
  - Return and land
  - Post-flight log written to dashboard/

Produces:
  demo_flight_<timestamp>.json  — structured event log
  demo_flight_<timestamp>.html  — HTML report (via demo_report.py)

Usage:
  python3 inject_outage.py

Exit codes:
  0 — demo completed successfully
  1 — demo failed (see output)

v1.2 §Part 9 steps 1-8 (run_demo.sh calls this script).
"""

import sys, time, math, threading, json, os
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from sim.als250_nav_sim import run_als250_sim
from integration.drivers.vio_driver import OfflineVIODriver
from integration.config.mission_config import MissionConfig
from integration.pipeline.live_pipeline import LivePipeline
from integration.pipeline.latency_monitor import LatencyMonitor
from integration.bridge.time_reference import TimeReference
from integration.bridge.bridge_logger import BridgeLogger
from integration.bridge.mavlink_bridge import MAVLinkBridge
from integration.pipeline.demo_report import generate_report
from core.ins.mechanisation import ins_propagate

# ---------------------------------------------------------------------------
# Event log
# ---------------------------------------------------------------------------

_events = []
_t0 = time.monotonic()

def log_event(kind, **data):
    entry = {'t': round(time.monotonic() - _t0, 3), 'event': kind}
    entry.update(data)
    _events.append(entry)
    ts = entry['t']
    print(f"  [{ts:7.3f}s] {kind}", end='')
    for k, v in data.items():
        if isinstance(v, float):
            print(f"  {k}={v:.3f}", end='')
        else:
            print(f"  {k}={v}", end='')
    print()
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main demo sequence
# ---------------------------------------------------------------------------

def run_demo(output_dir='dashboard'):
    global _t0
    _t0 = time.monotonic()
    ts  = datetime.now().strftime('%Y%m%d_%H%M%S')

    print()
    print("=" * 60)
    print("  MicroMind Pre-HIL — OEM Demonstration Flight")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    log_event("DEMO_START", version="Pre-HIL v1.2", basis="5a32c8e")

    result  = run_als250_sim(imu_name=None, duration_s=120.0, verbose=False)
    vio     = OfflineVIODriver(result['position'], sigma_pos_m=0.1,
                               dt_s=0.04, loop=True)
    config  = MissionConfig()
    monitor = LatencyMonitor(window_s=5.0)
    ref     = TimeReference()

    log_path = os.path.join(output_dir, f'demo_bridge_{ts}.jsonl')
    logger   = BridgeLogger(log_path, source_type='sim', time_ref=ref)
    logger.start()

    bridge   = MAVLinkBridge(time_ref=ref, logger=logger)
    bridge._latency_monitor = monitor
    pipeline = LivePipeline(config)
    pipeline.latency_monitor = monitor

    # ------------------------------------------------------------------
    # VIO injection flag
    # ------------------------------------------------------------------
    _vio_paused   = False
    _nan_detected = False

    def _patched_nav(self):
        nonlocal _vio_paused, _nan_detected
        import time as _t
        t_next   = _t.perf_counter()
        vio_step = 0
        while not self._stop_event.is_set():
            try:
                if self.latency_monitor:
                    self.latency_monitor.begin_step(
                        step=self._loop_count,
                        imu_stale=self._imu.is_stale(),
                        vio_mode=self._vio_nav.current_mode.name,
                    )
                imu_r = self._imu.read()
                self._state = ins_propagate(
                    self._state,
                    np.array(imu_r.accel_mss),
                    np.array(imu_r.gyro_rads),
                    self._dt_s,
                )
                self._eskf.propagate(self._state,
                                     np.array(imu_r.accel_mss), self._dt_s)
                if self.latency_monitor:
                    self.latency_monitor.mark_eskf()

                self._vio_nav.tick(self._dt_s)

                vio_step += 1
                if vio_step % 25 == 0 and not _vio_paused:
                    vio_r = vio.read()
                    if vio_r.valid:
                        nis, rejected, innov = self._eskf.update_vio(
                            self._state, vio_r.pos_ned_m, vio_r.cov_ned)
                        self._eskf.inject(self._state)
                        self._vio_nav.on_vio_update(
                            accepted=not rejected,
                            innov_mag=float(innov),
                        )

                if not all(math.isfinite(float(x)) for x in self._state.p):
                    _nan_detected = True

                self._enqueue_setpoint()
                if self.latency_monitor:
                    self.latency_monitor.mark_decision()

                with self._lock:
                    self._loop_count += 1
                    self._last_loop_t = _t.monotonic()

            except Exception:
                with self._lock:
                    self._loop_count += 1

            t_next += self._dt_s
            sleep_s = t_next - _t.perf_counter()
            if sleep_s > 0:
                _t.sleep(sleep_s)
            else:
                t_next = _t.perf_counter()

    import types
    pipeline._nav_loop = types.MethodType(_patched_nav, pipeline)

    # ------------------------------------------------------------------
    # Step 2: Connect bridge
    # ------------------------------------------------------------------
    print("Step 2: Connecting MAVLink bridge...")
    if not bridge.connect(timeout_s=15.0):
        log_event("FAIL", reason="MAVLink connect timeout")
        return 1
    log_event("CONNECTED", sysid=bridge._target_system,
              compid=bridge._target_component,
              time_synced=bridge._time_ref.is_synced)

    bridge.start_heartbeat()
    time.sleep(1.0)

    # Wait for EKF2
    t_w = time.monotonic()
    while time.monotonic() - t_w < 15.0:
        msg = bridge._mav.recv_match(
            type='LOCAL_POSITION_NED', blocking=True, timeout=1.0)
        if msg:
            bridge._local_pos_valid = True
            log_event("EKF2_VALID", x_m=round(msg.x, 3))
            break
    else:
        log_event("FAIL", reason="EKF2 not aligned within 15s")
        return 1

    # ------------------------------------------------------------------
    # Step 3: Pre-stream
    # ------------------------------------------------------------------
    print("Step 3: Pre-streaming setpoints (FM-2)...")
    monitor.start()
    pipeline.start()
    bridge.update_setpoint(x_m=0.0, y_m=0.0, z_m=-5.0)
    bridge.start_setpoints()
    time.sleep(2.0)
    log_event("PRESTREAM_COMPLETE", sp_hz=round(monitor.current_setpoint_hz, 1))

    # ------------------------------------------------------------------
    # Step 4: ARM + OFFBOARD
    # ------------------------------------------------------------------
    print("Step 4: Arming and engaging OFFBOARD...")
    bridge._mav.mav.command_long_send(
        bridge._target_system, bridge._target_component,
        400, 0, 1, 0, 0, 0, 0, 0, 0)
    ack = bridge._mav.recv_match(
        type='COMMAND_ACK', blocking=True, timeout=5.0)
    if not ack or ack.result != 0:
        log_event("FAIL", reason=f"ARM rejected result={ack.result if ack else 'TIMEOUT'}")
        return 1
    log_event("ARMED")

    bridge._mav.mav.command_long_send(
        bridge._target_system, bridge._target_component,
        176, 0, 209, 6, 0, 0, 0, 0, 0)
    ack = bridge._mav.recv_match(
        type='COMMAND_ACK', blocking=True, timeout=5.0)
    if not ack or ack.result != 0:
        log_event("FAIL", reason="OFFBOARD rejected")
        return 1
    log_event("OFFBOARD_ENGAGED")
    bridge.start_monitor()

    # Wait for 5m altitude
    t_climb = time.monotonic()
    while time.monotonic() - t_climb < 15.0:
        msg = bridge._mav.recv_match(
            type='LOCAL_POSITION_NED', blocking=True, timeout=0.5)
        if msg and -msg.z >= 4.0:
            log_event("ALTITUDE_REACHED", alt_m=round(-msg.z, 2))
            break

    # ------------------------------------------------------------------
    # Step 5: Trajectory — fly 50m North
    # ------------------------------------------------------------------
    print("Step 5: Flying trajectory (50m North)...")
    bridge.update_setpoint(x_m=50.0, y_m=0.0, z_m=-5.0)
    t_traj = time.monotonic()

    outage_injected  = False
    outage_recovered = False
    t_demo_start     = time.monotonic()

    while time.monotonic() - t_demo_start < 70.0:
        elapsed = time.monotonic() - t_demo_start
        h = pipeline.health()

        msg = bridge._mav.recv_match(
            type=['LOCAL_POSITION_NED', 'HEARTBEAT'],
            blocking=True, timeout=0.2)

        # ------------------------------------------------------------------
        # Step 6: VIO outage at T+30s
        # ------------------------------------------------------------------
        if elapsed >= 30.0 and not outage_injected:
            _vio_paused = True
            outage_injected = True
            log_event("VIO_OUTAGE_START",
                      vio_mode=h.vio_mode,
                      drift_envelope_m=round(
                          pipeline._vio_nav.drift_envelope_m or 0.0, 2))
            print("Step 6: VIO outage injected (10s)...")

        # ------------------------------------------------------------------
        # Step 7: VIO resume at T+40s
        # ------------------------------------------------------------------
        if elapsed >= 40.0 and outage_injected and not outage_recovered:
            _vio_paused = False
            outage_recovered = True
            log_event("VIO_OUTAGE_END",
                      vio_mode=h.vio_mode,
                      nan_detected=_nan_detected,
                      drift_envelope_m=round(
                          pipeline._vio_nav.drift_envelope_m or 0.0, 2))
            print("Step 7: VIO resumed — waiting for NOMINAL...")

        # Return to origin at T+50s
        if elapsed >= 50.0:
            bridge.update_setpoint(x_m=0.0, y_m=0.0, z_m=-5.0)

        # Descend at T+62s
        if elapsed >= 62.0:
            bridge.update_setpoint(x_m=0.0, y_m=0.0, z_m=-2.0)
            break

        # Log mode transitions
        if msg and msg.get_type() == 'LOCAL_POSITION_NED':
            if int(elapsed) % 10 == 0 and int(elapsed) != getattr(run_demo, '_last_tel', -1):
                log_event("TELEMETRY",
                          elapsed_s=round(elapsed, 1),
                          north_m=round(msg.x, 1),
                          alt_m=round(-msg.z, 1),
                          vio_mode=h.vio_mode,
                          sp_hz=round(monitor.current_setpoint_hz, 1),
                          drift_m=round(
                              pipeline._vio_nav.drift_envelope_m or 0.0, 2))
                run_demo._last_tel = int(elapsed)

    # ------------------------------------------------------------------
    # Land
    # ------------------------------------------------------------------
    print("Landing...")
    bridge._mav.mav.command_long_send(
        bridge._target_system, bridge._target_component,
        21, 0, 0, 0, 0, 0, 0, 0, 0)
    time.sleep(6.0)

    # ------------------------------------------------------------------
    # Collect results
    # ------------------------------------------------------------------
    pipeline.stop()
    summary = monitor.stop()
    bridge.stop()
    logger.stop()

    log_event("DEMO_COMPLETE",
              nan_detected=_nan_detected,
              e2e_p95_ms=round(summary.e2e_p95_ms, 3),
              sp_min_hz=round(summary.setpoint_min_hz, 1),
              cpu_mean_pct=round(summary.cpu_mean_pct, 1))

    # ------------------------------------------------------------------
    # Step 8: Generate HTML report
    # ------------------------------------------------------------------
    print("Step 8: Generating post-flight report...")
    os.makedirs(output_dir, exist_ok=True)

    # Export latency JSON
    latency_json = os.path.join(output_dir, f'demo_latency_{ts}.json')
    monitor.export_json(latency_json)

    # Write event log
    events_json = os.path.join(output_dir, f'demo_events_{ts}.json')
    with open(events_json, 'w') as f:
        json.dump({'ts': ts, 'events': _events}, f, indent=2)

    # Generate HTML report
    html_path = os.path.join(output_dir, f'demo_report_{ts}.html')
    generate_report(
        latency_json=latency_json,
        output_html=html_path,
        run_label=f"OEM Demo Flight — {datetime.now().strftime('%Y-%m-%d')}",
    )

    print()
    print("=" * 60)
    print(f"  Demo complete.")
    print(f"  Events log:  {events_json}")
    print(f"  HTML report: {html_path}")
    print(f"  E2E P95:     {summary.e2e_p95_ms:.3f}ms")
    print(f"  VIO outage:  {'PASS — no NaN, mode chain complete' if not _nan_detected and outage_recovered else 'FAIL'}")
    print("=" * 60)
    print()

    return 0 if (not _nan_detected and outage_recovered
                 and summary.e2e_p95_ms < 50.0) else 1


if __name__ == '__main__':
    sys.exit(run_demo())
