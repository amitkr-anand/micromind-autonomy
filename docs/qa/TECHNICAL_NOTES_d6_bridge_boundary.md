# Technical Notes — D6 / OffboardRecoveryFSM Bridge Boundary
Authority: Deputy 1 | Date: 26 April 2026 | Req: PX4-01 D1-D6

## T-MON Read-Only Constraint (CGM §1.3)

T-MON (_monitor_loop in MAVLinkBridge) is read-only by design.
It receives MAVLink messages. It does not send them.

**What T-MON does:**
- Reads HEARTBEAT messages from PX4
- Detects OFFBOARD loss (custom_mode change)
- Detects reboot (sequence number reset via RebootDetector)
- Detects HOLD mode (custom_mode == PX4_HOLD_CUSTOM_MODE)
- Logs events to event_log
- Dispatches registered callbacks via daemon threads

**What T-MON does NOT do:**
- Send any MAVLink command
- Modify any navigation state
- Call any PX4 command directly

## OffboardRecoveryFSM Boundary

The FSM receives:
- send_set_mode_fn: a callable that sends MAVLink SET_MODE
- event_log: the shared log
- clock_fn: monotonic time (NOT simulation clock)
- abort_fn: callable that signals mission abort

The FSM does NOT:
- Read from the Unified State Vector
- React to MAVLink-level events directly
- Hold any reference to MAVLinkBridge internals

**How the boundary works:**
T-MON detects OFFBOARD loss → calls registered callback
→ OffboardRecoveryFSM.on_offboard_loss() in daemon thread
→ FSM sends MAVLink commands via send_set_mode_fn
→ T-MON continues monitoring independently

The FSM is a consumer of detected events, not a monitor.
It never reads MAVLink directly.

## Deputy 2 Logic-Bleed Check

Agent 4 should verify:
1. OffboardRecoveryFSM imports: no pymavlink, no mavutil
2. HoldRecoveryHandler imports: no pymavlink, no mavutil
3. Both FSMs receive send_set_mode_fn as an injected callable
   — they do not import or instantiate MAVLink objects
4. T-MON's _monitor_loop has no direct calls to
   send_set_mode_fn or any command-sending function

If any of (1)-(4) fail, that is a genuine boundary violation
and Deputy 1 requires immediate notification.
