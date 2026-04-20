9. Mission Demonstration Tool
IMPLEMENTATION SEQUENCING NOTE: The Mission Demonstration Tool is not an immediate implementation priority. Demonstration capability is dependent on the successful completion of PX4 OFFBOARD continuity (EC-01), Checkpoint restore (EC-02), PX4 reboot recovery (EC-03), dynamic retasking (EC-04), 2-hour endurance stability (EC-05), queue behaviour (EC-08), and three-seed repeatability (EC-11). The visualisation and demonstration layer is a later-layer presentation system built on stable, proven mission behaviour. Development of the Live Mission Demo and tactical demonstration environment shall not begin until all prerequisite entry criteria in §17 are met.
Amit: This Section will be updated once we reach this Milestone (SB-5) after completing Tests in Section 1 thru 8. 

Section 9 should be revised to reflect that the Mission Demonstration Tool is not a short replay clip or post-processed video artifact.

The intended end-state is a live operator-driven demonstration environment that proves MicroMind can command a PX4-based vehicle stack dynamically during mission execution.

The demonstration architecture should be defined as a two-layer system:

1. Tactical Live Demonstration Layer
2. Long-Range Mission Demonstration Layer

Tactical Live Demonstration Layer

Purpose:
- Demonstrate real-time command authority over PX4
- Demonstrate live GNSS-denied navigation continuity
- Demonstrate dynamic mission replanning
- Demonstrate operator-in-the-loop mission updates
- Demonstrate the difference between standard PX4 autonomy and PX4 with MicroMind autonomy layer

This mode should use a Gazebo-class simulation environment or equivalent live SITL/HITL environment.

The live display should show:

- Vehicle A = standard PX4 behaviour
- Vehicle B = PX4 + MicroMind autonomy layer
- Both vehicles moving simultaneously in real time
- Operator commands injected during flight
- GNSS denial event during motion
- Visible divergence after GNSS denial
- Vehicle B maintaining route continuity
- Vehicle A drifting, failing corridor, or requiring manual intervention
- Dynamic retask event during flight
- Updated corridor, target, or exclusion zone
- Live status panels for:
    • navigation source
    • trust state
    • GNSS state
    • VIO/TRN state
    • EW state
    • PX4 mode
    • Mission Manager state
    • SHM state

The live demonstration should allow the operator to change:

- Target coordinate
- No-fly zone
- EW exclusion zone
- Route corridor width
- Mission timing constraints

Vehicle B should respond live and visibly.

Long-Range Mission Demonstration Layer

Purpose:

- Demonstrate endurance, terrain reasoning, EW-aware routing, corridor management, and comparative drift behaviour over 100–150 km missions
- Provide a TASL / DISC-facing business demonstration that can be understood within 2–3 minutes

This mode should use:

- DEM-style terrain underlay
- Srinagar Valley-inspired terrain profile
- Plains, ridge crossings, mountain ingress, industrial terminal area
- Live mission timeline synchronized across all panels
- Side-by-side Vehicle A and Vehicle B comparison
- Log-driven event overlays
- Corridor breach markers
- GNSS denial markers
- EW reroute markers
- TRN correction markers
- Retask markers
- SHM markers

The long-range demonstration must not behave like a static HTML report or pre-rendered video.

It should behave like a live mission playback environment where the operator can:

- Pause
- Resume
- Scrub timeline
- Change speed
- Switch camera view
- Toggle overlays
- Inspect events
- Re-render the mission from logs without rerunning the simulation

Mandatory Design Principle

The purpose of the demonstration environment is not only to prove technical correctness.

It must make the operational difference between Vehicle A and Vehicle B visually obvious within the first 30–60 seconds.

The viewer should not need a technical explanation to understand:

- when GNSS failed
- when Vehicle A began to drift
- why Vehicle B maintained the mission
- when retasking occurred
- how MicroMind changed the outcome

Gazebo-class live demonstration capability should therefore not be treated as optional or post-TASL scope. It is part of the intended MicroMind product narrative and should remain in the long-term roadmap even if implementation is phased after regression hardening.

9.1 Purpose and Architecture
The Mission Demonstration Tool is the external-facing presentation layer of the MicroMind programme. Its purpose is to make the operational difference between Vehicle A (standard PX4, no MicroMind) and Vehicle B (PX4 + MicroMind autonomy layer) visually obvious to a non-technical audience within 30–60 seconds, without requiring verbal explanation.
The intended end-state is not a short replay clip or a pre-rendered video. It is a live operator-driven demonstration environment that proves MicroMind can command a PX4-based vehicle stack dynamically during mission execution. This is a two-layer architecture:

Layer
Name
Purpose
Readiness Dependency
Layer 1
Static Mission Report
Post-run HTML/PNG output. Current SB-4 delivery. Provides technical evidence of comparative outcome.
Already delivered (VIZ-01). No further prerequisites.
Layer 2
Long-Range Mission Demonstration
Interactive log-driven mission playback environment over 100–150 km mission. DEM-style terrain underlay. Log-driven event overlays. Operator can pause, scrub, inspect. Designed for TASL/DISC leadership engagement.
Requires EC-01 through EC-11 (§17) before scope entry.
Layer 3
Tactical Live Demonstration
Real-time Gazebo-class SITL/HITL environment. Operator injects commands during flight. Both vehicles move simultaneously in real time. Live status panels for all subsystem states.
Long-term roadmap. Requires stable HIL integration as prerequisite.

The operational difference that the demonstration must communicate visibly:
    • When GNSS failed — the exact moment of denial, visible as a transition event, not a post-hoc annotation
    • When Vehicle A began to drift — cross-track separation growing over time, not a chart label
    • Why Vehicle B maintained the mission — TRN/VIO correction markers, corridor adherence visible
    • When retasking occurred — visible route change on Vehicle B, Vehicle A unable to respond
    • How MicroMind changed the outcome — decisive separation at the terminal phase

