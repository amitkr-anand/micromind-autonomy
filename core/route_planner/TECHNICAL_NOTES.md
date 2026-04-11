# core/route_planner/TECHNICAL_NOTES.md
**Module:** core/route_planner (HybridAstar + RoutePlanner)  
**Last updated:** 11 April 2026  
**SRS ref:** §5.2, §11.4, IT-PLN-01, IT-PLN-02, UT-PLN-02, GAP-02, GAP-03  
**Corrections implemented:** R-01 (terrain ordering), R-02 (EW staleness), R-03 (rollback completeness),
R-04 (waypoint upload order), R-05 (INS_ONLY rejection), R-06 (15 s timeout), PLN-03 (dead-end recovery),
RS-04 (route fragment cleanup)

---

## OODA-Loop Rationale — R-01 Terrain Before EW

**Why must terrain regeneration precede EW map refresh on every retask?**

The route planner operates on two overlaid cost spaces: the terrain corridor (which cells are
physically reachable given terrain, altitude, and manoeuvre constraints) and the EW cost overlay
(which cells are operationally preferred given threat intelligence).

The terrain corridor defines the **search space boundary**. EW costs are applied **within** that
boundary. If EW map refresh runs first, it may assign high penalties to cells that the subsequent
terrain regeneration will mark as physically unreachable. This wastes computation and, in
pathological cases where EW penalises all remaining navigable cells, can produce a spurious
dead-end detection that would not have occurred if terrain was updated first.

**Operational consequence of wrong order (EW first):**

1. EW refresh marks a grid region as high-cost.
2. A* avoids this region and finds an alternative path.
3. Terrain regen then reveals that the alternative path crosses a terrain feature it cannot clear.
4. A* is forced back toward the EW-penalised region, but the prior EW update biased the
   heuristic against it.
5. The planner may reject a viable route or converge more slowly, burning timeout budget.

**Enforcement:** `retask()` calls `_terrain_regen_fn()` first, logs `RETASK_TERRAIN_FIRST`, then
calls `_ew_refresh_fn()`. The log event provides an auditable ordering checkpoint for every retask.

---

## OODA-Loop Rationale — R-05 INS_ONLY Rejection

**Why is retask unsafe when nav mode is INS_ONLY?**

`INS_ONLY` means the vehicle's position estimate comes solely from IMU dead-reckoning. No GNSS,
no VIO/TRN corrections. IMU drift accumulates at approximately 1 m/km under nominal conditions
(STIM300 characterisation, Sprint S8) and can exceed 80 m at km 60 under worst-case initial
conditions (C-2 drift envelope, P99).

A retask requires the planner to find a route from the **current position** to a new goal. If the
current position estimate has accumulated significant drift:

1. **Start cell error**: The computed start grid cell may be offset from the vehicle's true
   position. A* finds an optimal path from the wrong starting point.
2. **Route validity gap**: Waypoints computed from a drifted start may collide with obstacles
   or EW threat zones the vehicle cannot avoid because it is not actually at the assumed position.
3. **Rollback ambiguity**: On retask failure, the pre-retask waypoints are restored. But those
   waypoints were themselves computed against a position estimate that has now drifted further.
   The vehicle ends up navigating against a route that has no valid reference.

Rejecting retask in INS_ONLY mode forces the vehicle to continue on the last validated route until
VIO/TRN corrections restore positional confidence. This is the safer operational choice — a known
route with bounded uncertainty is preferable to a newly planned route with unknown geometric error.

---

## OODA-Loop Rationale — R-06 15 s Timeout

**Why is 15 s the timeout boundary and what happens without rollback?**

**Timeout bound origin:** Route replan latency must meet KPI-E02 (≤ 1 s per replan). The retask
path tries up to three constraint relaxation levels (tight → relaxed → minimal). Under a
worst-case 50 000-cell grid with full EW penalties, each A* run can take up to 1 s. Three runs
give ~3 s nominal. The 15 s bound provides a 5× safety margin against slow convergence (e.g.
highly penalised grids near a dense jammer array) while remaining well within the mission timeline
budget for an ingress re-route decision (SRS §5.2 replanning latency budget).

**Without rollback, what does the vehicle do?**

If retask computation starts but never completes:

1. Terrain regen and EW refresh have already run — the planner's internal state has been modified.
2. The active waypoint list has not yet been updated (success path not reached).
3. The vehicle is navigating on the **pre-retask waypoints** but the planner's cost map reflects
   the **post-refresh EW state** — a split-brain condition.
4. The next route query (e.g. periodic re-optimisation) will start from an inconsistent state
   and may produce corrupted waypoints.

The rollback on timeout restores all three state components (EW map, terrain corridor, waypoints)
to their pre-retask snapshots, returning the planner to a fully consistent state. The vehicle
continues on its last valid route, which is always preferable to no route or an inconsistent one.

---

## OODA-Loop Rationale — PLN-03 Dead-End

**What does the dead-end condition mean operationally and why is returning to the last valid
waypoint the correct recovery?**

A dead-end occurs when the route planner exhausts all constraint relaxation levels without finding
a valid path to the new goal. This can happen when:

- The new goal lies across a dense EW threat zone with no navigable detour within the corridor.
- Terrain regeneration has contracted the corridor to the point where no cell-to-cell path exists
  from current position to the new goal.
- Both conditions coincide after a major route change request.

Dead-end is distinct from timeout: timeout means the route search took too long regardless of
whether a path might exist; dead-end means the search completed but no path was found.

**Why last valid waypoint?**

The vehicle must never be left with an empty route. An empty route means:

1. The autopilot has no next waypoint to navigate toward.
2. Depending on PX4 hold behaviour, the vehicle may enter an unplanned loiter or begin an
   uncommanded descent sequence.
3. Recovery from an unplanned loiter in GNSS-denied terrain is expensive — the vehicle may
   drift out of the terrain corridor before the mission manager can issue a corrective command.

Returning to the last valid waypoint achieves a known-safe state: the vehicle continues toward
the most recently validated destination, which was reachable from the pre-retask position.
This buys time for the mission planner to attempt a different goal or wait for the EW threat
environment to change (EW cost map decays at rate defined in EWEngine).

The last valid waypoint is always the **terminal waypoint of the previous successful route** —
the farthest point the vehicle was confirmed to be routed toward before the failed retask attempt.

---

## OODA-Loop Rationale — RS-04 Fragment Cleanup

**Why do route fragments accumulate during retask and why must they be explicitly cleared?**

During a retask, `RoutePlanner.retask()` attempts up to `RETASK_CONSTRAINT_LEVELS` (3) A\* replan
iterations in a constraint-relaxation loop (R-06). Each iteration calls `HybridAstar.replan()` and
receives a `ReplanResult` object containing a `waypoints` list. If that attempt fails (route not
found), the `ReplanResult.waypoints` list is a non-adopted intermediate fragment — it is not the
active route, but it was computed and allocated in memory.

Without explicit tracking and cleanup, these fragment lists become anonymous objects held only
by Python's reference-counting GC. In nominal operation on short missions, GC reclaims them
promptly. On long GNSS-denied missions with frequent retask events, the interaction of:

1. **High retask frequency** — threat environment changes force many consecutive retask attempts,
2. **Partial-path fragments** — each constraint relaxation level may produce a partial or empty
   `waypoints` list before failure is declared, and
3. **GC non-determinism** — CPython's cyclic GC does not guarantee immediate reclamation when
   objects participate in reference cycles (e.g., if `ReplanResult` ever holds a back-reference),

creates a credible unbounded accumulation path.

**The failure mode on a long GNSS-denied mission without cleanup:**

Consider a 100 km GNSS-denied segment with retask events every 30 s (realistic for a
jammer-dense environment where each EW update triggers a re-route attempt):

- 100 km at 150 km/h ≈ 2 400 s ≈ 80 retask events
- Each retask tries up to 3 constraint levels → up to 240 intermediate `ReplanResult` objects
- A large grid (50 000 cells, ~400 KB per cost map copy) means even partial fragment lists can
  accumulate 10s of MB if GC is delayed
- In Jetson Orin NX embedded context (OI-25: 8 GB RAM, memory margins unknown), unbounded
  growth of stale fragment lists competes with real-time navigation buffers and logger queues

**RS-04 v1.2 resolution:**

`RoutePlanner` tracks non-adopted replan attempts in `_intermediate_fragments` (a list of
waypoint lists). `_cleanup_route_fragments()` is called on **all exit paths** of `retask()`
— successful, failed/rollback, timeout/rollback, INS_ONLY rejection, TERMINAL rejection, and
dead-end. Each call clears `_intermediate_fragments`, computes a byte-freed estimate
(`ROUTE_FRAGMENT_BYTES_PER_WP = 24` bytes per waypoint: 3 × float64), and emits a
`ROUTE_FRAGMENT_CLEANUP` DEBUG event with `req_id='RS-04'`.

This makes cleanup deterministic (not GC-dependent), auditable (every retask produces a log
entry confirming 0 accumulated fragments), and verifiable under SB-07 gate (c) which asserts
`len(_intermediate_fragments) == 0` after each of 10 consecutive failed retasks.
