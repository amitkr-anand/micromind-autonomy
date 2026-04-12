# NavigationManager — Technical Notes

**Module:** `core/navigation/navigation_manager.py`  
**Req IDs:** NAV-02, NAV-03, EC-09, EC-10  
**SRS ref:** §2.2, §2.3, §10.1, §16  
**Governance:** Code Governance Manual v3.4  
**Created:** Gate 3 — 12 April 2026  

---

## OODA-Loop Rationale

### Why a fusion coordinator is needed

Three correction sources — GNSS, VIO, TRN — each produce corrections with
different accuracy, rate, and reliability. Without a coordinator, each source
would inject corrections independently with no arbitration. A weak VIO estimate
from featureless terrain would be weighted equally to a strong TRN match from
a textured ridge. The fusion coordinator applies confidence weighting so the
ESKF receives the most reliable signal available at each moment.

The requirement for a NavigationManager was foreseen in Gate 2 (`PhaseCorrelationTRN`
docstring referenced "NavigationManager — responsible caller") but the class did not
exist. Gate 3 materialises it as the authoritative fusion arbitration layer.

### Why confidence-weighted covariance (not a gating switch)

The ESKF processes measurements via their noise covariance. A measurement with
higher noise (lower confidence) produces less influence on the state estimate.
By encoding source confidence into the measurement covariance:

- GNSS: `R_eff = R_GNSS_NOMINAL / clip(trust_score, 0.1, 1.0)` — already in ESKF.
- TRN:  `R_eff = R_TRN_NOMINAL / clip(w, 0.1, 1.0)` — added Gate 3.
         `w = confidence × suitability_score` — combined terrain quality weight.
- VIO:  `R_VIO_NOMINAL / clip(confidence, 0.1, 1.0)` — encoded into cov_pos_ned.

This approach means the ESKF automatically down-weights unreliable corrections
without requiring a separate binary gating mechanism. A VIO estimate with
confidence=0.30 still contributes, just with 2.7× the nominal noise. A TRN
match with confidence=0.15 × suitability=0.60 = w=0.09 → rejected (w < 0.10
threshold) — the ESKF never sees it.

### Why nav_confidence drives SHM (Gate 3 addition)

Before Gate 3, SHM was triggered only by L10s-SE activation (terminal phase
event). This left a gap: a vehicle with no GNSS, failed VIO, and no TRN
corrections for 15+ km would continue flying on unconstrained INS drift with
no safety response.

The `nav_confidence` score closes this gap. When all correction sources fail
simultaneously, `nav_confidence` collapses to 0.0, which is below the
`NAV_CONFIDENCE_SHM_THRESHOLD = 0.20` boundary. The FSM transitions to
`SHM_ACTIVE` via trigger `SHM_ENTRY_LOW_NAV_CONFIDENCE`. The vehicle enters
electromagnetic silence before positional uncertainty becomes operationally
unsafe.

The threshold of 0.20 is derived from operational requirement: a vehicle at
INS-only with no correction for 15 km accumulates ~150 m drift (P99 at km 15
per C-2 drift envelopes). Below that confidence, position uncertainty exceeds
the terminal engagement radius. SHM forces the vehicle to cease manoeuvre and
await opportunistic terrain match or operator reinstatement.

### Why NAV_TRN_ONLY is a distinct FSM state

ST-03 `GNSS_DENIED` was a single state covering both "VIO+TRN active" and "TRN
only" navigation regimes. This masked a material difference: when VIO fails,
velocity estimation degrades rapidly and position grows less observable between
TRN fixes. `NAV_TRN_ONLY` (ST-03B) makes this explicitly trackable in the FSM
transition log, allowing post-mission analysis to distinguish VIO failure from
full navigation loss.

---

## Architecture Decision — Camera Loop Wiring

**Gate 2 open finding:** VIO confidence ceiling was 0.547 on DEM greyscale
hillshade tiles. Cause: synthetic hillshade tiles produce low-contrast features
with poor optical flow tracking.

**Gate 3 fix:** `NavigationManager.__init__()` registers `_on_camera_frame`
as a consumer of `NadirCameraFrameBridge`. When Gazebo renders real 3D terrain,
photographic frames arrive via the bridge callback and are processed by
`VIOFrameProcessor`. Real textured terrain produces higher feature counts and
consistency scores, pushing confidence above the 0.50 injection threshold.

The wiring is:
```
Gazebo camera → NadirCameraFrameBridge (gz.transport subscriber)
    → _on_camera_frame() callback
    → VIOFrameProcessor.process_frame()
    → VIOEstimate stored in _latest_vio
    → NavigationManager.update() injects into ESKF
```

At HIL, replace the Gazebo subscriber with a real EO camera driver. The
VIOEstimate interface is unchanged. No ESKF changes required.

---

## Sensor Substitution Contract

| Sensor | Current source | HIL replacement | Breaking change |
|--------|---------------|----------------|----------------|
| Camera | Gazebo gz.transport `/nadir_camera/image` | Real EO camera (ROS2 sensor_msgs/Image or V4L2) | NO |
| VIO processor | VIOFrameProcessor (Shi-Tomasi + CLAHE + LK) | OpenVINS (full ESKF-based VIO) | NO — same VIOEstimate interface |
| TRN | PhaseCorrelationTRN on DEM hillshade | Same module, real EO nadir frame as camera_tile | NO |

---

## Day/Night Extension Point

The Addendum v2 requires thermal sensing as a future VIO source in night mode.
The camera bridge consumer registration pattern supports multiple concurrent
consumers. When thermal is added:

```python
thermal_bridge.register_consumer(thermal_vio_processor.process_frame)
```

`NavigationManager.update()` selects between VIO sources based on
`observability_state` (day/night flag from mission envelope). No architectural
change required — the fusion arbitration logic already supports multiple
confidence inputs.

---

## Unified Nav Confidence Weighting

Active sources and their weights:

| Source | Weight | Rationale |
|--------|--------|-----------|
| GNSS   | 1.0    | Absolute position, globally referenced |
| VIO    | 0.7    | Relative tracking, drifts ~1 m/km, no global anchor |
| TRN    | 0.5    | Absolute reset, but sparse (5km interval) and terrain-dependent |

`nav_confidence = Σ(w_i × conf_i) / Σ(w_i)` — normalised by active weights only.

When no source is active, `nav_confidence = 0.0` (hard collapse, not a graceful
decay). This is intentional: a vehicle with no active correction sources must
not be assigned false confidence.

---

## Gate 3 Drift Evidence — 50km Shimla Corridor

Simulation parameters: seed=42, DRIFT_PSD=1.5 m/√s, bearing 055°, 100 km/h,
TRN interval 5km, 9 corrections applied.

| km | No correction | TRN only | Reduction |
|----|--------------|----------|-----------|
|  5 | 23.0 m       | 7.2 m    | 69 %      |
| 10 | 26.3 m       | 13.3 m   | 49 %      |
| 20 | 79.8 m       | 73.4 m   |  8 %      |
| 35 | 120.2 m      | 35.6 m   | 70 %      |
| 50 | 43.7 m       | 23.0 m   | 47 %      |

Notes:
- km 20 shows low reduction (8%) because random walk accumulated between the
  15km and 20km fixes with no TRN landing in that window. This is expected
  behaviour for a 5km correction interval on random walk drift.
- VIO+TRN column not separately measurable in this simulation because the
  corridor uses synthetic position drift, not a real optical flow pipeline.
  Live SITL measurement (VIO+TRN vs TRN-alone) is Gate 4 work.
- Drift model is single-run (not Monte Carlo). For programme performance claims,
  cite Gate 2 NAV-01 result: 37.17 m → 23.63 m over 35.5 km on real DEM
  self-match, which is a deterministic measurement.

---

## Known Limitations

| Item | Status |
|------|--------|
| VIO photographic confidence (>0.547) verified in live SITL only | Gate 2 caveat, Gate 4 closure target |
| `_on_camera_frame` write to `_latest_vio` not protected by a lock | Races benign in inject-only mode; real Gazebo subscription is daemon thread — review before HIL |
| NavigationManager imports from `integration/` (Logic Box boundary) | `NadirCameraFrameBridge` has no MAVLink/Gazebo module-level imports — compliant. Review if `gz.transport` import pattern changes. |
| Drift table is single-seed — not a Monte Carlo result | Acceptable for Gate 3 internal evidence. Monte Carlo N=300 required for TASL claim. |
