"""
core/dmrl/dmrl_stub.py
MicroMind Sprint S5 — DMRL (Discrimination & Multi-frame Recognition Logic)
Implements: FR-103, KPI-T01, KPI-T02, KPI-T03

Boundary conditions (Part Two V7 §1.9.3):
  - Lock confidence threshold to proceed    : ≥ 0.85
  - Decoy rejection confidence to abort     : ≥ 0.80 CNN probability over 3 consecutive frames
  - Minimum temporal association window     : ≥ 5 frames @ 25 FPS (= 200 ms dwell)
  - Minimum thermal ROI size                : 8×8 pixels at max engagement range
  - Aimpoint correction limit               : ±15° bearing per L10s-SE window
  - Target re-acquisition timeout           : EO lock lost > 1.5 s during terminal run → abort
  - L10s-SE abort/continue decision         : within ≤ 2 s of activation
"""

from __future__ import annotations

import random
import time
import math
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("DMRL")

# ─── Boundary Constants ───────────────────────────────────────────────────────
LOCK_CONFIDENCE_THRESHOLD   = 0.85   # FR-103 / §1.9.3
DECOY_ABORT_THRESHOLD       = 0.80   # CNN decoy probability to trigger abort
DECOY_ABORT_CONSECUTIVE     = 3      # frames of high decoy confidence required
MIN_DWELL_FRAMES            = 5      # ≥5 frames at 25 FPS = 200 ms
FRAME_RATE_HZ               = 25.0
MIN_THERMAL_ROI_PX          = 8      # 8×8 pixels at max engagement range
AIMPOINT_CORRECTION_LIMIT   = 15.0  # degrees
REACQUISITION_TIMEOUT_S     = 1.5   # seconds before declaring lock lost
L10S_DECISION_TIMEOUT_S     = 2.0   # must decide within 2 s of L10s-SE activation


# ─── Data Structures ──────────────────────────────────────────────────────────
@dataclass
class ThermalTarget:
    """Represents a thermal emitter in the synthetic scene."""
    target_id: str
    is_decoy: bool
    thermal_signature: float        # normalised 0–1 (real targets higher baseline)
    thermal_decay_rate: float       # deg/frame — decoys cool faster
    initial_roi_px: int             # pixel extent at acquisition range
    bearing_deg: float              # relative bearing to UAV boresight
    range_m: float                  # slant range


@dataclass
class DMRLFrameResult:
    """Per-frame output from DMRL processing pipeline."""
    frame_id: int
    timestamp_s: float
    target_id: str
    lock_confidence: float          # 0–1, ≥0.85 to proceed
    decoy_probability: float        # CNN proxy score 0–1
    thermal_roi_px: int             # current ROI size (shrinks with range)
    aimpoint_correction_deg: float  # bearing correction this frame
    is_decoy_flagged: bool          # True when decoy_probability ≥ threshold


@dataclass
class DMRLResult:
    """Final DMRL output handed to L10s-SE."""
    target_id: str
    lock_confidence: float
    is_decoy: bool
    decoy_confidence: float
    dwell_frames: int
    lock_acquired: bool             # confidence ≥ threshold sustained
    lock_lost_timeout: bool         # True if re-acq timeout exceeded
    frames: list[DMRLFrameResult] = field(default_factory=list)
    processing_latency_s: float = 0.0
    log: list[str] = field(default_factory=list)


# ─── Synthetic Thermal Scene Generator ────────────────────────────────────────

def generate_synthetic_scene(
    n_targets: int = 1,
    n_decoys: int = 1,
    seed: Optional[int] = None
) -> list[ThermalTarget]:
    """
    Generate a synthetic terminal thermal scene with real targets and decoys.

    Real targets: higher, stable thermal signature (low decay).
    Decoys (flares/thermal blankets): initial spike then rapid dissipation.
    """
    if seed is not None:
        random.seed(seed)

    targets: list[ThermalTarget] = []

    for i in range(n_targets):
        targets.append(ThermalTarget(
            target_id=f"TGT-{i+1:02d}",
            is_decoy=False,
            thermal_signature=random.uniform(0.87, 0.95),   # real: warm, stable, reliably above 0.85 gate
            thermal_decay_rate=random.uniform(0.001, 0.003), # slow decay
            initial_roi_px=random.randint(18, 32),
            bearing_deg=random.uniform(-5.0, 5.0),
            range_m=random.uniform(800.0, 2500.0),
        ))

    for i in range(n_decoys):
        targets.append(ThermalTarget(
            target_id=f"DCY-{i+1:02d}",
            is_decoy=True,
            thermal_signature=random.uniform(0.85, 0.99),   # decoy: initially hotter
            thermal_decay_rate=random.uniform(0.018, 0.035), # rapid dissipation
            initial_roi_px=random.randint(8, 14),            # smaller ROI
            bearing_deg=random.uniform(-20.0, 20.0),
            range_m=random.uniform(900.0, 2600.0),
        ))

    return targets


# ─── DMRL Processing Pipeline ─────────────────────────────────────────────────

def _thermal_signature_at_frame(target: ThermalTarget, frame_id: int) -> float:
    """Simulate thermal dissipation: real targets stable, decoys cool rapidly."""
    sig = target.thermal_signature - target.thermal_decay_rate * frame_id
    return max(0.0, min(1.0, sig))


def _compute_lock_confidence(
    target: ThermalTarget,
    frame_id: int,
    dwell_frames: int,
) -> float:
    """
    Lock confidence combines:
      - thermal signature strength (dominant term for real targets)
      - temporal association stability (ramps with dwell, gates lock until min frames)
      - ROI size adequacy
    Model: real targets stabilise above 0.85 once min dwell is reached;
           decoys degrade below threshold as thermal dissipation manifests.
    """
    sig = _thermal_signature_at_frame(target, frame_id)
    temporal_factor = min(1.0, dwell_frames / MIN_DWELL_FRAMES)  # 0→1 over first 5 frames
    roi_px = max(1, target.initial_roi_px - frame_id // 4)
    roi_factor = min(1.0, roi_px / MIN_THERMAL_ROI_PX)

    if target.is_decoy:
        # Decoy: initially plausible but rapidly dissipates below threshold
        raw = max(0.0, sig - target.thermal_decay_rate * frame_id * 3.0)
        noise = random.gauss(0.0, 0.035)
        confidence = min(1.0, (raw + noise) * temporal_factor * roi_factor * random.uniform(0.72, 0.84))
    else:
        # Real target: thermal signature plus small noise; temporal gates the result
        noise = random.gauss(0.0, 0.025)
        raw = min(1.0, sig + noise)          # sig ≥ 0.75 → raw ≈ 0.75–0.92
        # Once dwell ≥ 5 and roi adequate, confidence is directly driven by signature
        confidence = raw * temporal_factor * roi_factor

    return round(max(0.0, min(1.0, confidence)), 4)


def _compute_decoy_probability(
    target: ThermalTarget,
    frame_id: int,
    dwell_frames: int,
) -> float:
    """
    Rule-based CNN proxy for decoy probability.
    Uses thermal dissipation gradient: decoys cool fast across frames.
    Real targets: low and stable. Decoys: high and rising with dwell.
    """
    if dwell_frames < 2:
        return random.uniform(0.05, 0.25)  # insufficient data

    sig_now  = _thermal_signature_at_frame(target, frame_id)
    sig_prev = _thermal_signature_at_frame(target, max(0, frame_id - 1))
    decay    = sig_prev - sig_now   # positive → cooling

    if target.is_decoy:
        # Decoy: high decay gradient → high decoy probability as frames accumulate
        base = 0.60 + decay * 18.0 + (dwell_frames / 15.0) * 0.22
        prob = min(0.98, base + random.gauss(0.0, 0.025))
    else:
        # Real target: negligible decay → low decoy probability
        base = 0.05 + decay * 5.0
        prob = max(0.01, min(0.35, base + random.gauss(0.0, 0.02)))

    return round(max(0.0, min(1.0, prob)), 4)


def _compute_aimpoint_correction(target: ThermalTarget, frame_id: int) -> float:
    """Bearing correction within ±15° per L10s-SE window."""
    base_correction = target.bearing_deg * (1.0 - frame_id / 30.0)  # converges
    noise = random.gauss(0.0, 0.3)
    correction = base_correction + noise
    return round(max(-AIMPOINT_CORRECTION_LIMIT, min(AIMPOINT_CORRECTION_LIMIT, correction)), 3)


# ─── Main DMRL Processor ──────────────────────────────────────────────────────

class DMRLProcessor:
    """
    Multi-frame EO/IR target discrimination and lock logic.
    Implements FR-103 processing pipeline §1.9.2.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def _log(self, msg: str, result: DMRLResult):
        if self.verbose:
            logger.info(msg)
        result.log.append(msg)

    def process_target(
        self,
        target: ThermalTarget,
        max_frames: int = 30,
        inject_lock_loss_at: Optional[int] = None,  # frame # to simulate lock dropout
    ) -> DMRLResult:
        """
        Run DMRL pipeline on a single thermal target candidate.
        Returns DMRLResult for L10s-SE consumption.
        """
        t_start = time.perf_counter()
        result = DMRLResult(target_id=target.target_id, lock_confidence=0.0,
                            is_decoy=False, decoy_confidence=0.0,
                            dwell_frames=0, lock_acquired=False, lock_lost_timeout=False)

        self._log(f"[DMRL] Processing target {target.target_id} | is_decoy={target.is_decoy}", result)

        consecutive_decoy_frames = 0
        lock_lost_at: Optional[float] = None
        frame_time_s = 1.0 / FRAME_RATE_HZ

        for fid in range(max_frames):
            t_frame = fid * frame_time_s

            # Simulate lock loss injection
            if inject_lock_loss_at is not None and fid == inject_lock_loss_at:
                self._log(f"[DMRL] Frame {fid}: EO lock LOST (injected)", result)
                lock_lost_at = t_frame
                continue

            # Recover from lock loss
            if lock_lost_at is not None:
                elapsed_loss = t_frame - lock_lost_at
                if elapsed_loss > REACQUISITION_TIMEOUT_S:
                    result.lock_lost_timeout = True
                    self._log(
                        f"[DMRL] Re-acquisition TIMEOUT ({elapsed_loss:.2f}s > {REACQUISITION_TIMEOUT_S}s) → ABORT",
                        result
                    )
                    break
                else:
                    self._log(f"[DMRL] Frame {fid}: re-acquiring... ({elapsed_loss:.2f}s elapsed)", result)
                    continue  # attempting re-acquisition
            else:
                result.dwell_frames += 1

            # Compute decoy probability FIRST — detection is not gated by ROI adequacy
            decoy_prob  = _compute_decoy_probability(target, fid, result.dwell_frames)
            is_dec_flag = decoy_prob >= DECOY_ABORT_THRESHOLD

            # ROI size check — gates lock confidence only (not decoy detection)
            roi_px = max(1, target.initial_roi_px - fid // 8)  # slow ROI shrink
            roi_too_small = roi_px < MIN_THERMAL_ROI_PX

            if roi_too_small:
                self._log(
                    f"[DMRL] Frame {fid}: ROI {roi_px}px < {MIN_THERMAL_ROI_PX}px — "
                    f"lock gated; decoy check active", result
                )

            # Compute lock confidence only when ROI is adequate
            lock_conf  = _compute_lock_confidence(target, fid, result.dwell_frames) if not roi_too_small else 0.0
            aim_corr   = _compute_aimpoint_correction(target, fid)

            frame_res = DMRLFrameResult(
                frame_id=fid, timestamp_s=round(t_frame, 4),
                target_id=target.target_id, lock_confidence=lock_conf,
                decoy_probability=decoy_prob, thermal_roi_px=roi_px,
                aimpoint_correction_deg=aim_corr, is_decoy_flagged=is_dec_flag
            )
            result.frames.append(frame_res)
            if is_dec_flag:
                consecutive_decoy_frames += 1
            else:
                consecutive_decoy_frames = 0

            self._log(
                f"[DMRL] Frame {fid:02d} | lock={lock_conf:.4f} | decoy_p={decoy_prob:.4f} "
                f"| decoy_flag={is_dec_flag} | consec_decoy={consecutive_decoy_frames} "
                f"| aim_corr={aim_corr:+.2f}°",
                result
            )

            # Check decoy abort condition (3 consecutive frames ≥ threshold)
            if consecutive_decoy_frames >= DECOY_ABORT_CONSECUTIVE:
                result.is_decoy = True
                result.decoy_confidence = decoy_prob
                self._log(
                    f"[DMRL] DECOY DETECTED — {consecutive_decoy_frames} consecutive frames "
                    f"≥ {DECOY_ABORT_THRESHOLD} | confidence={decoy_prob:.4f} → ABORT signal to L10s-SE",
                    result
                )
                break

            # Check lock confidence gate (only after minimum dwell)
            if result.dwell_frames >= MIN_DWELL_FRAMES and lock_conf >= LOCK_CONFIDENCE_THRESHOLD:
                result.lock_confidence = lock_conf
                result.lock_acquired = True
                result.decoy_confidence = decoy_prob
                self._log(
                    f"[DMRL] LOCK ACQUIRED — confidence={lock_conf:.4f} ≥ {LOCK_CONFIDENCE_THRESHOLD} "
                    f"after {result.dwell_frames} frames → PROCEED signal to L10s-SE",
                    result
                )
                break

        # Final state if no early exit
        if not result.lock_acquired and not result.is_decoy and not result.lock_lost_timeout:
            # Use last frame confidence
            if result.frames:
                result.lock_confidence = result.frames[-1].lock_confidence
                result.decoy_confidence = result.frames[-1].decoy_probability
            self._log(
                f"[DMRL] Max frames reached — lock_conf={result.lock_confidence:.4f} | "
                f"lock_acquired={result.lock_acquired}",
                result
            )

        result.processing_latency_s = max(1e-6, round(time.perf_counter() - t_start, 6))
        self._log(
            f"[DMRL] Complete | latency={result.processing_latency_s*1000:.1f}ms | "
            f"lock={result.lock_acquired} | is_decoy={result.is_decoy}",
            result
        )
        return result

    def process_scene(
        self,
        targets: list[ThermalTarget],
        max_frames: int = 30,
    ) -> dict[str, DMRLResult]:
        """
        Process all targets in the scene. Returns per-target DMRLResult dict.
        In a real system this would be concurrent; here sequential for SIL.
        """
        results = {}
        for target in targets:
            results[target.target_id] = self.process_target(target, max_frames=max_frames)
        return results

    def select_primary_target(
        self,
        scene_results: dict[str, DMRLResult],
    ) -> Optional[DMRLResult]:
        """
        Select the best locked, non-decoy target from scene processing.
        Returns None if no valid lock found.
        """
        candidates = [
            r for r in scene_results.values()
            if r.lock_acquired and not r.is_decoy and not r.lock_lost_timeout
        ]
        if not candidates:
            return None
        # Highest lock confidence wins
        return max(candidates, key=lambda r: r.lock_confidence)


# ─── Standalone self-test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("=" * 70)
    print("DMRL Self-Test — Synthetic Terminal Scene")
    print("=" * 70)

    processor = DMRLProcessor(verbose=True)
    scene     = generate_synthetic_scene(n_targets=1, n_decoys=1, seed=42)

    print(f"\nScene: {len([t for t in scene if not t.is_decoy])} real target(s), "
          f"{len([t for t in scene if t.is_decoy])} decoy(s)\n")

    results = processor.process_scene(scene, max_frames=30)
    primary = processor.select_primary_target(results)

    print("\n--- Scene Summary ---")
    for tid, r in results.items():
        print(f"  {tid}: lock={r.lock_acquired} conf={r.lock_confidence:.4f} "
              f"is_decoy={r.is_decoy} decoy_conf={r.decoy_confidence:.4f} "
              f"dwell={r.dwell_frames}fr lock_lost={r.lock_lost_timeout}")

    if primary:
        print(f"\n✅ Primary target selected: {primary.target_id} "
              f"(confidence={primary.lock_confidence:.4f})")
    else:
        print("\n❌ No valid lock — L10s-SE will ABORT")
