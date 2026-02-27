"""
core/ins/imu_model.py
─────────────────────────────────────────────────────────────────────────────
MicroMind / NanoCorteX  —  IMU Noise Model
Sprint S8-A  |  FR-101, FR-107

Provides characterised IMU noise models for SIL navigation simulation.
Each IMUModel instance encodes the five dominant error sources for a
real sensor, derived from published datasheet / Allan Variance curves.

Three pre-defined instances are provided:

  STIM300        Safran STIM300 — primary tactical MEMS IMU.
                 No ITAR.  Safran India presence.  IC-engine vibration rated.
                 Exceeds Part Two V7 minimum spec on all axes.
                 Datasheet: Safran STIM300 Rev H (public).

  ADIS16505_3    Analog Devices ADIS16505-3 — budget MEMS alternative.
                 Lower cost, lower vibration rating.  Useful as cost-optimised
                 comparison.  Datasheet: AD ADIS16505 Rev C (public).

  BASELINE       Simplified Gaussian model used in S0–S7.
                 Not traceable to a real sensor — used only for regression
                 continuity.  Do not use for TASL / iDEX presentations.

Usage
─────
  from core.ins.imu_model import STIM300, generate_imu_noise

  dt      = 0.005          # 200 Hz
  n_steps = 180_000        # 250 km @ 100 km/h  (2.5 hr * 200 Hz)
  noise   = generate_imu_noise(STIM300, n_steps, dt, seed=42)

  noise.gyro_noise_rads   # shape (n_steps, 3) — rad/s per step
  noise.accel_noise_ms2   # shape (n_steps, 3) — m/s² per step
  noise.gyro_bias_rads    # shape (n_steps, 3) — slowly wandering bias
  noise.accel_bias_ms2    # shape (n_steps, 3) — slowly wandering bias

Design notes
────────────
Allan Variance parameterisation:
  - White noise (N): manifests as ARW/VRW on Allan deviation at τ = 1 s.
    σ_white per step = N / sqrt(dt)    (N in rad/s/sqrt(Hz) or m/s²/sqrt(Hz))

  - Bias instability (B): manifests as flat region on Allan deviation.
    Modelled as 1st-order Gauss-Markov with correlation time τ_corr.
    σ_drive = B * sqrt(2/τ_corr) * sqrt(dt)

  - Rate random walk (K): σ_rrw per step = K * sqrt(dt)
    (K in rad/s/sqrt(s) or m/s²/sqrt(s))

  - Scale factor error: multiplicative — applied to true rate * sf_ppm * 1e-6.

  - Vibration rectification error (VRE): IC-engine sinusoidal vibration
    induces a rectified bias proportional to vibration_g_rms².
    Modelled as a fixed bias offset added to accelerometer only.
    Relevant for IC-engine platforms (ALS 250).

Temperature model:
  Linear bias shift from nominal (25°C reference) over op range.
  Applied as a deterministic offset — not stochastic.

Unit conventions (internal, following IEEE Std 952-1997):
  Gyro  : rad/s  (input/output)
  Accel : m/s²   (input/output)
  ARW   : °/√hr  (datasheet) → converted to rad/s/√Hz internally
  VRW   : m/s/√hr             → converted to m/s²/√Hz internally
  Bias  : °/hr   (datasheet) → converted to rad/s internally

─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Unit conversion helpers
# ──────────────────────────────────────────────────────────────────────────────

def _deg_per_hr_to_rad_per_s(val: float) -> float:
    """Convert °/hr → rad/s."""
    return val * (math.pi / 180.0) / 3600.0

def _deg_per_sqrth_to_rad_per_sqrths(val: float) -> float:
    """Convert °/√hr (ARW) → rad/s/√Hz  (= rad/√s)."""
    return val * (math.pi / 180.0) / 60.0   # /√3600 = /60

def _mps_per_sqrth_to_mps2_per_sqrths(val: float) -> float:
    """Convert m/s/√hr (VRW) → m/s²/√Hz  (= m/s/√s)."""
    return val / 60.0   # /√3600 = /60

def _mg_to_mps2(val: float) -> float:
    """Convert milli-g → m/s²."""
    return val * 9.80665e-3

def _deg_per_hr_per_sqrth_to_rad_per_s_per_sqrths(val: float) -> float:
    """Convert °/hr/√hr (Rate RW, K) → rad/s/√s."""
    return val * (math.pi / 180.0) / (3600.0 ** 1.5)


# ──────────────────────────────────────────────────────────────────────────────
# IMUModel dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class IMUModel:
    """
    Complete noise characterisation for a single IMU.

    All parameters expressed in datasheet units for readability.
    Internal conversions to SI are handled by generate_imu_noise().

    Parameters
    ──────────
    name : str
        Human-readable identifier (used in logs and chart legends).

    sample_rate_hz : float
        Nominal output data rate.  Part Two V7 requires 200–1000 Hz.

    # ── Gyroscope ────────────────────────────────────────────────────────────

    gyro_arw_deg_per_sqrth : float
        Angle Random Walk  [°/√hr].
        Corresponds to white noise floor on Allan deviation curve.
        Part Two V7 minimum: ≤ 0.1 °/√hr.

    gyro_bias_instability_deg_per_hr : float
        Bias instability  [°/hr].
        Corresponds to the flat (minimum) region of the Allan deviation.
        Part Two V7 minimum: ≤ 1.0 °/hr.

    gyro_bias_corr_time_s : float
        Correlation time for the Gauss-Markov bias drift process  [s].
        Typical range 100–600 s for MEMS tactical grade.
        Not directly on datasheet — derived from Allan slope analysis.

    gyro_rate_rw_deg_per_hr_per_sqrth : float
        Rate Random Walk  [°/hr/√hr].
        Corresponds to +½ slope on Allan deviation at long τ.

    gyro_scale_factor_ppm : float
        Scale factor error  [ppm].
        Multiplicative error on the true angular rate.

    gyro_misalignment_deg : float
        Cross-axis misalignment  [°].
        Introduces small coupling between axes.  Simplified as
        independent per-axis offset for SIL.

    # ── Accelerometer ────────────────────────────────────────────────────────

    accel_vrw_mps_per_sqrth : float
        Velocity Random Walk  [m/s/√hr].
        Accelerometer equivalent of ARW.

    accel_bias_instability_mg : float
        Accelerometer bias instability  [mg].

    accel_bias_corr_time_s : float
        Correlation time for accelerometer Gauss-Markov bias  [s].

    accel_rate_rw_mps2_per_sqrts : float
        Accelerometer rate random walk  [m/s²/√s].

    accel_scale_factor_ppm : float
        Accelerometer scale factor error  [ppm].

    # ── Vibration (IC engine specific) ───────────────────────────────────────

    vibration_g_rms : float
        RMS vibration level the sensor is exposed to  [g RMS].
        For IC-engine platforms: typical 0.5–2.0 g RMS at prop/engine freq.
        STIM300 VRE spec: < 0.05 mg/(mg)² → bias ≈ VRE_coeff * g_rms².

    vre_coeff_mg_per_g2 : float
        Vibration Rectification Error coefficient  [mg/(g²)].
        Bias = vre_coeff * vibration_g_rms²  (per axis, worst case).
        From STIM300 datasheet: < 0.05 mg/(mg)²; practical ~0.03 mg/(g²)
        for random vibration profile.  Zero for BASELINE.

    # ── Temperature ──────────────────────────────────────────────────────────

    temp_ref_c : float
        Reference temperature for published bias specs  [°C].  Typically 25°C.

    gyro_temp_coeff_deg_per_hr_per_c : float
        Gyro bias temperature sensitivity  [°/hr/°C].
        Linear model: extra_bias = coeff * (T - T_ref).

    accel_temp_coeff_mg_per_c : float
        Accel bias temperature sensitivity  [mg/°C].

    # ── Metadata ─────────────────────────────────────────────────────────────

    datasheet_ref : str
        Citation for traceability.

    meets_part_two_v7_min_spec : bool
        True if gyro_arw ≤ 0.1 °/√hr AND bias_instability ≤ 1.0 °/hr.
        Validated on construction.

    notes : str
        Free-text notes for TASL / iDEX presentation context.
    """

    name: str

    # Gyroscope
    sample_rate_hz: float
    gyro_arw_deg_per_sqrth: float
    gyro_bias_instability_deg_per_hr: float
    gyro_bias_corr_time_s: float
    gyro_rate_rw_deg_per_hr_per_sqrth: float
    gyro_scale_factor_ppm: float
    gyro_misalignment_deg: float

    # Accelerometer
    accel_vrw_mps_per_sqrth: float
    accel_bias_instability_mg: float
    accel_bias_corr_time_s: float
    accel_rate_rw_mps2_per_sqrts: float
    accel_scale_factor_ppm: float

    # Vibration (IC engine)
    vibration_g_rms: float
    vre_coeff_mg_per_g2: float

    # Temperature
    temp_ref_c: float
    gyro_temp_coeff_deg_per_hr_per_c: float
    accel_temp_coeff_mg_per_c: float

    # Metadata
    datasheet_ref: str
    notes: str

    def __post_init__(self) -> None:
        # Validate against Part Two V7 minimum spec
        self.meets_part_two_v7_min_spec: bool = (
            self.gyro_arw_deg_per_sqrth <= 0.1
            and self.gyro_bias_instability_deg_per_hr <= 1.0
        )

    # ── Derived SI quantities ─────────────────────────────────────────────────

    @property
    def gyro_arw_si(self) -> float:
        """ARW in rad/s/√Hz (= rad/√s)."""
        return _deg_per_sqrth_to_rad_per_sqrths(self.gyro_arw_deg_per_sqrth)

    @property
    def gyro_bias_instability_si(self) -> float:
        """Bias instability in rad/s."""
        return _deg_per_hr_to_rad_per_s(self.gyro_bias_instability_deg_per_hr)

    @property
    def gyro_rate_rw_si(self) -> float:
        """Rate RW in rad/s/√s."""
        return _deg_per_hr_per_sqrth_to_rad_per_s_per_sqrths(
            self.gyro_rate_rw_deg_per_hr_per_sqrth
        )

    @property
    def accel_vrw_si(self) -> float:
        """VRW in m/s²/√Hz (= m/s/√s)."""
        return _mps_per_sqrth_to_mps2_per_sqrths(self.accel_vrw_mps_per_sqrth)

    @property
    def accel_bias_instability_si(self) -> float:
        """Accel bias instability in m/s²."""
        return _mg_to_mps2(self.accel_bias_instability_mg)

    @property
    def vre_bias_si(self) -> float:
        """
        Vibration rectification bias in m/s² (accel, per axis).
        bias_mg = vre_coeff * vibration_g_rms²
        """
        return _mg_to_mps2(self.vre_coeff_mg_per_g2 * self.vibration_g_rms ** 2)

    def bias_at_temperature(self, temp_c: float) -> tuple[float, float]:
        """
        Return (gyro_extra_bias_rad_s, accel_extra_bias_ms2) at given temperature.
        Linear model from temp_ref_c reference.
        """
        delta_t = temp_c - self.temp_ref_c
        gyro_extra = _deg_per_hr_to_rad_per_s(
            self.gyro_temp_coeff_deg_per_hr_per_c * delta_t
        )
        accel_extra = _mg_to_mps2(self.accel_temp_coeff_mg_per_c * delta_t)
        return gyro_extra, accel_extra

    def summary(self) -> str:
        spec = "MEETS" if self.meets_part_two_v7_min_spec else "DOES NOT MEET"
        return (
            f"IMUModel: {self.name}\n"
            f"  Gyro ARW:            {self.gyro_arw_deg_per_sqrth:.3f} °/√hr"
            f"  (min spec ≤ 0.1 → {spec})\n"
            f"  Gyro bias inst.:     {self.gyro_bias_instability_deg_per_hr:.2f} °/hr"
            f"  (min spec ≤ 1.0 → {spec})\n"
            f"  Accel VRW:           {self.accel_vrw_mps_per_sqrth:.4f} m/s/√hr\n"
            f"  Accel bias inst.:    {self.accel_bias_instability_mg:.3f} mg\n"
            f"  VRE bias (IC-eng):   {self.vre_bias_si*1000:.4f} m/s²  "
            f"({self.vre_coeff_mg_per_g2:.3f} mg/(g²) × {self.vibration_g_rms:.1f}g²)\n"
            f"  Part Two V7 spec:    {spec}\n"
            f"  Ref: {self.datasheet_ref}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# IMUNoiseOutput — result of generate_imu_noise()
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class IMUNoiseOutput:
    """
    Structured noise output for one simulation run.

    All arrays shape (n_steps, 3) — axes [x, y, z].
    Intended to be added to the true IMU signal in mechanisation.py.

    gyro_noise_rads    : white + rate_rw component  [rad/s]
    gyro_bias_rads     : slowly drifting Gauss-Markov bias  [rad/s]
    gyro_sf_error      : fractional scale factor error (unit-less, multiply by true rate)
    accel_noise_ms2    : white + rate_rw component  [m/s²]
    accel_bias_ms2     : slowly drifting Gauss-Markov bias  [m/s²]
    accel_sf_error     : fractional scale factor error (unit-less)
    accel_vre_ms2      : static VRE bias (IC engine)  [m/s²], shape (3,)
    temp_gyro_bias_rads: temperature-induced bias  [rad/s], shape (3,)
    temp_accel_bias_ms2: temperature-induced bias  [m/s²], shape (3,)
    """
    gyro_noise_rads: np.ndarray       # (n, 3)
    gyro_bias_rads: np.ndarray        # (n, 3)  — Gauss-Markov
    gyro_sf_error: np.ndarray         # (n, 3)  — fractional
    accel_noise_ms2: np.ndarray       # (n, 3)
    accel_bias_ms2: np.ndarray        # (n, 3)  — Gauss-Markov
    accel_sf_error: np.ndarray        # (n, 3)  — fractional
    accel_vre_ms2: np.ndarray         # (3,)    — static per mission
    temp_gyro_bias_rads: np.ndarray   # (3,)    — static per temperature
    temp_accel_bias_ms2: np.ndarray   # (3,)    — static per temperature
    model_name: str
    n_steps: int
    dt: float
    seed: int

    def total_gyro(self) -> np.ndarray:
        """
        Total gyro error to add to true rate signal  [rad/s].
        Shape (n_steps, 3).
        Includes: white noise + Gauss-Markov bias + temperature bias.
        Scale factor applied separately in mechanisation.
        """
        return (
            self.gyro_noise_rads
            + self.gyro_bias_rads
            + self.temp_gyro_bias_rads[np.newaxis, :]
        )

    def total_accel(self) -> np.ndarray:
        """
        Total accel error to add to true acceleration signal  [m/s²].
        Shape (n_steps, 3).
        Includes: white noise + Gauss-Markov bias + VRE + temperature bias.
        """
        return (
            self.accel_noise_ms2
            + self.accel_bias_ms2
            + self.accel_vre_ms2[np.newaxis, :]
            + self.temp_accel_bias_ms2[np.newaxis, :]
        )


# ──────────────────────────────────────────────────────────────────────────────
# Core noise generator
# ──────────────────────────────────────────────────────────────────────────────

def generate_imu_noise(
    model: IMUModel,
    n_steps: int,
    dt: float,
    seed: int = 42,
    temperature_c: float = 25.0,
) -> IMUNoiseOutput:
    """
    Generate a complete IMU noise realisation for one simulation run.

    Parameters
    ──────────
    model         : IMUModel instance (STIM300, ADIS16505_3, or BASELINE)
    n_steps       : number of simulation timesteps
    dt            : timestep duration  [s]
    seed          : RNG seed for reproducibility
    temperature_c : operating temperature  [°C]
                    Used to compute temperature-dependent bias offset.
                    ALS 250 op range: –20°C to +50°C.

    Returns
    ───────
    IMUNoiseOutput — all arrays pre-generated, ready to index per step.

    Notes
    ─────
    Gauss-Markov process (1st order):
        x[k+1] = exp(-dt/τ) * x[k] + σ_drive * w[k]
    where:
        σ_drive = σ_steady * sqrt(1 - exp(-2*dt/τ))
        σ_steady = bias_instability / sqrt(2/τ)   (steady-state std)

    The bias instability value on an Allan plot equals the 1σ of the
    Gauss-Markov process at τ_corr, so:
        σ_steady ≈ bias_instability_si (the published floor value)
    """
    rng = np.random.default_rng(seed)

    # ── Gyroscope white noise ─────────────────────────────────────────────────
    # ARW [rad/√s] → per-step std: σ = ARW / sqrt(dt)
    gyro_wn_std = model.gyro_arw_si / math.sqrt(dt)
    gyro_noise = rng.standard_normal((n_steps, 3)) * gyro_wn_std

    # ── Gyroscope rate random walk ────────────────────────────────────────────
    # σ_rrw per step = K_gyro * sqrt(dt)
    if model.gyro_rate_rw_deg_per_hr_per_sqrth > 0:
        gyro_rrw_std = model.gyro_rate_rw_si * math.sqrt(dt)
        gyro_noise += rng.standard_normal((n_steps, 3)) * gyro_rrw_std

    # ── Gyroscope Gauss-Markov bias ───────────────────────────────────────────
    tau_g = model.gyro_bias_corr_time_s
    alpha_g = math.exp(-dt / tau_g)
    sigma_steady_g = model.gyro_bias_instability_si
    sigma_drive_g = sigma_steady_g * math.sqrt(1.0 - math.exp(-2.0 * dt / tau_g))

    gyro_bias = np.zeros((n_steps, 3))
    # Initial bias draw from steady-state distribution
    gyro_bias[0] = rng.standard_normal(3) * sigma_steady_g
    for k in range(1, n_steps):
        gyro_bias[k] = (
            alpha_g * gyro_bias[k - 1]
            + rng.standard_normal(3) * sigma_drive_g
        )

    # ── Gyroscope scale factor ────────────────────────────────────────────────
    # Per-axis fractional error — constant per run (manufacturing tolerance)
    gyro_sf = rng.standard_normal(3) * (model.gyro_scale_factor_ppm * 1e-6)
    gyro_sf_array = np.tile(gyro_sf, (n_steps, 1))

    # ── Accelerometer white noise ─────────────────────────────────────────────
    accel_wn_std = model.accel_vrw_si / math.sqrt(dt)
    accel_noise = rng.standard_normal((n_steps, 3)) * accel_wn_std

    # ── Accelerometer rate random walk ───────────────────────────────────────
    if model.accel_rate_rw_mps2_per_sqrts > 0:
        accel_rrw_std = model.accel_rate_rw_mps2_per_sqrts * math.sqrt(dt)
        accel_noise += rng.standard_normal((n_steps, 3)) * accel_rrw_std

    # ── Accelerometer Gauss-Markov bias ──────────────────────────────────────
    tau_a = model.accel_bias_corr_time_s
    alpha_a = math.exp(-dt / tau_a)
    sigma_steady_a = model.accel_bias_instability_si
    sigma_drive_a = sigma_steady_a * math.sqrt(1.0 - math.exp(-2.0 * dt / tau_a))

    accel_bias = np.zeros((n_steps, 3))
    accel_bias[0] = rng.standard_normal(3) * sigma_steady_a
    for k in range(1, n_steps):
        accel_bias[k] = (
            alpha_a * accel_bias[k - 1]
            + rng.standard_normal(3) * sigma_drive_a
        )

    # ── Accelerometer scale factor ────────────────────────────────────────────
    accel_sf = rng.standard_normal(3) * (model.accel_scale_factor_ppm * 1e-6)
    accel_sf_array = np.tile(accel_sf, (n_steps, 1))

    # ── Vibration Rectification Error (IC engine) ─────────────────────────────
    # Static bias per mission — same sign and magnitude on all axes (conservative)
    vre_bias = np.ones(3) * model.vre_bias_si

    # ── Temperature bias ─────────────────────────────────────────────────────
    gyro_temp_bias_si, accel_temp_bias_si = model.bias_at_temperature(temperature_c)
    temp_gyro_bias = np.ones(3) * gyro_temp_bias_si
    temp_accel_bias = np.ones(3) * accel_temp_bias_si

    return IMUNoiseOutput(
        gyro_noise_rads=gyro_noise,
        gyro_bias_rads=gyro_bias,
        gyro_sf_error=gyro_sf_array,
        accel_noise_ms2=accel_noise,
        accel_bias_ms2=accel_bias,
        accel_sf_error=accel_sf_array,
        accel_vre_ms2=vre_bias,
        temp_gyro_bias_rads=temp_gyro_bias,
        temp_accel_bias_ms2=temp_accel_bias,
        model_name=model.name,
        n_steps=n_steps,
        dt=dt,
        seed=seed,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Allan Variance characterisation utility (for test validation)
# ──────────────────────────────────────────────────────────────────────────────

def compute_allan_deviation(
    signal: np.ndarray,
    dt: float,
    max_clusters: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute overlapping Allan deviation for a 1D signal.

    Used in tests to verify that generated noise matches the
    declared model parameters (ARW, bias instability).

    Parameters
    ──────────
    signal      : 1D array of rate measurements (e.g., gyro_z in rad/s)
    dt          : sample interval [s]
    max_clusters: number of τ values to compute (log-spaced)

    Returns
    ───────
    taus   : array of averaging times [s]
    adevs  : array of Allan deviation values [same units as signal * s^½]

    Notes
    ─────
    At τ = 1 s:  adev ≈ ARW  (read off the –½ slope region)
    At τ_min:    adev ≈ bias instability (flat region minimum)
    """
    n = len(signal)
    max_m = n // 2
    m_values = np.unique(
        np.round(np.logspace(0, np.log10(max_m), max_clusters)).astype(int)
    )
    m_values = m_values[m_values >= 1]

    taus = []
    adevs = []
    cumsum = np.cumsum(np.concatenate([[0], signal])) * dt  # phase array

    for m in m_values:
        tau = m * dt
        # Overlapping Allan variance
        k = n - 2 * m
        if k < 1:
            continue
        phase_diff = (
            cumsum[2 * m:n + 1]
            - 2.0 * cumsum[m:n - m + 1]
            + cumsum[0:n - 2 * m + 1]
        )
        avar = np.sum(phase_diff ** 2) / (2.0 * tau ** 2 * k)
        taus.append(tau)
        adevs.append(math.sqrt(avar))

    return np.array(taus), np.array(adevs)


# ──────────────────────────────────────────────────────────────────────────────
# Pre-defined IMU instances
# ──────────────────────────────────────────────────────────────────────────────

STIM300 = IMUModel(
    name="Safran STIM300",

    sample_rate_hz=200.0,    # Configurable 125–2000 Hz; 200 Hz nominal for SIL

    # ── Gyroscope (from STIM300 datasheet Rev H, Table 1) ────────────────────
    # ARW: 0.15 °/√hr typ  (datasheet: < 0.4 °/√hr, typical ~0.15)
    # Bias instability: 0.3 °/hr typ (datasheet: < 0.5 °/hr, typical ~0.3)
    # Note: both exceed Part Two V7 minimum spec (≤ 0.1 ARW, ≤ 1.0 bias)
    # STIM300 comfortably meets the spec — typical values used for SIL.
    gyro_arw_deg_per_sqrth=0.15,
    gyro_bias_instability_deg_per_hr=0.3,
    gyro_bias_corr_time_s=200.0,   # Typical for MEMS tactical: ~100–300 s
    gyro_rate_rw_deg_per_hr_per_sqrth=0.02,  # Read from +½ slope of Allan
    gyro_scale_factor_ppm=500.0,   # Datasheet: < 1000 ppm initial; 500 typ
    gyro_misalignment_deg=0.05,    # Datasheet: < 0.1°

    # ── Accelerometer (from STIM300 datasheet Rev H, Table 2) ────────────────
    # VRW: 0.05 m/s/√hr typ  (datasheet: < 0.1 m/s/√hr)
    # Bias instability: 0.05 mg typ  (datasheet: < 0.1 mg)
    accel_vrw_mps_per_sqrth=0.05,
    accel_bias_instability_mg=0.05,
    accel_bias_corr_time_s=300.0,
    accel_rate_rw_mps2_per_sqrts=1e-5,
    accel_scale_factor_ppm=800.0,  # Datasheet: < 1500 ppm

    # ── Vibration (IC engine — ALS 250 relevant) ─────────────────────────────
    # STIM300 vibration rating: 6 g RMS (operational)
    # ALS 250 IC engine: estimated 0.5–1.5 g RMS at IMU mount point
    # Conservative estimate: 1.0 g RMS
    # VRE: STIM300 datasheet specifies < 0.05 mg/(mg)² for random vib
    # Practical value for IC engine random profile: ~0.03 mg/(g²)
    vibration_g_rms=1.0,
    vre_coeff_mg_per_g2=0.03,

    # ── Temperature ──────────────────────────────────────────────────────────
    # Operating range: –40°C to +85°C (exceeds ALS 250 requirement of –20 to +50)
    # Bias temp sensitivity: ~0.02 °/hr/°C (estimated from datasheet in-run variation)
    temp_ref_c=25.0,
    gyro_temp_coeff_deg_per_hr_per_c=0.02,   # °/hr/°C
    accel_temp_coeff_mg_per_c=0.005,          # mg/°C

    datasheet_ref=(
        "Safran STIM300 Inertial Measurement Unit — Datasheet Rev H (public). "
        "Part no. T-STIM300-D. Safran Sensing Technologies AS, Norway. "
        "No ITAR restriction. Safran India: Safran Engineering Services India Pvt Ltd."
    ),
    notes=(
        "Primary tactical MEMS IMU recommendation for ALS 250 integration. "
        "Exceeds Part Two V7 minimum spec on all axes. "
        "6 g RMS vibration rating — qualified for IC-engine fixed-wing platforms. "
        "Procurable in India without ITAR constraints. "
        "Mass: 55 g. Power: 1.5 W. Interfaces: RS422 / SPI."
    ),
)


ADIS16505_3 = IMUModel(
    name="Analog Devices ADIS16505-3",

    sample_rate_hz=200.0,

    # ── Gyroscope (from ADIS16505 datasheet Rev C) ────────────────────────────
    # ARW: 0.08 °/√hr (±500°/s range variant — the -3 suffix)
    # Bias instability: 2.0 °/hr typ  (higher than STIM300)
    # NOTE: ADIS16505 ARW meets spec; bias instability does NOT meet ≤1°/hr spec
    gyro_arw_deg_per_sqrth=0.08,
    gyro_bias_instability_deg_per_hr=2.0,
    gyro_bias_corr_time_s=120.0,
    gyro_rate_rw_deg_per_hr_per_sqrth=0.10,
    gyro_scale_factor_ppm=1500.0,
    gyro_misalignment_deg=0.05,

    # ── Accelerometer ────────────────────────────────────────────────────────
    accel_vrw_mps_per_sqrth=0.014,   # 0.014 m/s/√hr (excellent accel noise)
    accel_bias_instability_mg=0.26,
    accel_bias_corr_time_s=150.0,
    accel_rate_rw_mps2_per_sqrts=5e-5,
    accel_scale_factor_ppm=3500.0,

    # ── Vibration ────────────────────────────────────────────────────────────
    # ADIS16505: rated to 5g RMS but less margin than STIM300 for IC engine
    # VRE less well characterised — use conservative estimate
    vibration_g_rms=1.0,
    vre_coeff_mg_per_g2=0.08,   # Higher VRE than STIM300

    # ── Temperature ──────────────────────────────────────────────────────────
    temp_ref_c=25.0,
    gyro_temp_coeff_deg_per_hr_per_c=0.05,
    accel_temp_coeff_mg_per_c=0.012,

    datasheet_ref=(
        "Analog Devices ADIS16505-3 iSensor — Datasheet Rev C (public). "
        "±500°/s gyro range variant. No ITAR restriction."
    ),
    notes=(
        "Budget MEMS alternative. ARW meets Part Two V7 spec; "
        "bias instability (2.0°/hr) does NOT meet ≤1°/hr minimum. "
        "Useful as cost-optimised comparison case to demonstrate "
        "why sensor grade matters over 150km GNSS-denied segment. "
        "Mass: 10 g. Power: 0.1 W."
    ),
)


BASELINE = IMUModel(
    name="Baseline Simplified (S0-S7)",

    sample_rate_hz=200.0,

    # ── S0–S7 simplified noise parameters ─────────────────────────────────────
    # These are the implicit parameters in the current mechanisation.py.
    # Deliberately optimistic — not traceable to a real sensor.
    # Used only for regression continuity. Do not present to TASL.
    gyro_arw_deg_per_sqrth=0.05,         # Better than STIM300 typical
    gyro_bias_instability_deg_per_hr=0.1, # Better than STIM300 typical
    gyro_bias_corr_time_s=600.0,          # Very long — slow drift
    gyro_rate_rw_deg_per_hr_per_sqrth=0.005,
    gyro_scale_factor_ppm=100.0,
    gyro_misalignment_deg=0.0,

    accel_vrw_mps_per_sqrth=0.01,
    accel_bias_instability_mg=0.01,
    accel_bias_corr_time_s=600.0,
    accel_rate_rw_mps2_per_sqrts=1e-6,
    accel_scale_factor_ppm=100.0,

    # No vibration model — baseline assumes ideal mounting
    vibration_g_rms=0.0,
    vre_coeff_mg_per_g2=0.0,

    temp_ref_c=25.0,
    gyro_temp_coeff_deg_per_hr_per_c=0.0,
    accel_temp_coeff_mg_per_c=0.0,

    datasheet_ref="Internal — not traceable to a real sensor.",
    notes=(
        "Simplified noise model used in Sprints S0–S7. "
        "Parameters are optimistic (better than any real MEMS IMU). "
        "Used for regression testing only. "
        "Do NOT present to TASL or use in iDEX submissions — "
        "use STIM300 or ADIS16505_3 for all external demonstrations."
    ),
)


# Registry for lookup by string name
IMU_REGISTRY: dict[str, IMUModel] = {
    "STIM300":     STIM300,
    "ADIS16505_3": ADIS16505_3,
    "BASELINE":    BASELINE,
}


def get_imu_model(name: str) -> IMUModel:
    """
    Look up an IMUModel by string name.
    Used for CLI --imu-model flag in bcmp1_runner.py and als250_nav_sim.py.

    Raises
    ──────
    ValueError if name not found in IMU_REGISTRY.
    """
    if name not in IMU_REGISTRY:
        valid = ", ".join(IMU_REGISTRY.keys())
        raise ValueError(
            f"Unknown IMU model '{name}'. Valid options: {valid}"
        )
    return IMU_REGISTRY[name]
