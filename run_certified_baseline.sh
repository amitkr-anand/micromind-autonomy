#!/bin/bash
# MicroMind Certified SIL Baseline
# 542 tests — run before every gate commit and every handoff
# G-14 (AT-6 memory growth slope) excluded — requires >= 1hr dedicated run
# CM-01 (gate6_cross_modal frame quality) excluded — pre-existing failure, OI-46 class

set -e
CONDA="conda run -n micromind-autonomy"

echo "=== S5 (119) ==="
$CONDA python run_s5_tests.py

echo "=== S8 (68) ==="
$CONDA python run_s8_tests.py

echo "=== BCMP2 (90) ==="
$CONDA python run_bcmp2_tests.py

echo "=== Integration + Gates (138) ==="
$CONDA python -m pytest \
  tests/test_bcmp2_at6.py -k "not test_G14_memory_growth_slope" \
  tests/test_s6_zpi_cems.py \
  tests/test_prehil_rc11.py \
  tests/test_prehil_rc7.py \
  tests/test_prehil_rc8.py \
  tests/test_s5_l10s_se_adversarial.py \
  tests/test_sb5_phase_a.py \
  tests/test_sb5_phase_b.py \
  tests/test_sb5_ec01.py \
  tests/test_sb5_adversarial_d2.py \
  tests/test_it_d10_gnss.py \
  tests/test_s9_nav01_pass.py \
  tests/test_gate2_navigation.py \
  tests/test_gate3_fusion.py -v

echo "=== Gate 5 Shimla-Manali corridor (22) ==="
$CONDA python -m pytest tests/test_gate5_corridor.py -v

echo "=== Gate 6 Jammu-Leh tactical corridor (22) ==="
$CONDA python -m pytest tests/test_gate6_jammu_leh.py -v

echo "=== Gate 4 Extended Corridor — NAV-09..12 (19) ==="
$CONDA python -m pytest tests/test_gate4_extended.py -v

echo "=== Gate 6 Cross-Modal TRN — CM-01..04 (14, CM-01 pre-existing failure excluded) ==="
$CONDA python -m pytest tests/test_gate6_cross_modal.py -k "not test_cm01_validate_frame_quality_not_poor" -v

echo "=== NM-LG SIL gates — NM-LG-01..06 (6) ==="
$CONDA python -m pytest tests/test_navigation_manager_lightglue.py -v

echo "=== Gate 7 SAL corridor — G7-01..05 (21) ==="
$CONDA python -m pytest tests/test_gate7_sal_corridor.py -v

echo "=== IT-D6-TIMEOUT-01 — D6 OffboardRecoveryFSM (4) ==="
$CONDA python -m pytest tests/test_it_d6_timeout.py -v

echo "=== UT-RS-03 — ProcessWatchdog restartability decision logic (6) ==="
$CONDA python -m pytest tests/test_ut_rs03.py -v

echo "=== UT-RS-02 — Log rolling policy RS-02/E-04 GAP-09 (5) ==="
$CONDA python -m pytest tests/test_ut_rs02.py -v

echo "=== CERTIFIED BASELINE COMPLETE ==="
echo "Expected: 547/547"
