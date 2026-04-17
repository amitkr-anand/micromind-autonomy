#!/bin/bash
# MicroMind Certified SIL Baseline
# 406 tests — run before every gate commit and every handoff
# G-14 (AT-6 memory growth slope) excluded — requires >= 1hr dedicated run

set -e
CONDA="conda run -n micromind-autonomy"

echo "=== S5 (119) ==="
$CONDA python run_s5_tests.py

echo "=== S8 (68) ==="
$CONDA python run_s8_tests.py

echo "=== BCMP2 (90) ==="
$CONDA python run_bcmp2_tests.py

echo "=== Integration + Gates (129) ==="
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
  tests/test_s9_nav01_pass.py \
  tests/test_gate2_navigation.py \
  tests/test_gate3_fusion.py -v

echo "=== Gate 5 Shimla-Manali corridor (22) ==="
$CONDA python -m pytest tests/test_gate5_corridor.py -v

echo "=== Gate 6 Jammu-Leh tactical corridor (22) ==="
$CONDA python -m pytest tests/test_gate6_jammu_leh.py -v

echo "=== CERTIFIED BASELINE COMPLETE ==="
echo "Expected: 450/450"
