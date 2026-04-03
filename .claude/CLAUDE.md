# MicroMind — Claude Code Session Instructions

Read docs/qa/MICROMIND_PROJECT_CONTEXT.md before doing anything else.

You are working on the MicroMind / NanoCorteX autonomy stack.
The context file contains programme state, frozen files, open items,
and QA standing rules. Do not ask the user to re-explain the programme.

## Frozen files — never modify without explicit instruction
- core/ekf/error_state_ekf.py
- core/fusion/vio_mode.py
- core/fusion/frame_utils.py
- core/bim/bim.py
- scenarios/bcmp1/bcmp1_runner.py

## Session start checklist (run before any new code is written)
python run_s5_tests.py      # must be 111/111
python run_s8_tests.py      # must be 68/68
python run_bcmp2_tests.py   # must be 90/90

## Context file location
docs/qa/MICROMIND_PROJECT_CONTEXT.md

## End of session — always do both
1. Update docs/qa/MICROMIND_PROJECT_CONTEXT.md Sections 6 and 8
2. Append a QA log entry to docs/qa/MICROMIND_QA_LOG.md
