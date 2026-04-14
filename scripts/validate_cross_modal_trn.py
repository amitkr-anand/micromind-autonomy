#!/usr/bin/env python3
"""
Cross-modal TRN validation.
Run when Blender frames are ready:

  python scripts/validate_cross_modal_trn.py
    --frames data/synthetic_imagery/shimla_corridor/
    --corridor shimla_local
    --output docs/qa/cross_modal_trn_results.md

Req IDs: NAV-02, AD-01, EC-13
"""
import argparse
import sys
sys.path.insert(0, '.')

from core.trn.dem_loader import DEMLoader
from core.trn.hillshade_generator import HillshadeGenerator
from core.trn.terrain_suitability import TerrainSuitabilityScorer
from core.trn.phase_correlation_trn import PhaseCorrelationTRN
from core.trn.blender_frame_ingestor import BlenderFrameIngestor
from core.trn.cross_modal_evaluator import CrossModalEvaluator
from core.navigation.corridors import SHIMLA_LOCAL, SHIMLA_MANALI
import time


def main():
    parser = argparse.ArgumentParser(
        description='Cross-modal TRN validation against Blender-rendered frames.'
    )
    parser.add_argument('--frames', required=True,
                        help='Directory containing frame_km*.png files')
    parser.add_argument('--corridor', default='shimla_local',
                        choices=['shimla_local', 'shimla_manali'])
    parser.add_argument('--output', default='docs/qa/cross_modal_trn_results.md')
    parser.add_argument('--altitude', type=float, default=150.0,
                        help='AGL altitude at which frames were rendered (m)')
    args = parser.parse_args()

    corridor = SHIMLA_LOCAL if args.corridor == 'shimla_local' else SHIMLA_MANALI

    print(f'Loading DEM from {corridor.terrain_dir}...')
    loader = DEMLoader.from_directory(corridor.terrain_dir)

    # Use Blender sun parameters (azimuth 135°, elevation 45°)
    gen = HillshadeGenerator(azimuth_deg=135.0, elevation_deg=45.0)
    scorer = TerrainSuitabilityScorer()
    trn = PhaseCorrelationTRN(
        loader, gen, scorer,
        clock_fn=lambda: int(time.monotonic() * 1000)
    )

    ingestor = BlenderFrameIngestor(
        frames_dir=args.frames,
        corridor=corridor,
        altitude_m=args.altitude,
    )

    evaluator = CrossModalEvaluator(loader, gen, scorer, trn)

    print(f'Frames available: {ingestor.get_available_kms()}')
    print()
    print('Running cross-modal evaluation...')
    print()

    results = evaluator.evaluate_corridor(ingestor)

    # Print results table
    print('Cross-Modal TRN Results')
    print('Query:     Blender RGB frames')
    print('Reference: DEM hillshade tiles')
    print(f'Corridor:  {corridor.name}')
    print(f'Altitude:  {args.altitude}m AGL')
    print()
    print(f'{"km":>4} | {"Status":>10} | {"Peak":>6} | {"Suit":>5} | '
          f'{"Error_m":>7} | Quality')
    print('-' * 55)

    for r in results.per_frame:
        print(
            f'{r.km:4.0f} | '
            f'{r.status:>10} | '
            f'{r.peak_value:6.4f} | '
            f'{r.suitability_score:5.3f} | '
            f'{r.localisation_error_m:7.1f} | '
            f'{r.quality}'
        )

    print()
    print('Summary:')
    print(f'  Accepted: {results.n_accepted}/{results.n_frames}')
    print(f'  Mean peak (accepted): '
          f'{results.mean_peak_accepted:.4f}' if results.n_accepted else
          f'  Mean peak (accepted): N/A')
    print(f'  P50 error: {results.p50_error_m:.1f}m'
          if not __import__('math').isnan(results.p50_error_m)
          else '  P50 error: N/A')
    print(f'  P95 error: {results.p95_error_m:.1f}m'
          if not __import__('math').isnan(results.p95_error_m)
          else '  P95 error: N/A')
    print(f'  P99 error: {results.p99_error_m:.1f}m'
          if not __import__('math').isnan(results.p99_error_m)
          else '  P99 error: N/A')
    print()
    print(f'Suggested min_peak_value: '
          f'{results.threshold_calibration["suggested_threshold"]:.3f}')
    print(f'Current threshold: {trn._min_peak:.3f}')

    # Write markdown report
    _write_report(results, args.output, corridor, args.altitude, trn)
    print(f'Report written: {args.output}')


def _write_report(results, output_path, corridor, altitude, trn):
    import os
    import math as _math
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('# Cross-Modal TRN Validation\n')
        f.write('## MicroMind Pre-HIL Evidence\n\n')
        f.write(f'**Date:** 15 April 2026\n')
        f.write(f'**Corridor:** {corridor.name}\n')
        f.write(
            f'**Query source:** Blender-rendered RGB frames '
            f'(Sentinel-2 texture + GLO-30 DEM)\n'
        )
        f.write(
            f'**Reference source:** DEM hillshade tiles '
            f'(Copernicus GLO-30)\n'
        )
        f.write(f'**Altitude:** {altitude}m AGL\n')
        f.write(f'**Sun:** azimuth 135deg, elevation 45deg\n\n')
        f.write('## Results\n\n')
        f.write(
            '| km | Status | Peak | Suitability | Error (m) | Quality |\n'
        )
        f.write(
            '|----|--------|------|-------------|-----------|--------|\n'
        )
        for r in results.per_frame:
            err_str = (
                f'{r.localisation_error_m:.1f}'
                if not _math.isnan(r.localisation_error_m)
                else 'N/A'
            )
            f.write(
                f'| {r.km:.0f} | {r.status} | {r.peak_value:.4f} |'
                f' {r.suitability_score:.3f} | {err_str} | {r.quality} |\n'
            )
        f.write('\n## Summary\n\n')
        f.write(f'- Accepted: {results.n_accepted}/{results.n_frames}\n')
        f.write(
            f'- Mean peak (accepted): '
            f'{results.mean_peak_accepted:.4f}\n'
            if not _math.isnan(results.mean_peak_accepted)
            else '- Mean peak (accepted): N/A\n'
        )
        f.write(
            f'- P50 error: {results.p50_error_m:.1f}m\n'
            if not _math.isnan(results.p50_error_m)
            else '- P50 error: N/A\n'
        )
        f.write(
            f'- P95 error: {results.p95_error_m:.1f}m\n'
            if not _math.isnan(results.p95_error_m)
            else '- P95 error: N/A\n'
        )
        f.write(
            f'- P99 error: {results.p99_error_m:.1f}m\n'
            if not _math.isnan(results.p99_error_m)
            else '- P99 error: N/A\n'
        )
        f.write(
            f'- Suggested threshold: '
            f'{results.threshold_calibration["suggested_threshold"]:.3f}\n'
        )
        f.write(f'- Current threshold: {trn._min_peak:.3f}\n\n')
        f.write('## Interpretation\n\n')
        f.write(
            'Peak values in cross-modal matching (Blender RGB vs DEM hillshade) '
            'are lower than self-match (1.0) by design. '
            'The CAS paper reports 0.3–0.7 over textured terrain. '
            'Values above the acceptance threshold indicate reliable TRN '
            'corrections in operational conditions.\n'
        )


if __name__ == '__main__':
    main()
