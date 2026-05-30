# Experiment Tracks

## Synthetic controlled shift

The synthetic suite generates source and target visual domains with controlled camera, lighting, clutter, and occlusion shifts. It provides a controlled setting for measuring coverage, set efficiency, abstention behavior, and calibration sensitivity.

## PushT visual-shift suite

The PushT suite exports public real robot frames, constructs source and target visual-shift domains, builds action-token codebooks with K in {32, 64, 128}, and evaluates conformal action sets across seeds, target-calibration fractions, and alpha levels.

## DROID_100 visual-shift suite

The DROID_100 suite follows the same evaluation structure on a larger public real-robot manipulation dataset. It uses K in {64, 128} and reports aggregate, per-domain, action-geometry, and episode-level diagnostics.

## AutoEval online-log diagnostic

The AutoEval track is optional and auxiliary. It parses public online real-robot evaluation logs when downloadable, constructs temporal source/target splits, and reports action-set coverage and failure-risk diagnostics. The strict exporter is configured to stop when too few usable frames are available unless an explicit under-minimum override is set.
