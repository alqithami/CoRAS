# CoRAS Appendix Checklist

The appendix should include the following items, all supported by the pipeline outputs.

## Dataset and split details

- Dataset card for each track: synthetic sim-to-shift, DROID_100, PushT.
- Target-domain definitions and held-out domains.
- Episode-block split procedure.
- Action-token codebook construction and quantization MSE.
- Train/tune/calibration/test counts.

## Metrics

- Marginal coverage at each alpha.
- Absolute coverage gap.
- Mean, median, and 90th-percentile set size.
- Top-1 accuracy.
- Singleton rate and abstention rate.
- Fail-to-abstain rate, used as unsafe singleton execution proxy.
- Per-domain coverage gap.
- ECE for top-1 probabilities.

## Baselines

- Top-1 singleton.
- Top-k calibrated.
- Vanilla split conformal.
- APS conformal.
- Temperature + conformal.
- Prompt-only top-1.
- CoRAS inverse-probability conformal.
- CoRAS APS.
- Mondrian/domain-conditional conformal.

## Sweeps

- Alpha: 0.05, 0.10, 0.20.
- Calibration fraction: 0.10, 0.25, 0.40.
- Seeds: at least 3 for initial appendix, 5 for final submission.
- Codebook sizes for public robot data: 32, 64, 128 if time permits.

## Failure analysis

Use `prediction_sets_compact_alpha*.csv` and `prediction_sets_coras_alpha*.csv` to show:

- false singleton failures;
- overlarge sets;
- target-domain-specific failures;
- target images where occlusion/camera shift inflates set size.

## Limitations text

State clearly that the work does not collect new real-robot rollouts. The real-robot evidence is offline evaluation on public robot trajectories; the controlled transfer evidence is simulation/domain-shift evaluation.
