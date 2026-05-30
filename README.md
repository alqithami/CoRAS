# Supplementary Material for CoRAS

This archive contains supplementary material for the submitted paper on Conformal Robot Action Sets under domain shift. It includes a standalone supplementary PDF, an anonymized implementation snapshot, configuration files, aggregate result tables, and reproduction instructions.

The archive does not include raw public datasets or trained model checkpoints. Public robot datasets are downloaded by the provided scripts. Aggregate CSV and TeX tables correspond to the reported results in the main paper and supplementary PDF.

## Contents

- `supplementary.pdf`: standalone supplementary material.
- `code/`: anonymized code snapshot for data export, action-codebook construction, model training, conformal evaluation, diagnostics, and aggregation.
- `results/`: aggregate CSV/TeX tables and diagnostic summaries used for the paper.
- `metadata/`: dataset summaries, manifests, and notes on excluded large artifacts.

## Reproduction levels

1. Smoke test: runs a small synthetic pipeline to verify the installation.
2. Table reproduction: uses the included aggregate CSV files to regenerate paper tables.
3. Full experiments: downloads public data and reruns the synthetic, PushT, DROID, and optional AutoEval tracks.

