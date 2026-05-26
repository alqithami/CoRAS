# Running the Complete CoRAS Experiments

## 1. Local Mac M4 Max validation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-mac.txt
bash scripts/run_fast_local_validation.sh
```

This confirms the full stack: data generation, source training, prompt adaptation, conformal evaluation, per-domain metrics, and aggregate table generation.

## 2. Full synthetic / controlled sim-to-shift suite

```bash
bash scripts/run_complete_synthetic_suite.sh
```

Recommended Mac M4 Max settings:

```bash
CORAS_SIM_N=9000 CORAS_SEEDS="0 1 2 3 4" CORAS_CALIB_FRACS="0.10 0.25 0.40" bash scripts/run_complete_synthetic_suite.sh
```

Expected duration depends on device and PyTorch/MPS support. The run uses `small_cnn` and is meant to be feasible locally.

## 3. Full DROID_100 offline real-robot suite

```bash
pip install -r requirements-robotdata.txt
bash scripts/run_droid100_complete.sh
```

RunPod / CUDA settings:

```bash
CORAS_DEVICE=cuda CORAS_ENCODER=resnet18 CORAS_BATCH_SIZE=128 CORAS_EPOCHS=8 CORAS_ADAPTER_EPOCHS=4 bash scripts/run_droid100_complete.sh
```

The script uses `lerobot/droid_100`, camera key `observation.images.exterior_image_1_left`, and `action`. It holds out task domains automatically using `scripts/write_config_from_npz.py`.

## 4. Full PushT suite

```bash
pip install -r requirements-robotdata.txt
bash scripts/run_pusht_complete.sh
```

PushT is useful for debugging and for a compact public-data appendix track. The DROID_100 track is more important for the no-new-real-robot evidence claim.

## 5. Inspecting a LeRobot dataset before export

```bash
python scripts/inspect_lerobot_dataset.py --repo-id lerobot/droid_100 --out results/droid100_keys.json
```

Use this if the LeRobot package changes key names.

## 6. Files to quote in the paper

Use these outputs:

```text
results/<track>/aggregate/main_table_alpha010.csv
results/<track>/aggregate/main_table_alpha010.tex
results/<track>/aggregate/aggregate_metrics.csv
results/<track>/aggregate/aggregate_domain_metrics.csv
results/<track>/aggregate/box_coverage_alpha010.png
results/<track>/aggregate/box_mean_set_size_alpha010.png
results/<track>/aggregate/box_fail_to_abstain_rate_alpha010.png
```

## 7. Reviewer-facing appendix coverage

The complete suite gives:

- repeated seeds;
- alpha sweeps: 0.05, 0.10, 0.20;
- calibration-size sweeps;
- source vs target split reporting;
- per-domain coverage and set size;
- method baselines;
- prediction-set CSVs for failure case inspection;
- latency proxy through collection timings in `eval_metadata_alpha*.json`.
