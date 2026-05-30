# Dataset Sources

This package does not redistribute raw public robot datasets. The scripts download data through public dataset interfaces.

- Synthetic controlled-shift data: generated locally by `scripts/make_simulated_affordance_data.py`.
- PushT: exported from the LeRobot PushT dataset via `scripts/export_lerobot_to_npz.py`.
- DROID_100: exported from the LeRobot DROID_100 dataset via `scripts/export_lerobot_to_npz.py`.
- AutoEval public logs: optional auxiliary online-log diagnostic exported by `scripts/export_autoeval_to_npz.py`.

Dataset summaries for the runs used in the paper are provided in `metadata/dataset_summaries.json`.
