# Environment Notes

The implementation was designed to run on Apple Silicon for smoke tests and moderate public-data runs, and on Linux/NVIDIA GPU machines for larger DROID and AutoEval runs.

Recommended macOS settings:

```bash
export CORAS_DEVICE=auto
export CORAS_NUM_WORKERS=0
export CORAS_LEROBOT_VIDEO_BACKEND=pyav
```

Recommended RunPod/Linux settings:

```bash
export CORAS_DEVICE=cuda
export CORAS_NUM_WORKERS=4
export CORAS_LEROBOT_VIDEO_BACKEND=pyav
```

No trained checkpoints are included. Full runs regenerate checkpoints locally.
