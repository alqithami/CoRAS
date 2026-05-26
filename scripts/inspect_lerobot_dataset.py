#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coras.utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect keys/features for a Hugging Face LeRobot dataset.")
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception as exc:
        raise RuntimeError("Install LeRobot first: pip install 'lerobot[video]' or follow Hugging Face LeRobot docs.") from exc
    ds = LeRobotDataset(args.repo_id)
    row = dict(ds[0])
    info = {"repo_id": args.repo_id, "len": len(ds), "sample_keys": list(row.keys())}
    # v2/v3 metadata varies by version; preserve whatever is accessible.
    meta = getattr(ds, "meta", None)
    features = None
    if meta is not None:
        for attr in ["features", "info"]:
            if hasattr(meta, attr):
                try:
                    val = getattr(meta, attr)
                    features = str(val)
                except Exception:
                    pass
    info["meta_repr"] = features
    image_like = [k for k, v in row.items() if "image" in k.lower() or "camera" in k.lower()]
    info["image_like_keys"] = image_like
    info["action_like_keys"] = [k for k in row.keys() if "action" in k.lower()]
    if args.out:
        write_json(args.out, info)
    print(info)


if __name__ == "__main__":
    main()
