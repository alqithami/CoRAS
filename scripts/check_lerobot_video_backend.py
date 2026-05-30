#!/usr/bin/env python
from __future__ import annotations
import argparse
import inspect
import os


def main() -> None:
    parser = argparse.ArgumentParser(description='Preflight LeRobot frame decoding for CoRAS exporters.')
    parser.add_argument('--repo-id', default='lerobot/pusht')
    parser.add_argument('--video-backend', default=os.environ.get('CORAS_LEROBOT_VIDEO_BACKEND', 'pyav'))
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    os.environ['VIDEO_BACKEND'] = args.video_backend
    os.environ['CORAS_LEROBOT_VIDEO_BACKEND'] = args.video_backend

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    kwargs = {}
    try:
        if 'video_backend' in inspect.signature(LeRobotDataset).parameters:
            kwargs['video_backend'] = args.video_backend
    except Exception:
        pass
    ds = LeRobotDataset(args.repo_id, **kwargs)
    row = dict(ds[args.index])
    image_keys = [k for k in row if 'image' in k.lower() or 'camera' in k.lower()]
    action_keys = [k for k in row if 'action' in k.lower()]
    print('OK loaded', args.repo_id, 'len=', len(ds), 'backend=', args.video_backend)
    print('image_keys=', image_keys[:10])
    print('action_keys=', action_keys[:10])
    for k in image_keys[:2] + action_keys[:2]:
        v = row[k]
        print(k, type(v), getattr(v, 'shape', None))


if __name__ == '__main__':
    main()
