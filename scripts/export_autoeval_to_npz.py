#!/usr/bin/env python
from __future__ import annotations

import argparse
import io
import json
import pickle
import re
import sys
import types
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


# -----------------------------------------------------------------------------
# Pickle compatibility for robot_eval_logger and wandb objects found in the
# public AutoEval trajectory pickles. These stubs preserve attributes without
# requiring the full robotics logging stack or wandb package during export.
# -----------------------------------------------------------------------------
@dataclass
class StepData:
    obs: Any = None
    action: Any = None
    joint_position: Any = None
    joint_velocity: Any = None
    end_effector_pose: Any = None
    gripper: Any = None
    joint_effort: Any = None


class TrajData:
    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["_pickle_state"] = state

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    @staticmethod
    def decode_images(obs: Dict[str, list]) -> Dict[str, List[np.ndarray]]:
        out: Dict[str, List[np.ndarray]] = {}
        for cam, xs in (obs or {}).items():
            seq = xs if isinstance(xs, list) else [xs]
            imgs = []
            for x in seq:
                if isinstance(x, bytes):
                    imgs.append(np.asarray(Image.open(io.BytesIO(x))))
                else:
                    imgs.append(np.asarray(x))
            out[cam] = imgs
        return out


class Histogram:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.__dict__.update(kwargs)

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["_pickle_state"] = state


def _install_pickle_stubs() -> None:
    root = types.ModuleType("robot_eval_logger")
    typing_mod = types.ModuleType("robot_eval_logger.typing")
    traj_mod = types.ModuleType("robot_eval_logger.typing.traj_data")
    traj_mod.TrajData = TrajData
    traj_mod.StepData = StepData
    sys.modules.setdefault("robot_eval_logger", root)
    sys.modules.setdefault("robot_eval_logger.typing", typing_mod)
    sys.modules.setdefault("robot_eval_logger.typing.traj_data", traj_mod)

    wandb_mod = types.ModuleType("wandb")
    sdk_mod = types.ModuleType("wandb.sdk")
    dt_mod = types.ModuleType("wandb.sdk.data_types")
    hist_mod = types.ModuleType("wandb.sdk.data_types.histogram")
    hist_mod.Histogram = Histogram
    sys.modules.setdefault("wandb", wandb_mod)
    sys.modules.setdefault("wandb.sdk", sdk_mod)
    sys.modules.setdefault("wandb.sdk.data_types", dt_mod)
    sys.modules.setdefault("wandb.sdk.data_types.histogram", hist_mod)


_install_pickle_stubs()


def parse_time(s: str | None) -> datetime:
    if not s:
        return datetime.min
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return datetime.min


def as_uint8_image(x: Any, resize: int | None = 96) -> np.ndarray:
    if isinstance(x, bytes):
        arr = np.asarray(Image.open(io.BytesIO(x)))
    else:
        arr = np.asarray(x)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in {1, 3, 4} and arr.shape[-1] not in {1, 3, 4}:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"Cannot convert image with shape {arr.shape}")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        finite = arr[np.isfinite(arr)]
        scale = 255.0 if finite.size and float(np.nanmax(finite)) <= 1.5 else 1.0
        arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
    if resize and (arr.shape[0] != resize or arr.shape[1] != resize):
        arr = np.asarray(Image.fromarray(arr).resize((resize, resize), Image.BILINEAR), dtype=np.uint8)
    return arr


def to_action_array(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    try:
        arr = np.asarray(x, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    return arr


def sequence_from_value(v: Any) -> List[Any]:
    if v is None:
        return []
    if isinstance(v, np.ndarray):
        if v.ndim >= 2:
            return [v[i] for i in range(v.shape[0])]
        return [v]
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


def first_attr(obj: Any, names: Sequence[str]) -> Any:
    if isinstance(obj, dict):
        for name in names:
            if name in obj:
                return obj[name]
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def action_sequence(traj: Any, allow_proxy: bool = True) -> Tuple[List[Any], str]:
    # Primary: logged policy/robot action.
    for name in ["action", "actions", "policy_action", "policy_actions", "commanded_action", "commands"]:
        v = first_attr(traj, [name])
        seq = sequence_from_value(v)
        if seq:
            return seq, name

    if not allow_proxy:
        return [], "none"

    # Fallback: executed-motion proxy. This is not a policy-command label; it is
    # useful for risk/uncertainty analysis when the public log omits action.
    for name in ["end_effector_pose", "eef_pose", "robot_eef_pose", "joint_position", "joint_positions"]:
        seq = sequence_from_value(first_attr(traj, [name]))
        arrs = []
        for x in seq:
            a = to_action_array(x)
            if a is not None:
                arrs.append(a)
        if len(arrs) >= 2:
            deltas = [arrs[i + 1] - arrs[i] for i in range(len(arrs) - 1)]
            return deltas, f"proxy_delta_{name}"
    return [], "none"


def choose_obs_camera(obs: Any, requested: str | None = None) -> Tuple[str, List[Any]]:
    if obs is None:
        raise ValueError("trajectory has no obs")
    if not isinstance(obs, dict):
        raise ValueError(f"obs is not a dict: {type(obs)}")
    keys = list(obs.keys())
    if not keys:
        raise ValueError("obs dict is empty")
    if requested:
        if requested not in obs:
            raise KeyError(f"Requested camera {requested!r} not found; available={keys}")
        key = requested
    else:
        priority = ["image_primary", "primary", "external", "exterior", "base", "front", "image", "camera", "cam"]
        key = keys[0]
        for pat in priority:
            matches = [k for k in keys if pat.lower() in str(k).lower()]
            if matches:
                key = matches[0]
                break
    vals = obs[key]
    if isinstance(vals, np.ndarray) and vals.ndim == 4:
        seq = [vals[i] for i in range(vals.shape[0])]
    elif isinstance(vals, list):
        seq = vals
    else:
        seq = [vals]
    return str(key), seq


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        header = f.read(256)
        f.seek(0)
        if header.startswith(b"version https://git-lfs") or header.startswith(b"version https://huggingface"):
            raise RuntimeError(
                f"{path.name} is an Xet/Git-LFS pointer, not the real pickle. "
                "Install/enable hf-xet in this environment and rerun the export."
            )
        if header[:4] == b"\x04\x22\x4d\x18":
            try:
                import lz4.frame  # type: ignore
            except Exception as exc:
                raise RuntimeError(f"{path} appears lz4-compressed; pip install lz4") from exc
            return pickle.loads(lz4.frame.decompress(f.read()))
        return pickle.load(f)


def list_repo_files_cached(repo_id: str) -> List[str]:
    from huggingface_hub import list_repo_files
    return list_repo_files(repo_id=repo_id, repo_type="dataset")


def list_eval_ids(repo_files: Sequence[str]) -> List[str]:
    ids = set()
    for f in repo_files:
        m = re.match(r"eval_data/([^/]+)/metadata\.json$", f)
        if m:
            ids.add(m.group(1))
    return sorted(ids)


def available_traj_files(repo_files: Sequence[str], eval_id: str) -> List[str]:
    prefix = f"eval_data/{eval_id}/traj_"
    files = [f for f in repo_files if f.startswith(prefix) and f.endswith(".pkl")]

    def key(f: str) -> int:
        m = re.search(r"traj_(\d+)\.pkl$", f)
        return int(m.group(1)) if m else 10**9

    return sorted(files, key=key)


def download_file(repo_id: str, filename: str, cache_dir: str | None = None) -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename, cache_dir=cache_dir))


def load_metadata(repo_id: str, eval_id: str, cache_dir: str | None = None) -> dict:
    path = download_file(repo_id, f"eval_data/{eval_id}/metadata.json", cache_dir)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(x: Any, default: float = np.nan) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        try:
            return float(bool(x))
        except Exception:
            return default


def main() -> None:
    ap = argparse.ArgumentParser(description="Export public AutoEval online real-robot logs to CoRAS NPZ.")
    ap.add_argument("--repo-id", default="zhouzypaul/auto_eval")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resize", type=int, default=96)
    ap.add_argument("--camera", type=str, default=None)
    ap.add_argument("--cache-dir", type=str, default=None)
    ap.add_argument("--max-evals", type=int, default=24, help="Target number of usable evaluation jobs to export.")
    ap.add_argument("--scan-evals", type=int, default=300, help="Candidate metadata dirs to scan while searching for usable logs.")
    ap.add_argument("--max-trajs-per-eval", type=int, default=2, help="Trajectory PKLs per eval job.")
    ap.add_argument("--max-traj-downloads", type=int, default=500, help="Safety cap on downloaded trajectory PKLs.")
    ap.add_argument("--frame-stride", type=int, default=4)
    ap.add_argument("--max-frames-per-traj", type=int, default=120)
    ap.add_argument("--source-frac", type=float, default=0.60)
    ap.add_argument("--min-action-dim", type=int, default=1)
    ap.add_argument("--min-exported-frames", type=int, default=200, help="Required minimum usable frames before export is accepted.")
    ap.add_argument("--allow-under-min", action=argparse.BooleanOptionalAction, default=False,
                    help="Save and continue even if fewer than --min-exported-frames are found. Use only for exploratory/preflight runs.")
    ap.add_argument("--max-consecutive-download-errors", type=int, default=40,
                    help="Stop scanning after this many consecutive trajectory download/pickle errors. This prevents extremely long noisy runs during network outages.")
    ap.add_argument("--prefer-recent", action="store_true", help="Search most recent eval jobs first.")
    ap.add_argument("--allow-action-proxy", action=argparse.BooleanOptionalAction, default=True,
                    help="If logged action is missing, use end-effector/joint deltas as an executed-motion proxy.")
    args = ap.parse_args()

    repo_files = list_repo_files_cached(args.repo_id)
    eval_ids = list_eval_ids(repo_files)
    if not eval_ids:
        raise RuntimeError(f"No eval_data/*/metadata.json entries found in {args.repo_id}")

    metas = []
    print(f"Found {len(eval_ids)} AutoEval metadata entries. Loading metadata...")
    for eid in eval_ids:
        try:
            md = load_metadata(args.repo_id, eid, args.cache_dir)
            md["eval_id"] = str(md.get("eval_id", eid))
            md["_eval_dir"] = eid
            md["_time_obj"] = parse_time(md.get("time") or md.get("collection_time") or md.get("timestamp"))
            metas.append(md)
        except Exception as exc:
            print(f"[skip metadata] {eid}: {exc}")
    metas.sort(key=lambda m: m.get("_time_obj", datetime.min), reverse=bool(args.prefer_recent))
    metas = metas[: max(1, args.scan_evals)]

    records: List[dict] = []
    usable_eval_dirs: List[str] = []
    action_dim: Optional[int] = None
    episode_counter = 0
    downloaded = 0
    skipped_no_action = 0
    skipped_no_obs = 0
    skipped_bad_pickle = 0
    proxy_frames = 0
    consecutive_download_errors = 0

    for md in metas:
        if len(usable_eval_dirs) >= args.max_evals and len(records) >= args.min_exported_frames:
            break
        eid = md["_eval_dir"]
        trajs = available_traj_files(repo_files, eid)[: max(1, args.max_trajs_per_eval)]
        if not trajs:
            continue
        eval_kept = 0
        for tfile in trajs:
            if downloaded >= args.max_traj_downloads:
                print(f"[stop] reached --max-traj-downloads={args.max_traj_downloads}")
                break
            downloaded += 1
            try:
                tpath = download_file(args.repo_id, tfile, args.cache_dir)
                traj = load_pickle(tpath)
            except Exception as exc:
                skipped_bad_pickle += 1
                consecutive_download_errors += 1
                print(f"[skip traj] {tfile}: pickle/download problem: {exc}")
                if consecutive_download_errors >= args.max_consecutive_download_errors:
                    print(f"[stop] reached --max-consecutive-download-errors={args.max_consecutive_download_errors}; stopping AutoEval scan early.")
                    break
                continue

            consecutive_download_errors = 0
            obs = first_attr(traj, ["obs", "observation", "observations", "images"])
            try:
                cam_key, img_seq = choose_obs_camera(obs, args.camera)
            except Exception as exc:
                skipped_no_obs += 1
                print(f"[skip traj] {tfile}: no usable obs/image ({exc})")
                continue

            act_list, action_source = action_sequence(traj, allow_proxy=args.allow_action_proxy)
            if not act_list:
                skipped_no_action += 1
                print(f"[skip traj] {tfile}: no action or proxy action")
                continue

            n = min(len(img_seq), len(act_list))
            if n <= 0:
                continue

            command = str(first_attr(traj, ["language_command", "command", "instruction", "task"]) or
                          md.get("eval_name") or md.get("task") or "unknown_command")
            succ_raw = first_attr(traj, ["success", "episode_success", "succeeded"])
            succ = safe_float(succ_raw, default=np.nan)
            ps = first_attr(traj, ["partial_success", "score"])
            ps_val = safe_float(ps, default=succ)
            collection_time = str(first_attr(traj, ["collection_time", "time"]) or md.get("time") or "")
            kept = 0
            for k in range(0, n, max(1, args.frame_stride)):
                if kept >= args.max_frames_per_traj:
                    break
                a = to_action_array(act_list[k])
                if a is None or a.size < args.min_action_dim:
                    continue
                if action_dim is None:
                    action_dim = int(a.size)
                if int(a.size) != action_dim:
                    continue
                try:
                    im = as_uint8_image(img_seq[k], resize=args.resize)
                except Exception:
                    continue
                records.append({
                    "image": im,
                    "action": a.astype(np.float32),
                    "episode": episode_counter,
                    "eval_dir": eid,
                    "task": re.sub(r"\s+", "_", command.lower())[:80] or "autoeval_task",
                    "success": succ,
                    "partial_success": ps_val,
                    "eval_id": str(md.get("eval_id", eid)),
                    "command": command,
                    "frame_idx": k,
                    "action_source": action_source,
                    "collection_time": collection_time,
                })
                kept += 1
            if kept:
                eval_kept += kept
                if action_source.startswith("proxy_delta"):
                    proxy_frames += kept
                print(f"[ok] {tfile}: kept={kept}, camera={cam_key}, action_source={action_source}, success={succ}")
                episode_counter += 1
        if eval_kept:
            usable_eval_dirs.append(eid)
        if downloaded >= args.max_traj_downloads:
            break
        if consecutive_download_errors >= args.max_consecutive_download_errors:
            break

    if len(records) == 0:
        raise RuntimeError(
            "No usable AutoEval frames/actions were exported. "
            f"Scanned {len(metas)} eval dirs and downloaded {downloaded} trajectories. "
            f"Skipped: no_action={skipped_no_action}, no_obs={skipped_no_obs}, bad_pickle={skipped_bad_pickle}. "
            "Try: CORAS_AUTOEVAL_PREFER_RECENT=1, increasing CORAS_AUTOEVAL_SCAN_EVALS, "
            "and ensuring hf-xet is installed/enabled."
        )

    if len(records) < args.min_exported_frames and not args.allow_under_min:
        raise RuntimeError(
            f"AutoEval export produced {len(records)} frames, below required --min-exported-frames={args.min_exported_frames}. "
            f"Usable eval dirs={len(set(r['eval_dir'] for r in records))}; downloaded={downloaded}; "
            f"skipped_bad_pickle_or_download={skipped_bad_pickle}, skipped_no_action={skipped_no_action}, skipped_no_obs={skipped_no_obs}. "
            "This usually indicates a network/Hugging Face/Xet download issue or that the requested subset is too large for the current connection. "
            "For exploratory runs only, rerun with CORAS_AUTOEVAL_ALLOW_UNDER_MIN=1 and report the smaller sample size explicitly."
        )
    if len(records) < args.min_exported_frames:
        print(f"[warn] exporting only {len(records)} frames below requested min {args.min_exported_frames} because --allow-under-min was set.")

    # Chronological split over the successfully exported eval dirs.
    # If prefer_recent was used for search, sort exported evals by metadata time before splitting.
    time_by_dir = {m["_eval_dir"]: m.get("_time_obj", datetime.min) for m in metas}
    exported_dirs = sorted(set(r["eval_dir"] for r in records), key=lambda d: time_by_dir.get(d, datetime.min))
    split_idx = max(1, int(round(args.source_frac * len(exported_dirs))))
    source_dirs = set(exported_dirs[:split_idx])

    domains = ["autoeval_online_source" if r["eval_dir"] in source_dirs else "autoeval_online_target_late" for r in records]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    labels = np.zeros(len(records), dtype=np.int64)
    np.savez_compressed(
        args.out,
        images=np.stack([r["image"] for r in records]),
        actions=np.stack([r["action"] for r in records]).astype(np.float32),
        labels=labels,
        domains=np.array(domains),
        episodes=np.array([r["episode"] for r in records], dtype=np.int64),
        tasks=np.array([r["task"] for r in records]),
        success=np.array([r["success"] for r in records], dtype=np.float32),
        partial_success=np.array([r["partial_success"] for r in records], dtype=np.float32),
        eval_ids=np.array([r["eval_id"] for r in records]),
        eval_dirs=np.array([r["eval_dir"] for r in records]),
        language_commands=np.array([r["command"] for r in records]),
        frame_indices=np.array([r["frame_idx"] for r in records], dtype=np.int64),
        action_sources=np.array([r["action_source"] for r in records]),
        collection_times=np.array([r["collection_time"] for r in records]),
    )
    unique, counts = np.unique(np.array(domains), return_counts=True)
    srcs, src_counts = np.unique(np.array([r["action_source"] for r in records]), return_counts=True)
    print(f"Wrote {args.out} with {len(records)} frames, action_dim={action_dim}, episodes={episode_counter}, usable_eval_dirs={len(exported_dirs)}")
    print("domains:", dict(zip(unique.tolist(), counts.astype(int).tolist())))
    print("action_sources:", dict(zip(srcs.tolist(), src_counts.astype(int).tolist())))
    if proxy_frames:
        print(f"[note] {proxy_frames} frames use executed-motion proxy actions. Report this as an AutoEval-log proxy label, not as raw policy commands.")


if __name__ == "__main__":
    main()
