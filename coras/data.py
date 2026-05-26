from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass
class DatasetSplits:
    train: np.ndarray
    tune: np.ndarray
    calib: np.ndarray
    test: np.ndarray

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {"train": self.train, "tune": self.tune, "calib": self.calib, "test": self.test}


class RobotNPZDataset(Dataset):
    """Unified CoRAS dataset.

    Required arrays:
      images: uint8 or float array with shape (N,H,W,3) or (N,3,H,W)
      labels: int64 action-token labels with shape (N,)

    Optional arrays:
      domains: str/int domain ID with shape (N,)
      episodes: int64 episode ID with shape (N,)
      actions: float continuous action vector with shape (N,D)
      tasks: str/int task ID with shape (N,)
      unsafe: bool indicator for optional safety diagnostics
    """

    def __init__(self, path: str | Path, indices: Optional[Sequence[int]] = None, image_size: int = 96):
        self.path = Path(path)
        self.data = np.load(self.path, allow_pickle=True)
        if "images" not in self.data or "labels" not in self.data:
            raise ValueError(f"{self.path} must include arrays named 'images' and 'labels'.")
        self.images = self.data["images"]
        self.labels = self.data["labels"].astype(np.int64)
        n = len(self.labels)
        self.domains = self.data["domains"].astype(str) if "domains" in self.data else np.array(["default"] * n)
        self.episodes = self.data["episodes"].astype(np.int64) if "episodes" in self.data else np.arange(n, dtype=np.int64)
        self.tasks = self.data["tasks"].astype(str) if "tasks" in self.data else np.array(["task"] * n)
        self.actions = self.data["actions"].astype(np.float32) if "actions" in self.data else None
        self.unsafe = self.data["unsafe"].astype(bool) if "unsafe" in self.data else np.zeros(n, dtype=bool)
        self.image_size = int(image_size)
        self.indices = np.asarray(indices if indices is not None else np.arange(n), dtype=np.int64)
        if len(self.images) != n:
            raise ValueError("images and labels length mismatch")

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def num_classes(self) -> int:
        return int(np.max(self.labels)) + 1

    def __getitem__(self, idx: int):
        real_idx = int(self.indices[idx])
        img = self.images[real_idx]
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=-1)
        if img.ndim != 3:
            raise ValueError(f"Expected image rank 3, got shape {img.shape}")
        if img.shape[0] in {1, 3} and img.shape[-1] not in {1, 3}:  # CHW
            x = torch.from_numpy(img).float()
        else:  # HWC
            x = torch.from_numpy(img).permute(2, 0, 1).float()
        if x.max() > 2.0:
            x = x / 255.0
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        if x.shape[0] > 3:
            x = x[:3]
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x.unsqueeze(0), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False).squeeze(0)
        out = {
            "image": x.contiguous(),
            "label": torch.tensor(int(self.labels[real_idx]), dtype=torch.long),
            "index": torch.tensor(real_idx, dtype=torch.long),
            "domain": str(self.domains[real_idx]),
            "episode": torch.tensor(int(self.episodes[real_idx]), dtype=torch.long),
            "unsafe": torch.tensor(bool(self.unsafe[real_idx])),
        }
        if self.actions is not None:
            out["action"] = torch.from_numpy(self.actions[real_idx])
        return out


def _episode_membership(indices: np.ndarray, episodes: np.ndarray, eps: set[int]) -> np.ndarray:
    return indices[np.array([int(e) in eps for e in episodes[indices]], dtype=bool)]


def episode_split(
    episodes: np.ndarray,
    domains: np.ndarray,
    target_domains: Optional[Iterable[str]] = None,
    tune_frac: float = 0.15,
    calib_frac: float = 0.25,
    test_frac: float = 0.25,
    seed: int = 0,
    include_remaining_target_in_train: bool = False,
    min_episodes_per_split: int = 1,
) -> DatasetSplits:
    """Episode-block split used by all experiments.

    If target_domains is provided, source domains form source training data, and
    target episodes are split into tune/calib/test. If target_domains is absent,
    all episodes are split into train/tune/calib/test.
    """
    rng = np.random.default_rng(seed)
    episodes = np.asarray(episodes).astype(np.int64)
    domains = np.asarray(domains).astype(str)
    all_idx = np.arange(len(episodes))

    if target_domains is None or len(list(target_domains)) == 0:
        target_mask = np.ones(len(all_idx), dtype=bool)
        source_idx = np.array([], dtype=np.int64)
    else:
        target_set = {str(x) for x in target_domains}
        target_mask = np.array([d in target_set for d in domains], dtype=bool)
        source_idx = all_idx[~target_mask]
        if target_mask.sum() == 0:
            raise ValueError(f"No samples matched target_domains={sorted(target_set)}. Available domains include {sorted(set(domains))[:20]}")

    target_idx = all_idx[target_mask]
    target_eps = np.unique(episodes[target_idx])
    rng.shuffle(target_eps)
    n_eps = len(target_eps)
    if n_eps < 4:
        # Fall back to sample-level split for extremely small exported datasets.
        idx = target_idx.copy(); rng.shuffle(idx)
        n_tune = max(1, int(round(tune_frac * len(idx))))
        n_calib = max(1, int(round(calib_frac * len(idx))))
        n_test = max(1, int(round(test_frac * len(idx))))
        tune_idx = idx[:n_tune]
        calib_idx = idx[n_tune:n_tune+n_calib]
        test_idx = idx[n_tune+n_calib:n_tune+n_calib+n_test]
        target_train_idx = idx[n_tune+n_calib+n_test:]
    else:
        n_tune = max(min_episodes_per_split, int(round(tune_frac * n_eps)))
        n_calib = max(min_episodes_per_split, int(round(calib_frac * n_eps)))
        n_test = max(min_episodes_per_split, int(round(test_frac * n_eps)))
        # Ensure there is at least one remaining episode if no source domain exists.
        while n_tune + n_calib + n_test >= n_eps and n_test > min_episodes_per_split:
            n_test -= 1
        while n_tune + n_calib + n_test >= n_eps and n_calib > min_episodes_per_split:
            n_calib -= 1
        while n_tune + n_calib + n_test >= n_eps and n_tune > min_episodes_per_split:
            n_tune -= 1
        tune_eps = set(int(e) for e in target_eps[:n_tune])
        calib_eps = set(int(e) for e in target_eps[n_tune:n_tune + n_calib])
        test_eps = set(int(e) for e in target_eps[n_tune + n_calib:n_tune + n_calib + n_test])
        remaining_eps = set(int(e) for e in target_eps[n_tune + n_calib + n_test:])
        tune_idx = _episode_membership(target_idx, episodes, tune_eps)
        calib_idx = _episode_membership(target_idx, episodes, calib_eps)
        test_idx = _episode_membership(target_idx, episodes, test_eps)
        target_train_idx = _episode_membership(target_idx, episodes, remaining_eps)

    if len(source_idx) == 0 or include_remaining_target_in_train:
        train_idx = np.concatenate([source_idx, target_train_idx])
    else:
        train_idx = source_idx
    if len(train_idx) == 0:
        # Last resort: use non-tune/calib/test target samples as train.
        used = set(map(int, np.concatenate([tune_idx, calib_idx, test_idx])))
        train_idx = np.array([i for i in target_idx if int(i) not in used], dtype=np.int64)
    for name, arr in {"train": train_idx, "tune": tune_idx, "calib": calib_idx, "test": test_idx}.items():
        if len(arr) == 0:
            raise ValueError(f"Empty {name} split. Adjust fractions or target_domains.")
    for arr in [train_idx, tune_idx, calib_idx, test_idx]:
        rng.shuffle(arr)
    return DatasetSplits(train=train_idx, tune=tune_idx, calib=calib_idx, test=test_idx)


def make_loaders(path: str | Path, cfg: Dict, split: DatasetSplits) -> Dict[str, DataLoader]:
    image_size = int(cfg.get("image_size", 96))
    batch_size = int(cfg.get("batch_size", 64))
    num_workers = int(cfg.get("num_workers", 0))
    pin_memory = bool(cfg.get("pin_memory", False))
    loaders: Dict[str, DataLoader] = {}
    for name, idx in split.as_dict().items():
        ds = RobotNPZDataset(path, idx, image_size=image_size)
        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(name in {"train", "tune"}),
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    return loaders


def describe_npz(path: str | Path) -> Dict[str, object]:
    ds = RobotNPZDataset(path)
    domains, domain_counts = np.unique(ds.domains.astype(str), return_counts=True)
    tasks, task_counts = np.unique(ds.tasks.astype(str), return_counts=True)
    return {
        "path": str(path),
        "n": int(len(ds.labels)),
        "image_shape": list(ds.images.shape),
        "num_classes": int(ds.num_classes),
        "num_episodes": int(len(np.unique(ds.episodes))),
        "domains": dict(zip(domains.tolist(), domain_counts.astype(int).tolist())),
        "tasks_top20": dict(zip(tasks[:20].tolist(), task_counts[:20].astype(int).tolist())),
        "has_actions": bool(ds.actions is not None),
    }
