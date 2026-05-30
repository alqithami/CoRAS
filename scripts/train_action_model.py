#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from coras.data import RobotNPZDataset, episode_split, make_loaders, load_robot_npz
from coras.models import ActionTokenModel
from coras.utils import ensure_dir, set_seed, select_device, write_json


def train_epoch(model, loader, opt, device, desc: str = "train"):
    model.train()
    total_loss, total_correct, total_n = 0.0, 0, 0
    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=5.0)
        opt.step()
        total_loss += float(loss.item()) * len(y)
        total_correct += int((logits.argmax(dim=-1) == y).sum().item())
        total_n += len(y)
        pbar.set_postfix(loss=total_loss / max(total_n, 1), acc=total_correct / max(total_n, 1))
    return {"loss": total_loss / max(total_n, 1), "acc": total_correct / max(total_n, 1), "n": total_n}


@torch.no_grad()
def eval_epoch(model, loader, device, desc: str = "eval"):
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for batch in tqdm(loader, desc=desc, leave=False):
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += float(loss.item()) * len(y)
        total_correct += int((logits.argmax(dim=-1) == y).sum().item())
        total_n += len(y)
    return {"loss": total_loss / max(total_n, 1), "acc": total_correct / max(total_n, 1), "n": total_n}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train source model and target prompt/adapter for CoRAS.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    set_seed(int(cfg.get("seed", 0)))
    torch.set_num_threads(int(cfg.get("torch_threads", 4)))
    device = select_device(args.device)
    out_dir = ensure_dir(cfg["output_dir"])

    base_arrays = load_robot_npz(cfg["data_path"])
    base_ds = RobotNPZDataset(cfg["data_path"], image_size=int(cfg.get("image_size", 96)), arrays=base_arrays)
    split = episode_split(
        base_arrays.episodes,
        base_arrays.domains,
        target_domains=cfg.get("target_domains"),
        tune_frac=float(cfg.get("tune_frac", 0.15)),
        calib_frac=float(cfg.get("calib_frac", 0.25)),
        test_frac=float(cfg.get("test_frac", 0.25)),
        seed=int(cfg.get("seed", 0)),
        include_remaining_target_in_train=bool(cfg.get("include_remaining_target_in_train", False)),
        exclude_source_with_target_eval_episodes=bool(cfg.get("exclude_source_with_target_eval_episodes", False)),
    )
    np.savez(out_dir / "split_indices.npz", **split.as_dict())
    write_json(out_dir / "split_summary.json", {
        "train": int(len(split.train)), "tune": int(len(split.tune)), "calib": int(len(split.calib)), "test": int(len(split.test)),
        "target_domains": cfg.get("target_domains"),
    })
    loaders = make_loaders(cfg["data_path"], cfg, split, arrays=base_arrays)

    model = ActionTokenModel(
        num_classes=base_ds.num_classes,
        encoder=cfg.get("encoder", "small_cnn"),
        pretrained=bool(cfg.get("pretrained", False)),
        adapter_rank=int(cfg.get("adapter_rank", 16)),
    ).to(device)

    if bool(cfg.get("freeze_encoder_from_start", False)):
        model.freeze_encoder()

    history = []
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=float(cfg.get("lr", 7e-4)), weight_decay=float(cfg.get("weight_decay", 1e-4)))
    epochs = int(cfg.get("epochs", cfg.get("source_epochs", 3)))
    for epoch in range(epochs):
        tr = train_epoch(model, loaders["train"], opt, device, desc=f"source {epoch+1}/{epochs}")
        va = eval_epoch(model, loaders["tune"], device, desc="tune-eval")
        row = {"stage": "source_train", "epoch": epoch, **{f"train_{k}": v for k, v in tr.items()}, **{f"tune_{k}": v for k, v in va.items()}}
        history.append(row)
        print(row, flush=True)

    ckpt_common = {"num_classes": base_ds.num_classes, "cfg": cfg}
    torch.save({"model": model.state_dict(), **ckpt_common}, out_dir / "model_base.pt")

    # Target prompt/adaptor tuning. Keep a true no-prompt baseline checkpoint above.
    if bool(cfg.get("target_prompt_tuning", True)):
        model.train_only_adapter()
        opt = torch.optim.AdamW(model.adapter.parameters(), lr=float(cfg.get("adapter_lr", 1e-3)), weight_decay=float(cfg.get("adapter_weight_decay", 1e-4)))
        adapter_epochs = int(cfg.get("adapter_epochs", 2))
        for epoch in range(adapter_epochs):
            tr = train_epoch(model, loaders["tune"], opt, device, desc=f"adapter {epoch+1}/{adapter_epochs}")
            ca = eval_epoch(model, loaders["calib"], device, desc="calib-eval")
            row = {"stage": "target_adapter", "epoch": epoch, **{f"tune_{k}": v for k, v in tr.items()}, **{f"calib_{k}": v for k, v in ca.items()}}
            history.append(row)
            print(row, flush=True)

    torch.save({"model": model.state_dict(), **ckpt_common}, out_dir / "model_prompt.pt")
    # Backward-compatible symlink/copy name.
    torch.save({"model": model.state_dict(), **ckpt_common}, out_dir / "model.pt")
    write_json(out_dir / "train_history.json", {"history": history, "device": str(device)})
    print(f"Saved checkpoints to {out_dir}")


if __name__ == "__main__":
    main()
