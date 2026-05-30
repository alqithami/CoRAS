#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from coras.conformal import (
    aps_scores, aps_sets, collect_logits, conformal_quantile, evaluate_sets,
    inverse_probability_scores, inverse_probability_sets, label_ranks, mondrian_thresholds,
    apply_mondrian_thresholds, softmax_np, topk_sets,
)
from coras.data import DatasetSplits, RobotNPZDataset, make_loaders, load_robot_npz
from coras.models import ActionTokenModel
from coras.utils import ensure_dir, select_device, write_json, to_builtin


def load_model(ckpt_path: Path, cfg: dict, device: torch.device) -> ActionTokenModel:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = ActionTokenModel(
        num_classes=int(ckpt["num_classes"]),
        encoder=cfg.get("encoder", "small_cnn"),
        pretrained=False,
        adapter_rank=int(cfg.get("adapter_rank", 16)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    return model


def fit_temperature(logits: np.ndarray, labels: np.ndarray, device: torch.device, max_iter: int = 80) -> float:
    """Fit scalar temperature on tune split by NLL minimization."""
    logits_t = torch.tensor(logits, dtype=torch.float32, device=device)
    labels_t = torch.tensor(labels.astype(int), dtype=torch.long, device=device)
    log_temp = torch.nn.Parameter(torch.zeros(() if logits_t.ndim else (1,), device=device))
    opt = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=True)
        temp = torch.exp(log_temp).clamp(0.05, 20.0)
        loss = F.cross_entropy(logits_t / temp, labels_t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_temp).detach().cpu().item())


def evaluate_method(method: str, alpha: float, calib_probs: np.ndarray, calib_labels: np.ndarray, calib_domains: np.ndarray,
                    test_probs: np.ndarray, test_labels: np.ndarray, test_domains: np.ndarray) -> tuple[object, np.ndarray]:
    if method in {"top1_singleton", "prompt_only_top1"}:
        return 1, topk_sets(test_probs, 1)
    if method == "topk_calibrated":
        ranks = label_ranks(calib_probs, calib_labels).astype(float)
        q = conformal_quantile(ranks, alpha)
        # With very small calibration sets, split conformal's finite-sample
        # quantile can be +inf because ceil((n+1)(1-alpha)) > n. For top-k,
        # +inf means the valid conservative set is the full action vocabulary,
        # not an integer conversion error.
        if not np.isfinite(q):
            k = int(calib_probs.shape[1])
        else:
            k = int(np.ceil(q))
        k = max(1, min(k, calib_probs.shape[1]))
        return k, topk_sets(test_probs, k)
    if method in {"vanilla_conformal", "coras", "temperature_conformal"}:
        scores = inverse_probability_scores(calib_probs, calib_labels)
        tau = conformal_quantile(scores, alpha)
        return tau, inverse_probability_sets(test_probs, tau)
    if method in {"aps_conformal", "coras_aps"}:
        scores = aps_scores(calib_probs, calib_labels)
        tau = conformal_quantile(scores, alpha)
        return tau, aps_sets(test_probs, tau)
    if method == "mondrian_domain":
        scores = inverse_probability_scores(calib_probs, calib_labels)
        thresholds = mondrian_thresholds(scores, calib_domains, alpha, min_group=10)
        return thresholds, apply_mondrian_thresholds(test_probs, test_domains, thresholds)
    raise ValueError(f"Unknown method: {method}")


def _safe_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if len(x) else float("nan")


def _safe_percentile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if len(x) else float("nan")


def action_metrics(
    set_mask: np.ndarray,
    probs: np.ndarray,
    labels: np.ndarray,
    test_idx: np.ndarray,
    calib_idx: np.ndarray,
    all_actions: np.ndarray | None,
    codebook_centers: np.ndarray | None,
    episodes: np.ndarray,
) -> Dict[str, float]:
    """Action-space diagnostics for discrete action-token prediction sets.

    Token coverage is necessary but not sufficient for robotics. These metrics
    report how close the *best action in the predicted set* is to the continuous
    demonstrator action, using the action-codebook centers as executable action
    representatives. They are offline metrics, not closed-loop success claims.
    """
    out: Dict[str, float] = {}
    labels = labels.astype(int)
    covered = set_mask[np.arange(len(labels)), labels].astype(bool)
    test_episodes = episodes[test_idx]

    # Trajectory/block-level diagnostics are useful even without continuous actions.
    ep_all, ep_mean, ep_first_miss_frac = [], [], []
    for ep in np.unique(test_episodes):
        loc = np.where(test_episodes == ep)[0]
        # Sort by original dataset index, which tracks temporal order for LeRobot exports.
        loc = loc[np.argsort(test_idx[loc])]
        c = covered[loc]
        ep_all.append(float(np.all(c)))
        ep_mean.append(float(np.mean(c)))
        miss = np.where(~c)[0]
        first_miss = len(c) if len(miss) == 0 else int(miss[0])
        ep_first_miss_frac.append(float(first_miss / max(len(c), 1)))
    out["episode_all_steps_covered"] = _safe_mean(np.array(ep_all))
    out["episode_mean_step_coverage"] = _safe_mean(np.array(ep_mean))
    out["episode_frac_before_first_miss"] = _safe_mean(np.array(ep_first_miss_frac))

    if all_actions is None or codebook_centers is None:
        return out
    actions = np.asarray(all_actions, dtype=np.float32)
    centers = np.asarray(codebook_centers, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[:, None]
    if centers.ndim == 1:
        centers = centers[:, None]
    if centers.shape[0] <= int(labels.max()):
        return out

    expert = actions[test_idx]
    calib_actions = actions[calib_idx]
    # Distances from expert continuous action to every action-token center.
    dists = np.linalg.norm(expert[:, None, :] - centers[None, :, :], axis=-1)
    top1 = probs.argmax(axis=1).astype(int)
    top1_l2 = dists[np.arange(len(top1)), top1]
    true_token_l2 = dists[np.arange(len(labels)), labels]
    masked = np.where(set_mask, dists, np.inf)
    oracle_l2 = masked.min(axis=1)

    # Calibrated near-action threshold from quantization residuals on calibration split.
    calib_labels = labels  # temporary initialization; overwritten below.
    # labels passed here are test labels, so recover calibration token residuals from arrays outside only if lengths match.
    # The threshold below is therefore based on all centers' nearest residual on calib continuous actions, independent of labels.
    calib_nearest = np.min(np.linalg.norm(calib_actions[:, None, :] - centers[None, :, :], axis=-1), axis=1)
    near_tau = float(np.quantile(calib_nearest, 0.90)) if len(calib_nearest) else float("nan")

    # Action-set diameter in continuous action space.
    center_pairwise = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    diam = np.zeros(len(labels), dtype=np.float32)
    for i, row in enumerate(set_mask):
        ids = np.flatnonzero(row)
        if len(ids) > 1:
            diam[i] = float(center_pairwise[np.ix_(ids, ids)].max())
        else:
            diam[i] = 0.0

    out.update({
        "top1_action_l2_mean": _safe_mean(top1_l2),
        "top1_action_l2_p90": _safe_percentile(top1_l2, 90),
        "true_token_action_l2_mean": _safe_mean(true_token_l2),
        "set_oracle_action_l2_mean": _safe_mean(oracle_l2),
        "set_oracle_action_l2_p90": _safe_percentile(oracle_l2, 90),
        "set_action_diameter_l2_mean": _safe_mean(diam),
        "set_action_diameter_l2_p90": _safe_percentile(diam, 90),
        "action_l2_improvement_vs_top1": float(_safe_mean(top1_l2) - _safe_mean(oracle_l2)),
        "action_l2_ratio_vs_top1": float(_safe_mean(oracle_l2) / max(_safe_mean(top1_l2), 1e-12)) if np.isfinite(_safe_mean(top1_l2)) else float("nan"),
        "near_action_coverage_q90_calib_residual": _safe_mean((oracle_l2 <= near_tau).astype(float)) if np.isfinite(near_tau) else float("nan"),
        "near_action_tau_q90_calib_residual": near_tau,
    })
    return out


def load_codebook_and_actions(npz_path: str | Path) -> Tuple[np.ndarray | None, np.ndarray | None]:
    with np.load(npz_path, allow_pickle=True) as data:
        actions = np.asarray(data["actions"], dtype=np.float32) if "actions" in data else None
        centers = np.asarray(data["codebook_centers"], dtype=np.float32) if "codebook_centers" in data else None
    return actions, centers


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all CoRAS baselines and conformal variants.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--methods", nargs="*", default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    torch.set_num_threads(int(cfg.get("torch_threads", 4)))
    device = select_device(args.device)
    alpha = float(args.alpha if args.alpha is not None else cfg.get("alpha", 0.10))
    out_dir = ensure_dir(cfg["output_dir"])

    split_npz = np.load(out_dir / "split_indices.npz")
    split = DatasetSplits(split_npz["train"], split_npz["tune"], split_npz["calib"], split_npz["test"])
    base_arrays = load_robot_npz(cfg["data_path"])
    base_ds = RobotNPZDataset(cfg["data_path"], image_size=int(cfg.get("image_size", 96)), arrays=base_arrays)
    all_actions, codebook_centers = load_codebook_and_actions(cfg["data_path"])
    loaders = make_loaders(cfg["data_path"], cfg, split, arrays=base_arrays)

    # Collect logits for base and prompt-adapted models.
    timings = {}
    t0 = time.perf_counter()
    base_model = load_model(out_dir / "model_base.pt", cfg, device)
    base_tune_logits, tune_labels, tune_idx, tune_unsafe = collect_logits(base_model, loaders["tune"], device)
    base_calib_logits, calib_labels, calib_idx, calib_unsafe = collect_logits(base_model, loaders["calib"], device)
    base_test_logits, test_labels, test_idx, test_unsafe = collect_logits(base_model, loaders["test"], device)
    timings["base_collect_seconds"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    prompt_model = load_model(out_dir / "model_prompt.pt", cfg, device)
    prompt_tune_logits, _, _, _ = collect_logits(prompt_model, loaders["tune"], device)
    prompt_calib_logits, _, _, _ = collect_logits(prompt_model, loaders["calib"], device)
    prompt_test_logits, _, _, _ = collect_logits(prompt_model, loaders["test"], device)
    timings["prompt_collect_seconds"] = time.perf_counter() - t1

    calib_domains = base_ds.domains.astype(str)[calib_idx]
    test_domains = base_ds.domains.astype(str)[test_idx]
    test_tasks = base_ds.tasks.astype(str)[test_idx]
    test_episodes = base_ds.episodes.astype(np.int64)[test_idx]

    base_tune_probs = softmax_np(base_tune_logits)
    base_calib_probs = softmax_np(base_calib_logits)
    base_test_probs = softmax_np(base_test_logits)
    prompt_tune_probs = softmax_np(prompt_tune_logits)
    prompt_calib_probs = softmax_np(prompt_calib_logits)
    prompt_test_probs = softmax_np(prompt_test_logits)

    temp = fit_temperature(base_tune_logits, tune_labels, device=device, max_iter=int(cfg.get("temperature_max_iter", 80)))
    temp_calib_probs = softmax_np(base_calib_logits, temperature=temp)
    temp_test_probs = softmax_np(base_test_logits, temperature=temp)

    method_probs = {
        "top1_singleton": (base_calib_probs, base_test_probs),
        "topk_calibrated": (base_calib_probs, base_test_probs),
        "vanilla_conformal": (base_calib_probs, base_test_probs),
        "aps_conformal": (base_calib_probs, base_test_probs),
        "temperature_conformal": (temp_calib_probs, temp_test_probs),
        "prompt_only_top1": (prompt_calib_probs, prompt_test_probs),
        "coras": (prompt_calib_probs, prompt_test_probs),
        "coras_aps": (prompt_calib_probs, prompt_test_probs),
        "mondrian_domain": (prompt_calib_probs, prompt_test_probs),
    }
    default_methods = list(method_probs.keys())
    methods = args.methods if args.methods else cfg.get("methods", default_methods)

    summary_rows = []
    domain_rows = []
    pred_frames = []
    set_masks_for_npz: List[np.ndarray] = []
    top1_for_npz: List[np.ndarray] = []
    prob_for_npz: List[np.ndarray] = []

    for method in methods:
        c_probs, t_probs = method_probs[method]
        tau, set_mask = evaluate_method(method, alpha, c_probs, calib_labels, calib_domains, t_probs, test_labels, test_domains)
        metrics = evaluate_sets(set_mask, test_labels, t_probs, unsafe=test_unsafe)
        metrics.update(action_metrics(set_mask, t_probs, test_labels, test_idx, calib_idx, all_actions, codebook_centers, base_ds.episodes))
        metrics["target_coverage"] = 1.0 - alpha
        metrics["coverage_gap"] = float(metrics["coverage"] - (1.0 - alpha))
        metrics["abs_coverage_gap"] = float(abs(metrics["coverage_gap"]))
        metrics["method"] = method
        metrics["alpha"] = alpha
        metrics["tau_or_k"] = str(tau)
        metrics["temperature"] = temp if method == "temperature_conformal" else np.nan
        metrics["device"] = str(device)
        metrics["seed"] = int(cfg.get("seed", -1))
        metrics["calib_frac"] = float(cfg.get("calib_frac", np.nan))
        metrics["n_calib"] = int(len(calib_labels))
        metrics["n_test"] = int(len(test_labels))
        summary_rows.append(metrics)

        for d in sorted(set(test_domains)):
            mask = test_domains == d
            dm = evaluate_sets(set_mask[mask], test_labels[mask], t_probs[mask], unsafe=test_unsafe[mask])
            # For domain action diagnostics, reuse global calibration threshold/actions.
            dm.update(action_metrics(set_mask[mask], t_probs[mask], test_labels[mask], test_idx[mask], calib_idx, all_actions, codebook_centers, base_ds.episodes))
            dm.update({"method": method, "alpha": alpha, "domain": d, "seed": int(cfg.get("seed", -1)), "target_coverage": 1.0 - alpha})
            dm["coverage_gap"] = float(dm["coverage"] - (1.0 - alpha)) if np.isfinite(dm["coverage"]) else np.nan
            domain_rows.append(dm)

        pred = pd.DataFrame({
            "index": test_idx,
            "episode": test_episodes,
            "domain": test_domains,
            "task": test_tasks,
            "label": test_labels,
            "top1": t_probs.argmax(axis=1),
            "top1_prob": t_probs.max(axis=1),
            "set_size": set_mask.sum(axis=1),
            "covered": set_mask[np.arange(len(test_labels)), test_labels],
            "unsafe": test_unsafe,
            "method": method,
            "alpha": alpha,
        })
        pred_frames.append(pred)
        set_masks_for_npz.append(set_mask.astype(np.bool_))
        top1_for_npz.append(t_probs.argmax(axis=1).astype(np.int16))
        prob_for_npz.append(t_probs.max(axis=1).astype(np.float32))

        # Store class-membership columns for every method by default. This enables
        # continuous-action diagnostics and set-valued action diagnostics. For very
        # large K/test splits, set save_full_sets: false in YAML.
        if bool(cfg.get("save_full_sets", True)):
            classes = [f"in_set_{j}" for j in range(set_mask.shape[1])]
            pd.concat([pred.reset_index(drop=True), pd.DataFrame(set_mask.astype(np.int8), columns=classes)], axis=1).to_csv(
                out_dir / f"prediction_sets_{method}_alpha{alpha:.2f}.csv", index=False
            )

    summary = pd.DataFrame(summary_rows)
    per_domain = pd.DataFrame(domain_rows)
    predictions = pd.concat(pred_frames, ignore_index=True)
    summary.to_csv(out_dir / f"metrics_summary_alpha{alpha:.2f}.csv", index=False)
    per_domain.to_csv(out_dir / f"metrics_by_domain_alpha{alpha:.2f}.csv", index=False)
    predictions.to_csv(out_dir / f"prediction_sets_compact_alpha{alpha:.2f}.csv", index=False)
    np.savez_compressed(
        out_dir / f"prediction_sets_all_alpha{alpha:.2f}.npz",
        methods=np.array(methods),
        indices=test_idx.astype(np.int64),
        episodes=test_episodes.astype(np.int64),
        labels=test_labels.astype(np.int64),
        domains=test_domains.astype(str),
        tasks=test_tasks.astype(str),
        set_masks=np.stack(set_masks_for_npz, axis=0),
        top1=np.stack(top1_for_npz, axis=0),
        top1_prob=np.stack(prob_for_npz, axis=0),
    )
    write_json(out_dir / f"eval_metadata_alpha{alpha:.2f}.json", to_builtin({
        "alpha": alpha,
        "temperature": temp,
        "timings": timings,
        "methods": methods,
        "n_tune": int(len(tune_labels)),
        "n_calib": int(len(calib_labels)),
        "n_test": int(len(test_labels)),
        "has_action_metrics": bool(all_actions is not None and codebook_centers is not None),
    }))
    display_cols = ["method", "coverage", "mean_set_size", "top1_accuracy", "set_oracle_action_l2_mean", "episode_mean_step_coverage", "fail_to_abstain_rate", "abs_coverage_gap"]
    display_cols = [c for c in display_cols if c in summary.columns]
    print(summary[display_cols].to_string(index=False))
    print(f"Saved evaluation files in {out_dir}")


if __name__ == "__main__":
    main()
