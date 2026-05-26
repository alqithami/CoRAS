#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

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
from coras.data import DatasetSplits, RobotNPZDataset, make_loaders
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
        k = int(np.ceil(conformal_quantile(ranks, alpha)))
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
    base_ds = RobotNPZDataset(cfg["data_path"], image_size=int(cfg.get("image_size", 96)))
    loaders = make_loaders(cfg["data_path"], cfg, split)

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
    for method in methods:
        c_probs, t_probs = method_probs[method]
        tau, set_mask = evaluate_method(method, alpha, c_probs, calib_labels, calib_domains, t_probs, test_labels, test_domains)
        metrics = evaluate_sets(set_mask, test_labels, t_probs, unsafe=test_unsafe)
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
        summary_rows.append(metrics)

        for d in sorted(set(test_domains)):
            mask = test_domains == d
            dm = evaluate_sets(set_mask[mask], test_labels[mask], t_probs[mask], unsafe=test_unsafe[mask])
            dm.update({"method": method, "alpha": alpha, "domain": d, "seed": int(cfg.get("seed", -1)), "target_coverage": 1.0 - alpha})
            dm["coverage_gap"] = float(dm["coverage"] - (1.0 - alpha)) if np.isfinite(dm["coverage"]) else np.nan
            domain_rows.append(dm)

        pred = pd.DataFrame({
            "index": test_idx,
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
        # Store class membership columns for CoRAS only; this is useful for debugging but can be large.
        if method == "coras":
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
    write_json(out_dir / f"eval_metadata_alpha{alpha:.2f}.json", to_builtin({
        "alpha": alpha,
        "temperature": temp,
        "timings": timings,
        "methods": methods,
        "n_tune": int(len(tune_labels)),
        "n_calib": int(len(calib_labels)),
        "n_test": int(len(test_labels)),
    }))
    print(summary[["method", "coverage", "mean_set_size", "top1_accuracy", "fail_to_abstain_rate", "abs_coverage_gap"]].to_string(index=False))
    print(f"Saved evaluation files in {out_dir}")


if __name__ == "__main__":
    main()
