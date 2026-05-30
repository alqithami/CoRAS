from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch


@dataclass
class MethodResult:
    method: str
    alpha: float
    tau: float | int | Dict[str, float]
    set_mask: np.ndarray


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Split-conformal quantile using ceil((n+1)(1-alpha))."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1:
        scores = scores.reshape(-1)
    n = len(scores)
    if n == 0:
        raise ValueError("Calibration scores are empty")
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    if rank > n:
        return float("inf")
    return float(np.sort(scores)[rank - 1])


def inverse_probability_scores(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    labels = labels.astype(int)
    return 1.0 - probs[np.arange(len(labels)), labels]


def inverse_probability_sets(probs: np.ndarray, tau: float) -> np.ndarray:
    return (1.0 - probs) <= tau


def label_ranks(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    order = np.argsort(-probs, axis=1)
    inv = np.empty_like(order)
    rows = np.arange(order.shape[0])[:, None]
    inv[rows, order] = np.arange(order.shape[1])[None, :]
    return inv[np.arange(len(labels)), labels] + 1


def topk_sets(probs: np.ndarray, k: int) -> np.ndarray:
    k = int(max(1, min(k, probs.shape[1])))
    order = np.argsort(-probs, axis=1)[:, :k]
    mask = np.zeros_like(probs, dtype=bool)
    mask[np.arange(len(probs))[:, None], order] = True
    return mask


def aps_scores(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Deterministic APS score: cumulative probability mass down to true label."""
    labels = labels.astype(int)
    order = np.argsort(-probs, axis=1)
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cumsum = np.cumsum(sorted_probs, axis=1)
    inv = np.empty_like(order)
    inv[np.arange(order.shape[0])[:, None], order] = np.arange(order.shape[1])[None, :]
    ranks0 = inv[np.arange(len(labels)), labels]
    return cumsum[np.arange(len(labels)), ranks0]


def aps_sets(probs: np.ndarray, tau: float) -> np.ndarray:
    order = np.argsort(-probs, axis=1)
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cumsum = np.cumsum(sorted_probs, axis=1)
    sorted_mask = cumsum <= tau
    # Always include the top prediction to avoid empty sets.
    sorted_mask[:, 0] = True
    mask = np.zeros_like(probs, dtype=bool)
    mask[np.arange(len(probs))[:, None], order] = sorted_mask
    return mask


def mondrian_thresholds(scores: np.ndarray, groups: np.ndarray, alpha: float, min_group: int = 20) -> Dict[str, float]:
    groups = groups.astype(str)
    global_tau = conformal_quantile(scores, alpha)
    out: Dict[str, float] = {"__global__": global_tau}
    for g in sorted(set(groups)):
        mask = groups == g
        if int(mask.sum()) >= int(min_group):
            out[g] = conformal_quantile(scores[mask], alpha)
    return out


def apply_mondrian_thresholds(probs: np.ndarray, groups: np.ndarray, thresholds: Dict[str, float]) -> np.ndarray:
    groups = groups.astype(str)
    mask = np.zeros_like(probs, dtype=bool)
    for g in sorted(set(groups)):
        tau = thresholds.get(g, thresholds["__global__"])
        idx = np.where(groups == g)[0]
        mask[idx] = inverse_probability_sets(probs[idx], tau)
    return mask


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if m.sum() > 0:
            ece += (m.mean()) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def evaluate_sets(set_mask: np.ndarray, labels: np.ndarray, probs: np.ndarray, unsafe: Optional[np.ndarray] = None) -> Dict[str, float]:
    labels = labels.astype(int)
    n = len(labels)
    k_classes = int(probs.shape[1]) if probs.ndim == 2 else 0
    sizes = set_mask.sum(axis=1)
    covered = set_mask[np.arange(n), labels]
    pred = probs.argmax(axis=1)
    top1_correct = pred == labels
    singleton = sizes == 1
    empty = sizes == 0
    fail_to_abstain = singleton & (~top1_correct)
    metrics: Dict[str, float] = {
        "n": int(n),
        "num_classes": int(k_classes),
        "coverage": float(covered.mean()) if n else float("nan"),
        "noncoverage_rate": float(1.0 - covered.mean()) if n else float("nan"),
        "mean_set_size": float(sizes.mean()) if n else float("nan"),
        "mean_set_size_norm": float(sizes.mean() / max(k_classes, 1)) if n else float("nan"),
        "median_set_size": float(np.median(sizes)) if n else float("nan"),
        "p90_set_size": float(np.quantile(sizes, 0.90)) if n else float("nan"),
        "top1_accuracy": float(top1_correct.mean()) if n else float("nan"),
        "singleton_rate": float(singleton.mean()) if n else float("nan"),
        "abstain_rate": float((sizes > 1).mean()) if n else float("nan"),
        "empty_set_rate": float(empty.mean()) if n else float("nan"),
        "fail_to_abstain_rate": float(fail_to_abstain.mean()) if n else float("nan"),
        "ece_top1": expected_calibration_error(probs, labels) if n else float("nan"),
    }
    # Planner-facing execute/defer diagnostics: execute when the conformal set is small.
    for budget in (1, 2, 4, 8):
        execute = sizes <= budget
        metrics[f"execute_rate_set_le_{budget}"] = float(execute.mean()) if n else float("nan")
        metrics[f"wrong_execute_rate_set_le_{budget}"] = float((execute & (~covered)).mean()) if n else float("nan")
        metrics[f"coverage_when_set_le_{budget}"] = float(covered[execute].mean()) if execute.sum() else float("nan")
    if unsafe is not None and len(unsafe) == n:
        unsafe = unsafe.astype(bool)
        metrics["unsafe_n"] = int(unsafe.sum())
        metrics["unsafe_coverage"] = float(covered[unsafe].mean()) if unsafe.sum() > 0 else float("nan")
        metrics["unsafe_fail_to_abstain_rate"] = float(fail_to_abstain[unsafe].mean()) if unsafe.sum() > 0 else float("nan")
    return metrics


def bootstrap_mean_ci(values: np.ndarray, seed: int = 0, n_boot: int = 1000, q: Tuple[float, float] = (0.025, 0.975)) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        stats[i] = rng.choice(values, size=len(values), replace=True).mean()
    lo, hi = np.quantile(stats, q)
    return float(lo), float(hi)


@torch.no_grad()
def collect_logits(model: torch.nn.Module, loader, device: torch.device):
    model.eval()
    logits_list, labels_list, idx_list, unsafe_list = [], [], [], []
    for batch in loader:
        x = batch["image"].to(device)
        logits = model(x).detach().cpu()
        logits_list.append(logits)
        labels_list.append(batch["label"].cpu())
        idx_list.append(batch["index"].cpu())
        unsafe_list.append(batch["unsafe"].cpu())
    return (
        torch.cat(logits_list, dim=0).numpy(),
        torch.cat(labels_list, dim=0).numpy(),
        torch.cat(idx_list, dim=0).numpy(),
        torch.cat(unsafe_list, dim=0).numpy().astype(bool),
    )


def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float64)
