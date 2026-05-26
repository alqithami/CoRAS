#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def _domain_probs(domains: list[str]) -> np.ndarray:
    # Source-heavy by default, but enough target episodes for tune/calib/test.
    base = {
        "sim_source_clean": 0.36,
        "sim_source_texture": 0.24,
        "sim_source_clutter": 0.18,
        "target_camera_shift": 0.11,
        "target_lighting_shift": 0.07,
        "target_occluded": 0.04,
    }
    p = np.array([base[d] for d in domains], dtype=np.float64)
    return p / p.sum()


def draw_frame(label: int, domain: str, size: int, grid: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, bool]:
    """Draw a top-down tabletop scene and return image, continuous action, unsafe flag.

    The label is the discretized end-effector waypoint / affordance cell. The object
    is drawn at the cell center; the action is a continuous normalized XY target plus
    a gripper-open scalar. Domain shifts alter rendering but not the semantic action.
    """
    cell = size / grid
    gx = label % grid
    gy = label // grid
    cx = (gx + 0.5) * cell
    cy = (gy + 0.5) * cell
    # Continuous action in normalized [-1, 1] coordinates.
    action = np.array([(cx / size) * 2 - 1, (cy / size) * 2 - 1, (label % 2) * 2 - 1], dtype=np.float32)
    unsafe = gx in {0, grid - 1} or gy in {0, grid - 1}

    if domain == "sim_source_clean":
        bg = (42, 45, 48); table = (72, 76, 82)
    elif domain == "sim_source_texture":
        bg = (34, 45, 48); table = (75, 69, 57)
    elif domain == "sim_source_clutter":
        bg = (40, 38, 44); table = (69, 72, 80)
    elif domain == "target_camera_shift":
        bg = (50, 38, 32); table = (84, 68, 52)
    elif domain == "target_lighting_shift":
        bg = (68, 54, 43); table = (105, 83, 61)
    else:  # target_occluded
        bg = (44, 47, 39); table = (75, 80, 70)

    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)
    margin = int(size * 0.08)
    draw.rounded_rectangle([margin, margin, size - margin, size - margin], radius=8, fill=table, outline=(140, 140, 135), width=1)

    # Table texture / grid lines are domain shift distractors.
    if domain in {"sim_source_texture", "target_lighting_shift"}:
        for _ in range(55):
            x0 = int(rng.integers(margin, size - margin))
            y0 = int(rng.integers(margin, size - margin))
            col = tuple(int(c + rng.integers(-10, 11)) for c in table)
            draw.point((x0, y0), fill=col)
    if domain == "sim_source_clutter":
        for _ in range(4):
            x0 = int(rng.integers(margin, size - margin - 10))
            y0 = int(rng.integers(margin, size - margin - 10))
            draw.rectangle([x0, y0, x0 + int(rng.integers(5, 12)), y0 + int(rng.integers(5, 12))], fill=(90, 95, 100))

    # Camera shift affects rendered location, but label remains the intended affordance target.
    jitter = rng.normal(0, cell * 0.06, size=2)
    if domain == "target_camera_shift":
        jitter += rng.normal(0, cell * 0.28, size=2) + np.array([cell * 0.18, -cell * 0.12])
    elif domain == "target_lighting_shift":
        jitter += rng.normal(0, cell * 0.14, size=2)
    elif domain == "target_occluded":
        jitter += rng.normal(0, cell * 0.18, size=2)
    ox = int(np.clip(cx + jitter[0], margin + 5, size - margin - 8))
    oy = int(np.clip(cy + jitter[1], margin + 5, size - margin - 8))

    # Draw target object and a gripper ray.
    color = ((53 * (label + 1)) % 230 + 20, (97 * (label + 3)) % 200 + 30, (149 * (label + 7)) % 170 + 45)
    radius = int(max(5, size / 28))
    draw.ellipse([ox - radius, oy - radius, ox + radius, oy + radius], fill=color, outline=(240, 240, 240), width=1)
    gripper_x = size // 2 + (5 if domain == "target_camera_shift" else 0)
    gripper_y = size - margin + 2
    draw.line([gripper_x, gripper_y, ox, oy], fill=(220, 220, 215), width=max(1, size // 64))
    draw.line([gripper_x - 5, gripper_y, gripper_x + 5, gripper_y], fill=(220, 220, 215), width=2)

    if domain == "target_occluded":
        # Deterministic-ish occluder at random side, approximating real clutter/hand occlusion.
        x0 = int(np.clip(ox + rng.integers(-8, 8), margin, size - margin - 12))
        y0 = int(np.clip(oy + rng.integers(-8, 8), margin, size - margin - 12))
        draw.rectangle([x0, y0, x0 + int(size * 0.18), y0 + int(size * 0.12)], fill=(25, 25, 25))

    if domain in {"target_camera_shift", "target_lighting_shift"}:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.35))

    arr = np.asarray(img).astype(np.float32)
    noise = 3.0 if domain.startswith("sim") else 8.0
    arr += rng.normal(0, noise, size=arr.shape)
    if domain == "target_lighting_shift":
        arr = arr * rng.uniform(0.85, 1.25) + rng.uniform(-8, 10)
    return np.clip(arr, 0, 255).astype(np.uint8), action, bool(unsafe)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a complete self-contained robot-affordance dataset.")
    parser.add_argument("--out", type=Path, default=Path("data/sim_affordance_v2.npz"))
    parser.add_argument("--n", type=int, default=9000)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--grid", type=int, default=4, help="grid^2 action-token labels")
    parser.add_argument("--episode-len", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.grid < 2:
        raise ValueError("grid must be >=2")
    rng = np.random.default_rng(args.seed)
    domains = [
        "sim_source_clean", "sim_source_texture", "sim_source_clutter",
        "target_camera_shift", "target_lighting_shift", "target_occluded",
    ]
    probs = _domain_probs(domains)
    images, labels, doms, episodes, actions, tasks, unsafe = [], [], [], [], [], [], []
    num_classes = args.grid * args.grid
    for i in range(args.n):
        label = int(rng.integers(0, num_classes))
        domain = str(rng.choice(domains, p=probs))
        image, action, is_unsafe = draw_frame(label, domain, args.size, args.grid, rng)
        images.append(image); labels.append(label); doms.append(domain)
        episodes.append(i // args.episode_len)
        actions.append(action); unsafe.append(is_unsafe)
        tasks.append("sim_pick_place_affordance")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        images=np.stack(images),
        labels=np.array(labels, dtype=np.int64),
        domains=np.array(doms),
        episodes=np.array(episodes, dtype=np.int64),
        actions=np.stack(actions).astype(np.float32),
        tasks=np.array(tasks),
        unsafe=np.array(unsafe, dtype=bool),
        metadata=np.array([f"grid={args.grid};size={args.size};episode_len={args.episode_len}"]),
    )
    unique, counts = np.unique(np.array(doms), return_counts=True)
    print(f"Wrote {args.out} with {args.n} samples, {num_classes} action tokens")
    print(dict(zip(unique.tolist(), counts.astype(int).tolist())))


if __name__ == "__main__":
    main()
