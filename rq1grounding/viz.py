"""Plotting helpers (matplotlib, headless)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


def overlay(ax, image: Image.Image, grid_map: np.ndarray | None, boxes=(), title="", cmap="jet", box_colors=("lime",)):
    """Image + upsampled token-grid heatmap + normalized xyxy boxes."""
    W, H = image.size
    ax.imshow(image)
    if grid_map is not None:
        m = grid_map.astype(np.float32)
        m = (m - m.min()) / (m.max() - m.min() + 1e-12)
        ax.imshow(m, cmap=cmap, alpha=0.5, extent=(0, W, H, 0), interpolation="nearest")
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        ax.add_patch(patches.Rectangle((x0 * W, y0 * H), (x1 - x0) * W, (y1 - y0) * H, fill=False,
                                       edgecolor=box_colors[min(i, len(box_colors) - 1)], linewidth=2))
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def layer_curves(curves: dict, ylabel: str, path: str, title: str = "", hline=None, x=None, xlabel="LLM layer"):
    """curves: label -> [n_samples, n_points] (mean +/- s.e.m.) or [n_points]."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for label, arr in curves.items():
        arr = np.asarray(arr, dtype=np.float64)
        if arr.size == 0:
            continue
        xs = np.arange(arr.shape[-1]) if x is None else np.asarray(x)
        if arr.ndim == 2:
            mu = np.nanmean(arr, 0)
            se = np.nanstd(arr, 0) / np.sqrt(max(1, len(arr)))
            ax.plot(xs, mu, label=f"{label} (n={len(arr)})")
            ax.fill_between(xs, mu - se, mu + se, alpha=0.2)
        else:
            ax.plot(xs, arr, marker=".", label=label)
    if hline is not None:
        ax.axhline(hline, color="gray", ls="--", lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def head_heatmap(score: np.ndarray, path: str, title: str = ""):
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(score, aspect="auto", cmap="viridis")
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
