"""VEA - Visual Evidence Augmentation ("Seeing but Not Believing", Liu et al., ICLR 2026) on Qwen3-VL.

A reproduction of the official code (github.com/ZhiningLiu1998/VEA: vea/profiling.py, attention.py,
highlight.py) with the paper's hyperparameters. One forward pass reads where the model looks, the rest of
the image is dimmed, and the model answers again on the highlighted image:
  A  profiling     per-layer AUROC of the last prompt token's head-averaged attention onto the image tokens
                   against the GT evidence cells, averaged over a calibration set; keep the top 10% of layers
  B  attribution   that attention averaged over the profiled layers -> one score per visual token   (Eq. 1)
  C  denoise       a token above lam x each of its 8 neighbours is an isolated spike -> their mean   (Eq. 2)
  D  smooth        paint the scores onto the model-input pixels, Gaussian blur with sigma = 0.5 x the
                   shorter side, min-max normalise to [0, 1]                                          (Eq. 3)
  E  highlight     every pixel x (alpha + (1 - alpha) x map): no evidence -> alpha x brightness    (Eq. 4)
The model then answers on the highlighted image, with a hint about the highlighted region in the prompt.

What differs from the original concerns the model and the data, not the method:
  - Qwen3-VL: one visual token = 32 x 32 px (Qwen2.5-VL: 28 x 28). The attention row comes from the kernel
    wrapper in attention.py (equal to eager output_attentions, tested), so no eager attention is needed.
  - Multiple choice, scored by the option-letter logits (the paper generates free-form answers, scored by EM).
  - The hint is reworded for multiple choice; the paper's ("Only use words from the picture, especially those
    in the highlighted region, to answer the question.") is written for text-reading VQA.
  - Profiling uses the calibration split of the evaluated dataset (the paper: 100 TextVQA questions).
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from .attention import AttentionRecorder
from .grounding import auroc_rows
from .model import forward

HINT = "Use the image, especially its highlighted region, to answer the question.\n"


@torch.no_grad()
def last_token_attention(model, prep) -> tuple[np.ndarray, torch.Tensor]:
    """The attribution pass: [layers, n_img] head-averaged attention of the answer position onto the main image
    (part of a softmax over the whole prompt), and the answer logits of the same pass."""
    with AttentionRecorder(model, {"last": [prep.answer_pos]}, prep.image_pos.cpu(), dtype=torch.float32) as rec:
        out = forward(model, prep)
    return rec.maps["last"].mean(1).numpy(), out.logits


def select_layers(layer_auroc, top_frac: float = 0.1) -> list[int]:
    """Step A: the top `top_frac` of the layers by mean evidence AUROC (at least one)."""
    score = np.nan_to_num(np.asarray(layer_auroc, dtype=np.float64), nan=-np.inf)
    n_keep = max(1, math.ceil(top_frac * len(score)))
    return sorted(int(i) for i in np.argsort(score)[-n_keep:])


def layer_span(n_layers: int, lo: float, hi: float) -> list[int]:
    """Layers in the relative depth range [lo, hi), as in the paper's attribution table (L_0-50% etc.)."""
    start, end = int(lo * n_layers), int(hi * n_layers)
    return list(range(start, max(end, start + 1)))


def layer_stats(attn: np.ndarray, evidence: np.ndarray, n_tokens: int) -> dict:
    """Per layer: evidence AUROC of the attention [layers, n_img], and the relative attention per token on the
    evidence / non-evidence cells (mean per token over the mean of the whole prompt row, i.e. x n_tokens;
    1 = uniform share) - the quantities behind the paper's correct-vs-wrong analysis."""
    pos = np.asarray(evidence) > 0
    nan = np.full(attn.shape[0], np.nan)
    return {"auroc": auroc_rows(attn, pos),
            "rapt_evd": attn[:, pos].mean(1) * n_tokens if pos.any() else nan,
            "rapt_nonevd": attn[:, ~pos].mean(1) * n_tokens if (~pos).any() else nan}


def denoise(grid: np.ndarray, lam: float = 10.0) -> np.ndarray:
    """Eq. 2: every token above lam x each of its 8 neighbours is replaced by their mean (edge-padded, so border
    tokens see copies of themselves; grids under 3 x 3 are left as they are)."""
    grid = np.array(grid, dtype=np.float32)
    if min(grid.shape) < 3:
        return grid
    h, w = grid.shape
    pad = np.pad(grid, 1, mode="edge")
    nb = np.stack([pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx])
    return np.where(grid > lam * nb.max(0), nb.mean(0), grid).astype(np.float32)


@lru_cache(maxsize=8)
def _paint_and_blur(n_cells: int, cell: int, sigma: float) -> np.ndarray:
    """[n_cells * cell, n_cells] linear map: paint one value per cell onto its pixels, then blur along that axis
    exactly as scipy's gaussian_filter does (reflect boundary, truncate 4). The blurred painted map is
    rows_map @ grid @ cols_map.T - exact, without filtering a megapixel map with a ~3000-tap kernel."""
    E = np.repeat(np.eye(n_cells), cell, axis=0)
    M = ndimage.gaussian_filter1d(E, sigma, axis=0) if sigma > 0 else E
    M.setflags(write=False)  # cached and shared
    return M


def normalize(x: np.ndarray) -> np.ndarray:
    """Min-max to [0, 1]; a constant map becomes all zeros."""
    lo, hi = float(x.min()), float(x.max())
    return np.zeros(x.shape, np.float32) if hi <= lo else ((x - lo) / (hi - lo)).astype(np.float32)


def evidence_map(scores, grid_h: int, grid_w: int, cell: int, lam: float = 10.0, sigma: float = 0.5,
                 apply_denoise: bool = True, apply_smooth: bool = True) -> np.ndarray:
    """Steps C-D: token scores [grid_h * grid_w] -> pixel map [grid_h * cell, grid_w * cell] in [0, 1].
    sigma <= 1 is relative to the shorter image side, > 1 is in pixels (as in the original)."""
    g = np.asarray(scores, dtype=np.float32).reshape(grid_h, grid_w)
    if apply_denoise:
        g = denoise(g, lam)
    H, W = grid_h * cell, grid_w * cell
    s = 0.0 if not apply_smooth or sigma <= 0 else float(sigma * min(H, W) if sigma <= 1 else sigma)
    pix = _paint_and_blur(grid_h, cell, s) @ g.astype(np.float64) @ _paint_and_blur(grid_w, cell, s).T
    return normalize(pix.astype(np.float32))  # float32 like the original, so rounding noise is not normalised up


def highlight(image: Image.Image, emap: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Eq. 4: scale every pixel by alpha + (1 - alpha) x map (map 1: unchanged, map 0: alpha x brightness)."""
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    assert emap.shape == arr.shape[:2], (emap.shape, arr.shape)
    gain = (alpha + (1.0 - alpha) * emap.astype(np.float32))[..., None]
    return Image.fromarray(np.clip(arr * gain, 0, 255).astype(np.uint8))
