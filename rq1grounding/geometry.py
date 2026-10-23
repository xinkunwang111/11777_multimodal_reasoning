"""Boxes and the visual-token grid.

Canonical box format everywhere in this package: **normalized xyxy** in [0, 1], relative to
the *original* image. Conversions:
  V* annotation   [x, y, w, h] in original pixels        -> from_xywh_px
  Qwen3-VL output [x1, y1, x2, y2] relative, 0..1000     -> from_qwen / to_qwen
  model input     pixels of the resized image            -> to_px
Because the model input is the original image resized as a whole, a normalized box means the
same region in the original image, in the resized image and on the token grid.

Token grid: Qwen3-VL emits one LLM visual token per 32x32 px cell of the (resized) input,
in raster order, so cell (r, c) is token index r * grid_w + c.
"""

from __future__ import annotations

import numpy as np

Box = tuple  # (x1, y1, x2, y2), normalized


def clip_box(b) -> Box | None:
    x1, y1, x2, y2 = (float(np.clip(v, 0.0, 1.0)) for v in b)
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def from_xywh_px(b, img_w: int, img_h: int) -> Box | None:
    x, y, w, h = map(float, b)
    return clip_box((x / img_w, y / img_h, (x + w) / img_w, (y + h) / img_h))


def from_qwen(b) -> Box | None:
    return clip_box([v / 1000.0 for v in b])


def to_qwen(b: Box) -> list[int]:
    return [int(round(v * 1000)) for v in b]


def to_px(b: Box, w: int, h: int) -> tuple[int, int, int, int]:
    return int(b[0] * w), int(b[1] * h), int(np.ceil(b[2] * w)), int(np.ceil(b[3] * h))


def area(b: Box) -> float:
    return (b[2] - b[0]) * (b[3] - b[1])


def center(b: Box) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def contains(b: Box, pt) -> bool:
    return b[0] <= pt[0] <= b[2] and b[1] <= pt[1] <= b[3]


def union_box(boxes) -> Box | None:
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def hflip(b: Box) -> Box:
    return (1 - b[2], b[1], 1 - b[0], b[3])


def vflip(b: Box) -> Box:
    return (b[0], 1 - b[3], b[2], 1 - b[1])


def match(pred: Box | None, gts: list[Box]) -> dict:
    """Score one predicted box against the GT instances of an object (best instance)."""
    if pred is None or not gts:
        return {"iou": 0.0, "center_hit": False}
    return {"iou": max(iou(pred, g) for g in gts), "center_hit": any(contains(g, center(pred)) for g in gts)}


def expand(b: Box, img_w: int, img_h: int, margin: float = 0.5, min_side: float = 0.15) -> Box:
    """Grow a box by `margin` x its size on every side, enforce a minimum side of `min_side` x the
    image's longer side (in pixels, so small targets keep some context) and shift it inside the image."""
    x1, y1, x2, y2 = b[0] * img_w, b[1] * img_h, b[2] * img_w, b[3] * img_h
    w = min(max((x2 - x1) * (1 + 2 * margin), min_side * max(img_w, img_h)), img_w)
    h = min(max((y2 - y1) * (1 + 2 * margin), min_side * max(img_w, img_h)), img_h)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    nx = float(np.clip(cx - w / 2, 0, img_w - w))
    ny = float(np.clip(cy - h / 2, 0, img_h - h))
    return nx / img_w, ny / img_h, (nx + w) / img_w, (ny + h) / img_h


# ---------------------------------------------------------------------------- token grid

def coverage(boxes, grid_h: int, grid_w: int) -> np.ndarray:
    """[grid_h, grid_w] fraction of each token cell covered by the union of the boxes."""
    cov = np.zeros((grid_h, grid_w), dtype=np.float32)
    xs, ys = np.linspace(0, 1, grid_w + 1), np.linspace(0, 1, grid_h + 1)
    for b in boxes:
        if b is None:
            continue
        ox = np.clip(np.minimum(xs[1:], b[2]) - np.maximum(xs[:-1], b[0]), 0, None) * grid_w
        oy = np.clip(np.minimum(ys[1:], b[3]) - np.maximum(ys[:-1], b[1]), 0, None) * grid_h
        cov = np.maximum(cov, np.outer(oy, ox))
    return cov


def region_tokens(boxes, grid_h: int, grid_w: int, pad: int = 0) -> np.ndarray:
    """Indices (raster order) of the token cells overlapping any box, optionally dilated by `pad` cells.
    Any overlap counts, so an object smaller than one cell still owns the cell it sits in."""
    m = coverage(boxes, grid_h, grid_w) > 0
    if pad > 0 and m.any():
        from scipy import ndimage

        m = ndimage.binary_dilation(m, structure=np.ones((3, 3)), iterations=pad)  # square ring per step
    return np.flatnonzero(m.reshape(-1))


def random_box_like(b: Box, avoid: list[Box], rng, tries: int = 100) -> Box:
    """Same-size box at a random location that does not overlap `avoid` (best effort) - spatial control region."""
    w, h = b[2] - b[0], b[3] - b[1]
    cand = None
    for _ in range(tries):
        x, y = rng.uniform(0, max(1e-6, 1 - w)), rng.uniform(0, max(1e-6, 1 - h))
        cand = (x, y, x + w, y + h)
        if all(iou(cand, a) == 0 for a in avoid):
            return cand
    return cand
