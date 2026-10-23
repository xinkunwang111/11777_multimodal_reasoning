"""Scoring a grounding map against V*-style GT boxes.

The map M lives on the merged token grid (gh x gw, one cell = 32x32 px of the model input).
GT is mapped onto the same grid as per-cell box coverage (data.bbox_coverage_grid);
"positive" cells are those overlapping a GT box (cov > 0).

Threshold-free (primary, no free parameter):
  pointing     argmax cell overlaps the GT box                      (Zhang et al., 2016)
  pointing_tol argmax within 1 cell of the GT box
  energy       share of the map's mass on GT cells (coverage weighted)
  lift         energy / GT area share (1 = chance, independent of box size)
  auroc        ranking quality of cells, positives = GT cells (0.5 = chance)
  ap           average precision of the same ranking
  own_gt_other energy on the queried object's box > energy on the other objects' boxes (compositional)

Thresholded ("the model is looking here"). Every rule is swept over a grid; a *fixed*
threshold is then chosen on a calibration split and applied to the test split
(MaxBoxAcc protocol of Choe et al., CVPR 2020 - never tune the threshold on test):
  rel@t   M >= t * max(M)                                   (CAM convention)
  mass@p  smallest set of cells holding a fraction p of the mass
  uni@k   M >= k / n_cells   ("k times more attention than uniform")
  otsu    per-map Otsu threshold (no parameter)
From the binary mask we take the connected component containing the peak, its bounding
box (in continuous image coordinates) and compare with GT:
  box_iou, box_acc@{0.3,0.5}, center_hit (GT centre cell inside the mask),
  recall (GT box area inside the predicted box - what a crop needs), tok_f1 (cell-level F1),
  area (mask size as a share of the image).
Note the quantization ceiling: a GT box smaller than a cell can never reach IoU 0.5 at
this resolution; `oracle_iou` (GT box snapped to cells) is reported for reference.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.stats import rankdata

RULES = {
    "rel": np.round(np.arange(0.05, 1.0, 0.05), 2),
    "mass": np.round(np.arange(0.1, 1.0, 0.1), 2),
    "uni": np.array([1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 50.0]),
    "otsu": np.array([0.0]),
}
SWEEP_METRICS = ("box_iou", "center_hit", "recall", "tok_f1", "area")


def auroc_rows(scores: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Vectorized AUROC of each row of scores [K, n] against boolean labels [n] (ties averaged)."""
    P, N = int(pos.sum()), int((~pos).sum())
    if P == 0 or N == 0:
        return np.full(scores.shape[0], np.nan)
    r = rankdata(scores, axis=1)
    return (r[:, pos].sum(1) - P * (P + 1) / 2) / (P * N)


def pointing_rows(scores: np.ndarray, pos: np.ndarray) -> np.ndarray:
    return pos[scores.argmax(1)].astype(np.float64)


def _ap(score: np.ndarray, pos: np.ndarray) -> float:
    if pos.sum() == 0 or (~pos).sum() == 0:
        return float("nan")
    order = np.argsort(-score, kind="stable")
    tp = np.cumsum(pos[order])
    prec = tp / np.arange(1, len(order) + 1)
    return float((prec * pos[order]).sum() / pos.sum())


def otsu(x: np.ndarray, bins: int = 64) -> float:
    hist, edges = np.histogram(x, bins=bins)
    mids = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * mids) / np.maximum(w0, 1)
    m1 = ((hist * mids).sum() - np.cumsum(hist * mids)) / np.maximum(w1, 1)
    return float(mids[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def binarize(M: np.ndarray, rule: str, t: float) -> np.ndarray:
    """Binary mask of 'attended' cells; always contains the peak, so a region exists for every threshold."""
    mask = _binarize(M, rule, t)
    mask[np.argmax(M)] = True
    return mask


def _binarize(M: np.ndarray, rule: str, t: float) -> np.ndarray:
    if rule == "rel":
        return M >= t * M.max()
    if rule == "mass":
        order = np.argsort(-M, kind="stable")
        k = int(np.searchsorted(np.cumsum(M[order]), t * M.sum())) + 1
        mask = np.zeros_like(M, dtype=bool)
        mask[order[:k]] = True
        return mask
    if rule == "uni":
        return M >= t / M.size
    if rule == "otsu":
        return M >= otsu(M)
    raise ValueError(rule)


def peak_component(mask2d: np.ndarray, M2d: np.ndarray) -> np.ndarray:
    """8-connected component of the mask that contains the map's peak (inside the mask)."""
    lab, n = ndimage.label(mask2d, structure=np.ones((3, 3)))
    if n == 0:
        return mask2d
    peak = np.unravel_index(np.argmax(np.where(mask2d, M2d, -np.inf)), M2d.shape)
    return lab == lab[peak]


def mask_box(mask2d: np.ndarray) -> tuple[float, float, float, float]:
    """Normalized xyxy box of the cells in the mask."""
    ys, xs = np.nonzero(mask2d)
    gh, gw = mask2d.shape
    return xs.min() / gw, ys.min() / gh, (xs.max() + 1) / gw, (ys.max() + 1) / gh


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def box_recall(pred, gt) -> float:
    ix = max(0.0, min(pred[2], gt[2]) - max(pred[0], gt[0]))
    iy = max(0.0, min(pred[3], gt[3]) - max(pred[1], gt[1]))
    area = (gt[2] - gt[0]) * (gt[3] - gt[1])
    return ix * iy / area if area > 0 else 0.0


def threshold_free(M: np.ndarray, cov: np.ndarray, gh: int, gw: int, other_cov: np.ndarray | None = None) -> dict:
    pos = cov > 0
    dil = ndimage.binary_dilation(pos.reshape(gh, gw), structure=np.ones((3, 3))).reshape(-1)
    energy = float((M * cov).sum())
    out = {
        "pointing": float(pos[M.argmax()]), "pointing_tol": float(dil[M.argmax()]),
        "energy": energy, "lift": energy / max(cov.mean(), 1e-6),
        "auroc": float(auroc_rows(M[None], pos)[0]), "ap": _ap(M, pos),
    }
    if other_cov is not None and (other_cov > 0).any():
        out["other_energy"] = float((M * other_cov).sum())
        out["own_gt_other"] = float(energy > out["other_energy"])
    return out


def thresholded(M: np.ndarray, cov: np.ndarray, gh: int, gw: int, gt_boxes: list, rule: str, t: float) -> dict:
    """gt_boxes: normalized xyxy boxes of the queried object(s)."""
    M2 = M.reshape(gh, gw)
    mask = peak_component(binarize(M, rule, t).reshape(gh, gw), M2)
    pb = mask_box(mask)
    gt_mask = (cov > 0).reshape(gh, gw)
    tp = (mask & gt_mask).sum()
    centers = [(min(int((b[1] + b[3]) / 2 * gh), gh - 1), min(int((b[0] + b[2]) / 2 * gw), gw - 1)) for b in gt_boxes]
    return {
        "box_iou": max(box_iou(pb, b) for b in gt_boxes),
        "center_hit": float(np.mean([mask[c] for c in centers])),
        "recall": float(np.mean([box_recall(pb, b) for b in gt_boxes])),
        "tok_f1": float(2 * tp / max(mask.sum() + gt_mask.sum(), 1)),
        "area": float(mask.mean()),
        "box": pb,
    }


def sweep(M, cov, gh, gw, gt_boxes) -> dict:
    """rule -> metric -> array over that rule's threshold grid."""
    res = {}
    for rule, grid in RULES.items():
        rows = [thresholded(M, cov, gh, gw, gt_boxes, rule, t) for t in grid]
        res[rule] = {m: np.array([r[m] for r in rows]) for m in SWEEP_METRICS}
    return res


def oracle_iou(cov: np.ndarray, gh: int, gw: int, gt_boxes: list) -> float:
    m = (cov > 0).reshape(gh, gw)
    return max(box_iou(mask_box(m), b) for b in gt_boxes) if m.any() else 0.0


def aggregate_sweeps(sweeps: list[dict]) -> dict:
    """Mean over samples, plus box_acc@{0.3,0.5} curves: rule -> metric -> [n_thresholds]."""
    agg = {}
    for rule in RULES:
        d = {m: np.nanmean(np.stack([s[rule][m] for s in sweeps]), 0) for m in SWEEP_METRICS}
        ious = np.stack([s[rule]["box_iou"] for s in sweeps])
        for delta in (0.3, 0.5):
            d[f"box_acc@{delta}"] = (ious >= delta).mean(0)
        agg[rule] = d
    return agg


def pick_thresholds(agg: dict, objective: str = "tok_f1") -> dict:
    """Best threshold per rule on the calibration split."""
    return {rule: float(RULES[rule][int(np.nanargmax(agg[rule][objective]))]) for rule in RULES}


def at_threshold(sweep_res: dict, rule: str, t: float) -> dict:
    i = int(np.argmin(np.abs(RULES[rule] - t)))
    return {m: float(sweep_res[rule][m][i]) for m in SWEEP_METRICS}
