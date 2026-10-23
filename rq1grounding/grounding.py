"""Attention grounding: where does the model look when it reads a question about an object?

Query groups (attention rows recorded onto the image tokens, per layer and head):
  object:<k>  all tokens of object k's name in the question ("... the color of the [glove]?")
  last        the answer position (the token whose logits pick the option letter)
"Grounding heads" are the few heads whose object-query attention localizes best on a calibration split.

Map metrics (threshold-free, against the per-cell GT coverage of the object's own box):
  pointing      the map's peak cell overlaps the GT box ("the attention found the object")
  auroc         ranking of cells, GT cells vs the rest (0.5 = chance)
  energy / lift share of the map on the GT box; lift = energy / GT area share (1 = chance)
  own_gt_other  more mass on its own object than on the other objects (compositional grounding)
A map becomes a region (box) by thresholding at t x its peak, keeping the connected component around
the peak and taking its bounding cells; t is calibrated (mean IoU with GT on the calibration split).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy import ndimage
from scipy.stats import rankdata

from .attention import AttentionRecorder
from .hooks import HiddenStateRecorder, sink_tokens

THRESHOLDS = np.round(np.arange(0.1, 1.0, 0.1), 2)


@dataclass
class AttnMaps:
    maps: dict  # query group -> [layers, heads, n_img] attention onto the main image (float16)
    sinks: np.ndarray  # [n_img] bool, massive-norm sink tokens
    logits: torch.Tensor


def query_groups(prep) -> dict:
    g = {"last": [prep.answer_pos]}
    g.update({f"object:{k}": p for k, p in enumerate(prep.object_pos)})
    return {k: v for k, v in g.items() if v}


@torch.no_grad()
def record_attention(model, prep, sink_k: float = 5.0, position_ids=None) -> AttnMaps:
    """One forward pass: attention of all query groups onto the main image + sink detection + answer logits."""
    with AttentionRecorder(model, query_groups(prep), prep.image_pos.cpu()) as rec, \
            HiddenStateRecorder(model, prep.image_pos.cpu(), norm_only=True) as nrec:
        out = model(**prep.inputs, position_ids=position_ids, use_cache=False, logits_to_keep=1)
    return AttnMaps(rec.maps, sink_tokens(nrec.stacked, sink_k), out.logits)


def aggregate(A: torch.Tensor, heads=None, layers=None, sinks=None) -> np.ndarray:
    """[layers, heads, n] -> one map [n] summing to 1 (sinks zeroed). heads: [(l, h)], layers: [l], or all."""
    if heads:
        M = torch.stack([A[l, h].float() for l, h in heads]).mean(0)
    elif layers is not None:
        M = A[list(layers)].float().mean((0, 1))
    else:
        M = A.float().mean((0, 1))
    M = M.numpy().astype(np.float64)
    if sinks is not None and (~sinks).any():
        M = np.where(sinks, 0.0, M)
    return M / max(M.sum(), 1e-12)


def _normalize_rows(X: np.ndarray, sinks) -> np.ndarray:
    if sinks is not None and (~sinks).any():
        X = np.where(sinks, 0.0, X)
    return X / X.sum(-1, keepdims=True).clip(1e-12)


def auroc_rows(scores: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """AUROC of every row of scores [K, n] against boolean labels [n] (ties averaged)."""
    P, N = int(pos.sum()), int((~pos).sum())
    if P == 0 or N == 0:
        return np.full(scores.shape[0], np.nan)
    r = rankdata(scores, axis=1)
    return (r[:, pos].sum(1) - P * (P + 1) / 2) / (P * N)


def pointing_rows(scores: np.ndarray, pos: np.ndarray) -> np.ndarray:
    return pos[scores.argmax(1)].astype(np.float64)


def head_scores(A: torch.Tensor, cov: np.ndarray, sinks=None) -> dict:
    """Pointing / AUROC of every single head [layers, heads] and of each layer's head average [layers]."""
    X = A.float().numpy()
    Ly, H, n = X.shape
    pos = cov > 0
    flat = _normalize_rows(X.reshape(Ly * H, n), sinks)
    lay = _normalize_rows(X.mean(1), sinks)
    return {"head_pointing": pointing_rows(flat, pos).reshape(Ly, H), "head_auroc": auroc_rows(flat, pos).reshape(Ly, H),
            "layer_pointing": pointing_rows(lay, pos), "layer_auroc": auroc_rows(lay, pos)}


def map_metrics(M: np.ndarray, cov: np.ndarray, other: np.ndarray | None = None) -> dict:
    pos = cov > 0
    energy = float((M * cov).sum())
    out = {"pointing": float(pos[M.argmax()]), "auroc": float(auroc_rows(M[None], pos)[0]),
           "energy": energy, "lift": energy / max(float(cov.mean()), 1e-6)}
    if other is not None and (other > 0).any():
        out["own_gt_other"] = float(energy > float((M * other).sum()))
    return out


def map_to_box(M: np.ndarray, grid_h: int, grid_w: int, t: float = 0.5):
    """Cells >= t x peak, connected component (8-neighbourhood) containing the peak -> normalized box."""
    M2 = M.reshape(grid_h, grid_w)
    lab, _ = ndimage.label(M2 >= t * M2.max(), structure=np.ones((3, 3)))
    ys, xs = np.nonzero(lab == lab[np.unravel_index(M2.argmax(), M2.shape)])
    return xs.min() / grid_w, ys.min() / grid_h, (xs.max() + 1) / grid_w, (ys.max() + 1) / grid_h


def peak_point(M: np.ndarray, grid_h: int, grid_w: int):
    r, c = divmod(int(M.argmax()), grid_w)
    return (c + 0.5) / grid_w, (r + 0.5) / grid_h


def select_heads(head_pointing: list, head_auroc: list, k: int) -> list:
    """Top-k heads by mean pointing (AUROC breaks ties) over calibration objects."""
    score = np.nanmean(head_pointing, 0) + 0.1 * np.nanmean(head_auroc, 0)
    flat = np.argsort(-np.nan_to_num(score, nan=-1).reshape(-1))[:k]
    return [[int(i // score.shape[1]), int(i % score.shape[1])] for i in flat]
