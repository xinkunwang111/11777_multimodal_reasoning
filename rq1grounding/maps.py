"""Grounding maps: where on the image token grid does the model put its evidence?

A single query token is a noisy probe, so attention is collected for *query sets*:
  answer    the position that emits the option letter
  question  every token of the question sentence
  object    every token of each target-object phrase in the question (one map per object,
            scored against that object's own box -> compositional, per-object grounding)
and aggregated over a chosen set of layers/heads (all, one layer, or calibrated
"grounding heads"). Visual attention sinks (image tokens with massive residual norm that
soak up attention regardless of the query) can be masked out before evaluation.
A gradient x input map on the visual embeddings is provided as a non-attention reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .hooks import AttentionRecorder, HiddenStateRecorder

QUERY_SETS = ("answer", "question", "object")


@dataclass
class MapBundle:
    attn: dict  # query-set name -> [layers, heads, n_img] attention (mean over the set's tokens); "object:k" per object
    norm_ratio: np.ndarray  # [layers+1, n_img] residual norm / per-layer median over image tokens
    logits: torch.Tensor
    n_obj: int

    def sinks(self, k: float) -> np.ndarray:
        """Image tokens whose residual norm is > k x median in at least half of the layers (skipping the first 2)."""
        r = self.norm_ratio[2:]
        return (r > k).mean(0) >= 0.5

    def has(self, qset: str) -> bool:
        return qset in self.attn


@torch.no_grad()
def collect(model, prep, store_dtype=torch.float16) -> MapBundle:
    """One forward pass recording attention of all query sets onto the (first) image + image-token norms."""
    groups = {"answer": [prep.answer_pos]}
    if prep.question_pos:
        groups["question"] = list(prep.question_pos)
    for k, sp in enumerate(prep.object_spans):
        if sp:
            groups[f"object:{k}"] = list(sp)
    qpos = sorted({p for v in groups.values() for p in v})
    col = {p: i for i, p in enumerate(qpos)}
    img = prep.image_pos.cpu()
    with AttentionRecorder(model, qpos, store_dtype) as rec, \
            HiddenStateRecorder(model, prep.image_pos, norm_only=True) as nrec:
        out = model(**prep.inputs, use_cache=False, logits_to_keep=1)
    A = rec.attn[..., img]  # [Ly, H, nQ, n_img]
    attn = {g: A[:, :, [col[p] for p in v]].float().mean(2) for g, v in groups.items()}
    objs = [v for g, v in attn.items() if g.startswith("object:")]
    if objs:
        attn["object"] = torch.stack(objs).mean(0)
    norms = nrec.stacked.numpy()
    ratio = norms / np.median(norms, axis=1, keepdims=True).clip(1e-6)
    return MapBundle(attn=attn, norm_ratio=ratio, logits=out.logits, n_obj=len(prep.object_spans))


def head_map(b: MapBundle, qset: str, heads=None, layers=None, sink_mask=None) -> np.ndarray | None:
    """Aggregate a query set's attention into one [n_img] map summing to 1.
    heads: list of (layer, head) | layers: list of layers | neither: all layers and heads."""
    if qset not in b.attn:
        return None
    A = b.attn[qset]
    if heads:
        M = torch.stack([A[l, h] for l, h in heads]).mean(0)
    elif layers is not None:
        M = A[list(layers)].mean((0, 1))
    else:
        M = A.mean((0, 1))
    M = M.numpy().astype(np.float64).copy()
    if sink_mask is not None and (~sink_mask).any():
        M[sink_mask] = 0.0
    return M / max(M.sum(), 1e-12)


def grad_input_map(model, prep) -> np.ndarray:
    """ReLU(sum_d grad * x) on the visual embeddings (main + every DeepStack level) for the log-prob
    of the *predicted* option (no ground truth used). Shape [n_img] summing to 1 (first image)."""
    cap = {}

    def hook(mod, args, out):
        main = out[0].detach().requires_grad_(True)
        ds = [d.detach().requires_grad_(True) for d in out[1]]
        cap["x"] = [main] + ds
        return (main, ds) + tuple(out[2:])

    h = model.model.visual.register_forward_hook(hook)
    try:
        with torch.enable_grad():
            out = model(**prep.inputs, use_cache=False, logits_to_keep=1)
            lp = out.logits[0, -1].float()[prep.letter_ids].log_softmax(-1)
            lp.max().backward()
    finally:
        h.remove()
    rel = sum((x.grad.float() * x.float()).sum(-1) for x in cap["x"]).clamp_min(0)[: prep.n_img]
    rel = rel.detach().cpu().numpy().astype(np.float64)
    return rel / max(rel.sum(), 1e-12)
