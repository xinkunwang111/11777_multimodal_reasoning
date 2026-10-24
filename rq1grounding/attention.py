"""Reading and editing Qwen3-VL's text-side attention to image tokens.

Both tools wrap the attention kernel of the LLM layers (sdpa / flash / eager), which receives the
post-RoPE queries/keys and the values. Only the rows of the chosen query positions are recomputed
in fp32, so no L x L map is ever materialised (V* inputs are ~1-4k tokens) and every other row
comes from the original kernel untouched.

AttentionRecorder  read : attention of query groups onto a set of key positions, per layer and head.
AttentionScaler    write: change how much the chosen query rows attend to a set of target keys.

Editing semantics (edit_probs), applied to the softmax probabilities of each (head, query) row:
  scale(a)   weights on the target keys x a, then renormalise.  a = 0 is attention knockout,
             0 < a < 1 suppresses, a > 1 amplifies. Side effect: the total attention on the
             image (and on text) changes too.
  shift(l)   move attention *within a pool* (default: the other image tokens) without touching
             anything else: l > 0 moves a fraction l of the pool's mass onto the target,
             l < 0 moves a fraction |l| of the target's mass back onto the pool (l = -1: knockout
             that keeps total image attention fixed). Text attention and the total image
             attention stay exactly the same, so the edit changes only *where in the image* the
             model reads, not *how much* it relies on the image. Within the target (and within
             the pool) the model's own relative weights are preserved.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import torch


def text_attention_modules(model):
    return [layer.self_attn for layer in model.model.language_model.layers]


class _TextAttentionPatch:
    """Context manager that routes every LLM self-attention call through `on_layer`."""

    def __init__(self, model):
        self.attn0 = text_attention_modules(model)[0]
        self.attn_cls = type(self.attn0)
        self.mod = sys.modules[self.attn_cls.__module__]
        cfg = model.config.text_config
        self.n_rep = cfg.num_attention_heads // cfg.num_key_value_heads

    def on_layer(self, layer_idx, q, k, v, scaling, out):  # pragma: no cover - overridden
        return out

    def _wrapper(self, module, query, key, value, attention_mask, *args, **kwargs):
        out, w = self._orig(module, query, key, value, attention_mask, *args, **kwargs)
        if isinstance(module, self.attn_cls) and query.shape[2] == key.shape[2] and query.shape[2] > 1:
            scaling = kwargs.get("scaling") or module.scaling
            out = self.on_layer(module.layer_idx, query, key, value, scaling, out)
        return out, w

    def __enter__(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        self.impl = self.attn0.config._attn_implementation
        if self.impl == "eager":
            self._orig = self.mod.eager_attention_forward
            self.mod.eager_attention_forward = self._wrapper
        else:
            self._orig = ALL_ATTENTION_FUNCTIONS[self.impl]
            ALL_ATTENTION_FUNCTIONS[self.impl] = self._wrapper
        return self

    def __exit__(self, *exc):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        if self.impl == "eager":
            self.mod.eager_attention_forward = self._orig
        else:
            ALL_ATTENTION_FUNCTIONS[self.impl] = self._orig

    def _rows(self, q, k, qpos: torch.Tensor, scaling: float):
        """Softmax attention probabilities [H, Q, L] (fp32) of the query positions qpos (batch 1, causal)."""
        qs = q[0, :, qpos].float()
        ks = k[0].float().repeat_interleave(self.n_rep, dim=0)
        s = qs @ ks.transpose(-1, -2) * scaling
        s.masked_fill_(torch.arange(ks.shape[1], device=s.device)[None, None, :] > qpos[None, :, None], float("-inf"))
        return s.softmax(-1)


class AttentionRecorder(_TextAttentionPatch):
    """Attention of query *groups* onto `keys`, averaged over each group's query rows.

    groups: name -> sequence positions of the query tokens (e.g. {"object:0": [...], "last": [L-1]}).
    After the forward: maps[name] -> [layers, heads, n_keys] (`dtype`, default float16, CPU) and
    mass[name] -> [layers, heads], the total attention the group puts on `keys`.
    """

    def __init__(self, model, groups: dict, keys, dtype=torch.float16):
        super().__init__(model)
        self.dtype = dtype
        self.groups = {g: list(map(int, p)) for g, p in groups.items() if len(p)}
        self.all_q = sorted({p for v in self.groups.values() for p in v})
        self.col = {p: i for i, p in enumerate(self.all_q)}
        if isinstance(keys, torch.Tensor):
            self.keys = keys.detach().cpu().long()
        else:
            self.keys = torch.as_tensor(np.asarray(keys), dtype=torch.long)
        self._maps: dict[str, list] = {}

    def on_layer(self, layer_idx, q, k, v, scaling, out):
        qpos = torch.tensor(self.all_q, device=q.device)
        p = self._rows(q, k, qpos, scaling)[:, :, self.keys.to(q.device)]  # [H, Q, n_keys]
        for g, ps in self.groups.items():
            self._maps.setdefault(g, []).append(p[:, [self.col[x] for x in ps]].mean(1).to(self.dtype).cpu())
        return out

    def __enter__(self):
        self._maps = {}
        return super().__enter__()

    @property
    def maps(self) -> dict:
        return {g: torch.stack(v) for g, v in self._maps.items()}

    @property
    def mass(self) -> dict:
        return {g: m.float().sum(-1) for g, m in self.maps.items()}


def edit_probs(p: torch.Tensor, target: torch.Tensor, mode: str, value: float, pool: torch.Tensor | None = None):
    """Edit attention probability rows p [..., L]; target / pool are boolean masks over L (see module doc)."""
    if mode == "scale":
        w = p.clone()
        w[..., target] *= value
        return w / w.sum(-1, keepdim=True).clamp_min(1e-30)
    if mode != "shift":
        raise ValueError(mode)
    assert pool is not None and not (pool & target).any()
    pt = p[..., target].sum(-1, keepdim=True)
    pp = p[..., pool].sum(-1, keepdim=True)
    move = value * pp if value >= 0 else value * pt  # signed mass that flows into the target
    w = p.clone()
    nt = int(target.sum())
    # rescale within each group; if the target had (numerically) no mass, spread what arrives uniformly
    w[..., target] = torch.where(pt > 1e-12, p[..., target] * (pt + move) / pt.clamp_min(1e-30), move / nt)
    w[..., pool] = p[..., pool] * ((pp - move) / pp.clamp_min(1e-30))
    return w


@dataclass
class AttnEdit:
    layers: list  # LLM layer indices
    queries: list  # sequence positions of the query rows to edit
    keys: list  # sequence positions of the target keys
    mode: str = "scale"  # "scale" | "shift"
    value: float = 0.0  # a for scale, l for shift
    pool: list | None = None  # shift: positions exchanging mass with the target (e.g. the other image tokens)


class AttentionScaler(_TextAttentionPatch):
    """Apply AttnEdits during the forward pass (several edits may target different layers / queries)."""

    def __init__(self, model, edits: list):
        super().__init__(model)
        self.edits = [e for e in edits if len(e.queries) and len(e.keys) and len(e.layers)]

    def on_layer(self, layer_idx, q, k, v, scaling, out):
        for e in self.edits:
            if layer_idx not in e.layers:
                continue
            L = k.shape[2]
            qpos = torch.tensor(sorted(set(map(int, e.queries))), device=q.device)
            target = torch.zeros(L, dtype=torch.bool, device=q.device)
            target_keys = e.keys.detach().long().to(q.device) if isinstance(e.keys, torch.Tensor) else torch.as_tensor(np.asarray(e.keys), dtype=torch.long, device=q.device)
            target[target_keys] = True
            pool = None
            if e.mode == "shift":
                pool = torch.zeros(L, dtype=torch.bool, device=q.device)
                pool_keys = e.pool.detach().long().to(q.device) if isinstance(e.pool, torch.Tensor) else torch.as_tensor(np.asarray(e.pool), dtype=torch.long, device=q.device)
                pool[pool_keys] = True
                pool &= ~target
            p = edit_probs(self._rows(q, k, qpos, scaling), target, e.mode, e.value, pool)
            vs = v[0].float().repeat_interleave(self.n_rep, dim=0)  # [H, L, d]
            out = out.clone()
            out[0, qpos] = (p @ vs).transpose(0, 1).to(out.dtype)  # out: [B, L, H, d]
        return out
