"""Instrumentation for Qwen3-VL's alignment pathway.

Qwen3-VL specific pieces probed here:
  * DeepStack: features from ViT blocks (8/16/24 for the 8B model) are merged and
    *added* to the visual-token residual stream after LLM layers 0..k-1.
  * Interleaved-MRoPE: image tokens get (t, h, w) rotary positions; spatial layout
    enters the LLM only through these rotary angles, not through the residual.

All tools assume batch size 1 and a prefill (no KV-cache) forward.
"""

from __future__ import annotations

import sys
import torch


def _text_layers(model):
    return model.model.language_model.layers


def _modeling_module(model):
    return sys.modules[type(_text_layers(model)[0].self_attn).__module__]


# ---------------------------------------------------------------------------
# Attention rows for selected query positions (works with sdpa / flash attention)
# ---------------------------------------------------------------------------

class AttentionRecorder:
    """Recomputes softmax(q k^T) for a few query positions in every LLM layer.

    The text attention calls the module-level `apply_rotary_pos_emb` right
    before the attention kernel, so wrapping it gives post-RoPE q/k without
    needing `attn_implementation="eager"` (which would materialise L x L maps
    for ~2k-token V* inputs). Result: `self.attn` of shape [layers, heads, n_query, seq_len].
    """

    def __init__(self, model, query_positions: list[int], store_dtype=torch.float16):
        self.mod = _modeling_module(model)
        self.qpos = list(query_positions)
        self.store_dtype = store_dtype
        cfg = model.config.text_config
        self.n_rep = cfg.num_attention_heads // cfg.num_key_value_heads
        self.rows: list[torch.Tensor] = []

    def _wrapped(self, q, k, cos, sin, *args, **kwargs):
        q_e, k_e = self._orig(q, k, cos, sin, *args, **kwargs)
        if q_e.shape[2] > 1:
            qs = q_e[0, :, self.qpos].float()  # [H, nQ, d]
            ks = k_e[0].float().repeat_interleave(self.n_rep, dim=0)  # [H, L, d]
            scores = qs @ ks.transpose(-1, -2) * qs.shape[-1] ** -0.5
            L = ks.shape[1]
            qp = torch.tensor(self.qpos, device=scores.device)
            causal = torch.arange(L, device=scores.device)[None, :] > qp[:, None]
            scores.masked_fill_(causal[None], float("-inf"))
            self.rows.append(scores.softmax(-1).to(self.store_dtype).cpu())
        return q_e, k_e

    def __enter__(self):
        self._orig = self.mod.apply_rotary_pos_emb
        self.mod.apply_rotary_pos_emb = self._wrapped
        self.rows = []
        return self

    def __exit__(self, *exc):
        self.mod.apply_rotary_pos_emb = self._orig

    @property
    def attn(self) -> torch.Tensor:
        return torch.stack(self.rows)


# ---------------------------------------------------------------------------
# Residual stream at selected positions (input of every layer + final pre-norm)
# ---------------------------------------------------------------------------

class HiddenStateRecorder:
    """states[l] = residual stream entering LLM layer l (after DeepStack injection of
    layer l-1); states[n_layers] = final pre-norm state. Shape [n_layers+1, n_pos, D]."""

    def __init__(self, model, positions, dtype=torch.float32):
        self.model = model
        self.pos = torch.as_tensor(positions)
        self.dtype = dtype
        self.handles = []
        self.states: list[torch.Tensor] = []

    def _hook(self, module, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        self.states.append(h[0, self.pos.to(h.device)].to(self.dtype).cpu())

    def __enter__(self):
        lm = self.model.model.language_model
        self.states = []
        for layer in lm.layers:
            self.handles.append(layer.register_forward_pre_hook(self._hook, with_kwargs=True))
        self.handles.append(lm.norm.register_forward_pre_hook(self._hook, with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []

    @property
    def stacked(self) -> torch.Tensor:
        return torch.stack(self.states)


@torch.no_grad()
def logit_lens(model, h: torch.Tensor) -> torch.Tensor:
    """Decode residual states [..., D] with the final norm + LM head."""
    lm = model.model.language_model
    w = model.lm_head.weight
    return model.lm_head(lm.norm(h.to(w.device, w.dtype))).float()


# ---------------------------------------------------------------------------
# Editing what the vision tower hands to the LLM (main embeds + DeepStack embeds)
# ---------------------------------------------------------------------------

class VisualFeatureEditor:
    """Forward hook on `model.model.visual`, whose output is (merged_embeds, [deepstack_embeds...]).

    zero_deepstack: None | "all" | list of DeepStack levels to zero (level i is injected after LLM layer i).
    ablate_tokens: indices (raster order over the merged grid) to replace by the per-image mean
                   ("mean") or zeros ("zero") in the main embeds and, if ablate_deepstack, every DeepStack level.
    record: keep a CPU copy of the (edited) outputs in `self.captured`.
    """

    def __init__(self, model, zero_deepstack=None, ablate_tokens=None, ablate_mode="mean",
                 ablate_deepstack=True, record=False):
        self.model = model
        self.zero_ds = zero_deepstack
        self.ablate = None if ablate_tokens is None else torch.as_tensor(ablate_tokens, dtype=torch.long)
        self.ablate_mode = ablate_mode
        self.ablate_ds = ablate_deepstack
        self.record = record
        self.captured = None

    def _edit_rows(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        idx = self.ablate.to(x.device)
        x[idx] = x.mean(0, keepdim=True).expand(len(idx), -1) if self.ablate_mode == "mean" else 0
        return x

    def _hook(self, module, args, output):
        main, ds = output[0], list(output[1])
        if self.ablate is not None and len(self.ablate):
            main = self._edit_rows(main)
            if self.ablate_ds:
                ds = [self._edit_rows(d) for d in ds]
        if self.zero_ds is not None:
            levels = range(len(ds)) if self.zero_ds == "all" else self.zero_ds
            for i in levels:
                ds[i] = torch.zeros_like(ds[i])
        if self.record:
            self.captured = {"main": main.float().cpu(), "deepstack": [d.float().cpu() for d in ds]}
        return (main, ds) + tuple(output[2:])

    def __enter__(self):
        self.handle = self.model.model.visual.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        self.handle.remove()


class DeepStackRecorder:
    """Measures how strongly each DeepStack injection perturbs the visual residual stream:
    per level, mean ||injected|| / ||residual|| and cosine(injected, residual) over image tokens."""

    def __init__(self, model):
        self.lm = model.model.language_model
        self.stats: list[dict] = []

    def __enter__(self):
        orig = self.lm._deepstack_process
        self.stats = []

        def wrapped(hidden_states, visual_pos_masks, visual_embeds):
            h = hidden_states[visual_pos_masks.to(hidden_states.device)].float()
            v = visual_embeds.to(h.device).float()
            self.stats.append({
                "norm_ratio": float((v.norm(dim=-1) / h.norm(dim=-1).clamp_min(1e-6)).mean()),
                "cosine": float(torch.nn.functional.cosine_similarity(v, h, dim=-1).mean()),
            })
            return orig(hidden_states, visual_pos_masks, visual_embeds)

        self.lm._deepstack_process = wrapped
        return self

    def __exit__(self, *exc):
        del self.lm._deepstack_process


# ---------------------------------------------------------------------------
# MRoPE position edits for image tokens
# ---------------------------------------------------------------------------

POSITION_MODES = ("default", "collapse", "shuffle", "hflip", "vflip")


def mrope_position_ids(model, prep, mode: str = "default", seed: int = 0) -> torch.Tensor:
    """3 x 1 x L (t, h, w) position ids with the image tokens' spatial ids edited.

    collapse: every image token gets the same (t, h, w) -> 2D layout removed from RoPE.
    shuffle:  (h, w) pairs randomly permuted across tokens -> RoPE layout decoupled from content.
    hflip / vflip: w -> w_max - w (h likewise) -> RoPE says the image is mirrored, pixels are not.
    """
    inp = prep.inputs
    pos, _ = model.model.get_rope_index(inp["input_ids"], inp["image_grid_thw"], None, inp.get("attention_mask"))
    if mode == "default":
        return pos
    pos = pos.clone()
    idx = prep.image_pos.to(pos.device)
    t, h, w = pos[0, 0, idx], pos[1, 0, idx], pos[2, 0, idx]
    if mode == "collapse":
        pos[1, 0, idx] = t
        pos[2, 0, idx] = t
    elif mode == "shuffle":
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(idx), generator=g).to(pos.device)
        pos[1, 0, idx] = h[perm]
        pos[2, 0, idx] = w[perm]
    elif mode == "hflip":
        pos[2, 0, idx] = w.min() + w.max() - w
    elif mode == "vflip":
        pos[1, 0, idx] = h.min() + h.max() - h
    else:
        raise ValueError(mode)
    return pos

