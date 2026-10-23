"""Instrumentation of Qwen3-VL's own alignment pathway (attention lives in attention.py).

  * DeepStack: features from ViT blocks (8/16/24 for the 8B model) are merged and *added* to the
    visual-token residual stream after LLM layers 0..k-1  -> VisualFeatureEditor, DeepStackRecorder
  * Interleaved-MRoPE: image tokens get (t, h, w) rotary positions; the 2D layout enters the LLM
    only through these rotary angles, never through the residual  -> mrope_position_ids
  * residual stream / massive-norm "attention sink" tokens  -> HiddenStateRecorder

All tools assume batch size 1.
"""

from __future__ import annotations

import numpy as np
import torch

from .model import rope_positions


class HiddenStateRecorder:
    """states[l] = residual stream entering LLM layer l (after the DeepStack injection of layer l-1);
    states[n_layers] = final pre-norm state. Shape [n_layers+1, n_pos, D], or [n_layers+1, n_pos]
    with norm_only=True. Only the prefill pass is recorded."""

    def __init__(self, model, positions, dtype=torch.float32, norm_only=False):
        self.model = model
        if isinstance(positions, torch.Tensor):
            self.pos = positions.detach().cpu().long()
        else:
            self.pos = torch.as_tensor(np.asarray(positions), dtype=torch.long)
        self.dtype = dtype
        self.norm_only = norm_only
        self.handles, self.states = [], []

    def _hook(self, module, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        if h.shape[1] <= int(self.pos.max()):
            return  # decoding step
        h = h[0, self.pos.to(h.device)].float()
        self.states.append((h.norm(dim=-1) if self.norm_only else h).to(self.dtype).cpu())

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


def sink_tokens(norms: torch.Tensor, k: float = 5.0) -> np.ndarray:
    """Visual attention sinks from residual norms [layers+1, n_img]: tokens whose norm exceeds k x the
    per-layer median in at least half of the layers (the first two layers are skipped)."""
    r = norms[2:] / norms[2:].median(dim=1, keepdim=True).values.clamp_min(1e-6)
    return ((r > k).float().mean(0) >= 0.5).numpy()


@torch.no_grad()
def logit_lens(model, h: torch.Tensor) -> torch.Tensor:
    """Decode residual states [..., D] with the final norm + LM head."""
    w = model.lm_head.weight
    return model.lm_head(model.model.language_model.norm(h.to(w.device, w.dtype))).float()


class VisualFeatureEditor:
    """Forward hook on `model.model.visual`, whose output is (merged_embeds, [deepstack_embeds...]).

    zero_deepstack: None | "all" | list of DeepStack levels to zero (level i is injected after LLM layer i).
    ablate_tokens:  token indices (raster order over the merged grid) replaced by the per-image mean
                    ("mean") or zeros ("zero"), in the main embeds and, if ablate_deepstack, every level.
    record:         keep a CPU copy of the (edited) outputs in `self.captured`.
    """

    def __init__(self, model, zero_deepstack=None, ablate_tokens=None, ablate_mode="mean",
                 ablate_deepstack=True, record=False):
        self.model = model
        self.zero_ds = zero_deepstack
        if ablate_tokens is None:
            self.ablate = None
        elif isinstance(ablate_tokens, torch.Tensor):
            self.ablate = ablate_tokens.detach().cpu().long()
        else:
            self.ablate = torch.as_tensor(np.asarray(ablate_tokens), dtype=torch.long)
        self.ablate_mode = ablate_mode
        self.ablate_ds = ablate_deepstack
        self.record = record
        self.captured = None

    def _edit_rows(self, x):
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
            for i in (range(len(ds)) if self.zero_ds == "all" else self.zero_ds):
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
    """Per DeepStack level: mean ||injected|| / ||residual|| and cosine(injected, residual) over image tokens."""

    def __init__(self, model):
        self.lm = model.model.language_model
        self.stats: list[dict] = []

    def __enter__(self):
        orig = self.lm._deepstack_process
        self.stats = []

        def wrapped(hidden_states, visual_pos_masks, visual_embeds):
            h = hidden_states[visual_pos_masks.to(hidden_states.device)].float()
            v = visual_embeds.to(h.device).float()
            self.stats.append({"norm_ratio": float((v.norm(dim=-1) / h.norm(dim=-1).clamp_min(1e-6)).mean()),
                               "cosine": float(torch.nn.functional.cosine_similarity(v, h, dim=-1).mean())})
            return orig(hidden_states, visual_pos_masks, visual_embeds)

        self.lm._deepstack_process = wrapped
        return self

    def __exit__(self, *exc):
        del self.lm._deepstack_process


POSITION_MODES = ("default", "collapse", "shuffle", "hflip", "vflip")


def mrope_position_ids(model, inputs, image_pos, mode: str = "default", seed: int = 0) -> torch.Tensor:
    """3 x 1 x L (t, h, w) position ids with one image's spatial ids edited (text untouched).
    image_pos: sequence positions of that image's tokens (Prepared.image_pos / ImageSpan.pos).

    collapse: every image token gets the same (t, h, w) -> 2D layout removed from RoPE
    shuffle:  (h, w) pairs permuted across tokens  -> RoPE layout decoupled from content
    hflip / vflip: w -> w_max - w (h likewise)     -> RoPE says "mirrored", the pixels are not
    """
    pos = rope_positions(model, inputs)
    if mode == "default":
        return pos
    pos = pos.clone()
    idx = image_pos.to(pos.device)
    t, h, w = pos[0, 0, idx], pos[1, 0, idx], pos[2, 0, idx]
    if mode == "collapse":
        pos[1, 0, idx], pos[2, 0, idx] = t, t
    elif mode == "shuffle":
        perm = torch.randperm(len(idx), generator=torch.Generator().manual_seed(seed)).to(pos.device)
        pos[1, 0, idx], pos[2, 0, idx] = h[perm], w[perm]
    elif mode == "hflip":
        pos[2, 0, idx] = w.min() + w.max() - w
    elif mode == "vflip":
        pos[1, 0, idx] = h.min() + h.max() - h
    else:
        raise ValueError(mode)
    return pos
