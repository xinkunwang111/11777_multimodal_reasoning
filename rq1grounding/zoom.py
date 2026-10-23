"""Self-Grounded Zoom (SGZ) with MRoPE-aligned re-injection - a training-free fix for small-target grounding.

1. Localize with the model itself: one forward pass at the normal budget, read the attention
   of the *object-phrase* tokens through the few grounding heads selected on a calibration
   split (05_grounding_bench.py), mask visual attention sinks, binarize with the calibrated
   threshold and keep the component around the peak. Relation questions: one region per
   object, then their union, so both arguments of the relation stay in view.
2. Zoom: crop that region (plus margin) from the *original* high-resolution image.
3. Re-inject: answer again with [global image, crop]. Qwen3-VL gives image tokens 3D
   (t, h, w) rotary positions, so instead of letting the crop look like an unrelated second
   image we can give its tokens the (fractional) h/w positions of the region they come from
   in the global image, on the next temporal slice (t + dt): RoPE then says "same place, finer
   view". Rotary angles are continuous, so fractional positions are well defined.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from . import grounding_eval as ge
from .data import build_prompt
from .maps import head_map
from .model import Prepared, build_inputs, letter_token_ids, resize_to_budget, token_factor

ZOOM_HINT = "Image 1 is the full image. Image 2 is a zoomed-in view of a region of Image 1.\n"


def region_from_map(M: np.ndarray, gh: int, gw: int, rule: str, t: float):
    mask = ge.peak_component(ge.binarize(M, rule, t).reshape(gh, gw), M.reshape(gh, gw))
    return ge.mask_box(mask)


def union_box(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def expand_box(box, W: int, H: int, margin: float = 0.5, min_frac: float = 0.2):
    """Grow a normalized xyxy box by `margin` x its size on each side, enforce a minimum side of
    min_frac x the image's longer side (context for tiny targets), and shift it inside the image."""
    x0, y0, x1, y1 = box[0] * W, box[1] * H, box[2] * W, box[3] * H
    w, h = (x1 - x0) * (1 + 2 * margin), (y1 - y0) * (1 + 2 * margin)
    side = min_frac * max(W, H)
    w, h = min(max(w, side), W), min(max(h, side), H)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    nx0 = float(np.clip(cx - w / 2, 0, W - w))
    ny0 = float(np.clip(cy - h / 2, 0, H - h))
    return nx0 / W, ny0 / H, (nx0 + w) / W, (ny0 + h) / H


def random_box_like(box, rng):
    w, h = box[2] - box[0], box[3] - box[1]
    x0, y0 = rng.uniform(0, 1 - w), rng.uniform(0, 1 - h)
    return x0, y0, x0 + w, y0 + h


def crop(orig: Image.Image, box) -> Image.Image:
    W, H = orig.size
    return orig.crop((int(box[0] * W), int(box[1] * H), int(np.ceil(box[2] * W)), int(np.ceil(box[3] * H))))


def sgz_box(bundle, prep, cfg: dict, rule: str | None = None):
    """Region proposal from calibrated grounding heads. Returns (normalized box, query set used) or (None, None)."""
    rule = rule or cfg.get("default_rule", "rel")
    sink = bundle.sinks(cfg.get("sink_k", 5.0))
    for q in ("object", "question", "answer"):
        name = f"{q}/heads+ns"
        if q not in cfg.get("heads", {}) or name not in cfg.get("thresholds", {}):
            continue
        heads = [tuple(x) for x in cfg["heads"][q]]
        t = cfg["thresholds"][name][rule]
        keys = [f"object:{k}" for k in range(bundle.n_obj) if bundle.has(f"object:{k}")] if q == "object" else [q]
        boxes = [region_from_map(head_map(bundle, k, heads=heads, sink_mask=sink), prep.grid_h, prep.grid_w, rule, t)
                 for k in keys if bundle.has(k)]
        if boxes:
            return union_box(boxes), q
    return None, None


def attn_all_box(bundle, prep, cfg: dict | None, rule: str = "rel"):
    """Baseline region: answer-token attention averaged over all layers and heads, no sink removal."""
    t = (cfg or {}).get("thresholds", {}).get("answer/all", {}).get(rule, 0.5)
    return region_from_map(head_map(bundle, "answer"), prep.grid_h, prep.grid_w, rule, t)


def prepare_images(model, processor, sample, images: list[Image.Image], hint: bool) -> Prepared:
    """Prepared for an arbitrary list of images (only the fields needed for answering)."""
    prompt = (ZOOM_HINT if hint and len(images) > 1 else "") + build_prompt(sample)
    inputs, spans = build_inputs(model, processor, images, prompt)
    sp = spans[0]
    return Prepared(sample=sample, inputs=inputs, image=images[0], grid_h=sp.grid_h, grid_w=sp.grid_w,
                    image_pos=sp.pos, answer_pos=inputs["input_ids"].shape[1] - 1,
                    coverage=np.zeros((sp.grid_h, sp.grid_w), np.float32),
                    letter_ids=letter_token_ids(processor.tokenizer, len(sample.options)), spans=spans)


def zoom_inputs(model, processor, sample, global_img: Image.Image, orig: Image.Image, box, crop_pixels: int,
                hint: bool = True) -> Prepared:
    f = token_factor(model)
    # small crops are upsampled to at least half the budget so the target spans more ViT patches
    c = resize_to_budget(crop(orig, box), crop_pixels, crop_pixels // 2, f)
    return prepare_images(model, processor, sample, [global_img, c], hint)


def aligned_position_ids(model, prep: Prepared, box, dt: float = 1.0) -> torch.Tensor:
    """MRoPE ids where the crop (image 2) tokens sit at the h/w of their source region in image 1, at t + dt."""
    inp = prep.inputs
    pos, _ = model.model.get_rope_index(inp["input_ids"], inp["image_grid_thw"], None, inp.get("attention_mask"))
    pos = pos.float()
    g, c = prep.spans[0], prep.spans[1]
    first = g.pos[0]
    t0, h0, w0 = pos[0, 0, first], pos[1, 0, first], pos[2, 0, first]
    ii = torch.arange(c.grid_h, device=pos.device, dtype=pos.dtype)
    jj = torch.arange(c.grid_w, device=pos.device, dtype=pos.dtype)
    hy = box[1] * g.grid_h + (ii + 0.5) * (box[3] - box[1]) * g.grid_h / c.grid_h - 0.5
    wx = box[0] * g.grid_w + (jj + 0.5) * (box[2] - box[0]) * g.grid_w / c.grid_w - 0.5
    idx = c.pos.to(pos.device)
    pos[0, 0, idx] = t0 + dt
    pos[1, 0, idx] = (h0 + hy[:, None].expand(c.grid_h, c.grid_w)).reshape(-1)
    pos[2, 0, idx] = (w0 + wx[None, :].expand(c.grid_h, c.grid_w)).reshape(-1)
    return pos


def crop_quality(box, gt_boxes) -> dict:
    """Does the crop contain the target(s)?  gt_boxes: normalized xyxy."""
    centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in gt_boxes]
    return {"center_in": float(np.mean([box[0] <= x <= box[2] and box[1] <= y <= box[3] for x, y in centers])),
            "recall": float(np.mean([ge.box_recall(box, b) for b in gt_boxes])),
            "area": float((box[2] - box[0]) * (box[3] - box[1]))}
