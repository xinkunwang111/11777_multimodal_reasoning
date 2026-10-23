"""Ways to tell the model *where the object is* when it answers (the method and its baselines).

All take `boxes`: one list of normalized boxes per question object (aligned with sample.objects),
from any source (the model's own explicit grounding, its attention, or GT).

  text   coordinates in the prompt, in Qwen3-VL's native grounding format (0-1000 relative xyxy)
  draw   box outlines drawn on the image. The colour is picked so that it is *not* one of the answer
         options (a red box around an object would otherwise bias "what colour is it?")
  crop   [full image, zoomed crop around the boxes] - input-side zoom; costs extra visual tokens
  steer  (ours) no input change: during the forward pass the text tokens' attention is moved onto the
         visual tokens inside the boxes, in the layers where the answer reads the object (probe 2)
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from . import geometry as G
from .attention import AttnEdit
from .model import resize_to_budget, token_factor

DRAW_COLORS = [("red", (255, 0, 0)), ("green", (0, 200, 0)), ("blue", (0, 90, 255)), ("magenta", (255, 0, 255)),
               ("cyan", (0, 220, 220)), ("yellow", (255, 220, 0))]
MODES = ("text", "draw", "crop", "steer")


def _names(sample, boxes):
    return [o.name for o, b in zip(sample.objects, boxes) if b]


def _join(names):
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def text_cue(sample, boxes) -> str:
    parts = [f"the {o.name} is at {G.to_qwen(b[0]) if len(b) == 1 else [G.to_qwen(x) for x in b]}"
             for o, b in zip(sample.objects, boxes) if b]
    if not parts:
        return ""
    return "Hint: " + "; ".join(parts) + " (bounding boxes [x1, y1, x2, y2], relative coordinates 0-1000).\n"


def draw_cue(image: Image.Image, sample, boxes):
    """image: the resized model input. Returns (image with boxes, prompt prefix)."""
    names = _names(sample, boxes)
    if not names:
        return image, ""
    opts = " ".join(sample.options).lower()
    cname, rgb = next(((n, c) for n, c in DRAW_COLORS if n not in opts), DRAW_COLORS[-1])
    img = image.copy()
    d = ImageDraw.Draw(img)
    width = max(2, round(0.004 * max(img.size)))
    for b in (x for bs in boxes if bs for x in bs):
        x1, y1, x2, y2 = G.to_px(b, *img.size)
        d.rectangle([x1 - width, y1 - width, x2 + width, y2 + width], outline=rgb, width=width)
    verb = "is" if len(names) == 1 else "are"
    return img, f"Hint: the {_join(names)} {verb} marked with {cname} box{'es' if len(names) > 1 else ''}.\n"


def crop_cue(model, original: Image.Image, sample, boxes, crop_pixels: int, margin: float = 0.5, min_side: float = 0.15):
    """Zoomed crop of the original (full-resolution) image around all boxes, resized to the crop budget."""
    names = _names(sample, boxes)
    ub = G.union_box([x for bs in boxes if bs for x in bs])
    if ub is None:
        return None, "", None
    W, H = original.size
    box = G.expand(ub, W, H, margin, min_side)
    crop = original.crop(G.to_px(box, W, H))
    crop = resize_to_budget(crop, crop_pixels, crop_pixels, token_factor(model))  # small crops are upsampled
    return crop, f"Hint: the second image is a zoomed-in view of the region around the {_join(names)} in the first image.\n", box


def steer_edit(prep, boxes, layers, mode: str = "shift", value: float = 0.5, pad: int = 0) -> AttnEdit | None:
    """Move the text tokens' attention onto the visual tokens of the boxes (pool = the other image tokens)."""
    flat = [x for bs in boxes if bs for x in bs]
    if not flat:
        return None
    keys = prep.tokens_of(flat, pad)
    pool = np.setdiff1d(prep.image_pos.cpu().numpy(), keys)
    return AttnEdit(layers=list(layers), queries=prep.text_pos, keys=keys.tolist(), mode=mode, value=value,
                    pool=pool.tolist())
