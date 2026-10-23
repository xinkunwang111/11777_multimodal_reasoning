"""Loading Qwen3-VL, building inputs and the token layout, answering.

Qwen3-VL image tokens: ViT patch = 16 px, 2x2 spatial merge -> one LLM token per 32x32 px cell,
emitted in raster order. Images are pre-resized to a multiple of 32 under a pixel budget, so the
processor does not resize again and the token grid is known exactly; a normalized GT box maps
onto that grid directly (geometry.coverage).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image

from . import geometry as G
from .data import LETTERS, Sample, build_prompt

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def load_model(model_id: str = DEFAULT_MODEL, dtype: str = "bfloat16", device_map: str = "auto",
               attn_implementation: str = "sdpa"):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=getattr(torch, dtype), device_map=device_map, attn_implementation=attn_implementation
    ).eval()
    model.requires_grad_(False)
    return model, AutoProcessor.from_pretrained(model_id)


def token_factor(model) -> int:
    v = model.config.vision_config
    return v.patch_size * v.spatial_merge_size


def resize_to_budget(img: Image.Image, max_pixels: int, min_pixels: int, factor: int) -> Image.Image:
    w, h = img.size
    scale = 1.0
    if w * h > max_pixels:
        scale = math.sqrt(max_pixels / (w * h))
    elif w * h < min_pixels:
        scale = math.sqrt(min_pixels / (w * h))
    nw = max(factor, int(w * scale) // factor * factor)
    nh = max(factor, int(h * scale) // factor * factor)
    return img if (nw, nh) == img.size else img.resize((nw, nh), Image.BICUBIC)


@dataclass
class ImageSpan:
    pos: torch.Tensor  # sequence positions of the image's tokens, raster order over its merged grid
    grid_h: int
    grid_w: int


def build_inputs(model, processor, images: list[Image.Image], prompt: str):
    """Chat-formatted inputs for [image_1, ..., image_k, prompt] plus the token span of every image."""
    content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
    text = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                         add_generation_prompt=True)
    inputs = processor(text=[text], images=list(images), return_tensors="pt").to(model.device)
    merge = model.config.vision_config.spatial_merge_size
    all_pos = (inputs["input_ids"][0] == model.config.image_token_id).nonzero().squeeze(-1)
    spans, o = [], 0
    for _, h, w in inputs["image_grid_thw"].tolist():
        spans.append(ImageSpan(all_pos[o:o + (h // merge) * (w // merge)], h // merge, w // merge))
        o += (h // merge) * (w // merge)
    assert o == all_pos.numel(), (o, all_pos.numel())
    return inputs, spans


@dataclass
class Prepared:
    """Model inputs for one question plus everything needed to relate tokens to GT."""
    sample: Sample
    inputs: dict
    image: Image.Image  # main image as fed to the model (resized)
    spans: list  # ImageSpan per image; spans[0] is the main image
    answer_pos: int  # last prompt position: its logits give the option letter
    letter_ids: list
    question_pos: list = field(default_factory=list)  # tokens of the question sentence
    object_pos: list = field(default_factory=list)  # per sample.objects: tokens of its name inside the question

    @property
    def grid_h(self) -> int:
        return self.spans[0].grid_h

    @property
    def grid_w(self) -> int:
        return self.spans[0].grid_w

    @property
    def n_img(self) -> int:
        return self.grid_h * self.grid_w

    @property
    def image_pos(self) -> torch.Tensor:
        return self.spans[0].pos

    @property
    def text_pos(self) -> list:
        """Text positions after the last image (question, options, chat suffix, answer position)."""
        return list(range(min(int(self.spans[-1].pos[-1]) + 2, self.answer_pos), self.answer_pos + 1))

    # ---- GT on the main image's token grid
    def object_cov(self, k: int) -> np.ndarray:
        return G.coverage(self.sample.objects[k].boxes, self.grid_h, self.grid_w).reshape(-1)

    def gt_cov(self) -> np.ndarray:
        return G.coverage(self.sample.boxes, self.grid_h, self.grid_w).reshape(-1)

    def other_cov(self, k: int) -> np.ndarray | None:
        """Coverage of everything that is *not* object k: other question objects + annotated distractors."""
        boxes = [b for j, o in enumerate(self.sample.objects) if j != k for b in o.boxes]
        boxes += [tuple(b) for b in self.sample.meta.get("other_boxes", [])]
        return G.coverage(boxes, self.grid_h, self.grid_w).reshape(-1) if boxes else None

    def tokens_of(self, boxes, pad: int = 0) -> np.ndarray:
        """Sequence positions of the main-image tokens overlapping the boxes."""
        idx = G.region_tokens(boxes, self.grid_h, self.grid_w, pad)
        return self.image_pos.cpu().numpy()[idx]


def _char_spans(tokenizer, ids):
    spans, text = [], ""
    for i in ids:
        piece = tokenizer.decode([i])
        spans.append((len(text), len(text) + len(piece)))
        text += piece
    return text, spans


def locate_question(tokenizer, input_ids: list[int], question: str, names: list[str], start: int):
    """Token positions of the question sentence and of each object name *inside* the question
    (so an option such as "The cup is left of the dog" is never taken as the object query)."""
    text, spans = _char_spans(tokenizer, input_ids[start:])
    low, q = text.lower(), question.lower().strip()
    q0 = low.find(q)
    lo, hi = (q0, q0 + len(q)) if q0 >= 0 else (0, len(low))

    def pos(k, n):
        return [start + j for j, (a, b) in enumerate(spans) if a < k + n and b > k]

    obj = []
    for name in names:
        hit = []
        for cand in (name.lower(), name.lower().split()[-1]):
            k = low.find(cand, lo, hi)
            if k >= 0:
                hit = pos(k, len(cand))
                break
        obj.append(hit)
    return (pos(q0, len(q)) if q0 >= 0 else []), obj


def letter_token_ids(tokenizer, n: int) -> list[int]:
    return [tokenizer.encode(LETTERS[i], add_special_tokens=False)[0] for i in range(n)]


def prepare(model, processor, sample: Sample, max_pixels: int, min_pixels: int = 64 * 32 * 32,
            image: Image.Image | None = None, prefix: str = "", extra_images=()) -> Prepared:
    """image: main image to use instead of the file (e.g. with a drawn box); extra_images are
    appended after it as-is (already resized), e.g. a zoomed crop."""
    img = image if image is not None else Image.open(sample.image_path).convert("RGB")
    img = resize_to_budget(img, max_pixels, min_pixels, token_factor(model))
    inputs, spans = build_inputs(model, processor, [img, *extra_images], build_prompt(sample, prefix))
    ids = inputs["input_ids"][0].tolist()
    q_pos, obj_pos = locate_question(processor.tokenizer, ids, sample.question, [o.name for o in sample.objects],
                                     int(spans[-1].pos[-1]) + 1)
    return Prepared(sample, inputs, img, spans, len(ids) - 1, letter_token_ids(processor.tokenizer, len(sample.options)),
                    q_pos, obj_pos)


@torch.no_grad()
def forward(model, prep_or_inputs, position_ids: torch.Tensor | None = None, **kw):
    inputs = prep_or_inputs.inputs if isinstance(prep_or_inputs, Prepared) else prep_or_inputs
    return model(**inputs, position_ids=position_ids, use_cache=False, logits_to_keep=1, **kw)


def answer_from_logits(logits: torch.Tensor, prep: Prepared) -> dict:
    """Multiple-choice prediction restricted to the option letters."""
    lp = logits[0, -1].float()[prep.letter_ids].log_softmax(-1)
    pred = int(lp.argmax())
    return {"pred": pred, "correct": pred == prep.sample.answer_idx, "probs": lp.exp().tolist(),
            "logp_correct": float(lp[prep.sample.answer_idx])}


def rope_positions(model, inputs) -> torch.Tensor:
    pos, _ = model.model.get_rope_index(inputs["input_ids"], inputs["image_grid_thw"], None, inputs.get("attention_mask"))
    return pos
