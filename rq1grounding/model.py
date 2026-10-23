"""Loading Qwen3-VL and turning a Sample into model inputs plus a token layout.

Qwen3-VL image tokens: ViT patch = 16px, a 2x2 spatial merge -> one LLM token per
32x32 pixel cell, emitted in raster order of the merged grid. We pre-resize the
image to a multiple of 32 under a pixel budget so the processor does not resize
again and the token grid is known exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image

from .data import LETTERS, Sample, bbox_coverage_grid, build_prompt

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def load_model(model_id: str = DEFAULT_MODEL, dtype: str = "bfloat16", device_map: str = "auto",
               attn_implementation: str = "sdpa"):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=getattr(torch, dtype), device_map=device_map, attn_implementation=attn_implementation
    ).eval()
    model.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def resize_to_budget(img: Image.Image, max_pixels: int, min_pixels: int, factor: int) -> Image.Image:
    w, h = img.size
    scale = 1.0
    if w * h > max_pixels:
        scale = math.sqrt(max_pixels / (w * h))
    elif w * h < min_pixels:
        scale = math.sqrt(min_pixels / (w * h))
    nw = max(factor, int(w * scale) // factor * factor)
    nh = max(factor, int(h * scale) // factor * factor)
    return img.resize((nw, nh), Image.BICUBIC)


def token_factor(model) -> int:
    v = model.config.vision_config
    return v.patch_size * v.spatial_merge_size


@dataclass
class ImageSpan:
    pos: torch.Tensor  # sequence positions of this image's tokens (raster order over the merged grid)
    grid_h: int
    grid_w: int


def build_inputs(model, processor, images: list[Image.Image], prompt: str):
    """Chat-formatted inputs for [image_1, ..., image_k, prompt] and the token span of every image."""
    content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
    text = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                         add_generation_prompt=True)
    inputs = processor(text=[text], images=list(images), return_tensors="pt").to(model.device)
    merge = model.config.vision_config.spatial_merge_size
    all_pos = (inputs["input_ids"][0] == model.config.image_token_id).nonzero().squeeze(-1)
    spans, o = [], 0
    for _, h, w in inputs["image_grid_thw"].tolist():
        gh, gw = h // merge, w // merge
        spans.append(ImageSpan(all_pos[o:o + gh * gw], gh, gw))
        o += gh * gw
    assert o == all_pos.numel(), (o, all_pos.numel())
    return inputs, spans


@dataclass
class Prepared:
    sample: Sample
    inputs: dict
    image: Image.Image  # the resized image actually fed to the model
    grid_h: int
    grid_w: int
    image_pos: torch.Tensor  # sequence positions of the gh*gw image tokens (raster order)
    answer_pos: int  # last prompt position, whose logits give the answer letter
    coverage: np.ndarray  # [gh, gw] fraction of each token cell inside the union of target boxes
    object_pos: dict = field(default_factory=dict)  # object name -> positions of its tokens in the question
    letter_ids: list = field(default_factory=list)
    distractor_coverage: np.ndarray | None = None
    question_pos: list = field(default_factory=list)  # all tokens of the question sentence
    object_spans: list = field(default_factory=list)  # aligned with sample.target_objects
    object_cov: list = field(default_factory=list)  # per-object coverage grid (own box), aligned likewise
    spans: list = field(default_factory=list)  # ImageSpan per image (1 unless zoom inputs)

    @property
    def n_img(self) -> int:
        return self.grid_h * self.grid_w

    @property
    def cov_flat(self) -> np.ndarray:
        return self.coverage.reshape(-1)


def _token_char_spans(tokenizer, ids: list[int]):
    spans, text = [], ""
    for i in ids:
        piece = tokenizer.decode([i])
        spans.append((len(text), len(text) + len(piece)))
        text += piece
    return text, spans


def _positions_in(spans, start: int, k: int, n: int) -> list[int]:
    return [start + j for j, (a, b) in enumerate(spans) if a < k + n and b > k]


def find_word_positions(tokenizer, input_ids: list[int], word: str, start: int) -> list[int]:
    """Positions (>= start) of the tokens spelling `word` (last occurrence in the prompt)."""
    text, spans = _token_char_spans(tokenizer, input_ids[start:])
    low = text.lower()
    for cand in (word.lower(), word.lower().split()[-1]):
        k = low.rfind(cand)
        if k >= 0:
            return _positions_in(spans, start, k, len(cand))
    return []


def locate_question(tokenizer, input_ids: list[int], question: str, objects: list[str], start: int):
    """Token positions of the question sentence and of each object phrase *inside the question*
    (options such as "The cup is left of the dog" must not be picked up as the object query)."""
    text, spans = _token_char_spans(tokenizer, input_ids[start:])
    low, q = text.lower(), question.lower().strip()
    q0 = low.find(q)
    q_pos = _positions_in(spans, start, q0, len(q)) if q0 >= 0 else []
    obj_spans = []
    for o in objects:
        hit = []
        for cand in (o.lower(), o.lower().split()[-1]):
            k = low.find(cand, max(q0, 0), (q0 + len(q)) if q0 >= 0 else None)
            if k >= 0:
                hit = _positions_in(spans, start, k, len(cand))
                break
        obj_spans.append(hit or find_word_positions(tokenizer, input_ids, o, start))
    return q_pos, obj_spans


def letter_token_ids(tokenizer, n: int) -> list[int]:
    return [tokenizer.encode(LETTERS[i], add_special_tokens=False)[0] for i in range(n)]


def prepare(model, processor, sample: Sample, max_pixels: int = 1024 * 32 * 32,
            min_pixels: int = 256 * 32 * 32, image: Image.Image | None = None) -> Prepared:
    orig = image if image is not None else Image.open(sample.image_path).convert("RGB")
    img = resize_to_budget(orig, max_pixels, min_pixels, token_factor(model))
    inputs, spans = build_inputs(model, processor, [img], build_prompt(sample))
    sp = spans[0]
    gh, gw = sp.grid_h, sp.grid_w

    W, H = orig.size
    tok = processor.tokenizer
    id_list = inputs["input_ids"][0].tolist()
    after_img = int(sp.pos[-1]) + 1
    q_pos, obj_spans = locate_question(tok, id_list, sample.question, sample.target_objects, after_img)
    boxes = sample.bboxes
    per_obj = [bbox_coverage_grid([b], W, H, gh, gw) for b in boxes] if len(boxes) == len(sample.target_objects) else []
    dcov = None
    if sample.meta.get("distractor_bboxes"):
        dcov = bbox_coverage_grid(sample.meta["distractor_bboxes"], W, H, gh, gw)
    return Prepared(
        sample=sample, inputs=inputs, image=img, grid_h=gh, grid_w=gw, image_pos=sp.pos,
        answer_pos=len(id_list) - 1, coverage=bbox_coverage_grid(boxes, W, H, gh, gw),
        object_pos={o: s for o, s in zip(sample.target_objects, obj_spans)},
        letter_ids=letter_token_ids(tok, len(sample.options)), distractor_coverage=dcov,
        question_pos=q_pos, object_spans=obj_spans, object_cov=per_obj, spans=spans,
    )


@torch.no_grad()
def forward(model, prep: Prepared, position_ids: torch.Tensor | None = None, **kw):
    return model(**prep.inputs, position_ids=position_ids, use_cache=False, logits_to_keep=1, **kw)


def answer_from_logits(logits: torch.Tensor, prep: Prepared) -> dict:
    """Multiple-choice prediction restricted to the option letters."""
    lg = logits[0, -1].float()[prep.letter_ids]
    p = torch.softmax(lg, -1)
    pred = int(p.argmax())
    return {"pred": pred, "correct": pred == prep.sample.answer_idx, "probs": p.tolist(),
            "p_correct": float(p[prep.sample.answer_idx])}
