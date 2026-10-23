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


@dataclass
class Prepared:
    sample: Sample
    inputs: dict
    image: Image.Image  # the resized image actually fed to the model
    grid_h: int
    grid_w: int
    image_pos: torch.Tensor  # sequence positions of the gh*gw image tokens (raster order)
    answer_pos: int  # last prompt position, whose logits give the answer letter
    coverage: np.ndarray  # [gh, gw] fraction of each token cell inside target box(es)
    object_pos: dict = field(default_factory=dict)  # object name -> positions of its tokens in the question
    letter_ids: list = field(default_factory=list)
    distractor_coverage: np.ndarray | None = None

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


def find_word_positions(tokenizer, input_ids: list[int], word: str, start: int) -> list[int]:
    """Positions (>= start) of the tokens spelling `word` (last occurrence in the prompt)."""
    text, spans = _token_char_spans(tokenizer, input_ids[start:])
    low = text.lower()
    for cand in (word.lower(), word.lower().split()[-1]):
        k = low.rfind(cand)
        if k >= 0:
            return [start + j for j, (a, b) in enumerate(spans) if a < k + len(cand) and b > k]
    return []


def letter_token_ids(tokenizer, n: int) -> list[int]:
    return [tokenizer.encode(LETTERS[i], add_special_tokens=False)[0] for i in range(n)]


def prepare(model, processor, sample: Sample, max_pixels: int = 1024 * 32 * 32,
            min_pixels: int = 256 * 32 * 32) -> Prepared:
    vcfg = model.config.vision_config
    factor = vcfg.patch_size * vcfg.spatial_merge_size
    orig = Image.open(sample.image_path).convert("RGB")
    img = resize_to_budget(orig, max_pixels, min_pixels, factor)

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": build_prompt(sample)}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to(model.device)

    thw = inputs["image_grid_thw"][0].tolist()
    gh, gw = thw[1] // vcfg.spatial_merge_size, thw[2] // vcfg.spatial_merge_size
    ids = inputs["input_ids"][0]
    image_pos = (ids == model.config.image_token_id).nonzero().squeeze(-1)
    assert image_pos.numel() == gh * gw, (image_pos.numel(), gh, gw)

    W, H = orig.size
    tok = processor.tokenizer
    id_list = ids.tolist()
    after_img = int(image_pos[-1]) + 1
    obj_pos = {o: find_word_positions(tok, id_list, o, after_img) for o in sample.target_objects}
    dcov = None
    if sample.meta.get("distractor_bboxes"):
        dcov = bbox_coverage_grid(sample.meta["distractor_bboxes"], W, H, gh, gw)
    return Prepared(
        sample=sample, inputs=inputs, image=img, grid_h=gh, grid_w=gw, image_pos=image_pos,
        answer_pos=len(id_list) - 1, coverage=bbox_coverage_grid(sample.bboxes, W, H, gh, gw),
        object_pos=obj_pos, letter_ids=letter_token_ids(tok, len(sample.options)), distractor_coverage=dcov,
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
