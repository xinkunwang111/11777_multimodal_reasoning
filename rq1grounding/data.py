"""Datasets for grounding probes: V* Bench and a synthetic attribute-binding set.

Every sample carries the ground-truth box(es) of the queried object(s), which is
what all probes need: we compare where the model "looks" / what it encodes at
those image tokens against everything else.
"""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import asdict, dataclass, field

import numpy as np

LETTERS = "ABCDEFGH"
PROMPT_SUFFIX = "Answer with the option's letter from the given choices directly."


@dataclass
class Sample:
    qid: str
    category: str  # direct_attributes / relative_position / synth_attr / synth_rel
    image_path: str
    question: str
    options: list[str]
    answer_idx: int
    target_objects: list[str]
    bboxes: list[list[float]]  # xywh in original image pixels, aligned with target_objects
    meta: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)


def build_prompt(sample: Sample) -> str:
    opts = "\n".join(f"({LETTERS[i]}) {o}" for i, o in enumerate(sample.options))
    return f"{sample.question}\n{opts}\n{PROMPT_SUFFIX}"


# ---------------------------------------------------------------------------
# V* Bench
# ---------------------------------------------------------------------------

VSTAR_REPO = "craigwu/vstar_bench"
VSTAR_CATEGORIES = ("direct_attributes", "relative_position")


def _as_box_list(b) -> list[list[float]]:
    if b is None:
        return []
    if len(b) == 4 and all(isinstance(v, (int, float)) for v in b):
        return [list(map(float, b))]
    return [list(map(float, x)) for x in b]


def load_vstar(root: str | None = None, categories=VSTAR_CATEGORIES, limit: int | None = None,
               seed: int = 0) -> list[Sample]:
    """Load V* Bench. Annotation json: {question, options (options[0] is correct), target_object, bbox (xywh)}.

    Options are shuffled deterministically so the correct answer is not always (A).
    """
    if root is None or not os.path.isdir(root):
        from huggingface_hub import snapshot_download

        root = snapshot_download(VSTAR_REPO, repo_type="dataset", local_dir=root)

    samples: list[Sample] = []
    for cat in categories:
        folder = os.path.join(root, cat)
        stems = sorted(f[:-5] for f in os.listdir(folder) if f.endswith(".json"))
        for stem in stems:
            ann = json.load(open(os.path.join(folder, stem + ".json")))
            img = next((os.path.join(folder, stem + ext) for ext in (".jpg", ".jpeg", ".png")
                        if os.path.exists(os.path.join(folder, stem + ext))), None)
            if img is None:
                continue
            options = list(ann["options"])
            rng = random.Random(f"{seed}-{cat}-{stem}")
            order = list(range(len(options)))
            rng.shuffle(order)
            objs = ann.get("target_object", [])
            objs = [objs] if isinstance(objs, str) else list(objs)
            samples.append(Sample(
                qid=f"{cat}/{stem}", category=cat, image_path=img, question=ann["question"],
                options=[options[i] for i in order], answer_idx=order.index(0),
                target_objects=objs, bboxes=_as_box_list(ann.get("bbox")),
            ))
    if limit:
        samples = samples[:limit] if len(categories) == 1 else _balanced_limit(samples, limit)
    return samples


def _balanced_limit(samples: list[Sample], limit: int) -> list[Sample]:
    by_cat: dict[str, list[Sample]] = {}
    for s in samples:
        by_cat.setdefault(s.category, []).append(s)
    per = max(1, limit // len(by_cat))
    return [s for v in by_cat.values() for s in v[:per]]


def load_jsonl(path: str, categories=None, limit: int | None = None) -> list[Sample]:
    base = os.path.dirname(os.path.abspath(path))
    out = []
    for line in open(path):
        d = json.loads(line)
        if categories and d["category"] not in categories:
            continue
        if not os.path.isabs(d["image_path"]):
            d["image_path"] = os.path.join(base, d["image_path"])
        out.append(Sample(**d))
    return out[:limit] if limit else out


def load_dataset(name: str, root: str | None, categories=None, limit=None) -> list[Sample]:
    if name == "vstar":
        return load_vstar(root, categories or VSTAR_CATEGORIES, limit)
    if name == "synthetic":
        return load_jsonl(os.path.join(root, "samples.jsonl"), categories, limit)
    if name.endswith(".jsonl"):
        return load_jsonl(name, categories, limit)
    raise ValueError(name)


# ---------------------------------------------------------------------------
# Geometry: boxes -> merged-token grid
# ---------------------------------------------------------------------------

def bbox_coverage_grid(bboxes, img_w: int, img_h: int, grid_h: int, grid_w: int) -> np.ndarray:
    """Fraction of each visual token's cell covered by the union of the boxes, shape [grid_h, grid_w].

    Works in normalized coordinates, so it is valid regardless of how the image
    was resized before patchification (the resize maps image -> full canvas).
    """
    cov = np.zeros((grid_h, grid_w), dtype=np.float32)
    xs = np.linspace(0, 1, grid_w + 1)
    ys = np.linspace(0, 1, grid_h + 1)
    for x, y, w, h in bboxes:
        x0, x1 = np.clip([x / img_w, (x + w) / img_w], 0, 1)
        y0, y1 = np.clip([y / img_h, (y + h) / img_h], 0, 1)
        ox = np.clip(np.minimum(xs[1:], x1) - np.maximum(xs[:-1], x0), 0, None) * grid_w
        oy = np.clip(np.minimum(ys[1:], y1) - np.maximum(ys[:-1], y0), 0, None) * grid_h
        cov = np.maximum(cov, np.outer(oy, ox))
    return cov


# ---------------------------------------------------------------------------
# Option helpers
# ---------------------------------------------------------------------------

_WORD = re.compile(r"[a-zA-Z]+")
MIRROR = {"left": "right", "right": "left", "above": "below", "below": "above", "top": "bottom",
          "bottom": "top"}


def option_keywords(options: list[str]) -> list[str]:
    """The word that distinguishes each option ("The cup is on the left" -> "left")."""
    toks = [[w.lower() for w in _WORD.findall(o)] for o in options]
    common = set.intersection(*map(set, toks)) if toks and all(toks) else set()
    out = []
    for t in toks:
        diff = [w for w in t if w not in common]
        out.append(diff[0] if diff else (t[-1] if t else ""))
    return out


def mirror_option_idx(options: list[str], answer_idx: int, axis: str) -> int | None:
    """Index of the option a model should switch to if it reads the scene mirrored along `axis`."""
    kws = option_keywords(options)
    pair = {"h": ("left", "right"), "v": ("above", "below", "top", "bottom")}[axis]
    a = kws[answer_idx]
    if a not in pair or MIRROR[a] not in kws:
        return None
    return kws.index(MIRROR[a])
