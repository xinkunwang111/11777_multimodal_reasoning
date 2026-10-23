"""Samples with ground-truth object boxes: V* Bench and the synthetic binding set.

Every sample lists the objects its question is about (`objects`, in question order), each with
one or more GT boxes in normalized xyxy (see geometry.py). For V*:
  direct_attributes   one object ("What is the color of the glove?")
  relative_position   two objects, [subject, reference] ("Is the cup on the left or right of the dog?")

V* sources (craigwu/vstar_bench, downloaded on first use):
  <category>/<stem>.json  {"target_object": [names], "bbox": [[x, y, w, h], ...] (original pixels),
                           "question": str, "options": [correct, ...]}
  test_questions.jsonl    official prompts; if present its option order / answer letter is used so
                          accuracies are comparable with lmms-eval, otherwise options are shuffled
                          deterministically (in the json the correct option is always first).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from dataclasses import asdict, dataclass, field

from PIL import Image

from .geometry import from_xywh_px

LETTERS = "ABCDEFGH"
PROMPT_SUFFIX = "Answer with the option's letter from the given choices directly."
VSTAR_REPO = "craigwu/vstar_bench"
VSTAR_CATEGORIES = ("direct_attributes", "relative_position")


@dataclass
class GTObject:
    name: str
    boxes: list  # normalized xyxy, one per annotated instance


@dataclass
class Sample:
    qid: str
    category: str
    image_path: str
    question: str
    options: list
    answer_idx: int
    objects: list  # list[GTObject], in question order
    image_size: tuple = (0, 0)  # original (W, H)
    meta: dict = field(default_factory=dict)

    @property
    def boxes(self) -> list:
        """All GT boxes of all question objects."""
        return [b for o in self.objects for b in o.boxes]

    @property
    def has_gt(self) -> bool:
        return bool(self.boxes)

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(d: dict) -> "Sample":
        d = dict(d)
        d["objects"] = [GTObject(o["name"], [tuple(b) for b in o["boxes"]]) for o in d.get("objects", [])]
        d["image_size"] = tuple(d.get("image_size", (0, 0)))
        return Sample(**d)


def build_prompt(sample: Sample, prefix: str = "") -> str:
    opts = "\n".join(f"({LETTERS[i]}) {o}" for i, o in enumerate(sample.options))
    return f"{prefix}{sample.question}\n{opts}\n{PROMPT_SUFFIX}"


# ---------------------------------------------------------------------------- V*

def _gt_objects(names, raw_boxes, W, H) -> tuple[list[GTObject], str | None]:
    names = [names] if isinstance(names, str) else list(names or [])
    raw = raw_boxes or []
    if len(raw) == 4 and all(isinstance(v, (int, float)) for v in raw):
        raw = [raw]
    boxes = [from_xywh_px(b, W, H) for b in raw]
    warn = None
    if len(names) == len(boxes):
        pairs = list(zip(names, boxes))
    elif len(names) == 1:
        pairs = [(names[0], b) for b in boxes]
    else:  # unexpected: pair in order, extra boxes go to the last name
        warn = f"{len(names)} names vs {len(boxes)} boxes"
        pairs = [(names[min(i, len(names) - 1)] if names else "object", b) for i, b in enumerate(boxes)]
    objs: dict[str, GTObject] = {}
    for n, b in pairs:  # identical names = instances of one object (the question cannot tell them apart)
        objs.setdefault(n, GTObject(n, []))
        if b is not None:
            objs[n].boxes.append(b)
    return list(objs.values()), warn


def _official_order(root: str) -> dict:
    """image relative path -> (options, answer_idx) from test_questions.jsonl, if available."""
    path = os.path.join(root, "test_questions.jsonl")
    if not os.path.exists(path):
        return {}
    out = {}
    for line in open(path):
        d = json.loads(line)
        opts = [m.group(2).strip() for m in (re.match(r"^\(([A-H])\)\s*(.+)$", s.strip()) for s in d["text"].split("\n")) if m]
        if opts and d.get("label", "").strip() in LETTERS[: len(opts)]:
            out[d["image"]] = (opts, LETTERS.index(d["label"].strip()))
    return out


def load_vstar(root: str | None = "data/vstar", categories=VSTAR_CATEGORIES, limit: int | None = None,
               seed: int = 0, option_order: str = "auto") -> list[Sample]:
    """option_order: auto (official if test_questions.jsonl exists, else shuffle) | official | shuffle."""
    root = root or "data/vstar"
    if not os.path.isdir(os.path.join(root, VSTAR_CATEGORIES[0])):
        from huggingface_hub import snapshot_download

        snapshot_download(VSTAR_REPO, repo_type="dataset", local_dir=root)
    official = _official_order(root) if option_order in ("auto", "official") else {}
    samples = []
    for cat in categories:
        folder = os.path.join(root, cat)
        for stem in sorted(f[:-5] for f in os.listdir(folder) if f.endswith(".json")):
            img = next((f"{cat}/{stem}{e}" for e in (".jpg", ".jpeg", ".png") if os.path.exists(os.path.join(folder, stem + e))), None)
            if img is None:
                continue
            ann = json.load(open(os.path.join(folder, stem + ".json")))
            W, H = Image.open(os.path.join(root, img)).size
            objects, warn = _gt_objects(ann.get("target_object"), ann.get("bbox"), W, H)
            options, correct = list(ann["options"]), ann["options"][0]
            if img in official and correct in official[img][0]:
                options, answer = official[img]
                order = "official"
            else:
                rng = random.Random(f"{seed}-{cat}-{stem}")
                rng.shuffle(options)
                answer, order = options.index(correct), "shuffled"
            meta = {"option_order": order, **({"gt_warning": warn} if warn else {})}
            samples.append(Sample(f"{cat}/{stem}", cat, os.path.join(root, img), ann["question"], options, answer,
                                  objects, (W, H), meta))
    return _limit(samples, limit)


def _limit(samples, limit):
    if not limit:
        return samples
    by_cat: dict[str, list] = {}
    for s in samples:
        by_cat.setdefault(s.category, []).append(s)
    per = max(1, limit // len(by_cat))
    return [s for v in by_cat.values() for s in v[:per]]


def load_jsonl(path: str, categories=None, limit: int | None = None) -> list[Sample]:
    base = os.path.dirname(os.path.abspath(path))
    out = []
    for line in open(path):
        s = Sample.from_json(json.loads(line))
        if categories and s.category not in categories:
            continue
        if not os.path.isabs(s.image_path):
            s.image_path = os.path.join(base, s.image_path)
        out.append(s)
    return _limit(out, limit)


def load_dataset(name: str, root: str | None = None, categories=None, limit=None) -> list[Sample]:
    if name == "vstar":
        return load_vstar(root, categories or VSTAR_CATEGORIES, limit)
    if name == "synthetic":
        return load_jsonl(os.path.join(root or "data/synthetic", "samples.jsonl"), categories, limit)
    if name.endswith(".jsonl"):
        return load_jsonl(name, categories, limit)
    raise ValueError(f"unknown dataset {name}")


def split(samples: list[Sample], calib_frac: float) -> tuple[list[Sample], list[Sample]]:
    """Deterministic calibration / test split by qid hash (identical across scripts and runs).
    Colour-swapped synthetic twins share a pair_id and always land on the same side."""
    def h(s):
        key = s.meta.get("pair_id", s.qid)
        return int(hashlib.md5(key.encode()).hexdigest(), 16) % 10000 / 10000

    return [s for s in samples if h(s) < calib_frac], [s for s in samples if h(s) >= calib_frac]


# ---------------------------------------------------------------------------- options

_WORD = re.compile(r"[a-zA-Z]+")
MIRROR = {"left": "right", "right": "left", "above": "below", "below": "above", "top": "bottom", "bottom": "top"}


def option_keywords(options: list[str]) -> list[str]:
    """The word that distinguishes each option ("The cup is on the left of the dog" -> "left")."""
    toks = [[w.lower() for w in _WORD.findall(o)] for o in options]
    common = set.intersection(*map(set, toks)) if toks and all(toks) else set()
    out = []
    for t in toks:
        diff = [w for w in t if w not in common]
        out.append(diff[0] if diff else (t[-1] if t else ""))
    return out


def mirror_option_idx(options: list[str], answer_idx: int, axis: str) -> int | None:
    """Index of the option the answer should flip to if the scene were mirrored along axis 'h' or 'v'."""
    kws = option_keywords(options)
    pair = {"h": ("left", "right"), "v": ("above", "below", "top", "bottom")}[axis]
    a = kws[answer_idx]
    if a not in pair or MIRROR[a] not in kws:
        return None
    return kws.index(MIRROR[a])
