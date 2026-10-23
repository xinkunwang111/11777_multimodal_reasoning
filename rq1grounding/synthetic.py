"""Controlled attribute-binding / spatial-relation images (the "red cube, blue sphere" setting of RQ1).

Each image has 2-4 shapes, each with a unique shape and a unique color, so a
wrong color answer can be classified as
  * binding error   - the model reported the color of *another object in the image*
  * hallucination   - the model reported a color that is not in the image at all.
Color-swapped twins (same layout, target/distractor colors exchanged) share a
`pair_id`, so binding consistency can be measured on minimal pairs.
"""

from __future__ import annotations

import json
import os
import random

from PIL import Image, ImageDraw

from .data import Sample

COLORS = {
    "red": (220, 30, 30), "blue": (30, 70, 220), "green": (30, 160, 50), "yellow": (240, 210, 20),
    "purple": (130, 40, 170), "orange": (245, 130, 20), "black": (20, 20, 20), "pink": (240, 120, 180),
    "brown": (120, 70, 30), "gray": (128, 128, 128),
}
SHAPES = ("square", "circle", "triangle", "diamond")


def _draw(d: ImageDraw.ImageDraw, shape: str, box, color):
    x, y, w, h = box
    if shape == "square":
        d.rectangle([x, y, x + w, y + h], fill=color)
    elif shape == "circle":
        d.ellipse([x, y, x + w, y + h], fill=color)
    elif shape == "triangle":
        d.polygon([(x + w / 2, y), (x, y + h), (x + w, y + h)], fill=color)
    elif shape == "diamond":
        d.polygon([(x + w / 2, y), (x + w, y + h / 2), (x + w / 2, y + h), (x, y + h / 2)], fill=color)


def _place(rng, n, W, H, size_range, tries=500):
    boxes = []
    for _ in range(tries):
        if len(boxes) == n:
            break
        s = int(rng.uniform(*size_range) * min(W, H))
        x, y = rng.randint(0, W - s), rng.randint(0, H - s)
        if all(x + s + 8 < bx or bx + bs + 8 < x or y + s + 8 < by or by + bs + 8 < y for bx, by, bs, _ in boxes):
            boxes.append((x, y, s, s))
    return boxes if len(boxes) == n else None


def _render(path, W, H, objs, rng, clutter):
    img = Image.new("RGB", (W, H), (235, 235, 230))
    d = ImageDraw.Draw(img)
    for _ in range(clutter):  # faint background clutter so the task is not trivially pop-out
        cx, cy, r = rng.randint(0, W), rng.randint(0, H), rng.randint(2, 6)
        g = rng.randint(190, 225)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(g, g, g))
    for o in objs:
        _draw(d, o["shape"], o["box"], COLORS[o["color"]])
    img.save(path)


def generate(out_dir: str, n_images: int = 300, size=(1024, 768), size_range=(0.04, 0.12),
             n_obj_range=(2, 4), n_options=4, clutter=150, swap_pairs=True, seed=0) -> list[Sample]:
    rng = random.Random(seed)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    W, H = size
    samples: list[Sample] = []
    for i in range(n_images):
        n = rng.randint(*n_obj_range)
        boxes = _place(rng, n, W, H, size_range)
        if boxes is None:
            continue
        shapes = rng.sample(SHAPES, n)
        colors = rng.sample(list(COLORS), n)
        objs = [{"shape": s, "color": c, "box": [float(v) for v in b]} for s, c, b in zip(shapes, colors, boxes)]

        # --- attribute (binding) question, optionally with a color-swapped twin
        t, dist = rng.sample(range(n), 2)
        variants = [objs]
        if swap_pairs:
            swapped = [dict(o) for o in objs]
            swapped[t]["color"], swapped[dist]["color"] = objs[dist]["color"], objs[t]["color"]
            variants.append(swapped)
        absent_pool = [c for c in COLORS if c not in colors]
        absent = rng.sample(absent_pool, max(0, n_options - n))
        for v, vobjs in enumerate(variants):
            path = f"images/{i:05d}_v{v}.png"
            _render(os.path.join(out_dir, path), W, H, vobjs, random.Random(f"{seed}-{i}"), clutter)
            in_image = [o["color"] for o in vobjs]
            opts = (in_image + absent)[:max(n_options, 2)]
            if vobjs[t]["color"] not in opts:
                opts[-1] = vobjs[t]["color"]
            random.Random(f"{seed}-{i}-opt").shuffle(opts)
            src = {}
            for k, c in enumerate(opts):
                owner = next((j for j, o in enumerate(vobjs) if o["color"] == c), None)
                src[str(k)] = "target" if owner == t else ("in_image" if owner is not None else "absent")
            samples.append(Sample(
                qid=f"synth_attr/{i:05d}_v{v}", category="synth_attr", image_path=path,
                question=f"What is the color of the {vobjs[t]['shape']}?", options=opts,
                answer_idx=opts.index(vobjs[t]["color"]), target_objects=[vobjs[t]["shape"]],
                bboxes=[vobjs[t]["box"]],
                meta={"pair_id": f"{i:05d}", "option_source": src,
                      "distractor_objects": [o["shape"] for j, o in enumerate(vobjs) if j != t],
                      "distractor_bboxes": [o["box"] for j, o in enumerate(vobjs) if j != t],
                      "objects": vobjs},
            ))

        # --- spatial relation question on the un-swapped image
        a, b = rng.sample(range(n), 2)
        (ax, ay, aw, ah), (bx, by, bw, bh) = objs[a]["box"], objs[b]["box"]
        dx, dy = (ax + aw / 2) - (bx + bw / 2), (ay + ah / 2) - (by + bh / 2)
        if abs(dx) >= abs(dy):
            q = f"Is the {objs[a]['shape']} on the left or right side of the {objs[b]['shape']}?"
            opts, ans = ["left", "right"], 0 if dx < 0 else 1
        else:
            q = f"Is the {objs[a]['shape']} above or below the {objs[b]['shape']}?"
            opts, ans = ["above", "below"], 0 if dy < 0 else 1
        if rng.random() < 0.5:
            opts, ans = opts[::-1], 1 - ans
        samples.append(Sample(
            qid=f"synth_rel/{i:05d}", category="synth_rel", image_path=f"images/{i:05d}_v0.png",
            question=q, options=opts, answer_idx=ans,
            target_objects=[objs[a]["shape"], objs[b]["shape"]], bboxes=[objs[a]["box"], objs[b]["box"]],
            meta={"objects": objs},
        ))

    with open(os.path.join(out_dir, "samples.jsonl"), "w") as f:
        for s in samples:
            f.write(json.dumps(s.to_json()) + "\n")
    return samples
