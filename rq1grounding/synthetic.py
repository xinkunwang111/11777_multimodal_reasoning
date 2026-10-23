"""Controlled binding scenes ("red cube next to a blue sphere"), with GT boxes in the V* format.

Each scene has 6-10 small shapes. Shapes and colours repeat across objects, except:
  * the target's shape is unique (so "the square" is well defined) and no other object has its colour;
  * a *neighbour* of a different shape and colour is placed right next to the target - the classic
    setting for attribute leakage.
Attribute question: "What is the color of the <target>?" with 4 options whose source is known:
  target | neighbor (binding error) | in_image (another object's colour) | absent (hallucination).
Every attribute scene also has a colour-swapped twin (target and neighbour exchange colours, same
layout and same options, so the answer must change) sharing a `pair_id`.
Relation question: "Is the <target> on the left or right side of the <reference>?" (or above /
below), where the reference shape is also unique; objects = [target, reference] as in V*.
"""

from __future__ import annotations

import json
import math
import os
import random

from PIL import Image, ImageDraw

from .data import GTObject, Sample

COLORS = {
    "red": (220, 30, 30), "blue": (30, 70, 220), "green": (30, 160, 50), "yellow": (240, 210, 20),
    "purple": (130, 40, 170), "orange": (245, 130, 20), "pink": (240, 120, 180), "brown": (120, 70, 30),
    "gray": (128, 128, 128), "black": (20, 20, 20),
}
SHAPES = ("square", "circle", "triangle", "diamond", "star", "cross")


def _polygon(shape, x, y, s):
    if shape == "triangle":
        return [(x + s / 2, y), (x, y + s), (x + s, y + s)]
    if shape == "diamond":
        return [(x + s / 2, y), (x + s, y + s / 2), (x + s / 2, y + s), (x, y + s / 2)]
    if shape == "star":
        pts = []
        for k in range(10):
            r = s / 2 if k % 2 == 0 else s / 5
            a = -math.pi / 2 + k * math.pi / 5
            pts.append((x + s / 2 + r * math.cos(a), y + s / 2 + r * math.sin(a)))
        return pts
    if shape == "cross":
        t = s / 3
        return [(x + t, y), (x + 2 * t, y), (x + 2 * t, y + t), (x + s, y + t), (x + s, y + 2 * t), (x + 2 * t, y + 2 * t),
                (x + 2 * t, y + s), (x + t, y + s), (x + t, y + 2 * t), (x, y + 2 * t), (x, y + t), (x + t, y + t)]
    raise ValueError(shape)


def _draw(d: ImageDraw.ImageDraw, shape: str, box_px, color):
    x, y, s = box_px[0], box_px[1], box_px[2] - box_px[0]
    if shape == "square":
        d.rectangle([x, y, x + s, y + s], fill=color)
    elif shape == "circle":
        d.ellipse([x, y, x + s, y + s], fill=color)
    else:
        d.polygon(_polygon(shape, x, y, s), fill=color)


def _free(box, placed, gap):
    return all(box[2] + gap < b[0] or b[2] + gap < box[0] or box[3] + gap < b[1] or b[3] + gap < box[1] for b in placed)


def _layout(rng, W, H, n, size_range, tries=2000):
    """Pixel boxes: [target, neighbour, others...]; the neighbour sits next to the target."""
    smin = min(W, H)
    for _ in range(50):
        s = rng.uniform(*size_range) * smin
        t = (rng.uniform(0, W - s), rng.uniform(0, H - s))
        boxes = [(t[0], t[1], t[0] + s, t[1] + s)]
        sn = rng.uniform(*size_range) * smin
        a = rng.uniform(0, 2 * math.pi)
        dist = (s + sn) / 2 + rng.uniform(0.2, 0.8) * min(s, sn)
        cx, cy = t[0] + s / 2 + dist * math.cos(a), t[1] + s / 2 + dist * math.sin(a)
        nb = (cx - sn / 2, cy - sn / 2, cx + sn / 2, cy + sn / 2)
        if nb[0] < 0 or nb[1] < 0 or nb[2] > W or nb[3] > H or not _free(nb, boxes, 2):
            continue
        boxes.append(nb)
        for _ in range(tries):
            if len(boxes) == n:
                return boxes
            so = rng.uniform(*size_range) * smin
            x, y = rng.uniform(0, W - so), rng.uniform(0, H - so)
            b = (x, y, x + so, y + so)
            if _free(b, boxes, 0.3 * so):
                boxes.append(b)
    return None


def _render(path, W, H, objs, seed, clutter):
    rng = random.Random(seed)
    img = Image.new("RGB", (W, H), (235, 235, 230))
    d = ImageDraw.Draw(img)
    for _ in range(clutter):  # faint background specks so the scene is not trivially pop-out
        cx, cy, r, g = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(1.5, 5), rng.randint(185, 225)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(g, g, g))
    for o in objs:
        _draw(d, o["shape"], o["px"], COLORS[o["color"]])
    img.save(path)


def generate(out_dir: str, n_images: int = 300, size=(1024, 768), n_objects=(6, 10), obj_size=(0.03, 0.07),
             clutter=120, swap_pairs: bool = True, seed: int = 0) -> list[Sample]:
    rng = random.Random(seed)
    W, H = size
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    samples: list[Sample] = []
    for i in range(n_images):
        n = rng.randint(*n_objects)
        px = _layout(rng, W, H, n, obj_size)
        if px is None:
            continue
        t_shape, r_shape = rng.sample(SHAPES, 2)  # unique target and reference shapes
        rest_shapes = [s for s in SHAPES if s not in (t_shape, r_shape)]
        t_color, n_color = rng.sample(list(COLORS), 2)
        objs = [{"shape": t_shape, "color": t_color}, {"shape": rng.choice(rest_shapes), "color": n_color}]
        ref_idx = rng.randrange(2, n)  # one of the "others" becomes the unique reference shape
        for k in range(2, n):
            shape = r_shape if k == ref_idx else rng.choice(rest_shapes)
            objs.append({"shape": shape, "color": rng.choice([c for c in COLORS if c != t_color])})
        for o, b in zip(objs, px):
            o["px"] = b
            o["box"] = (b[0] / W, b[1] / H, b[2] / W, b[3] / H)

        # ---- attribute question + colour-swapped twin (same options, answer changes)
        others = [o["color"] for o in objs[2:] if o["color"] not in (t_color, n_color)]
        other_c = rng.choice(others) if others else None
        absent = [c for c in COLORS if c not in {o["color"] for o in objs}]
        opts = [t_color, n_color] + ([other_c] if other_c else [])
        opts += rng.sample(absent, min(len(absent), 4 - len(opts)))
        opts += [c for c in others if c not in opts][: 4 - len(opts)]
        rng.shuffle(opts)
        variants = [objs]
        if swap_pairs:
            sw = [dict(o) for o in objs]
            sw[0]["color"], sw[1]["color"] = n_color, t_color
            variants.append(sw)
        for v, vo in enumerate(variants):
            path = f"images/{i:05d}_v{v}.png"
            _render(os.path.join(out_dir, path), W, H, vo, f"{seed}-{i}", clutter)
            src = {}
            for k, c in enumerate(opts):
                src[str(k)] = ("target" if c == vo[0]["color"] else "neighbor" if c == vo[1]["color"]
                               else "in_image" if c in {o["color"] for o in vo} else "absent")
            samples.append(Sample(
                qid=f"synth_attr/{i:05d}_v{v}", category="synth_attr", image_path=path,
                question=f"What is the color of the {t_shape}?", options=list(opts),
                answer_idx=opts.index(vo[0]["color"]), objects=[GTObject(t_shape, [vo[0]["box"]])], image_size=(W, H),
                meta={"pair_id": f"{i:05d}", "option_source": src,
                      "neighbor": {"name": vo[1]["shape"], "box": vo[1]["box"], "color": vo[1]["color"]},
                      "other_boxes": [o["box"] for o in vo[1:]]},
            ))

        # ---- relation question between the two unique shapes (on the un-swapped image)
        ref = objs[ref_idx]
        (tx, ty), (rx, ry) = [((o["box"][0] + o["box"][2]) / 2, (o["box"][1] + o["box"][3]) / 2) for o in (objs[0], ref)]
        dx, dy = tx - rx, ty - ry
        if max(abs(dx), abs(dy)) < 0.05:
            continue
        if abs(dx) >= abs(dy):
            q, opts_r, ans = f"Is the {t_shape} on the left or right side of the {r_shape}?", ["left", "right"], int(dx > 0)
        else:
            q, opts_r, ans = f"Is the {t_shape} above or below the {r_shape}?", ["above", "below"], int(dy > 0)
        if rng.random() < 0.5:
            opts_r, ans = opts_r[::-1], 1 - ans
        samples.append(Sample(
            qid=f"synth_rel/{i:05d}", category="synth_rel", image_path=f"images/{i:05d}_v0.png", question=q,
            options=opts_r, answer_idx=ans,
            objects=[GTObject(t_shape, [objs[0]["box"]]), GTObject(r_shape, [ref["box"]])], image_size=(W, H),
            meta={"pair_id": f"{i:05d}",  # same image as the attribute twins -> same side of every split
                  "other_boxes": [o["box"] for k, o in enumerate(objs) if k not in (0, ref_idx)]},
        ))

    with open(os.path.join(out_dir, "samples.jsonl"), "w") as f:
        for s in samples:
            f.write(json.dumps(s.to_json()) + "\n")
    return samples
