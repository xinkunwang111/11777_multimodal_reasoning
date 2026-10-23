"""Method - Self-Grounded Zoom (SGZ) with MRoPE-aligned re-injection, against baselines and controls.

Variants (accuracy per category, image tokens used):
  base                 single image at --max_pixels (N tokens)
  base_hi              single image resized to N + M tokens (equal-compute baseline for every zoom variant)
  <src>:crop           crop only (M tokens)
  <src>:zoom           [global, crop] as two ordinary images
  <src>:zoom_aligned   [global, crop] with crop tokens placed at their source region in MRoPE (ours)
Region sources <src>:
  sgz       calibrated grounding heads, object-phrase queries, sinks masked (ours; needs --config)
  attn_all  answer-token attention averaged over all layers/heads (prior-work style baseline)
  gt        ground-truth boxes (upper bound)        random   random box of the sgz size (control)
  grad      gradient x input map (with --grad)
Confidence gating is evaluated post hoc: zoom only when the base answer's probability < tau.

Outputs: summary.csv, crop_quality.csv, gating.csv, per_sample.jsonl
"""

import csv
import json
import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rq1grounding import zoom as Z  # noqa: E402
from rq1grounding.cli import common_parser, setup  # noqa: E402
from rq1grounding.maps import collect, grad_input_map  # noqa: E402
from rq1grounding.model import answer_from_logits, forward, prepare  # noqa: E402
from rq1grounding.viz import boxes_to_norm  # noqa: E402


def main():
    p = common_parser(__doc__)
    p.add_argument("--config", default=None, help="grounding_config.json from 05_grounding_bench.py (needed for sgz)")
    p.add_argument("--crop_pixels", type=int, default=1024 * 32 * 32, help="token budget M of the crop (in pixels)")
    p.add_argument("--sources", nargs="*", default=["sgz", "attn_all", "gt", "random"])
    p.add_argument("--modes", nargs="*", default=["crop", "zoom", "zoom_aligned"])
    p.add_argument("--margin", type=float, default=0.5)
    p.add_argument("--min_frac", type=float, default=0.2, help="min crop side as a share of the image's longer side")
    p.add_argument("--dt", type=float, default=1.0, help="temporal offset of the aligned crop")
    p.add_argument("--rule", default=None, help="binarization rule (default: the config's default_rule)")
    p.add_argument("--no_hint", action="store_true")
    p.add_argument("--grad", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    model, processor, samples = setup(args)
    cfg = json.load(open(args.config)) if args.config else None
    if cfg is None and "sgz" in args.sources:
        print("no --config: dropping the sgz source (run 05_grounding_bench.py first)")
        args.sources = [s for s in args.sources if s != "sgz"]
    sources = args.sources + (["grad"] if args.grad and "grad" not in args.sources else [])
    rng = np.random.default_rng(args.seed)

    rows = []
    for s in tqdm(samples):
        orig = Image.open(s.image_path).convert("RGB")
        W, H = orig.size
        prep = prepare(model, processor, s, args.max_pixels, args.min_pixels, image=orig)
        bundle = collect(model, prep)
        base = answer_from_logits(bundle.logits, prep)
        gt = [tuple(np.clip(b, 0, 1)) for b in boxes_to_norm(s.bboxes, W, H)]
        res = {"base": {**base, "tokens": prep.n_img}}

        budget = args.max_pixels + args.crop_pixels  # resized (up if needed) to N + M tokens: equal compute
        hi = prepare(model, processor, s, budget, budget, image=orig)
        res["base_hi"] = {**answer_from_logits(forward(model, hi).logits, hi), "tokens": hi.n_img}

        raw, sgz_query = {}, None
        if "sgz" in sources:
            raw["sgz"], sgz_query = Z.sgz_box(bundle, prep, cfg, args.rule)
        if "attn_all" in sources:
            raw["attn_all"] = Z.attn_all_box(bundle, prep, cfg, args.rule or "rel")
        if "gt" in sources and gt:
            raw["gt"] = Z.union_box(gt)
        if "grad" in sources:
            raw["grad"] = Z.region_from_map(grad_input_map(model, prep), prep.grid_h, prep.grid_w, "mass", 0.5)
        boxes = {k: Z.expand_box(v, W, H, args.margin, args.min_frac) for k, v in raw.items() if v is not None}
        if "random" in sources:
            ref = boxes.get("sgz") or boxes.get("attn_all") or Z.expand_box((0.45, 0.45, 0.55, 0.55), W, H, 0, args.min_frac)
            boxes["random"] = Z.random_box_like(ref, rng)
        quality = {k: Z.crop_quality(b, gt) for k, b in boxes.items()} if gt else {}

        for src, box in boxes.items():
            for mode in args.modes:
                if mode == "crop":
                    c = Z.crop(orig, box)
                    zp = prepare(model, processor, s, args.crop_pixels, args.crop_pixels // 2, image=c)
                    out, tokens = forward(model, zp), zp.n_img
                else:
                    zp = Z.zoom_inputs(model, processor, s, prep.image, orig, box, args.crop_pixels, hint=not args.no_hint)
                    pos = Z.aligned_position_ids(model, zp, box, args.dt) if mode == "zoom_aligned" else None
                    out, tokens = forward(model, zp, position_ids=pos), sum(sp.grid_h * sp.grid_w for sp in zp.spans)
                res[f"{src}:{mode}"] = {**answer_from_logits(out.logits, zp), "tokens": tokens}
        rows.append({"qid": s.qid, "category": s.category, "boxes": boxes, "crop_quality": quality,
                     "results": res, "image_size": [W, H], "sgz_query": sgz_query})

    out_dir = args.out
    with open(os.path.join(out_dir, "per_sample.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    variants = list(dict.fromkeys(v for r in rows for v in r["results"]))
    cats = sorted({r["category"] for r in rows}) + ["ALL"]
    table = []
    for v in variants:
        line = {"variant": v}
        for c in cats:
            sub = [r for r in rows if (c == "ALL" or r["category"] == c) and v in r["results"]]
            line[f"acc_{c}"] = round(float(np.mean([r["results"][v]["correct"] for r in sub])), 4) if sub else None
        line["mean_tokens"] = round(float(np.mean([r["results"][v]["tokens"] for r in rows if v in r["results"]])), 1)
        table.append(line)
    _write_csv(os.path.join(out_dir, "summary.csv"), table)

    srcs = sorted({k for r in rows for k in r["crop_quality"]})
    _write_csv(os.path.join(out_dir, "crop_quality.csv"), [
        {"source": k, **{m: round(float(np.mean([r["crop_quality"][k][m] for r in rows if k in r["crop_quality"]])), 4)
                         for m in ("center_in", "recall", "area")}} for k in srcs])

    gate = []
    for v in variants:
        if v in ("base", "base_hi"):
            continue
        for tau in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01):
            use = [max(r["results"]["base"]["probs"]) < tau for r in rows]
            acc = np.mean([r["results"][v]["correct"] if u else r["results"]["base"]["correct"] for r, u in zip(rows, use)])
            gate.append({"variant": v, "tau": tau, "zoom_rate": round(float(np.mean(use)), 4), "acc": round(float(acc), 4)})
    _write_csv(os.path.join(out_dir, "gating.csv"), gate)
    for t in table:
        print(t)


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
