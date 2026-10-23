"""Method - "look where you found it": self-grounded attention steering vs. other ways to pass the box.

Per question, two fixed steps (no training, no tools, no agent loop):
  1. find    the model boxes each question object itself (explicit grounding, same image)
  2. answer  the original question on the original image, while in the reading layers (probe 2) the text
             tokens' attention is moved onto the visual tokens inside that box: shift(lambda) moves a
             fraction lambda of the attention on the rest of the image onto the box; text attention and
             total visual attention are unchanged.

The comparison is a grid:
  how the box is passed   base (none) | text | draw | crop | steer (ours)
                          + base_hi: full image at the base + crop token budget (equal-compute reference for crop)
  where the box is from   self (explicit grounding) | attn (probe-1 grounding heads) | gt (upper bound)
"text works as well as steer" -> telling the model in words suffices; "steer works, text does not" -> the
model's text-side coordinates are not wired to its visual read-out (a cross-modal alignment gap).

Steer hyperparameters (layer set, lambda) are chosen on the calibration split (self boxes by default,
--calib_source gt); candidate layer sets are probe-2's reading windows computed on that same split
(+ middle third, all layers). When a source gives no box the question falls back to the base answer
(flagged no_box; its rate is in box_quality.csv).

Outputs: calibration.json, per_sample.jsonl, summary.json, summary.csv (accuracy: mode x source),
         summary_by_category.csv, box_quality.csv, by_found.csv
"""

import csv
import json
import os
import sys

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rq1grounding import geometry as G  # noqa: E402
from rq1grounding.attention import AttentionScaler  # noqa: E402
from rq1grounding.cli import (add_split_args, common_parser, load_p1_config, load_samples, nanmean, reading_layers,  # noqa: E402
                              setup, to_jsonable, write_jsonl)
from rq1grounding.cues import MODES, crop_cue, draw_cue, steer_edit, text_cue  # noqa: E402
from rq1grounding.grounding import aggregate, ground_explicit, map_to_box, record_attention  # noqa: E402
from rq1grounding.model import answer_from_logits, forward, prepare  # noqa: E402

SOURCES = ("self", "attn", "gt")


def get_boxes(model, processor, prep, source, cfg, args) -> list:
    """One list of normalized boxes per question object ([] when the source gives none)."""
    s = prep.sample
    if source == "gt":
        return [list(o.boxes) for o in s.objects]
    if source == "self":
        out = []
        for o in s.objects:
            b = ground_explicit(model, processor, prep.image, o.name, args.max_new_tokens)["box"]
            out.append([b] if b is not None else [])
        return out
    if source == "attn":
        am = record_attention(model, prep, cfg["sink_k"])
        out = []
        for k in range(len(s.objects)):
            A = am.maps.get(f"object:{k}")
            M = aggregate(A, heads=cfg["heads"], sinks=am.sinks) if A is not None else None
            out.append([map_to_box(M, prep.grid_h, prep.grid_w, cfg["box_threshold"])] if M is not None else [])
        return out
    raise ValueError(source)


def answer(model, prep, edit=None):
    if edit is None:
        return answer_from_logits(forward(model, prep).logits, prep)
    with AttentionScaler(model, [edit]):
        return answer_from_logits(forward(model, prep).logits, prep)


def run_mode(model, processor, prep, orig, mode, boxes, steer, args) -> dict | None:
    s = prep.sample
    if not any(boxes):
        return None
    if mode == "text":
        p = prepare(model, processor, s, args.max_pixels, args.min_pixels, image=prep.image, prefix=text_cue(s, boxes))
        return {**answer(model, p), "tokens": p.n_img}
    if mode == "draw":
        img, prefix = draw_cue(prep.image, s, boxes)
        p = prepare(model, processor, s, args.max_pixels, args.min_pixels, image=img, prefix=prefix)
        return {**answer(model, p), "tokens": p.n_img}
    if mode == "crop":
        crop, prefix, _ = crop_cue(model, orig, s, boxes, args.crop_pixels, args.crop_margin, args.crop_min_side)
        p = prepare(model, processor, s, args.max_pixels, args.min_pixels, image=prep.image, prefix=prefix, extra_images=[crop])
        return {**answer(model, p), "tokens": sum(sp.grid_h * sp.grid_w for sp in p.spans)}
    if mode == "steer":
        edit = steer_edit(prep, boxes, steer["layers"], args.steer_mode, steer["value"], args.pad)
        return {**answer(model, prep, edit), "tokens": prep.n_img}
    raise ValueError(mode)


def layer_candidates(args, n_layers, calib_qids) -> dict:
    cands = {"mid_third": list(range(n_layers // 3, 2 * n_layers // 3)), "all": list(range(n_layers))}
    for top in (1, 2):
        layers, _ = reading_layers(args.p2, calib_qids, args.p2_group, top)
        if layers:
            cands[f"p2_top{top}"] = layers
    return cands


def calibrate(model, processor, calib, args, n_layers) -> dict:
    cands = layer_candidates(args, n_layers, {s.qid for s in calib})
    grid = {(name, v): [] for name in cands for v in args.lambdas}
    for s in tqdm(calib, desc="calibrate steer"):
        prep = prepare(model, processor, s, args.max_pixels, args.min_pixels)
        boxes = get_boxes(model, processor, prep, args.calib_source, None, args)
        for (name, v), res in grid.items():
            r = run_mode(model, processor, prep, None, "steer", boxes, {"layers": cands[name], "value": v}, args)
            r = r or answer(model, prep)
            res.append((r["correct"], r["logp_correct"]))
    scores = {k: (nanmean([c for c, _ in v]), nanmean([lp for _, lp in v])) for k, v in grid.items()} if calib else {}
    if scores:
        best = max(scores, key=lambda k: scores[k])
    else:  # no calibration data: default strength; probe-2 windows are only used when calibration qids exist
        best = ("p2_top1" if "p2_top1" in cands else "mid_third", 0.5)
    return {"layers": cands[best[0]], "layer_set": best[0], "value": best[1], "mode": args.steer_mode,
            "candidates": cands, "grid": [{"layer_set": k[0], "value": k[1], "acc": a, "mean_logp": lp}
                                          for k, (a, lp) in scores.items()]}


def write_csv(path, rows):
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader()
            w.writerows(to_jsonable(rows))


def main():
    p = common_parser(__doc__)
    add_split_args(p)
    p.add_argument("--p1", default=None, help="probe-1 output dir (grounding heads for the attn source)")
    p.add_argument("--p2", default=None, help="probe-2 output dir (reading layers)")
    p.add_argument("--p2_group", default="all")
    p.add_argument("--sources", nargs="*", default=list(SOURCES), choices=SOURCES)
    p.add_argument("--modes", nargs="*", default=list(MODES), choices=MODES)
    p.add_argument("--steer_mode", default="shift", choices=["shift", "scale"])
    p.add_argument("--lambdas", type=float, nargs="*", default=[0.25, 0.5, 0.75],
                   help="steer strengths tried on the calibration split (shift: fraction; scale: factor)")
    p.add_argument("--calib_source", default="self", choices=["self", "gt"],
                   help="boxes used to tune steer: self = as deployed (default); gt = cleaner signal if self-grounding often fails")
    p.add_argument("--pad", type=int, default=0, help="dilate the steered region by this many token cells")
    p.add_argument("--crop_pixels", type=int, default=256 * 32 * 32, help="token budget of the zoomed crop")
    p.add_argument("--crop_margin", type=float, default=0.5)
    p.add_argument("--crop_min_side", type=float, default=0.15)
    p.add_argument("--max_new_tokens", type=int, default=128)
    args = p.parse_args()
    calib, test = load_samples(args)
    model, processor = setup(args)
    cfg = load_p1_config(args.p1)
    sources = [x for x in args.sources if x != "attn" or cfg]
    if "attn" in args.sources and not cfg:
        print("no --p1 config: skipping the attn box source")
    print(f"calibration {len(calib)} / test {len(test)} questions")

    steer = calibrate(model, processor, calib, args, model.config.text_config.num_hidden_layers) if "steer" in args.modes else None
    json.dump(to_jsonable(steer), open(os.path.join(args.out, "calibration.json"), "w"), indent=2)
    if steer:
        print(f"steer: layers {steer['layer_set']} {steer['layers']}, {args.steer_mode} {steer['value']}")

    budget = args.max_pixels + args.crop_pixels
    rows = []
    for s in tqdm(test, desc="test"):
        orig = Image.open(s.image_path).convert("RGB")
        prep = prepare(model, processor, s, args.max_pixels, args.min_pixels, image=orig)
        hi = prepare(model, processor, s, budget, budget, image=orig)
        res = {"base": {**answer(model, prep), "tokens": prep.n_img}, "base_hi": {**answer(model, hi), "tokens": hi.n_img}}
        boxes, quality = {}, {}
        for src in sources:
            boxes[src] = get_boxes(model, processor, prep, src, cfg, args)
            quality[src] = [G.match(b[0] if b else None, o.boxes) for b, o in zip(boxes[src], s.objects)]
            for mode in args.modes:
                r = run_mode(model, processor, prep, orig, mode, boxes[src], steer, args)
                res[f"{mode}|{src}"] = r if r is not None else {**res["base"], "no_box": True}
        found = all(q["center_hit"] for q in quality.get("self", [{"center_hit": False}]))
        rows.append({"qid": s.qid, "category": s.category, "results": res, "boxes": boxes, "box_quality": quality,
                     "self_found": found})
    write_jsonl(os.path.join(args.out, "per_sample.jsonl"), rows)

    # ---- accuracy grid: mode x source
    def acc(key, rs):
        return nanmean([r["results"][key]["correct"] for r in rs if key in r["results"]])

    table, by_cat = [], []
    cats = sorted({r["category"] for r in rows})
    for mode in ["base", "base_hi", *args.modes]:
        keys = {src: (mode if mode in ("base", "base_hi") else f"{mode}|{src}") for src in sources}
        line = {"mode": mode, **{src: acc(k, rows) for src, k in keys.items()},
                "mean_tokens": nanmean([r["results"][k]["tokens"] for r in rows for k in keys.values()
                                        if not r["results"][k].get("no_box")])}
        table.append(line)
        for c in cats:
            by_cat.append({"category": c, "mode": mode, **{src: acc(k, [r for r in rows if r["category"] == c])
                                                           for src, k in keys.items()}})
    write_csv(os.path.join(args.out, "summary.csv"), table)
    write_csv(os.path.join(args.out, "summary_by_category.csv"), by_cat)
    write_csv(os.path.join(args.out, "box_quality.csv"), [
        {"source": src, "center_hit": nanmean([q["center_hit"] for r in rows for q in r["box_quality"][src]]),
         "iou": nanmean([q["iou"] for r in rows for q in r["box_quality"][src]]),
         "no_box": nanmean([not any(r["boxes"][src]) for r in rows])} for src in sources])
    found_rows = []
    for f in (True, False):
        rs = [r for r in rows if r["self_found"] == f]
        found_rows.append({"self_found": f, "n": len(rs), "base": acc("base", rs),
                           **{m: acc(f"{m}|self", rs) for m in args.modes if "self" in sources}})
    write_csv(os.path.join(args.out, "by_found.csv"), found_rows)
    json.dump(to_jsonable({"n": len(rows), "steer": steer, "table": table, "by_found": found_rows}),
              open(os.path.join(args.out, "summary.json"), "w"), indent=2)
    for t in table:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in t.items()})


if __name__ == "__main__":
    main()
