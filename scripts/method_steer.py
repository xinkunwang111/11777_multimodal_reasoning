"""Method - amplify the attention the model already puts on the object (self-grounded attention steering).

Premise (probes 1 and 2): the object-name attention usually lands on the object even when the answer is
wrong, and the answer depends on the attention on the object. So, per question, two forward passes
(no training, no tools, no generation):
  1. normal pass  -> base answer + attention of the object-name tokens through the grounding heads
                     (probe 1) -> region = component around the attention peak (calibrated threshold)
  2. steered pass -> same input; in the reading layers (probe 2) a fraction lambda of the text tokens'
                     attention on the rest of the image is moved onto the region's visual tokens
                     (text attention and total visual attention unchanged; see rq1grounding/attention.py)

Comparison (summary.csv: rows = how the region is used, columns = where it comes from)
  rows     base | base_hi (full image at base + crop tokens: equal compute for crop) |
           steer (ours) | crop ([image, zoomed crop of the region], ViCrop-style input zoom) |
           optional: text (region coordinates in the prompt), draw (box drawn on the image)
  columns  attn (the model's own attention, ours) | gt (V* GT box: upper bound for the region)
by_found.csv splits the gain by whether the attention found the object (peak on the GT box): the method
should help where it did and cannot help where it did not.

Steer layers and lambda are chosen on the calibration split only; candidate layer sets are probe-2's
reading windows computed on those same calibration questions (+ middle third, all layers).

Outputs: calibration.json, per_sample.jsonl, summary.json, summary.csv, summary_by_category.csv,
         region_quality.csv, by_found.csv
"""

import json
import os
import sys

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rq1grounding import geometry as G  # noqa: E402
from rq1grounding.attention import AttentionScaler  # noqa: E402
from rq1grounding.cli import (add_split_args, append_jsonl, common_parser, load_p1_config, load_samples,
                              nanmean, read_jsonl, reading_layers, setup, to_jsonable, write_csv, write_jsonl)
from rq1grounding.cues import MODES, crop_cue, draw_cue, steer_edit, text_cue  # noqa: E402
from rq1grounding.grounding import aggregate, map_metrics, map_to_box, record_attention  # noqa: E402
from rq1grounding.model import answer_from_logits, forward, prepare  # noqa: E402

SOURCES = ("attn", "gt")


def attention_pass(model, prep, cfg) -> dict:
    """Pass 1: base answer + one region per question object from the grounding-heads attention."""
    am = record_attention(model, prep, cfg["sink_k"])
    regions, found = [], []
    for k in range(len(prep.sample.objects)):
        A = am.maps.get(f"object:{k}")
        if A is None:  # object name not in the question text
            regions.append([])
            found.append(False)
            continue
        M = aggregate(A, heads=cfg["heads"], sinks=am.sinks)
        regions.append([map_to_box(M, prep.grid_h, prep.grid_w, cfg["box_threshold"])])
        found.append(map_metrics(M, prep.object_cov(k))["pointing"] > 0)
    return {"base": answer_from_logits(am.logits, prep), "regions": regions, "found": bool(found) and all(found)}


def answer(model, prep, edit=None):
    if edit is None:
        return answer_from_logits(forward(model, prep).logits, prep)
    with AttentionScaler(model, [edit]):
        return answer_from_logits(forward(model, prep).logits, prep)


def run_mode(model, processor, prep, orig, mode, regions, steer, args) -> dict | None:
    """Answer with the region passed in the given way (None if there is no region)."""
    s = prep.sample
    if not any(regions):
        return None
    if mode == "steer":
        edit = steer_edit(prep, regions, steer["layers"], args.steer_mode, steer["value"], args.pad)
        return {**answer(model, prep, edit), "tokens": prep.n_img}
    if mode == "crop":
        crop, prefix, _ = crop_cue(model, orig, s, regions, args.crop_pixels, args.crop_margin, args.crop_min_side)
        p = prepare(model, processor, s, args.max_pixels, args.min_pixels, image=prep.image, prefix=prefix, extra_images=[crop])
        return {**answer(model, p), "tokens": sum(sp.grid_h * sp.grid_w for sp in p.spans)}
    if mode == "text":
        p = prepare(model, processor, s, args.max_pixels, args.min_pixels, image=prep.image, prefix=text_cue(s, regions))
        return {**answer(model, p), "tokens": p.n_img}
    if mode == "draw":
        img, prefix = draw_cue(prep.image, s, regions)
        p = prepare(model, processor, s, args.max_pixels, args.min_pixels, image=img, prefix=prefix)
        return {**answer(model, p), "tokens": p.n_img}
    raise ValueError(mode)


def calibrate(model, processor, calib, args, cfg, n_layers) -> dict:
    qids = {s.qid for s in calib}
    cands = {"mid_third": list(range(n_layers // 3, 2 * n_layers // 3)), "all": list(range(n_layers))}
    for top in (1, 2):
        layers, _ = reading_layers(args.p2, qids, args.p2_group, top)
        if layers:
            cands[f"p2_top{top}"] = layers
    grid = {(name, v): [] for name in cands for v in args.lambdas}
    for s in tqdm(calib, desc="calibrate steer"):
        prep = prepare(model, processor, s, args.max_pixels, args.min_pixels)
        ap = attention_pass(model, prep, cfg)
        regions = ap["regions"] if args.calib_source == "attn" else [list(o.boxes) for o in s.objects]
        for (name, v), res in grid.items():
            r = run_mode(model, processor, prep, None, "steer", regions, {"layers": cands[name], "value": v}, args) or ap["base"]
            res.append((r["correct"], r["logp_correct"]))
    scores = {k: (nanmean([c for c, _ in v]), nanmean([lp for _, lp in v])) for k, v in grid.items()} if calib else {}
    best = max(scores, key=lambda k: scores[k]) if scores else ("mid_third", 0.5)  # no calibration data: defaults
    return {"layers": cands[best[0]], "layer_set": best[0], "value": best[1], "mode": args.steer_mode,
            "candidates": cands, "grid": [{"layer_set": k[0], "value": k[1], "acc": a, "mean_logp": lp}
                                          for k, (a, lp) in scores.items()]}


def main():
    p = common_parser(__doc__)
    add_split_args(p)
    p.add_argument("--p1", default=None, help="probe-1 output dir (grounding heads, region threshold)")
    p.add_argument("--p2", default=None, help="probe-2 output dir (reading layers)")
    p.add_argument("--p2_group", default="all")
    p.add_argument("--sources", nargs="*", default=list(SOURCES), choices=SOURCES)
    p.add_argument("--modes", nargs="*", default=["steer", "crop"], choices=MODES)
    p.add_argument("--steer_mode", default="shift", choices=["shift", "scale"])
    p.add_argument("--lambdas", type=float, nargs="*", default=[0.25, 0.5, 0.75],
                   help="steer strengths tried on the calibration split (shift: fraction; scale: factor)")
    p.add_argument("--calib_source", default="attn", choices=["attn", "gt"], help="regions used to tune steer")
    p.add_argument("--pad", type=int, default=0, help="dilate the steered region by this many token cells")
    p.add_argument("--crop_pixels", type=int, default=256 * 32 * 32, help="token budget of the zoomed crop")
    p.add_argument("--crop_margin", type=float, default=0.5)
    p.add_argument("--crop_min_side", type=float, default=0.15)
    p.add_argument("--sink_k", type=float, default=5.0)
    args = p.parse_args()
    calib, test = load_samples(args)
    model, processor = setup(args)
    cfg = load_p1_config(args.p1) or {"heads": [], "box_threshold": 0.5}
    if not args.p1:
        print("no --p1: using all heads and threshold 0.5 for the attention region")
    cfg.setdefault("sink_k", args.sink_k)
    print(f"calibration {len(calib)} / test {len(test)} questions")

    steer = calibrate(model, processor, calib, args, cfg, model.config.text_config.num_hidden_layers) if "steer" in args.modes else None
    json.dump(to_jsonable(steer), open(os.path.join(args.out, "calibration.json"), "w"), indent=2)
    if steer:
        print(f"steer: layers {steer['layer_set']} {steer['layers']}, {args.steer_mode} {steer['value']}")

    budget = args.max_pixels + args.crop_pixels
    out_file = os.path.join(args.out, "per_sample.jsonl")
    done_rows = read_jsonl(out_file)
    done_map = {r["qid"]: r for r in done_rows if "results" in r}
    rows = []
    for s in tqdm(test, desc="test"):
        if s.qid in done_map:
            rows.append(done_map[s.qid])
            continue
        orig = Image.open(s.image_path).convert("RGB")
        prep = prepare(model, processor, s, args.max_pixels, args.min_pixels, image=orig)
        ap = attention_pass(model, prep, cfg)
        hi = prepare(model, processor, s, budget, budget, image=orig)
        res = {"base": {**ap["base"], "tokens": prep.n_img}, "base_hi": {**answer(model, hi), "tokens": hi.n_img}}
        regions = {"attn": ap["regions"], "gt": [list(o.boxes) for o in s.objects]}
        quality = {src: [G.match(r[0] if r else None, o.boxes) for r, o in zip(regions[src], s.objects)] for src in args.sources}
        for src in args.sources:
            for mode in args.modes:
                r = run_mode(model, processor, prep, orig, mode, regions[src], steer, args)
                res[f"{mode}|{src}"] = r if r is not None else {**res["base"], "no_region": True}
        row = {"qid": s.qid, "category": s.category, "results": res, "regions": {k: regions[k] for k in args.sources},
               "region_quality": quality, "attn_found": ap["found"]}
        rows.append(row)
        append_jsonl(out_file, row)
    write_jsonl(out_file, rows)

    def acc(key, rs):
        return nanmean([r["results"][key]["correct"] for r in rs if key in r["results"]])

    table, by_cat = [], []
    cats = sorted({r["category"] for r in rows})
    for mode in ["base", "base_hi", *args.modes]:
        keys = {src: (mode if mode in ("base", "base_hi") else f"{mode}|{src}") for src in args.sources}
        table.append({"mode": mode, **{src: acc(k, rows) for src, k in keys.items()},
                      "mean_tokens": nanmean([r["results"][k]["tokens"] for r in rows for k in keys.values()
                                              if not r["results"][k].get("no_region")])})
        by_cat += [{"category": c, "mode": mode, **{src: acc(k, [r for r in rows if r["category"] == c]) for src, k in keys.items()}}
                   for c in cats]
    write_csv(os.path.join(args.out, "summary.csv"), table)
    write_csv(os.path.join(args.out, "summary_by_category.csv"), by_cat)
    write_csv(os.path.join(args.out, "region_quality.csv"), [
        {"source": src, "center_hit": nanmean([q["center_hit"] for r in rows for q in r["region_quality"][src]]),
         "iou": nanmean([q["iou"] for r in rows for q in r["region_quality"][src]]),
         "no_region": nanmean([not any(r["regions"][src]) for r in rows])} for src in args.sources])
    found_rows = [{"attention_found_object": f, "n": len([r for r in rows if r["attn_found"] == f]),
                   "base": acc("base", [r for r in rows if r["attn_found"] == f]),
                   **{m: acc(f"{m}|attn", [r for r in rows if r["attn_found"] == f]) for m in args.modes if "attn" in args.sources}}
                  for f in (True, False)]
    write_csv(os.path.join(args.out, "by_found.csv"), found_rows)
    json.dump(to_jsonable({"n": len(rows), "steer": steer, "table": table, "by_found": found_rows}),
              open(os.path.join(args.out, "summary.json"), "w"), indent=2)
    for t in table:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in t.items()})


if __name__ == "__main__":
    main()
