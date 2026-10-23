"""Probe 1 - Does Qwen3-VL find the object it is asked about, and does finding it predict answering right?

For every object named in a question (V*: its GT boxes) we measure
  explicit   the box the model outputs when asked "Locate the <object> ..." on the same image
             -> centre-in-GT (found), IoU
  implicit   the attention of the object's name tokens onto the image tokens, sinks removed
             -> pointing (found), AUROC, lift, own-vs-other-object; per layer, per head, and for the
                grounding heads chosen on the calibration split (+ the box they imply)
  agreement  is the attention peak inside the model's own box?  (does it look where it says?)
and cross "found" with answer correctness. The found x correct table separates errors made *after*
the object was found (binding / read-out: RQ1) from errors where it was never found (localization).

Protocol: grounding heads and the map->box threshold are fixed on the calibration split
(--calib_frac or --calib_dataset); all reported numbers are on the test split.

Outputs: config.json (heads, threshold, sink_k - reused by probe 3 and the method), per_sample.jsonl,
         summary.json, layers.png, heads.png, threshold.png, vis/*.png
"""

import json
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rq1grounding import geometry as G  # noqa: E402
from rq1grounding.cli import add_split_args, common_parser, load_samples, nanmean, setup, to_jsonable, write_jsonl  # noqa: E402
from rq1grounding.data import LETTERS  # noqa: E402
from rq1grounding.grounding import (THRESHOLDS, aggregate, ground_explicit, head_scores, map_metrics,  # noqa: E402
                                    map_to_box, peak_point, record_attention, select_heads)
from rq1grounding.model import answer_from_logits, prepare  # noqa: E402
from rq1grounding.viz import head_heatmap, layer_curves, overlay, plt  # noqa: E402

ERROR_TYPES = {"neighbor": "binding:neighbor", "in_image": "binding:other_object", "absent": "hallucination"}


def error_type(s, pred):
    if pred == s.answer_idx:
        return "correct"
    src = s.meta.get("option_source", {}).get(str(pred))
    return ERROR_TYPES.get(src, "wrong")


def analyse(model, processor, s, args, heads=None, t=0.5, keep=False):
    prep = prepare(model, processor, s, args.max_pixels, args.min_pixels)
    am = record_attention(model, prep, args.sink_k)
    ans = answer_from_logits(am.logits, prep)
    gh, gw = prep.grid_h, prep.grid_w
    rec = {"qid": s.qid, "category": s.category, **ans, "error_type": error_type(s, ans["pred"]),
           "grid": [gh, gw], "sink_frac": float(am.sinks.mean()), "objects": [], "layers": {}}
    kept = []
    for k, obj in enumerate(s.objects):
        cov, other = prep.object_cov(k), prep.other_cov(k)
        ex = ground_explicit(model, processor, prep.image, obj.name, args.max_new_tokens)
        o = {"name": obj.name, "gt_area": float(sum(G.area(b) for b in obj.boxes)),
             "explicit": {**ex, **G.match(ex["box"], obj.boxes)}}
        A = am.maps.get(f"object:{k}")
        if A is not None:
            sc = head_scores(A, cov, am.sinks)
            o["implicit_all"] = map_metrics(aggregate(A, sinks=am.sinks), cov, other)
            o["layer_pointing"], o["layer_auroc"] = sc["layer_pointing"], sc["layer_auroc"]
            if keep:
                kept.append({"A": A, "sinks": am.sinks, "boxes": obj.boxes, "grid": (gh, gw), **sc})
            if heads:
                M = aggregate(A, heads=heads, sinks=am.sinks)
                box = map_to_box(M, gh, gw, t)
                o["implicit_heads"] = {**map_metrics(M, cov, other), "box": box, **G.match(box, obj.boxes)}
                if ex["box"] is not None:
                    o["agree"] = G.contains(ex["box"], peak_point(M, gh, gw))
        rec["objects"].append(o)
    for g in ("question", "last"):
        if g in am.maps:
            sc = head_scores(am.maps[g], prep.gt_cov(), am.sinks)
            rec["layers"][g] = {"pointing": sc["layer_pointing"], "auroc": sc["layer_auroc"]}
    obj_layers = [o for o in rec["objects"] if "layer_pointing" in o]
    if obj_layers:
        rec["layers"]["object"] = {m: np.mean([o[f"layer_{m}"] for o in obj_layers], 0) for m in ("pointing", "auroc")}
    rec["found_explicit"] = all(o["explicit"]["center_hit"] for o in rec["objects"])
    imp = [o.get("implicit_heads", o.get("implicit_all")) for o in rec["objects"]]
    rec["found_implicit"] = bool(imp) and all(i is not None and i["pointing"] > 0 for i in imp)
    return rec, prep, am, kept


def calibrate(kept, args):
    if not kept:  # empty calibration split: no head selection, all heads + default threshold downstream
        print("warning: empty calibration split - using all heads and t=0.5")
        return {"heads": [], "box_threshold": 0.5, "sink_k": args.sink_k, "threshold_curve": {}, "head_pointing_mean": [[0.0]]}
    heads = select_heads([x["head_pointing"] for x in kept], [x["head_auroc"] for x in kept], args.top_heads)
    curve = []
    for t in THRESHOLDS:
        curve.append(nanmean([G.match(map_to_box(aggregate(x["A"], heads=heads, sinks=x["sinks"]), *x["grid"], t),
                                      x["boxes"])["iou"] for x in kept]))
    t = float(THRESHOLDS[int(np.nanargmax(curve))])
    return {"heads": heads, "box_threshold": t, "sink_k": args.sink_k, "threshold_curve": dict(zip(THRESHOLDS.tolist(), curve)),
            "head_pointing_mean": np.nanmean([x["head_pointing"] for x in kept], 0)}


def found_table(recs, key):
    t = {f"{'found' if f else 'not_found'}&{'correct' if c else 'wrong'}": 0 for f in (1, 0) for c in (1, 0)}
    for r in recs:
        t[f"{'found' if r[key] else 'not_found'}&{'correct' if r['correct'] else 'wrong'}"] += 1
    wrong = t["found&wrong"] + t["not_found&wrong"]
    found = t["found&correct"] + t["found&wrong"]
    t["acc_if_found"] = t["found&correct"] / found if found else None
    t["acc_if_not_found"] = t["not_found&correct"] / (len(recs) - found) if len(recs) > found else None
    t["share_of_errors_found"] = t["found&wrong"] / wrong if wrong else None
    return t


def summarize(recs):
    objs = [o for r in recs for o in r["objects"]]
    out = {"n_questions": len(recs), "n_objects": len(objs), "accuracy": nanmean([r["correct"] for r in recs]),
           "accuracy_by_category": {c: nanmean([r["correct"] for r in recs if r["category"] == c])
                                    for c in sorted({r["category"] for r in recs})},
           "explicit": {"center_hit": nanmean([o["explicit"]["center_hit"] for o in objs]),
                        "iou": nanmean([o["explicit"]["iou"] for o in objs]),
                        "iou>=0.5": nanmean([o["explicit"]["iou"] >= 0.5 for o in objs]),
                        "no_box": nanmean([o["explicit"]["box"] is None for o in objs])}}
    for kind in ("implicit_all", "implicit_heads"):
        xs = [o[kind] for o in objs if kind in o]
        if xs:
            out[kind] = {m: nanmean([x.get(m) for x in xs]) for m in ("pointing", "auroc", "lift", "own_gt_other", "iou", "center_hit")
                         if any(m in x for x in xs)}
    out["agreement_peak_in_own_box"] = nanmean([o["agree"] for o in objs if "agree" in o])
    out["found_x_correct"] = {"explicit": found_table(recs, "found_explicit"), "implicit": found_table(recs, "found_implicit")}
    out["error_types"] = {e: sum(r["error_type"] == e for r in recs) for e in sorted({r["error_type"] for r in recs})}
    out["mean_sink_frac"] = nanmean([r["sink_frac"] for r in recs])
    pairs = {}
    for r in recs:
        if r["category"] == "synth_attr":
            pairs.setdefault(r["qid"].rsplit("_v", 1)[0], []).append(r["correct"])
    pairs = [v for v in pairs.values() if len(v) == 2]
    if pairs:
        out["twin_both_correct"] = nanmean([all(v) for v in pairs])
    return out


def visualize(prep, rec, am, cfg, path):
    """GT (green) + the model's own box (blue) | grounding-heads map + its box (white) | all-heads map."""
    s = prep.sample
    A = am.maps.get("object:0")
    panels = [("no object tokens", None, None)] * 2
    if A is not None:
        Mh = aggregate(A, heads=cfg["heads"], sinks=am.sinks)
        panels = [(f"grounding heads ({len(cfg['heads'])})", Mh, map_to_box(Mh, prep.grid_h, prep.grid_w, cfg["box_threshold"])),
                  ("all layers x heads", aggregate(A, sinks=am.sinks), None)]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    gt = s.boxes
    ex = [o["explicit"]["box"] for o in rec["objects"] if o["explicit"]["box"]]
    overlay(axes[0], prep.image, None, gt + ex, f"{s.qid}\npred {LETTERS[rec['pred']]} / gt {LETTERS[s.answer_idx]}",
            box_colors=("lime",) * len(gt) + ("deepskyblue",) * len(ex))
    for ax, (title, M, box) in zip(axes[1:], panels):
        overlay(ax, prep.image, None if M is None else M.reshape(prep.grid_h, prep.grid_w), gt + ([box] if box else []),
                title, box_colors=("lime",) * len(gt) + ("white",))
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    p = common_parser(__doc__)
    add_split_args(p)
    p.add_argument("--top_heads", type=int, default=8)
    p.add_argument("--sink_k", type=float, default=5.0, help="sink token: residual norm > k x median")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--n_vis", type=int, default=12)
    args = p.parse_args()
    calib, test = load_samples(args)
    model, processor = setup(args)
    print(f"calibration {len(calib)} / test {len(test)} questions")

    kept = []
    for s in tqdm(calib, desc="calibration"):
        kept += analyse(model, processor, s, args, keep=True)[3]
    cfg = calibrate(kept, args)
    cfg["calib_qids"] = [s.qid for s in calib]
    json.dump(to_jsonable(cfg), open(os.path.join(args.out, "config.json"), "w"), indent=2)
    head_heatmap(cfg["head_pointing_mean"], os.path.join(args.out, "heads.png"), "object query: per-head pointing (calibration)")
    layer_curves({"attention box IoU": np.array(list(cfg["threshold_curve"].values()))}, "mean IoU with GT",
                 os.path.join(args.out, "threshold.png"), "map -> box threshold t (x peak), calibration", x=THRESHOLDS)

    recs = []
    os.makedirs(os.path.join(args.out, "vis"), exist_ok=True)
    for i, s in enumerate(tqdm(test, desc="test")):
        rec, prep, am, _ = analyse(model, processor, s, args, cfg["heads"], cfg["box_threshold"])
        recs.append(rec)
        if i < args.n_vis:
            visualize(prep, rec, am, cfg, os.path.join(args.out, "vis", f"{i:04d}_{s.qid.replace('/', '_')}.png"))
    write_jsonl(os.path.join(args.out, "per_sample.jsonl"), recs)

    summary = summarize(recs)
    json.dump(to_jsonable(summary), open(os.path.join(args.out, "summary.json"), "w"), indent=2)
    for m, yl, h in (("pointing", "pointing (peak in GT)", None), ("auroc", "AUROC (0.5 = chance)", 0.5)):
        curves = {g: np.stack([r["layers"][g][m] for r in recs if g in r["layers"]]) for g in ("object", "question", "last")
                  if any(g in r["layers"] for r in recs)}
        layer_curves(curves, yl, os.path.join(args.out, f"layers_{m}.png"), "head-averaged attention per layer (sinks removed)", hline=h)
    print(json.dumps(to_jsonable({k: summary[k] for k in ("accuracy", "explicit", "found_x_correct")}), indent=1))


if __name__ == "__main__":
    main()
