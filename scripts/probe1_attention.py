"""Probe 1 - Does the attention find the object, even when the answer is wrong?

For every object named in a question we take the attention of its name tokens onto the image tokens
(visual attention sinks removed) and score it against the object's GT box (V*: its annotated bbox):
  pointing (= "found": the attention peak lies on the object), AUROC, lift, own-vs-other-object,
per layer and per head. The few heads that localize best on the calibration split are kept as
"grounding heads", with a calibrated threshold that turns their map into a region (used by the method).
Then "attention found the object" is crossed with "answer correct".

This is the premise of the method: if the attention finds the object in many questions the model still
answers wrong, the model knows where to look but does not rely on it enough -> amplify that attention.
(Probe 2 checks that the attention on the object is actually used: cutting it hurts the answer.)

Protocol: heads and threshold are fixed on the calibration split (--calib_frac or --calib_dataset);
every reported number is on the test split.

Outputs: config.json (heads, threshold, sink_k - used by the method), per_sample.jsonl, summary.json,
         layers.png (pointing per layer, correct vs wrong answers), heads.png, threshold.png, vis/*.png
"""

import json
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rq1grounding import geometry as G  # noqa: E402
from rq1grounding.cli import (add_split_args, append_jsonl, common_parser, load_samples,
                              nanmean, read_jsonl, setup, to_jsonable, write_jsonl)  # noqa: E402
from rq1grounding.data import LETTERS  # noqa: E402
from rq1grounding.grounding import (THRESHOLDS, aggregate, head_scores, map_metrics, map_to_box,  # noqa: E402
                                    record_attention, select_heads)
from rq1grounding.model import answer_from_logits, prepare  # noqa: E402
from rq1grounding.viz import head_heatmap, layer_curves, overlay, plt  # noqa: E402

ERROR_TYPES = {"neighbor": "binding:neighbor", "in_image": "binding:other_object", "absent": "hallucination"}


def error_type(s, pred):
    if pred == s.answer_idx:
        return "correct"
    return ERROR_TYPES.get(s.meta.get("option_source", {}).get(str(pred)), "wrong")


def analyse(model, processor, s, args, heads=None, t=0.5, keep=False):
    """One forward pass: answer + attention of every query group onto the image."""
    prep = prepare(model, processor, s, args.max_pixels, args.min_pixels)
    am = record_attention(model, prep, args.sink_k)
    ans = answer_from_logits(am.logits, prep)
    rec = {"qid": s.qid, "category": s.category, **ans, "error_type": error_type(s, ans["pred"]),
           "sink_frac": float(am.sinks.mean()), "objects": [], "layers": {}}
    kept = []
    for k, obj in enumerate(s.objects):
        A = am.maps.get(f"object:{k}")
        o = {"name": obj.name, "gt_area": float(sum(G.area(b) for b in obj.boxes)), "name_tokens_found": A is not None}
        if A is not None:
            cov = prep.object_cov(k)
            sc = head_scores(A, cov, am.sinks)
            o["all_heads"] = map_metrics(aggregate(A, sinks=am.sinks), cov, prep.other_cov(k))
            o["layer_pointing"], o["layer_auroc"] = sc["layer_pointing"], sc["layer_auroc"]
            if keep:
                kept.append({"A": A, "sinks": am.sinks, "boxes": obj.boxes, "grid": (prep.grid_h, prep.grid_w), **sc})
            if heads:
                M = aggregate(A, heads=heads, sinks=am.sinks)
                box = map_to_box(M, prep.grid_h, prep.grid_w, t)
                o["heads"] = {**map_metrics(M, cov, prep.other_cov(k)), "box": box, **G.match(box, obj.boxes)}
        rec["objects"].append(o)
    if "last" in am.maps:
        rec["layers"]["last"] = head_scores(am.maps["last"], prep.gt_cov(), am.sinks)["layer_pointing"]
    objs = [o for o in rec["objects"] if "layer_pointing" in o]
    if objs:
        rec["layers"]["object"] = np.mean([o["layer_pointing"] for o in objs], 0)
    key = "heads" if heads else "all_heads"
    rec["attn_found"] = bool(rec["objects"]) and all(o.get(key, {}).get("pointing", 0) > 0 for o in rec["objects"])
    return rec, prep, am, kept


def calibrate(kept, args):
    if not kept:  # empty calibration split: no head selection -> all heads, default threshold
        print("warning: empty calibration split - using all heads and t=0.5")
        return {"heads": [], "box_threshold": 0.5, "sink_k": args.sink_k, "threshold_curve": {}, "head_pointing_mean": [[0.0]]}
    heads = select_heads([x["head_pointing"] for x in kept], [x["head_auroc"] for x in kept], args.top_heads)
    curve = [nanmean([G.match(map_to_box(aggregate(x["A"], heads=heads, sinks=x["sinks"]), *x["grid"], t), x["boxes"])["iou"]
                      for x in kept]) for t in THRESHOLDS]
    return {"heads": heads, "box_threshold": float(THRESHOLDS[int(np.nanargmax(curve))]), "sink_k": args.sink_k,
            "threshold_curve": dict(zip(THRESHOLDS.tolist(), curve)),
            "head_pointing_mean": np.nanmean([x["head_pointing"] for x in kept], 0)}


def summarize(recs):
    objs = [o for r in recs for o in r["objects"]]
    out = {"n_questions": len(recs), "n_objects": len(objs), "accuracy": nanmean([r["correct"] for r in recs]),
           "accuracy_by_category": {c: nanmean([r["correct"] for r in recs if r["category"] == c])
                                    for c in sorted({r["category"] for r in recs})},
           "name_tokens_found": nanmean([o["name_tokens_found"] for o in objs])}
    for key in ("heads", "all_heads"):
        xs = [o[key] for o in objs if key in o]
        if xs:
            out[f"attention_{key}"] = {m: nanmean([x.get(m) for x in xs]) for m in ("pointing", "auroc", "lift", "own_gt_other", "iou")
                                       if any(m in x for x in xs)}
    # the premise of the method
    t = {f"{'found' if f else 'not_found'}&{'correct' if c else 'wrong'}": sum(r["attn_found"] == f and r["correct"] == c for r in recs)
         for f in (True, False) for c in (True, False)}
    wrong, right = [r for r in recs if not r["correct"]], [r for r in recs if r["correct"]]
    out["attention_found_x_correct"] = {
        **t,
        "found_rate_when_correct": nanmean([r["attn_found"] for r in right]),
        "found_rate_when_wrong": nanmean([r["attn_found"] for r in wrong]),
        "accuracy_if_found": nanmean([r["correct"] for r in recs if r["attn_found"]]),
        "accuracy_if_not_found": nanmean([r["correct"] for r in recs if not r["attn_found"]]),
    }
    binding = [r for r in recs if r["error_type"].startswith("binding")]
    if binding:  # synthetic: wrong colour from another object although the attention was on the right one?
        out["binding_errors_with_attention_on_target"] = nanmean([r["attn_found"] for r in binding])
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
    """GT (green) | grounding-heads map + its region (white) | all-heads map."""
    s = prep.sample
    A = am.maps.get("object:0")
    panels = [("object name not found in question", None, None)] * 2
    if A is not None:
        Mh = aggregate(A, heads=cfg["heads"], sinks=am.sinks)
        panels = [(f"grounding heads ({len(cfg['heads']) or 'all'})", Mh, map_to_box(Mh, prep.grid_h, prep.grid_w, cfg["box_threshold"])),
                  ("all layers x heads", aggregate(A, sinks=am.sinks), None)]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    gt = s.boxes
    overlay(axes[0], prep.image, None, gt, f"{s.qid}\npred {LETTERS[rec['pred']]} / gt {LETTERS[s.answer_idx]}")
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
    p.add_argument("--n_vis", type=int, default=12)
    args = p.parse_args()
    calib, test = load_samples(args)
    model, processor = setup(args)
    print(f"calibration {len(calib)} / test {len(test)} questions")

    kept = []
    for s in tqdm(calib, desc="calibration"):
        kept += analyse(model, processor, s, args, keep=True)[3]
    cfg = {**calibrate(kept, args), "calib_qids": [s.qid for s in calib]}
    json.dump(to_jsonable(cfg), open(os.path.join(args.out, "config.json"), "w"), indent=2)
    head_heatmap(np.asarray(cfg["head_pointing_mean"]), os.path.join(args.out, "heads.png"),
                 "object query: per-head pointing (calibration)")
    if cfg["threshold_curve"]:
        layer_curves({"attention region": np.array(list(cfg["threshold_curve"].values()))}, "mean IoU with GT",
                     os.path.join(args.out, "threshold.png"), "map -> region threshold (calibration)", x=THRESHOLDS,
                     xlabel="t (x peak)")

    out_file = os.path.join(args.out, "per_sample.jsonl")
    done_recs = read_jsonl(out_file)
    done_map = {r["qid"]: r for r in done_recs if "layers" in r}
    recs = []
    os.makedirs(os.path.join(args.out, "vis"), exist_ok=True)
    for i, s in enumerate(tqdm(test, desc="test")):
        if s.qid in done_map:
            recs.append(done_map[s.qid])
            continue
        rec, prep, am, _ = analyse(model, processor, s, args, cfg["heads"], cfg["box_threshold"])
        recs.append(rec)
        append_jsonl(out_file, rec)
        if i < args.n_vis:
            visualize(prep, rec, am, cfg, os.path.join(args.out, "vis", f"{i:04d}_{s.qid.replace('/', '_')}.png"))
    write_jsonl(out_file, recs)

    summary = summarize(recs)
    json.dump(to_jsonable(summary), open(os.path.join(args.out, "summary.json"), "w"), indent=2)
    curves = {}
    for g in ("object", "last"):
        for name, sel in (("correct", True), ("wrong", False)):
            arr = [r["layers"][g] for r in recs if g in r["layers"] and r["correct"] == sel]
            if arr:
                curves[f"{g} query, {name} answers"] = np.stack(arr)
    layer_curves(curves, "pointing (attention peak on the object)", os.path.join(args.out, "layers.png"),
                 "head-averaged attention per layer (sinks removed)")
    print(json.dumps(to_jsonable({k: summary[k] for k in ("accuracy", "attention_found_x_correct")}), indent=1))


if __name__ == "__main__":
    main()
