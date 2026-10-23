"""Probe 5 - A rigorous grounding benchmark for attention maps (calibrate on one split, test on another).

What is evaluated (see rq1grounding/grounding_eval.py for the metric definitions):
  query sets  answer / question (all tokens) / object (all tokens of each object phrase, scored
              per object against its own box; other objects + distractors are the "other" boxes)
  maps        {q}/all          all layers x heads
              {q}/best_layer   head-average of the layer chosen on the calibration split
              {q}/heads        mean of the top-k "grounding heads" chosen on the calibration split
              +ns suffix       visual attention sinks (massive-norm image tokens) masked out
              grad             gradient x input on the visual embeddings (non-attention reference)
  metrics     threshold-free (pointing, auroc, ap, energy, lift, own>other) and, for every
              binarization rule (rel / mass / uni / otsu), metrics at the threshold fixed on the
              calibration split plus the test-oracle MaxBoxAcc for reference.

Protocol
  --mode calibrate : per-head and per-layer scores -> grounding heads, best layers, thresholds
                     -> <out>/grounding_config.json
  --mode eval      : fixed config (--config) on the test split -> bench_table.csv, plots, vis
  --mode both      : calibrate on --calib_* data (default: a --calib_frac split of --dataset),
                     then evaluate on the rest
"""

import csv
import hashlib
import json
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rq1grounding import grounding_eval as ge  # noqa: E402
from rq1grounding.cli import common_parser, to_jsonable  # noqa: E402
from rq1grounding.data import load_dataset  # noqa: E402
from rq1grounding.maps import QUERY_SETS, collect, grad_input_map, head_map  # noqa: E402
from rq1grounding.metrics import topk_heads  # noqa: E402
from rq1grounding.model import answer_from_logits, load_model, prepare  # noqa: E402
from rq1grounding.viz import boxes_to_norm, head_heatmap, layer_curves, overlay, plt  # noqa: E402

FREE = ("pointing", "pointing_tol", "energy", "lift", "auroc", "ap", "own_gt_other")


def nmean(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


def gt_boxes_norm(s, which=None):
    from PIL import Image

    W, H = Image.open(s.image_path).size
    boxes = s.bboxes if which is None else [s.bboxes[which]]
    return [tuple(np.clip(b, 0, 1)) for b in boxes_to_norm(boxes, W, H)]


def targets(prep, qset):
    """(map query-set key, coverage, other coverage, gt boxes) for every evaluation target of a query set."""
    s = prep.sample
    if qset != "object":
        return [(qset, prep.cov_flat, None, gt_boxes_norm(s))]
    out = []
    for k, sp in enumerate(prep.object_spans):
        if not sp or k >= len(prep.object_cov):
            continue
        others = [c for j, c in enumerate(prep.object_cov) if j != k]
        if prep.distractor_coverage is not None:
            others.append(prep.distractor_coverage)
        other = np.max(np.stack(others), 0).reshape(-1) if others else None
        out.append((f"object:{k}", prep.object_cov[k].reshape(-1), other, gt_boxes_norm(s, k)))
    return out


def map_specs(cfg):
    """name -> (query set, kwargs for head_map, drop sinks)."""
    specs = {}
    for q in QUERY_SETS:
        specs[f"{q}/all"] = (q, {}, False)
        specs[f"{q}/all+ns"] = (q, {}, True)
        if cfg and q in cfg.get("best_layer", {}):
            specs[f"{q}/best_layer"] = (q, {"layers": [cfg["best_layer"][q]]}, True)
        if cfg and q in cfg.get("heads", {}):
            hs = [tuple(x) for x in cfg["heads"][q]]
            specs[f"{q}/heads"] = (q, {"heads": hs}, False)
            specs[f"{q}/heads+ns"] = (q, {"heads": hs}, True)
    return specs


def per_layer_head_scores(b, prep, qset, sink):
    """Threshold-free scores for every layer (head-avg) and every head: dict of [Ly] / [Ly, H] arrays."""
    res = {}
    for key, cov, _, _ in targets(prep, qset):
        if key not in b.attn:
            continue
        A = b.attn[key].float().numpy()  # [Ly, H, n]
        Ly, H, n = A.shape
        pos = cov > 0
        for tag, mask in (("raw", None), ("ns", sink)):
            X = A.copy()
            if mask is not None and (~mask).any():
                X[..., mask] = 0
            X /= X.sum(-1, keepdims=True).clip(1e-12)
            lay = X.mean(1)
            r = {"layer_auroc": ge.auroc_rows(lay, pos), "layer_pointing": ge.pointing_rows(lay, pos),
                 "layer_lift": (lay * cov).sum(-1) / max(cov.mean(), 1e-6)}
            if tag == "raw":
                flat = X.reshape(Ly * H, n)
                r["head_auroc"] = ge.auroc_rows(flat, pos).reshape(Ly, H)
                r["head_pointing"] = ge.pointing_rows(flat, pos).reshape(Ly, H)
                r["head_img_mass"] = A.sum(-1)
            for k, v in r.items():
                res.setdefault(f"{tag}_{k}", []).append(v)
    return {k: np.nanmean(np.stack(v), 0) for k, v in res.items()}


def run(model, processor, samples, args, cfg, out_dir, vis=0):
    """One pass: per-layer/head scores, every map's metrics. Returns per-sample records."""
    specs = map_specs(cfg)
    recs = []
    vis_dir = os.path.join(out_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)
    for si, s in enumerate(tqdm(samples)):
        prep = prepare(model, processor, s, args.max_pixels, args.min_pixels)
        if prep.cov_flat.max() == 0:
            continue
        b = collect(model, prep)
        ans = answer_from_logits(b.logits, prep)
        sink = b.sinks(args.sink_k)
        rec = {"qid": s.qid, "category": s.category, "correct": ans["correct"], "pred": ans["pred"],
               "sink_frac": float(sink.mean()), "oracle_iou": ge.oracle_iou(prep.cov_flat, prep.grid_h, prep.grid_w,
                                                                             gt_boxes_norm(s)),
               "layers": {}, "maps": {}}
        for q in QUERY_SETS:
            sc = per_layer_head_scores(b, prep, q, sink)
            if sc:
                rec["layers"][q] = sc
        maps_for_vis = {}
        for name, (q, kw, ns) in specs.items():
            per_t = []
            for key, cov, other, boxes in targets(prep, q):
                M = head_map(b, key, sink_mask=sink if ns else None, **kw)
                if M is None:
                    continue
                per_t.append((ge.threshold_free(M, cov, prep.grid_h, prep.grid_w, other),
                              ge.sweep(M, cov, prep.grid_h, prep.grid_w, boxes)))
                maps_for_vis.setdefault(name, M)
            if per_t:
                rec["maps"][name] = _merge_targets(per_t)
        if args.grad:
            M = grad_input_map(model, prep)
            rec["maps"]["grad"] = _merge_targets([(ge.threshold_free(M, prep.cov_flat, prep.grid_h, prep.grid_w),
                                                   ge.sweep(M, prep.cov_flat, prep.grid_h, prep.grid_w, gt_boxes_norm(s)))])
            maps_for_vis["grad"] = M
        recs.append(rec)
        if si < vis:
            _visualize(prep, maps_for_vis, cfg, os.path.join(vis_dir, f"{si:04d}_{s.qid.replace('/', '_')}.png"))
    return recs


def _merge_targets(per_t):
    free = {k: nmean([f.get(k, np.nan) for f, _ in per_t]) for k in FREE}
    sw = {rule: {m: np.mean([s[rule][m] for _, s in per_t], 0) for m in ge.SWEEP_METRICS} for rule in ge.RULES}
    return {"free": free, "sweep": sw}


def _visualize(prep, maps, cfg, path):
    names = [n for n in ("answer/all", "question/all+ns", "object/all+ns", "object/heads+ns", "answer/heads+ns", "grad")
             if n in maps]
    fig, axes = plt.subplots(1, len(names) + 1, figsize=(3 * (len(names) + 1), 3.2))
    boxes = gt_boxes_norm(prep.sample)
    s = prep.sample
    overlay(axes[0], prep.image, None, boxes, f"{s.qid}\n{s.question[:40]}")
    rule, t = (cfg or {}).get("default_rule", "otsu"), 0.0
    for ax, n in zip(axes[1:], names):
        if cfg and n in cfg.get("thresholds", {}):
            t = cfg["thresholds"][n][rule]
        M = maps[n]
        mask = ge.peak_component(ge.binarize(M, rule, t).reshape(prep.grid_h, prep.grid_w), M.reshape(prep.grid_h, prep.grid_w))
        pb = ge.mask_box(mask)
        overlay(ax, prep.image, M.reshape(prep.grid_h, prep.grid_w), boxes + [pb], f"{n}\n{rule}@{t:g}",
                box_colors=("lime",) * len(boxes) + ("white",))
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def calibrate(recs, args, n_layers):
    cfg = {"sink_k": args.sink_k, "heads": {}, "best_layer": {}, "default_rule": args.default_rule,
           "calib_objective": args.calib_objective, "n_calib": len(recs)}
    for q in QUERY_SETS:
        rows = [r["layers"][q] for r in recs if q in r["layers"]]
        if not rows:
            continue
        hp = np.nanmean([r["raw_head_pointing"] for r in rows], 0)
        ha = np.nanmean([r["raw_head_auroc"] for r in rows], 0)
        mass = np.nanmean([r["raw_head_img_mass"] for r in rows], 0)
        score = np.where(mass >= args.min_img_mass, hp + 0.1 * ha, -np.inf)
        cfg["heads"][q] = [[l, h] for l, h, _ in topk_heads(score, args.top_heads)]
        la = np.nanmean([r["ns_layer_auroc"] for r in rows], 0)
        cfg["best_layer"][q] = int(np.nanargmax(la))
    return cfg


def calibrate_thresholds(recs, cfg, args):
    cfg["thresholds"] = {}
    for name in recs[0]["maps"]:
        sw = [r["maps"][name]["sweep"] for r in recs if name in r["maps"]]
        cfg["thresholds"][name] = ge.pick_thresholds(ge.aggregate_sweeps(sw), args.calib_objective)
    return cfg


def report(recs, cfg, out_dir, n_layers):
    names = sorted({n for r in recs for n in r["maps"]})
    table = []
    for n in names:
        rr = [r["maps"][n] for r in recs if n in r["maps"]]
        row = {"map": n, "n": len(rr)}
        for k in FREE:
            row[k] = round(nmean([x["free"][k] for x in rr]), 4)
        agg = ge.aggregate_sweeps([x["sweep"] for x in rr])
        for rule in ge.RULES:
            t = cfg.get("thresholds", {}).get(n, {}).get(rule)
            if t is not None:
                vals = [ge.at_threshold(x["sweep"], rule, t) for x in rr]
                for m in ("tok_f1", "center_hit", "recall", "box_iou", "area"):
                    row[f"{rule}@cal:{m}"] = round(float(np.mean([v[m] for v in vals])), 4)
                row[f"{rule}@cal:box_acc@0.5"] = round(float(np.mean([v["box_iou"] >= 0.5 for v in vals])), 4)
                row[f"{rule}:threshold"] = t
            row[f"{rule}:MaxBoxAcc@0.5(oracle)"] = round(float(np.max(agg[rule]["box_acc@0.5"])), 4)
        table.append(row)
    keys = list(dict.fromkeys(k for r in table for k in r))
    with open(os.path.join(out_dir, "bench_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(table)

    # breakdowns of the threshold-free metrics
    groups = {"all": recs, "correct": [r for r in recs if r["correct"]], "incorrect": [r for r in recs if not r["correct"]]}
    for c in sorted({r["category"] for r in recs}):
        groups[c] = [r for r in recs if r["category"] == c]
    breakdown = {g: {n: {k: nmean([r["maps"][n]["free"][k] for r in rs if n in r["maps"]])
                         for k in ("pointing", "auroc", "lift", "own_gt_other")} for n in names}
                 for g, rs in groups.items() if rs}

    # per-layer curves and head maps
    for q in QUERY_SETS:
        rows = [r["layers"][q] for r in recs if q in r["layers"]]
        if not rows:
            continue
        for m, yl, h in (("auroc", "AUROC (0.5 = chance)", 0.5), ("pointing", "pointing-game hit", None),
                         ("lift", "in-box mass / area (1 = chance)", 1.0)):
            curves = {"raw": np.stack([r[f"raw_layer_{m}"] for r in rows]),
                      "sinks removed": np.stack([r[f"ns_layer_{m}"] for r in rows])}
            layer_curves(curves, yl, os.path.join(out_dir, f"layer_{q}_{m}.png"), f"{q} query set", hline=h)
        head_heatmap(np.nanmean([r["raw_head_auroc"] for r in rows], 0), os.path.join(out_dir, f"heads_{q}_auroc.png"),
                     f"{q}: per-head AUROC")
        head_heatmap(np.nanmean([r["raw_head_pointing"] for r in rows], 0),
                     os.path.join(out_dir, f"heads_{q}_pointing.png"), f"{q}: per-head pointing")

    # threshold sweeps of the best map
    best = max(table, key=lambda r: r["auroc"] if not np.isnan(r["auroc"]) else -1)["map"]
    agg = ge.aggregate_sweeps([r["maps"][best]["sweep"] for r in recs if best in r["maps"]])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for ax, rule in zip(axes, ("rel", "mass", "uni")):
        for m in ("tok_f1", "center_hit", "recall", "box_acc@0.5", "area"):
            ax.plot(ge.RULES[rule], agg[rule][m], marker=".", label=m)
        ax.set_xlabel({"rel": "t (x max)", "mass": "p (mass share)", "uni": "k (x uniform)"}[rule])
        ax.set_xscale("log" if rule == "uni" else "linear")
        ax.set_title(f"{best}: {rule}", fontsize=9)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "threshold_sweep.png"), dpi=150)
    plt.close(fig)

    summary = {"n": len(recs), "accuracy": float(np.mean([r["correct"] for r in recs])),
               "mean_sink_frac": float(np.mean([r["sink_frac"] for r in recs])),
               "mean_oracle_iou": float(np.mean([r["oracle_iou"] for r in recs])),
               "best_map_by_auroc": best, "breakdown": breakdown}
    json.dump(to_jsonable(summary), open(os.path.join(out_dir, "bench_summary.json"), "w"), indent=2)
    with open(os.path.join(out_dir, "per_sample.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(to_jsonable({k: r[k] for k in ("qid", "category", "correct", "sink_frac", "oracle_iou")}
                                           | {"free": {n: r["maps"][n]["free"] for n in r["maps"]}})) + "\n")
    print(json.dumps({k: summary[k] for k in ("n", "accuracy", "mean_sink_frac", "mean_oracle_iou", "best_map_by_auroc")}))
    for row in sorted(table, key=lambda r: -r["auroc"] if not np.isnan(r["auroc"]) else 0)[:8]:
        print({k: row[k] for k in ("map", "pointing", "auroc", "lift", "own_gt_other")})


def split(samples, frac):
    def h(s):
        return int(hashlib.md5(s.qid.encode()).hexdigest(), 16) % 1000 / 1000

    return [s for s in samples if h(s) < frac], [s for s in samples if h(s) >= frac]


def main():
    p = common_parser(__doc__)
    p.add_argument("--mode", choices=("calibrate", "eval", "both"), default="both")
    p.add_argument("--config", default=None, help="grounding_config.json for --mode eval")
    p.add_argument("--calib_dataset", default=None)
    p.add_argument("--calib_root", default=None)
    p.add_argument("--calib_limit", type=int, default=None)
    p.add_argument("--calib_frac", type=float, default=0.3, help="if no --calib_dataset: share of --dataset for calibration")
    p.add_argument("--top_heads", type=int, default=8)
    p.add_argument("--min_img_mass", type=float, default=0.0, help="ignore heads with less attention on the image")
    p.add_argument("--sink_k", type=float, default=5.0, help="sink = residual norm > k x median in >= half the layers")
    p.add_argument("--calib_objective", default="tok_f1", choices=("tok_f1", "center_hit", "box_iou", "box_acc@0.5"))
    p.add_argument("--default_rule", default="rel", choices=tuple(ge.RULES))
    p.add_argument("--grad", action="store_true", help="also evaluate gradient x input maps (needs a backward pass)")
    p.add_argument("--n_vis", type=int, default=12)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"), indent=2)

    samples = load_dataset(args.dataset, args.data_root, args.categories, args.limit)
    if args.calib_dataset:
        calib = load_dataset(args.calib_dataset, args.calib_root, None, args.calib_limit)
        test = samples
    else:
        calib, test = split(samples, args.calib_frac)
    model, processor = load_model(args.model, dtype=args.dtype)
    n_layers = model.config.text_config.num_hidden_layers

    cfg = json.load(open(args.config)) if args.config else None
    if args.mode in ("calibrate", "both"):
        cdir = os.path.join(args.out, "calibration")
        print(f"calibration on {len(calib)} samples")
        recs = run(model, processor, calib, args, None, cdir)
        cfg = calibrate(recs, args, n_layers)
        # second pass over the calibration split: score the selected-head / best-layer maps to fix thresholds
        recs = run(model, processor, calib, args, cfg, cdir)
        cfg = calibrate_thresholds(recs, cfg, args)
        json.dump(cfg, open(os.path.join(args.out, "grounding_config.json"), "w"), indent=2)
        report(recs, cfg, cdir, n_layers)
    if args.mode in ("eval", "both"):
        assert cfg is not None, "--mode eval needs --config"
        print(f"evaluation on {len(test)} samples")
        recs = run(model, processor, test, args, cfg, args.out, vis=args.n_vis)
        report(recs, cfg, args.out, n_layers)


if __name__ == "__main__":
    main()
