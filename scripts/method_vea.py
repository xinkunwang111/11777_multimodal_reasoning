"""Method B - VEA (Visual Evidence Augmentation; Liu et al., "Seeing but Not Believing", ICLR 2026) on Qwen3-VL.

The paper's finding is probe 1's premise: deep-layer attention lands on the evidence even when the answer is
wrong. VEA brings that attention back through the *input* (method A, method_steer.py, edits it inside the
model): one forward pass reads where the profiled layers look, the rest of the image is dimmed, and the model
answers again on the highlighted image. Training-free; the steps, the paper's hyperparameters and what differs
from the original code are in rq1grounding/vea.py.

  calibration split   step A: profile the layers (evidence AUROC of the last prompt token's attention per
                      layer; keep the top 10%), or take them from --layers
  test split          per question: 1 attribution pass (it also gives the base answer) + 1 pass per variant

Variants (rows of summary.csv: the paper's comparison and ablations, plus a GT upper bound)
  base              original image, plain prompt
  inst              original image, prompt with the highlight hint        prompt-only control
  vea               highlighted image + hint                              the method
  vea_no_prompt     highlighted image, plain prompt                       visual effect alone
  vea_no_denoise    w/o step C;   vea_no_smooth  w/o step D;   vea_no_profiling  all layers
  vea_gt            highlight built from the V* GT boxes instead of the attention (no denoise: it would erase
                    a one-cell object)                                    upper bound for the evidence map
fixed / broken count the questions a variant turns wrong -> right / right -> wrong compared with base.

The paper's analyses, on the test split
  attribution.csv   evidence AUROC and pointing of the attention scores: layer spans L0-50%, L50-100%, L0-100%
                    vs the profiled layers (+ denoise, as VEA uses them)
  layers_*.png      per layer, correct vs wrong answers: evidence AUROC, and the attention per token on evidence
                    vs non-evidence cells relative to the prompt average ("seeing but not believing")
by_found.csv splits the accuracy by whether the VEA scores peak on the object.

Outputs: profile.json, profile.png, per_sample.jsonl, summary.json, summary.csv, attribution.csv, by_found.csv,
         layers_auroc.png, layers_rapt.png, vis/*.png
"""

import json
import math
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rq1grounding import vea as V  # noqa: E402
from rq1grounding.cli import (add_split_args, append_jsonl, common_parser, load_samples,
                              nanmean, read_jsonl, setup, to_jsonable, write_csv, write_jsonl)  # noqa: E402
from rq1grounding.data import LETTERS  # noqa: E402
from rq1grounding.grounding import auroc_rows  # noqa: E402
from rq1grounding.model import answer_from_logits, forward, prepare, token_factor  # noqa: E402
from rq1grounding.viz import layer_curves, overlay, plt  # noqa: E402

# variant -> (evidence map the image is highlighted with, None = original image; hint in the prompt)
VARIANTS = {"base": (None, False), "inst": (None, True), "vea": ("vea", True), "vea_no_prompt": ("vea", False),
            "vea_no_denoise": ("no_denoise", True), "vea_no_smooth": ("no_smooth", True),
            "vea_no_profiling": ("no_profiling", True), "vea_gt": ("gt", True)}
IMAGES = {None: "original", "vea": "highlighted", "no_denoise": "highlighted, w/o denoise",
          "no_smooth": "highlighted, w/o smoothing", "no_profiling": "highlighted, all layers",
          "gt": "highlighted from GT boxes"}
VEA_SCORES = "profiled + denoise"


def profile(model, processor, calib, args) -> dict:
    """Step A on the calibration split: mean evidence AUROC per layer -> the top `top_frac` of the layers."""
    n_layers = model.config.text_config.num_hidden_layers
    if args.layers:
        return {"layers": sorted(args.layers), "source": "--layers"}
    rows = []
    for s in tqdm(calib, desc="profile layers"):
        prep = prepare(model, processor, s, args.max_pixels, args.min_pixels)
        attn, _ = V.last_token_attention(model, prep)
        rows.append(auroc_rows(attn, prep.gt_cov() > 0))
    rows = [r for r in rows if np.isfinite(r).any()]  # AUROC is undefined if every cell is evidence
    if not rows:
        print("warning: no calibration questions - using all layers (the paper's w/o-profiling setting)")
        return {"layers": list(range(n_layers)), "source": "all layers (no calibration data)"}
    auroc = np.array([nanmean(c) for c in np.stack(rows).T])
    layers = V.select_layers(auroc, args.top_frac)
    return {"layers": layers, "source": f"top {args.top_frac:.0%} of layers by evidence AUROC", "n_samples": len(rows),
            "layer_auroc": auroc, "mean_auroc_all": nanmean(auroc), "mean_auroc_selected": nanmean(auroc[layers]),
            "calib_qids": [s.qid for s in calib]}


def evidence_maps(attn, prep, layers, names, args, cell) -> dict:
    """Steps B-D for the requested maps: name -> pixel map of the model input image, in [0, 1]."""
    prof = attn[layers].mean(0)
    spec = {"vea": (prof, {}), "no_denoise": (prof, {"apply_denoise": False}),
            "no_smooth": (prof, {"apply_smooth": False}), "no_profiling": (attn.mean(0), {}),
            "gt": ((prep.gt_cov() > 0).astype(np.float32), {"apply_denoise": False})}
    return {m: V.evidence_map(spec[m][0], prep.grid_h, prep.grid_w, cell, lam=args.lam, sigma=args.sigma, **spec[m][1])
            for m in names}


def run_question(model, processor, s, layers, spans, args, cell):
    prep = prepare(model, processor, s, args.max_pixels, args.min_pixels)
    attn, logits = V.last_token_attention(model, prep)  # the attribution pass is the base answer's pass
    res = {"base": answer_from_logits(logits, prep)}
    maps = evidence_maps(attn, prep, layers, {VARIANTS[v][0] for v in args.variants} - {None}, args, cell)
    images = {m: V.highlight(prep.image, e, args.alpha) for m, e in maps.items()}
    for v in args.variants[1:]:
        m, hint = VARIANTS[v]
        p = prepare(model, processor, s, args.max_pixels, args.min_pixels, image=images.get(m, prep.image),
                    prefix=args.hint if hint else "")
        res[v] = answer_from_logits(forward(model, p).logits, p)
    gt = prep.gt_cov() > 0
    scores = {name: attn[ls].mean(0) for name, ls in spans.items()}
    scores[VEA_SCORES] = V.denoise(attn[layers].mean(0).reshape(prep.grid_h, prep.grid_w), args.lam).reshape(-1)
    attribution = {k: {"auroc": float(auroc_rows(sc[None], gt)[0]), "pointing": float(gt[sc.argmax()])}
                   for k, sc in scores.items()}
    rec = {"qid": s.qid, "category": s.category, "results": res, "found": attribution[VEA_SCORES]["pointing"] > 0,
           "attribution": attribution, **V.layer_stats(attn, gt, prep.answer_pos + 1)}
    return rec, prep, maps, images


def summarize(recs, variants, n_layers):
    cats = sorted({r["category"] for r in recs})
    right = [r for r in recs if r["results"]["base"]["correct"]]
    wrong = [r for r in recs if not r["results"]["base"]["correct"]]

    def acc(v, rs):
        return nanmean([r["results"][v]["correct"] for r in rs])

    table = [{"method": v, "image": IMAGES[VARIANTS[v][0]], "prompt": "hint" if VARIANTS[v][1] else "plain",
              "accuracy": acc(v, recs), **{c: acc(v, [r for r in recs if r["category"] == c]) for c in cats},
              "fixed": sum(r["results"][v]["correct"] for r in wrong),
              "broken": sum(not r["results"][v]["correct"] for r in right)} for v in variants]
    attribution = [{"scores": k, "auroc": nanmean([r["attribution"][k]["auroc"] for r in recs]),
                    "pointing": nanmean([r["attribution"][k]["pointing"] for r in recs])}
                   for k in (recs[0]["attribution"] if recs else [])]
    deep = range(math.ceil(n_layers / 2), n_layers)  # the paper's "deeper half": layer >= n_layers / 2
    snb = {"n_correct": len(right), "n_wrong": len(wrong)}
    for name, rs in (("correct", right), ("wrong", wrong)):
        snb[f"evidence_auroc_when_{name}"] = nanmean([r["attribution"][VEA_SCORES]["auroc"] for r in rs])
        snb[f"found_rate_when_{name}"] = nanmean([r["found"] for r in rs])
        snb[f"evidence_over_nonevidence_deep_when_{name}"] = nanmean([r["rapt_evd"][l] / r["rapt_nonevd"][l]
                                                                      for r in rs for l in deep])
    by_found = []
    for f in (True, False):
        rs = [r for r in recs if r["found"] == f]
        by_found.append({"vea_scores_peak_on_object": f, "n": len(rs),
                         **{v: acc(v, rs) for v in variants if v in ("base", "inst", "vea", "vea_gt")}})
    return table, attribution, snb, by_found


def plots(recs, prof, out):
    if "layer_auroc" in prof:
        layer_curves({"calibration questions": np.asarray(prof["layer_auroc"])}, "evidence AUROC",
                     os.path.join(out, "profile.png"), f"layer profiling (last token, head average): selected {prof['layers']}",
                     hline=0.5)
    groups = {k: [r for r in recs if r["results"]["base"]["correct"] == (k == "correct")] for k in ("correct", "wrong")}
    auroc = {f"{k} answers": np.stack([r["auroc"] for r in rs]) for k, rs in groups.items() if rs}
    rapt = {f"{part}, {k} answers": np.stack([r[key] for r in rs]) for k, rs in groups.items() if rs
            for part, key in (("evidence", "rapt_evd"), ("non-evidence", "rapt_nonevd"))}
    if auroc:
        layer_curves(auroc, "evidence AUROC", os.path.join(out, "layers_auroc.png"),
                     "last prompt token -> image (head average), test split", hline=0.5)
        layer_curves(rapt, "attention per token / prompt average", os.path.join(out, "layers_rapt.png"),
                     "seeing but not believing: evidence vs non-evidence cells", hline=1.0)


def visualize(prep, rec, maps, images, path):
    """Model input + GT (green) | VEA evidence map | highlighted image the model answers on."""
    s, r = prep.sample, rec["results"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    overlay(axes[0], prep.image, None, s.boxes, f"{s.qid}\ngt ({LETTERS[s.answer_idx]}), base ({LETTERS[r['base']['pred']]})")
    overlay(axes[1], prep.image, maps["vea"], s.boxes, "VEA evidence map")
    overlay(axes[2], images["vea"], None, s.boxes, f"highlighted -> vea ({LETTERS[r['vea']['pred']]})")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    p = common_parser(__doc__)
    add_split_args(p)
    p.add_argument("--top_frac", type=float, default=0.1, help="share of the layers kept by profiling (paper: 10%%)")
    p.add_argument("--layers", type=int, nargs="*", default=None, help="use these layers instead of profiling")
    p.add_argument("--lam", type=float, default=10.0, help="denoise: spike = above lam x every neighbour (paper: 10)")
    p.add_argument("--sigma", type=float, default=0.5, help="smoothing; <= 1: x the shorter image side (paper: 0.5)")
    p.add_argument("--alpha", type=float, default=0.5, help="brightness left where there is no evidence (paper: 0.5)")
    p.add_argument("--hint", default=V.HINT, help="prompt prefix of the variants with the hint")
    p.add_argument("--variants", nargs="*", default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument("--n_vis", type=int, default=12)
    args = p.parse_args()
    args.variants = list(dict.fromkeys(["base", *args.variants]))
    calib, test = load_samples(args)
    model, processor = setup(args)
    n_layers, cell = model.config.text_config.num_hidden_layers, token_factor(model)
    print(f"calibration {len(calib)} / test {len(test)} questions")

    prof = profile(model, processor, calib, args)
    json.dump(to_jsonable(prof), open(os.path.join(args.out, "profile.json"), "w"), indent=2)
    print(f"layers {prof['layers']} ({prof['source']})")
    spans = {"L0-50%": V.layer_span(n_layers, 0, 0.5), "L50-100%": V.layer_span(n_layers, 0.5, 1),
             "L0-100%": list(range(n_layers))}

    out_file = os.path.join(args.out, "per_sample.jsonl")
    done_recs = read_jsonl(out_file)
    done_map = {r["qid"]: r for r in done_recs if "variants" in r}
    recs = []
    os.makedirs(os.path.join(args.out, "vis"), exist_ok=True)
    for i, s in enumerate(tqdm(test, desc="test")):
        if s.qid in done_map:
            recs.append(done_map[s.qid])
            continue
        rec, prep, maps, images = run_question(model, processor, s, prof["layers"], spans, args, cell)
        recs.append(rec)
        append_jsonl(out_file, rec)
        if i < args.n_vis and "vea" in maps:
            visualize(prep, rec, maps, images, os.path.join(args.out, "vis", f"{i:04d}_{s.qid.replace('/', '_')}.png"))
    write_jsonl(out_file, recs)

    table, attribution, snb, by_found = summarize(recs, args.variants, n_layers)
    write_csv(os.path.join(args.out, "summary.csv"), table)
    write_csv(os.path.join(args.out, "attribution.csv"), attribution)
    write_csv(os.path.join(args.out, "by_found.csv"), by_found)
    hyper = {"top_frac": args.top_frac, "lam": args.lam, "sigma": args.sigma, "alpha": args.alpha, "hint": args.hint}
    json.dump(to_jsonable({"n": len(recs), "profile": {k: v for k, v in prof.items() if k != "calib_qids"},
                           "hyperparameters": hyper, "table": table, "seeing_not_believing": snb,
                           "attribution": attribution, "by_found": by_found}),
              open(os.path.join(args.out, "summary.json"), "w"), indent=2)
    plots(recs, prof, args.out)
    print(json.dumps(to_jsonable(snb), indent=1))
    for t in table:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in t.items()})


if __name__ == "__main__":
    main()
