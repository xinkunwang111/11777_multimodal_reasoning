"""Appendix B - Where is the attribute / position information, and is it bound to the right tokens?

Extracts per-layer residual features and trains linear probes (cross-validated):

  attribute/relation probe: predict the correct answer word (e.g. "red", "left") from
    inbox      mean of image tokens inside the target box      <- where binding *should* live
    outbox     mean of an equal number of random other tokens  <- control
    distractor mean of tokens on the *other* objects (synthetic) <- leakage control
    objword    the object-name token in the question           <- has the text token pulled the attribute in?
    answer     the last prompt token                           <- what the decision reads from
  and on the vision side (what DeepStack hands over): vit_ds{i} (DeepStack level i) and vit_merged.

  position probe (MRoPE): ridge-regress a token's (row, col) on the merged grid from its
    residual state. Qwen3-VL puts 2D layout into rotary angles, not the residual, so this
    shows how much explicit position the visual tokens carry and how it evolves.

Usage: run once to extract (features.npz), re-run with --skip_extract to re-train.
"""

import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from rq1grounding.cli import common_parser, to_jsonable  # noqa: E402
from rq1grounding.data import option_keywords  # noqa: E402
from rq1grounding.viz import layer_curves  # noqa: E402

TOKEN_FEATS = ("inbox", "outbox", "distractor", "objword", "answer")


def extract(args):
    from rq1grounding.cli import load_samples, setup
    from rq1grounding import geometry as G
    from rq1grounding.hooks import HiddenStateRecorder, VisualFeatureEditor
    from rq1grounding.model import answer_from_logits, forward, prepare

    samples = load_samples(args)
    model, processor = setup(args)
    rng = np.random.default_rng(args.seed)
    feats = {k: [] for k in TOKEN_FEATS}
    vit, pos_X, pos_y, pos_g, meta = {}, [], [], [], []
    for si, s in enumerate(tqdm(samples)):
        prep = prepare(model, processor, s, args.max_pixels, args.min_pixels)
        cov = prep.gt_cov()
        inb, rest = np.flatnonzero(cov > 0), np.flatnonzero(cov == 0)
        if len(inb) == 0:
            continue
        inb = rng.choice(inb, min(len(inb), args.max_tokens), replace=False)
        outb = rng.choice(rest, min(len(rest), len(inb)), replace=False)
        dis = np.array([], dtype=int)
        dis_boxes = [tuple(s.meta["neighbor"]["box"])] if "neighbor" in s.meta else [tuple(b) for b in s.meta.get("other_boxes", [])]
        if dis_boxes:
            d = np.setdiff1d(G.region_tokens(dis_boxes, prep.grid_h, prep.grid_w), np.flatnonzero(cov > 0))
            dis = rng.choice(d, min(len(d), args.max_tokens), replace=False) if len(d) else dis
        pp = rng.choice(prep.n_img, min(prep.n_img, args.pos_tokens), replace=False)
        obj = [v[-1] for v in prep.object_pos if v][:1]
        groups = {"inbox": inb, "outbox": outb, "distractor": dis, "pos": pp}
        img_pos = prep.image_pos.cpu().numpy()
        seq = np.concatenate([img_pos[g] for g in groups.values()] + [np.array(obj + [prep.answer_pos])])
        with HiddenStateRecorder(model, torch.as_tensor(seq)) as rec, \
                VisualFeatureEditor(model, record=True) as vfe:
            out = forward(model, prep)
        Hs = rec.stacked[::args.layer_stride] if args.layer_stride > 1 else rec.stacked  # [Ly', n, D]
        o = 0
        sl = {}
        for k, g in groups.items():
            sl[k] = slice(o, o + len(g))
            o += len(g)
        nan = torch.full_like(Hs[:, 0], float("nan"))
        feats["inbox"].append(Hs[:, sl["inbox"]].mean(1))
        feats["outbox"].append(Hs[:, sl["outbox"]].mean(1))
        feats["distractor"].append(Hs[:, sl["distractor"]].mean(1) if len(dis) else nan)
        feats["objword"].append(Hs[:, o] if obj else nan)
        feats["answer"].append(Hs[:, -1])
        pos_X.append(Hs[::args.pos_layer_stride, sl["pos"]].transpose(0, 1).half())  # [k, Lp, D]
        pos_y.append(np.stack([pp // prep.grid_w / max(1, prep.grid_h - 1), pp % prep.grid_w / max(1, prep.grid_w - 1)], 1))
        pos_g.append(np.full(len(pp), si))
        cap = vfe.captured
        for name, x in [("vit_merged", cap["main"])] + [(f"vit_ds{i}", d) for i, d in enumerate(cap["deepstack"])]:
            vit.setdefault(name + "_inbox", []).append(x[inb].mean(0))
            vit.setdefault(name + "_outbox", []).append(x[outb].mean(0))
            vit.setdefault(name + "_pos", []).append(x[pp].half())
        ans = answer_from_logits(out.logits, prep)
        meta.append({"qid": s.qid, "category": s.category, "label": option_keywords(s.options)[s.answer_idx],
                     "group": s.meta.get("pair_id", s.qid), "correct": ans["correct"]})

    np.savez_compressed(
        os.path.join(args.out, "features.npz"),
        **{k: torch.stack(v).half().numpy() for k, v in feats.items()},
        **{k: torch.stack(v).half().numpy() for k, v in vit.items() if not k.endswith("_pos")},
        **{k: torch.cat(v).numpy() for k, v in vit.items() if k.endswith("_pos")},
        pos_X=torch.cat(pos_X).numpy(), pos_y=np.concatenate(pos_y), pos_g=np.concatenate(pos_g),
    )
    with open(os.path.join(args.out, "meta.jsonl"), "w") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")


def cv_probe(X, y, groups, C, n_splits=5, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ok = ~np.isnan(X).any(1)
    X, y, groups = X[ok], y[ok], groups[ok]
    if len(y) == 0:
        return np.nan
    _, counts = np.unique(y, return_counts=True)
    k = int(min(n_splits, counts.min()))
    if k < 2 or len(np.unique(y)) < 2:
        return np.nan
    accs = []
    for tr, te in StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(X, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
        clf.fit(X[tr], y[tr])
        accs.append((clf.predict(X[te]) == y[te]).mean())
    return float(np.mean(accs))


def cv_position(X, y, groups, alpha=10.0):
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import GroupKFold

    pred = np.zeros_like(y)
    for tr, te in GroupKFold(5).split(X, y, groups):
        pred[te] = Ridge(alpha=alpha).fit(X[tr], y[tr]).predict(X[te])
    return float(r2_score(y[:, 0], pred[:, 0])), float(r2_score(y[:, 1], pred[:, 1]))


def train(args):
    F = np.load(os.path.join(args.out, "features.npz"))
    meta = [json.loads(line) for line in open(os.path.join(args.out, "meta.jsonl"))]
    cat = np.array([m["category"] for m in meta])
    labels = np.array([m["label"] for m in meta])
    groups = np.array([m["group"] for m in meta])
    tasks = {"attribute": np.isin(cat, ["direct_attributes", "synth_attr"]),
             "relation": np.isin(cat, ["relative_position", "synth_rel"])}
    res = {}
    for task, mask in tasks.items():
        if mask.sum() < 10:
            continue
        y, g = labels[mask], groups[mask]
        keep = np.isin(y, [c for c, n in zip(*np.unique(y, return_counts=True)) if n >= args.min_class_count])
        y, g = y[keep], g[keep]
        if len(y) < 10:
            print(f"skip {task}: only {len(y)} samples in classes with >= {args.min_class_count} members")
            continue
        majority = float(max(np.unique(y, return_counts=True)[1]) / len(y)) if len(y) else np.nan
        r = {"n": int(len(y)), "classes": sorted(set(y.tolist())), "majority_baseline": majority, "llm": {}, "vit": {}}
        for k in TOKEN_FEATS:
            X = F[k][mask][keep].astype(np.float32)  # [N, Ly, D]
            r["llm"][k] = [cv_probe(X[:, l], y, g, args.C) for l in range(X.shape[1])]
        for k in F.files:
            if k.startswith("vit_") and not k.endswith("_pos"):
                r["vit"][k] = cv_probe(F[k][mask][keep].astype(np.float32), y, g, args.C)
        res[task] = r
        layer_curves({k: np.array(v) for k, v in r["llm"].items()}, "CV probe accuracy",
                     os.path.join(args.out, f"probe_{task}.png"),
                     f"{task} probe (majority={majority:.2f}, n={len(y)}); x = recorded layer index", hline=majority)
        print(task, json.dumps(to_jsonable({"vit": r["vit"], "best_inbox": np.nanmax(r["llm"]["inbox"])})))

    # position probe
    X, y, g = F["pos_X"].astype(np.float32), F["pos_y"], F["pos_g"]
    pos = {"llm_r2_row_col": [cv_position(X[:, l], y, g) for l in range(X.shape[1])]}
    for k in F.files:
        if k.endswith("_pos"):
            pos[k + "_r2_row_col"] = cv_position(F[k].astype(np.float32), y, g)
    res["position"] = pos
    layer_curves({"row R2": np.array([a for a, _ in pos["llm_r2_row_col"]]),
                  "col R2": np.array([b for _, b in pos["llm_r2_row_col"]])}, "R2",
                 os.path.join(args.out, "probe_position.png"),
                 f"grid position from residual (every {args.layer_stride * args.pos_layer_stride} layers)")
    json.dump(to_jsonable(res), open(os.path.join(args.out, "probe_results.json"), "w"), indent=2)


def main():
    p = common_parser(__doc__)
    p.add_argument("--max_tokens", type=int, default=64)
    p.add_argument("--pos_tokens", type=int, default=8, help="random image tokens per image for the position probe")
    p.add_argument("--layer_stride", type=int, default=1)
    p.add_argument("--pos_layer_stride", type=int, default=4)
    p.add_argument("--C", type=float, default=0.05, help="inverse L2 strength of the logistic probe")
    p.add_argument("--min_class_count", type=int, default=4)
    p.add_argument("--skip_extract", action="store_true")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if not args.skip_extract:
        extract(args)
    train(args)


if __name__ == "__main__":
    main()
