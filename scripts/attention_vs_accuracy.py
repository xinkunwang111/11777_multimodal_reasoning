"""Does Qwen3-VL answer correctly when its attention looks at the right place? (all V* questions)

One forward pass per question; nothing is tuned on the answers and there is no calibration split:
  attention    the question's object-name tokens ("... the color of the [glove]?") -> image tokens,
               averaged over all heads of the layers below; attention-sink tokens removed
  looks right  the attention peak lies on the object's GT box (on both boxes for relation questions)
  AUROC        how well the map ranks the GT box cells above the rest (0.5 = chance, 1 = perfect)
  layers       the top 25% of layers by mean AUROC over all questions. This uses the GT boxes but never
               the answers, so it cannot create a link between attention and correctness (--layers to fix)
Result: accuracy when the attention looks right vs elsewhere (Fisher exact test), accuracy per AUROC
tercile, AUROC of correct vs wrong answers (Mann-Whitney U); overall and per category, since relation
questions are both harder to answer and harder to "look right" on (two objects).

    python scripts/attention_vs_accuracy.py --model checkpoints/Qwen3-VL-8B-Instruct --data_root data/vstar \
        --dtype float16 --out outputs/attention_vs_accuracy        # V100: float16; A100 / H100: bfloat16

Outputs
  summary.json                the numbers above (95% Wilson intervals on every accuracy)
  accuracy_vs_attention.png   the main figure: accuracy by where the attention looks / by AUROC tercile
  layers.png                  per layer: share of questions whose attention peak is on the object, correct vs wrong
  gallery.png                 examples of each case (correct / wrong x looks right / elsewhere): image + GT box,
                              attention map, overlay, zoom on the object; '+' marks the attention peak
  vis/*.png                   the same figure for every question;  per_sample.jsonl
"""

import json
import os
import sys
import textwrap

import matplotlib.patches as patches
import numpy as np
import torch
from PIL import Image
from scipy.stats import fisher_exact, mannwhitneyu
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rq1grounding import geometry as G  # noqa: E402
from rq1grounding.cli import common_parser, load_samples, nanmean, setup, to_jsonable, write_jsonl  # noqa: E402
from rq1grounding.data import LETTERS  # noqa: E402
from rq1grounding.grounding import auroc_rows, map_metrics, peak_point, pointing_rows, record_attention  # noqa: E402
from rq1grounding.model import answer_from_logits, prepare, resize_to_budget, token_factor  # noqa: E402
from rq1grounding.viz import plt  # noqa: E402

CASES = [(True, True, "correct, attention on the object"), (False, False, "wrong, attention elsewhere"),
         (False, True, "wrong, attention on the object"), (True, False, "correct, attention elsewhere")]


@torch.no_grad()
def attention_pass(model, processor, s, args):
    """Answer + per-layer head-averaged attention map [layers, n_img] of every object named in the question."""
    prep = prepare(model, processor, s, args.max_pixels, args.min_pixels)
    am = record_attention(model, prep, args.sink_k)
    rec = {"qid": s.qid, "category": s.category, "question": s.question, "options": s.options,
           "answer": s.answer_idx, "grid": [prep.grid_h, prep.grid_w], "objects": []}
    if not torch.isfinite(am.logits[0, -1]).all():  # e.g. a float16 overflow: never score such a question
        rec["error"] = "non-finite logits"
        return rec, []
    rec.update(answer_from_logits(am.logits, prep))
    maps = []
    for k, o in enumerate(s.objects):
        A = am.maps.get(f"object:{k}")
        if A is None:  # the object's name is not in the question text
            continue
        M = A.float().mean(1).numpy().astype(np.float64)
        if (~am.sinks).any():
            M[:, am.sinks] = 0.0
        M /= M.sum(-1, keepdims=True).clip(1e-12)
        rec["objects"].append({"name": o.name, "boxes": o.boxes})
        maps.append((M, prep.object_cov(k)))
    return rec, maps


def wilson(k: int, n: int, z: float = 1.96) -> dict:
    if n == 0:
        return {"acc": float("nan"), "ci": [float("nan")] * 2, "n": 0}
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {"acc": p, "ci": [max(0.0, c - h), min(1.0, c + h)], "n": n}


def statistics(rs: list) -> dict:
    right = [r for r in rs if r["looks_right"]]
    elsewhere = [r for r in rs if not r["looks_right"]]
    t = [[sum(r["correct"] for r in g), sum(not r["correct"] for r in g)] for g in (right, elsewhere)]
    a, b = t[0]
    c, d = t[1]
    out = {"n": len(rs), "accuracy": wilson(a + c, len(rs)), "looks_right_rate": nanmean([r["looks_right"] for r in rs]),
           "accuracy_if_looks_right": wilson(a, a + b), "accuracy_if_looks_elsewhere": wilson(c, c + d),
           "table": {"looks_right": {"correct": a, "wrong": b}, "looks_elsewhere": {"correct": c, "wrong": d}},
           "fisher_p": float(fisher_exact(t)[1]) if rs else float("nan"),
           "odds_ratio": (a + 0.5) * (d + 0.5) / ((b + 0.5) * (c + 0.5))}  # +0.5: finite with empty cells
    au_c = [r["auroc"] for r in rs if r["correct"] and np.isfinite(r["auroc"])]
    au_w = [r["auroc"] for r in rs if not r["correct"] and np.isfinite(r["auroc"])]
    out.update({"auroc_median_correct": float(np.median(au_c)) if au_c else float("nan"),
                "auroc_median_wrong": float(np.median(au_w)) if au_w else float("nan"),
                "mannwhitney_p": float(mannwhitneyu(au_c, au_w).pvalue) if au_c and au_w else float("nan")})
    return out


def terciles(rs: list) -> list:
    au = np.array([r["auroc"] for r in rs])
    ok = np.isfinite(au)
    if ok.sum() < 3:
        return []
    edges = np.quantile(au[ok], [0, 1 / 3, 2 / 3, 1])
    out = []
    for i in range(3):
        g = [r for r, x in zip(rs, au) if np.isfinite(x) and edges[i] <= x and (x < edges[i + 1] or i == 2)]
        out.append({"auroc_range": [float(edges[i]), float(edges[i + 1])], **wilson(sum(r["correct"] for r in g), len(g))})
    return out


# ---------------------------------------------------------------------------- figures

def main_figure(S: dict, terc: list, path: str):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.6, 1]})
    groups = list(S)
    x = np.arange(len(groups))
    for j, (key, label, color) in enumerate([("accuracy_if_looks_right", "attention peak on the object", "#2a9d8f"),
                                             ("accuracy_if_looks_elsewhere", "attention peak elsewhere", "#e76f51")]):
        v = [S[g][key] for g in groups]
        acc = np.array([e["acc"] for e in v])
        err = np.array([[e["acc"] - e["ci"][0], e["ci"][1] - e["acc"]] for e in v]).T
        xs = x + (j - 0.5) * 0.38
        a1.bar(xs, acc, 0.38, yerr=err, capsize=3, color=color, label=label)
        for xi, e in zip(xs, v):
            a1.text(xi, 0.02, f"n={e['n']}", ha="center", fontsize=7, color="white")
    a1.set_xticks(x, [f"{g}\nFisher p = {S[g]['fisher_p']:.2g}" for g in groups], fontsize=8)
    a1.set_ylim(0, 1.05)
    a1.set_ylabel("answer accuracy (95% CI)")
    a1.set_title("Accuracy by where the object-name attention looks", fontsize=9)
    a1.legend(fontsize=7, loc="lower right")
    if terc:
        acc = np.array([t["acc"] for t in terc])
        err = np.array([[t["acc"] - t["ci"][0], t["ci"][1] - t["acc"]] for t in terc]).T
        a2.errorbar(range(3), acc, yerr=err, marker="o", capsize=3, color="#264653")
        a2.set_xticks(range(3), [f"{t['auroc_range'][0]:.2f}-{t['auroc_range'][1]:.2f}\n(n={t['n']})" for t in terc], fontsize=8)
        a2.set_xlim(-0.4, 2.4)
        a2.set_ylim(0, 1.05)
        a2.set_xlabel("attention AUROC against the GT box (tercile)", fontsize=8)
        a2.set_title("Accuracy by how sharply the attention hits the object", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def layer_figure(rs: list, layers: list, path: str):
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    for name, sel, color in (("correct answers", True, "#2a9d8f"), ("wrong answers", False, "#e76f51")):
        arr = np.array([r["layer_pointing"] for r in rs if r["correct"] == sel])
        if len(arr):
            mu, se = arr.mean(0), arr.std(0) / np.sqrt(len(arr))
            ax.plot(mu, color=color, label=f"{name} (n={len(arr)})")
            ax.fill_between(range(len(mu)), mu - se, mu + se, color=color, alpha=0.2)
    for i in layers:
        ax.axvspan(i - 0.5, i + 0.5, color="gray", alpha=0.12, lw=0)
    ax.set_xlabel("LLM layer (shaded: layers used for the map)")
    ax.set_ylabel("attention peak on the object")
    ax.set_title("Object-name attention, all heads averaged, per layer", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def blend(img: Image.Image, m: np.ndarray, alpha: float = 0.65) -> np.ndarray:
    """Attention fused into the image: colour where the model looks, the plain image elsewhere."""
    m = m / max(float(m.max()), 1e-12)
    up = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR), np.float32) / 255
    a = (alpha * up)[..., None]
    return np.clip(np.asarray(img, np.float32) / 255 * (1 - a) + plt.get_cmap("jet")(up)[..., :3] * a, 0, 1)


def _boxes(ax, boxes, W, H):
    for x0, y0, x1, y1 in boxes:
        ax.add_patch(patches.Rectangle((x0 * W, y0 * H), (x1 - x0) * W, (y1 - y0) * H, fill=False, edgecolor="lime", lw=1.5))


def draw_row(axes, img: Image.Image, r: dict, m: np.ndarray):
    W, H = img.size
    gh, gw = r["grid"]
    gt = [tuple(b) for o in r["objects"] for b in o["boxes"]]
    peaks = [(o["peak"][0] * W, o["peak"][1] * H) for o in r["objects"]]
    over = blend(img, m.reshape(gh, gw))
    verdict = "correct" if r["correct"] else "WRONG"
    where = "on the object" if r["looks_right"] else "elsewhere"
    axes[0].imshow(img)
    axes[0].set_title(f"{r['qid']}\n{textwrap.shorten(r['question'], 70)}\npred ({LETTERS[r['pred']]}), "
                      f"gt ({LETTERS[r['answer']]}): {verdict}", fontsize=7)
    axes[1].imshow(m.reshape(gh, gw), cmap="jet", interpolation="nearest", extent=(0, W, H, 0))
    axes[1].set_title(f"attention of '{' / '.join(o['name'] for o in r['objects'])}'", fontsize=7)
    axes[2].imshow(over)
    axes[2].set_title(f"overlay: peak {where}, AUROC {r['auroc']:.2f}", fontsize=7)
    zb = G.to_px(G.expand(G.union_box(gt), W, H, margin=1.5, min_side=0.2), W, H)
    axes[3].imshow(over[zb[1]:zb[3], zb[0]:zb[2]], extent=(zb[0], zb[2], zb[3], zb[1]))
    axes[3].set_title("zoom on the object", fontsize=7)
    for ax in axes:
        _boxes(ax, gt, W, H)
        if ax is not axes[0]:
            for px, py in peaks:
                ax.plot(px, py, "+", color="white", ms=9, mew=2)
        ax.axis("off")
    axes[3].set_xlim(zb[0], zb[2])
    axes[3].set_ylim(zb[3], zb[1])


def figure(rows: list, path: str):
    fig, axes = plt.subplots(len(rows), 4, figsize=(13, 3.2 * len(rows)), squeeze=False)
    for ax_row, (img, r, m, label) in zip(axes, rows):
        draw_row(ax_row, img, r, m)
        if label:
            ax_row[0].text(-0.05, 0.5, label, transform=ax_row[0].transAxes, rotation=90, ha="right", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------- main

def main():
    p = common_parser(__doc__)
    p.add_argument("--top_frac", type=float, default=0.25, help="share of layers whose attention is averaged")
    p.add_argument("--layers", type=int, nargs="*", default=None, help="use these layers instead")
    p.add_argument("--sink_k", type=float, default=5.0, help="sink token: residual norm > k x median")
    p.add_argument("--gallery_per_case", type=int, default=2)
    p.add_argument("--n_vis", type=int, default=-1, help="per-question figures (-1: all, 0: none)")
    args = p.parse_args()
    samples = load_samples(args)
    model, processor = setup(args)
    print(f"{len(samples)} questions")

    recs, maps = [], []
    for s in tqdm(samples, desc="attention"):
        r, m = attention_pass(model, processor, s, args)
        recs.append(r)
        maps.append(m)
    use = [i for i, r in enumerate(recs) if "error" not in r and maps[i]]
    print(f"scored {len(use)} / {len(recs)} (non-finite logits: {sum('error' in r for r in recs)}, "
          f"no object name in the question: {sum('error' not in r and not maps[i] for i, r in enumerate(recs))})")

    # layers: best localisation over all questions, using the GT boxes only (never the answers)
    layer_auroc = np.array([nanmean(c) for c in np.stack([auroc_rows(M, cov > 0) for i in use for M, cov in maps[i]]).T])
    if args.layers:
        layers = sorted(args.layers)
    else:
        n = max(1, round(args.top_frac * len(layer_auroc)))
        layers = sorted(int(i) for i in np.argsort(np.nan_to_num(layer_auroc, nan=-1))[-n:])
    print(f"layers {layers}")

    shown = {}
    for i in use:
        r = recs[i]
        gh, gw = r["grid"]
        ms = []
        for o, (M, cov) in zip(r["objects"], maps[i]):
            m = M[layers].mean(0)
            m /= max(m.sum(), 1e-12)
            o.update(map_metrics(m, cov))
            o["peak"] = peak_point(m, gh, gw)
            o["layer_pointing"] = pointing_rows(M, cov > 0)
            ms.append(m)
        r["looks_right"] = all(o["pointing"] > 0 for o in r["objects"])
        r["auroc"] = nanmean([o["auroc"] for o in r["objects"]])
        r["layer_pointing"] = np.mean([o["layer_pointing"] for o in r["objects"]], 0)
        shown[i] = np.mean(ms, 0)
    write_jsonl(os.path.join(args.out, "per_sample.jsonl"), recs)

    rs = [recs[i] for i in use]
    S = {"all": statistics(rs), **{c: statistics([r for r in rs if r["category"] == c]) for c in sorted({r["category"] for r in rs})}}
    terc = terciles(rs)
    summary = {"n_questions": len(recs), "n_scored": len(rs), "layers": layers, "layer_auroc": layer_auroc,
               "dtype": args.dtype, "max_pixels": args.max_pixels, "by_category": S, "accuracy_by_auroc_tercile": terc}
    json.dump(to_jsonable(summary), open(os.path.join(args.out, "summary.json"), "w"), indent=2)
    if rs:
        main_figure(S, terc, os.path.join(args.out, "accuracy_vs_attention.png"))
        layer_figure(rs, layers, os.path.join(args.out, "layers.png"))

    by_qid = {s.qid: s for s in samples}
    factor = token_factor(model)

    def image(r):
        s = by_qid[r["qid"]]
        return resize_to_budget(Image.open(s.image_path).convert("RGB"), args.max_pixels, args.min_pixels, factor)

    rows = []
    for correct, right, label in CASES:
        picks = [i for i in use if recs[i]["correct"] == correct and recs[i]["looks_right"] == right]
        rows += [(image(recs[i]), recs[i], shown[i], label) for i in picks[: args.gallery_per_case]]
    if rows:
        figure(rows, os.path.join(args.out, "gallery.png"))
    os.makedirs(os.path.join(args.out, "vis"), exist_ok=True)
    for i in use[: len(use) if args.n_vis < 0 else args.n_vis]:
        r = recs[i]
        case = ("correct" if r["correct"] else "wrong") + ("_on_object" if r["looks_right"] else "_elsewhere")
        figure([(image(r), r, shown[i], None)], os.path.join(args.out, "vis", f"{case}__{r['qid'].replace('/', '_')}.png"))

    for g, st in S.items():
        a, b = st["accuracy_if_looks_right"], st["accuracy_if_looks_elsewhere"]
        print(f"{g:>18}: n={st['n']}  accuracy {st['accuracy']['acc']:.3f} | attention on the object: "
              f"{a['acc']:.3f} (n={a['n']}) vs elsewhere: {b['acc']:.3f} (n={b['n']}), Fisher p={st['fisher_p']:.3g} | "
              f"AUROC median correct {st['auroc_median_correct']:.3f} vs wrong {st['auroc_median_wrong']:.3f}, "
              f"Mann-Whitney p={st['mannwhitney_p']:.3g}")


if __name__ == "__main__":
    main()
