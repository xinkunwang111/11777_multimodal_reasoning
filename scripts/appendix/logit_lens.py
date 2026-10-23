"""Appendix A - Is the attribute readable at the object's visual tokens?  Logit lens on image tokens.

Each image token's residual state at every LLM layer is decoded with the final
norm + LM head. Comparing tokens inside the GT box with an equal number of
tokens outside it tells us, layer by layer,
  * object identity : best rank of the object name (e.g. "glove") among in-box vs out-box tokens
  * attribute binding: averaging over in-box tokens, does the correct option word
    (e.g. "red") outscore the distractor options? (out-box tokens = control)
Layer 0..k are where DeepStack injects ViT features, so watch the early layers.

Outputs: arrays.npz, summary.json, layer_*.png, top_tokens.jsonl (qualitative)
"""

import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from rq1grounding.cli import common_parser, load_samples, setup, to_jsonable  # noqa: E402
from rq1grounding.data import option_keywords  # noqa: E402
from rq1grounding.hooks import HiddenStateRecorder, logit_lens  # noqa: E402
from rq1grounding.model import answer_from_logits, forward, prepare  # noqa: E402
from rq1grounding.viz import layer_curves  # noqa: E402


def word_variants(tok, word: str) -> list[int]:
    ids = set()
    for w in (word, " " + word, word.capitalize(), " " + word.capitalize()):
        enc = tok.encode(w, add_special_tokens=False)
        if enc:
            ids.add(enc[0])
    return sorted(ids)


def main():
    p = common_parser(__doc__)
    p.add_argument("--max_tokens", type=int, default=96, help="max in-box (and out-box) tokens decoded per image")
    p.add_argument("--n_top", type=int, default=20, help="samples for which top decoded words are dumped")
    args = p.parse_args()
    samples = load_samples(args)
    model, processor = setup(args)
    tok = processor.tokenizer
    rng = np.random.default_rng(args.seed)

    A = {k: [] for k in ("obj_rank_in", "obj_rank_out", "bind_in", "bind_out", "margin_in", "margin_out")}
    recs, tops = [], []
    for si, s in enumerate(tqdm(samples)):
        prep = prepare(model, processor, s, args.max_pixels, args.min_pixels)
        cov = prep.gt_cov()
        inb, outb = np.flatnonzero(cov > 0), np.flatnonzero(cov == 0)
        if len(inb) == 0 or len(outb) == 0:
            continue
        inb = rng.choice(inb, min(len(inb), args.max_tokens), replace=False)
        outb = rng.choice(outb, min(len(outb), len(inb)), replace=False)
        pos = prep.image_pos.cpu()[np.concatenate([inb, outb])]
        with HiddenStateRecorder(model, pos) as rec:
            out = forward(model, prep)
        ans = answer_from_logits(out.logits, prep)
        H = rec.stacked  # [Ly+1, n, D]
        n_in = len(inb)

        obj_ids = sorted({i for o in s.objects for i in word_variants(tok, o.name.split()[-1])})
        opt_ids = [word_variants(tok, w) for w in option_keywords(s.options)]
        row = {k: [] for k in A}
        top_layers = {}
        for l in range(H.shape[0]):
            lp = logit_lens(model, H[l]).log_softmax(-1)  # [n, V]
            ranks = (lp[:, None, :] > lp[:, obj_ids, None]).sum(-1).min(-1).values.float()  # [n]
            row["obj_rank_in"].append(float(torch.log10(ranks[:n_in].min() + 1)))
            row["obj_rank_out"].append(float(torch.log10(ranks[n_in:].min() + 1)))
            opt = torch.stack([lp[:, ids].max(-1).values for ids in opt_ids], -1)  # [n, n_opt]
            for tag, sl in (("in", slice(0, n_in)), ("out", slice(n_in, None))):
                sc = opt[sl].mean(0)
                others = torch.cat([sc[:s.answer_idx], sc[s.answer_idx + 1:]])
                row[f"bind_{tag}"].append(float(sc.argmax() == s.answer_idx))
                row[f"margin_{tag}"].append(float(sc[s.answer_idx] - others.max()))
            if si < args.n_top and l % 4 == 0:
                mean_lp = lp[:n_in].mean(0)
                top_layers[l] = [tok.decode([i]) for i in mean_lp.topk(10).indices.tolist()]
        for k in A:
            A[k].append(row[k])
        recs.append({"qid": s.qid, "category": s.category, **ans, "n_in": n_in})
        if top_layers:
            tops.append({"qid": s.qid, "question": s.question, "answer": s.options[s.answer_idx],
                         "objects": [o.name for o in s.objects], "in_box_top_tokens_by_layer": top_layers})

    A = {k: np.array(v) for k, v in A.items()}
    np.savez_compressed(os.path.join(args.out, "arrays.npz"), **A)
    with open(os.path.join(args.out, "top_tokens.jsonl"), "w") as f:
        for t in tops:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "per_sample.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    layer_curves({"in-box": A["obj_rank_in"], "out-box": A["obj_rank_out"]}, "log10 best rank of object name",
                 os.path.join(args.out, "layer_obj_rank.png"), "logit lens: object identity at image tokens")
    attr = np.array([r["category"] in ("direct_attributes", "synth_attr") for r in recs])
    for sub, mask in (("attr", attr), ("all", np.ones_like(attr))):
        if mask.any():
            layer_curves({"in-box": A["bind_in"][mask], "out-box": A["bind_out"][mask]},
                         "option-word accuracy", os.path.join(args.out, f"layer_binding_acc_{sub}.png"),
                         f"logit lens binding ({sub})")
            layer_curves({"in-box": A["margin_in"][mask], "out-box": A["margin_out"][mask]},
                         "logp(correct) - max logp(distractor)", os.path.join(args.out, f"layer_binding_margin_{sub}.png"),
                         f"logit lens binding margin ({sub})", hline=0.0)
    summary = {"n": len(recs), "model_accuracy": float(np.mean([r["correct"] for r in recs])),
               **{f"{k}_layer_mean": A[k].mean(0) for k in A}}
    json.dump(to_jsonable(summary), open(os.path.join(args.out, "summary.json"), "w"), indent=2)
    print(f"done: {len(recs)} samples -> {args.out}")


if __name__ == "__main__":
    main()
