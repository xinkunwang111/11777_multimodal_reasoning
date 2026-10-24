"""Pre-flight check before the real runs (about a minute on one GPU).

Loads the model, answers a couple of questions, records the object-name attention, runs one attention
knockout and one VEA highlight, printing what it sees. If this passes, every experiment script will run.

    python scripts/check_setup.py --model checkpoints/Qwen3-VL-8B-Instruct --data_root data/vstar
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rq1grounding.attention import AttentionScaler, AttnEdit  # noqa: E402
from rq1grounding.data import LETTERS, load_dataset  # noqa: E402
from rq1grounding.grounding import aggregate, map_metrics, record_attention  # noqa: E402
from rq1grounding.model import DEFAULT_MODEL, answer_from_logits, forward, load_model, prepare, token_factor  # noqa: E402
from rq1grounding.vea import HINT, evidence_map, highlight, last_token_attention  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--dataset", default="vstar")
    p.add_argument("--data_root", default="data/vstar")
    p.add_argument("--max_pixels", type=int, default=1024 * 32 * 32)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--n", type=int, default=2)
    args = p.parse_args()

    import transformers

    print(f"torch {torch.__version__} | transformers {transformers.__version__} | "
          f"CUDA devices: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] or 'none'}")
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    if (major, minor) < (4, 57):
        sys.exit("transformers >= 4.57 is required for Qwen3-VL: pip install -U 'transformers>=4.57'")

    samples = [s for s in load_dataset(args.dataset, args.data_root, None, None) if s.has_gt]
    print(f"{args.dataset}: {len(samples)} questions with GT boxes, e.g. {sorted({s.category for s in samples})}")
    model, processor = load_model(args.model, dtype=args.dtype)
    c = model.config.text_config
    print(f"model {args.model}: {c.num_hidden_layers} layers x {c.num_attention_heads} heads on {model.device}")

    picks = [next(s for s in samples if s.category == cat) for cat in sorted({s.category for s in samples})][: args.n]
    for s in picks:
        prep = prepare(model, processor, s, args.max_pixels)
        am = record_attention(model, prep)
        ans = answer_from_logits(am.logits, prep)
        print(f"\n[{s.qid}] {s.question}")
        print(f"  options {s.options} | pred ({LETTERS[ans['pred']]}) gt ({LETTERS[s.answer_idx]})")
        print(f"  image {prep.image.size} -> {prep.grid_h}x{prep.grid_w} visual tokens, sinks {int(am.sinks.sum())}")
        for k, o in enumerate(s.objects):
            A = am.maps.get(f"object:{k}")
            if A is None:
                print(f"  object '{o.name}': name not found in the question tokens")
                continue
            m = map_metrics(aggregate(A, sinks=am.sinks), prep.object_cov(k))
            print(f"  object '{o.name}': {len(prep.object_pos[k])} name tokens, GT covers "
                  f"{int((prep.object_cov(k) > 0).sum())} cells, attention pointing {m['pointing']:.0f}, AUROC {m['auroc']:.2f}")
        keys = prep.tokens_of(s.boxes).tolist()
        with AttentionScaler(model, [AttnEdit(list(range(c.num_hidden_layers)), prep.text_pos, keys, "scale", 0.0)]):
            ko = answer_from_logits(forward(model, prep).logits, prep)
        print(f"  knockout of the object in all layers: log p(correct) {ans['logp_correct']:.3f} -> {ko['logp_correct']:.3f}")
        assert np.isfinite(ko["logp_correct"])
        attn, _ = last_token_attention(model, prep)
        hl = highlight(prep.image, evidence_map(attn.mean(0), prep.grid_h, prep.grid_w, token_factor(model)))
        p = prepare(model, processor, s, args.max_pixels, image=hl, prefix=HINT)
        va = answer_from_logits(forward(model, p).logits, p)
        print(f"  VEA (all layers): answer on the highlighted image ({LETTERS[va['pred']]}), "
              f"log p(correct) {va['logp_correct']:.3f}")
        assert p.n_img == prep.n_img and np.isfinite(va["logp_correct"])
    print("\nOK - setup works")


if __name__ == "__main__":
    main()
