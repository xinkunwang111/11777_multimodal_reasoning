"""Generate the controlled binding set (colored shapes, color-swapped twins, left/right/above/below)."""

import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rq1grounding.synthetic import generate  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--n_images", type=int, default=300)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--min_size", type=float, default=0.04, help="object side / short image side")
    p.add_argument("--max_size", type=float, default=0.12)
    p.add_argument("--clutter", type=int, default=150)
    p.add_argument("--no_swap_pairs", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    s = generate(a.out, a.n_images, (a.width, a.height), (a.min_size, a.max_size), clutter=a.clutter,
                 swap_pairs=not a.no_swap_pairs, seed=a.seed)
    print(len(s), "samples ->", a.out, dict(collections.Counter(x.category for x in s)))
