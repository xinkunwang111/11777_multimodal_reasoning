"""Generate the controlled binding set (many small shapes, a neighbour next to the target, colour-swapped twins)."""

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
    p.add_argument("--n_objects", type=int, nargs=2, default=[6, 10], metavar=("MIN", "MAX"))
    p.add_argument("--obj_size", type=float, nargs=2, default=[0.03, 0.07], metavar=("MIN", "MAX"),
                   help="object side as a share of the short image side")
    p.add_argument("--clutter", type=int, default=120)
    p.add_argument("--no_swap_pairs", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    s = generate(a.out, a.n_images, (a.width, a.height), tuple(a.n_objects), tuple(a.obj_size), a.clutter,
                 not a.no_swap_pairs, a.seed)
    print(len(s), "questions ->", a.out, dict(collections.Counter(x.category for x in s)))
