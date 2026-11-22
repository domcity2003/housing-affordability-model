"""Nested-CV check of NNLS blends over saved OOF predictions (weights fit on 4 folds, scored on the 5th).
Run: python src/blend.py name1 name2 ...   (names without the oof_ prefix; 'v3old' = v2/v3 blend)
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

OUT = Path(__file__).resolve().parents[1] / "outputs"


def oof(n):
    if n == "v3old":
        return np.column_stack([np.load(OUT / f"oof_{m}.npy") for m in ["lgb_reg", "lgb_logit", "cat", "cat_logit"]]) \
            @ np.array([.303, .155, .394, .149])
    return np.load(OUT / f"oof_{n}.npy")


def main(names):
    y, folds, _, _ = pickle.loads((OUT / "mats.pkl").read_bytes())
    P = np.column_stack([oof(n) for n in names])
    nested = np.zeros(len(y))
    for a, b in folds:
        w, _ = nnls(P[a], y[a])
        nested[b] = P[b] @ w
    w, _ = nnls(P, y)
    print(dict(zip(names, (w / w.sum()).round(3))))
    print(f"in-sample {np.sqrt(np.mean((y - P @ w) ** 2)):.3f}  nested {np.sqrt(np.mean((y - nested) ** 2)):.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
