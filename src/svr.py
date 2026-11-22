"""SVR (RBF) on the quantile-normalised mats.pkl features: a smooth learner for blend variety.
Run: python src/svr.py [--full]
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.svm import SVR

from nn import prep

OUT = Path(__file__).resolve().parents[1] / "outputs"
C, EPS = 30.0, 1.0


def model(d):
    return SVR(C=C, epsilon=EPS, gamma=1.0 / d, cache_size=2000)


def main():
    y, folds, mats, _ = pickle.loads((OUT / "mats.pkl").read_bytes())
    t0, oof = time.time(), np.zeros(len(y))
    for (a, b), (Xa, Xb) in zip(folds, mats):
        A, B = prep(Xa, Xb)
        oof[b] = model(A.shape[1]).fit(A, y[a]).predict(B)
        print(f"  fold done {time.time() - t0:.0f}s", flush=True)
    oof = np.clip(oof, 0, 100)
    np.save(OUT / "oof_p_svr.npy", oof)
    print(f"p_svr RMSE {np.sqrt(np.mean((y - oof) ** 2)):.3f} ({time.time() - t0:.0f}s)", flush=True)


def full():
    y, A, T = pickle.loads((OUT / "full_mats.pkl").read_bytes())
    a, t = prep(A, T)
    np.save(OUT / "new_pred_svr.npy", np.clip(model(a.shape[1]).fit(a, y).predict(t), 0, 100))
    print("saved new_pred_svr.npy", flush=True)


if __name__ == "__main__":
    full() if "--full" in sys.argv else main()
