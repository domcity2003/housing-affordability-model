"""MLP on outputs/mats.pkl features (quantile-normalised), averaged over seeds. Saves oof_p_nn.npy.
Run: python src/nn.py
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import QuantileTransformer

OUT = Path(__file__).resolve().parents[1] / "outputs"
SEEDS = (0, 1, 2)
CFG = {"nn": dict(h=(256, 128), alpha=1e-2, bs=256), "nn_big": dict(h=(512, 256, 128), alpha=3e-3, bs=128),
       "nn_small": dict(h=(128, 64), alpha=1e-1, bs=512),
       "nn_big2": dict(h=(512, 256, 128), alpha=3e-3, bs=128, seeds=(3, 4, 5)),
       "nn_deep": dict(h=(384, 384, 192, 96), alpha=1e-2, bs=128),
       "nn_deep2": dict(h=(384, 384, 192, 96), alpha=1e-2, bs=128, seeds=(3, 4, 5)),
       "nn_logit": dict(h=(384, 384, 192, 96), alpha=1e-2, bs=128, logit=True),
       "nn_logit2": dict(h=(384, 384, 192, 96), alpha=1e-2, bs=128, logit=True, seeds=(3, 4, 5)),
       "nn_logit_big": dict(h=(512, 256, 128), alpha=3e-3, bs=128, logit=True),
       "nn_logit_wide": dict(h=(1024, 256), alpha=3e-2, bs=256, logit=True),
       "nn_logit_deep5": dict(h=(256, 256, 256, 128, 64), alpha=1e-2, bs=128, logit=True)}
NAME = next((a for a in sys.argv[1:] if a in CFG), "nn")
SEEDS = CFG[NAME].get("seeds", SEEDS)


def to_t(y):
    """Training target on 0-1 scale, or a logit of it (stretches 0% and 100% ZIPs) for logit configs."""
    p = (y + 1) / 102
    return np.log(p / (1 - p)) / 4 if CFG[NAME].get("logit") else y / 100


def from_t(t):
    return (102 / (1 + np.exp(-4 * t)) - 1) if CFG[NAME].get("logit") else t * 100


def prep(Xa, Xb):
    med = Xa.median()
    qt = QuantileTransformer(n_quantiles=200, output_distribution="normal", subsample=10**6, random_state=0)
    A = qt.fit_transform(Xa.fillna(med))
    B = qt.transform(Xb.fillna(med))
    return A, B


def model(seed):
    c = CFG[NAME]
    return MLPRegressor(hidden_layer_sizes=c["h"], alpha=c["alpha"],
                        learning_rate_init=1e-3, batch_size=c["bs"], max_iter=200, early_stopping=True,
                        validation_fraction=0.1, n_iter_no_change=15, random_state=seed)


def main():
    y, folds, mats, _ = pickle.loads((OUT / "mats.pkl").read_bytes())
    t0, oof = time.time(), np.zeros(len(y))
    for (a, b), (Xa, Xb) in zip(folds, mats):
        A, B = prep(Xa, Xb)
        oof[b] = np.mean([from_t(model(s).fit(A, to_t(y[a])).predict(B)) for s in SEEDS], axis=0)
    oof = np.clip(oof, 0, 100)
    np.save(OUT / f"oof_p_{NAME}.npy", oof)
    print(f"p_{NAME} RMSE {np.sqrt(np.mean((y - oof) ** 2)):.3f} ({time.time() - t0:.0f}s)", flush=True)


def full():
    y, A, T = pickle.loads((OUT / "full_mats.pkl").read_bytes())
    a, t = prep(A, T)
    pred = np.mean([from_t(model(s).fit(a, to_t(y)).predict(t)) for s in SEEDS], axis=0)
    np.save(OUT / f"new_pred_{NAME}.npy", np.clip(pred, 0, 100))
    print(f"saved new_pred_{NAME}.npy", flush=True)


if __name__ == "__main__":
    full() if "--full" in sys.argv else main()
