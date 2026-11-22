"""LightGBM with the Optuna-best params (outputs/tune_best.json), averaged over 3 seeds.
Run: python src/tuned.py         -> oof_p_tuned.npy (CV on mats.pkl)
     python src/tuned.py --full  -> new_pred_tuned.npy (fit on full_mats.pkl)
"""
import json
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "outputs"
BEST = json.loads((OUT / "tune_best.json").read_text())
SEEDS = (0, 1, 2)


def params(seed, n=6000):
    p = {k: v for k, v in BEST.items() if k not in ("iters", "cv")}
    return dict(objective="regression", learning_rate=0.03, verbose=-1, n_estimators=n, subsample_freq=1,
                random_state=seed, **p)


def main():
    y, folds, mats, _ = pickle.loads((OUT / "mats.pkl").read_bytes())
    oof, iters = np.zeros(len(y)), []
    for (a, b), (Xa, Xb) in zip(folds, mats):
        ps = []
        for s in SEEDS:
            m = lgb.LGBMRegressor(**params(s)).fit(Xa, y[a], eval_set=[(Xb, y[b])],
                                                   callbacks=[lgb.early_stopping(200, verbose=False)])
            ps.append(m.predict(Xb, num_iteration=m.best_iteration_))
            iters.append(m.best_iteration_)
        oof[b] = np.clip(np.mean(ps, axis=0), 0, 100)
    np.save(OUT / "oof_p_tuned.npy", oof)
    np.save(OUT / "iters_p_tuned.npy", np.array(iters))
    print(f"p_tuned RMSE {np.sqrt(np.mean((y - oof) ** 2)):.3f} mean iters {np.mean(iters):.0f}", flush=True)


def full():
    y, A, T = pickle.loads((OUT / "full_mats.pkl").read_bytes())
    n = int(np.mean(np.load(OUT / "iters_p_tuned.npy")) * 1.1)
    pred = np.mean([lgb.LGBMRegressor(**params(s, n)).fit(A, y).predict(T) for s in SEEDS], axis=0)
    np.save(OUT / "new_pred_tuned.npy", np.clip(pred, 0, 100))
    print(f"saved new_pred_tuned.npy ({n} rounds)", flush=True)


if __name__ == "__main__":
    full() if "--full" in sys.argv else main()
