"""Rank model: predict a ZIP's target percentile within its group (Metro / Zip3 / State) from
outputs/mats.pkl features, then map it through the group's training-target quantiles.
Run: python src/rankmodel.py <group ...>
"""
import pickle
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import load  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
MIN_N = 8  # groups smaller than this fall back to State


def group_of(tr, g, fit_idx):
    """Group label per row; small groups (in the fit set) fall back to State."""
    lab = tr[g].astype(str).to_numpy()
    n = pd.Series(tr["Zip"].to_numpy()[fit_idx]).groupby(lab[fit_idx]).nunique()
    small = ~pd.Series(lab).map(n).fillna(0).ge(MIN_N).to_numpy()
    return np.where(small, "S|" + tr["State"].to_numpy(), lab)


def pct_labels(y, grp):
    return pd.Series(y).groupby(grp).rank(pct=True, method="average").to_numpy()


def mapq(pred_pct, grp_apply, y_fit, grp_fit):
    srt = pd.Series(y_fit).groupby(grp_fit).apply(lambda s: np.sort(s.to_numpy()))
    allq = np.sort(y_fit)
    out = np.empty(len(pred_pct))
    for i, (q, g) in enumerate(zip(np.clip(pred_pct, 0, 1), grp_apply)):
        ys = srt.get(g, allq)
        out[i] = np.quantile(ys, q)
    return out


def params(seed=0):
    return dict(objective="regression", learning_rate=0.03, num_leaves=15, min_child_samples=40,
                colsample_bytree=0.4, subsample=0.8, subsample_freq=1, reg_lambda=5.0,
                n_estimators=6000, random_state=seed, verbose=-1)


def main(groups):
    tr, _ = load(ROOT / "data")
    y, folds, mats, _ = pickle.loads((OUT / "mats.pkl").read_bytes())
    for g in groups:
        t0, oof, iters = time.time(), np.zeros(len(y)), []
        for (a, b), (Xa, Xb) in zip(folds, mats):
            grp = group_of(tr, g, a)
            ya = pct_labels(y[a], grp[a])
            # early-stop on the val rows' percentile within their own group (fit-set quantiles)
            m = lgb.LGBMRegressor(**params()).fit(Xa, ya, eval_set=[(Xb, pct_labels(y[b], grp[b]))],
                                                  callbacks=[lgb.early_stopping(200, verbose=False)])
            iters.append(m.best_iteration_)
            oof[b] = mapq(m.predict(Xb, num_iteration=m.best_iteration_), grp[b], y[a], grp[a])
        np.save(OUT / f"oof_p_rank_{g}.npy", oof)
        np.save(OUT / f"iters_p_rank_{g}.npy", np.array(iters))
        print(f"p_rank_{g:<6} {np.sqrt(np.mean((y - oof) ** 2)):.3f} iters {iters} ({time.time() - t0:.0f}s)",
              flush=True)


def full(groups):
    tr, _ = load(ROOT / "data")
    y, A, T = pickle.loads((OUT / "full_mats.pkl").read_bytes())
    te = pd.read_csv(ROOT / "data" / "test.csv")
    te["Metro"] = te["Metro"].fillna("NONE|" + te["State"])
    te["Zip3"] = te["Zip"] // 100
    both = pd.concat([tr[["Zip", "State", "Metro", "Zip3"]], te[["Zip", "State", "Metro", "Zip3"]]], ignore_index=True)
    fit = np.arange(len(tr))
    for g in groups:
        grp = group_of(both, g, fit)
        n_iter = int(np.mean(np.load(OUT / f"iters_p_rank_{g}.npy")) * 1.1)
        m = lgb.LGBMRegressor(**{**params(), "n_estimators": n_iter}).fit(A, pct_labels(y, grp[fit]))
        pred = mapq(m.predict(T), grp[len(tr):], y, grp[fit])
        np.save(OUT / f"new_pred_rank_{g}.npy", pred)
        print(f"saved new_pred_rank_{g}.npy ({n_iter} rounds)", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--full"]
    (full if "--full" in sys.argv else main)(args or ["Metro", "Zip3", "State"])
