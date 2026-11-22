"""Model zoo for new ZIPs on the feature set from features.py. Saves out-of-fold
predictions per model to outputs/oof_<name>.npy so blends can be fit afterwards.

Run: python src/models.py [model ...]
"""
import sys
import time
from pathlib import Path

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from experiments import cols_for, rmse  # noqa: E402
from features import TARGET, add_context, load, target_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TE_KEEP = ("te_", "n_", "knn")


def logit(y):
    p = np.clip(y, 0.5, 99.5) / 100
    return np.log(p / (1 - p))


def inv_logit(z):
    return 100 / (1 + np.exp(-z))


def lgb_model(lr=0.02, leaves=15, mcs=40, ff=0.5, l2=5.0, seed=42):
    return lgb.LGBMRegressor(objective="regression", learning_rate=lr, num_leaves=leaves,
                             min_child_samples=mcs, feature_fraction=ff, bagging_fraction=0.8,
                             bagging_freq=1, lambda_l2=l2, n_estimators=8000, random_state=seed, verbose=-1)


MODELS = {
    "lgb_base": dict(kind="lgb", params=dict(lr=0.05, leaves=31, mcs=20, ff=0.7, l2=1.0)),
    "lgb_reg": dict(kind="lgb", params={}),
    "lgb_logit": dict(kind="lgb", params={}, logit=True),
    "cat": dict(kind="cat"),
    "cat_logit": dict(kind="cat", logit=True),
    "lgb_reg_s1": dict(kind="lgb", params=dict(seed=1)),
    "lgb_reg_s2": dict(kind="lgb", params=dict(seed=2)),
    "lgb_logit_s1": dict(kind="lgb", params=dict(seed=1), logit=True),
    "lgb_deep": dict(kind="lgb", params=dict(leaves=63, mcs=30, ff=0.4, l2=10.0, seed=3)),
    "xgb": dict(kind="xgb"),
    "xgb_logit": dict(kind="xgb", logit=True),
}


def fit_one(spec, Xa, ya, Xb=None, yb=None, n_iter=None):
    """Fit; with a validation set use early stopping, else use n_iter rounds."""
    tf = spec.get("logit", False)
    ya_t = logit(ya) if tf else ya
    yb_t = (logit(yb) if tf else yb) if yb is not None else None
    if spec["kind"] == "lgb":
        m = lgb_model(**spec["params"])
        if n_iter:
            m.set_params(n_estimators=n_iter)
            m.fit(Xa, ya_t)
            best = n_iter
        else:
            m.fit(Xa, ya_t, eval_set=[(Xb, yb_t)], callbacks=[lgb.early_stopping(300, verbose=False)])
            best = m.best_iteration_
        pred = lambda X: m.predict(X, num_iteration=best)  # noqa: E731
    elif spec["kind"] == "xgb":
        m = xgb.XGBRegressor(n_estimators=n_iter or 8000, learning_rate=0.02, max_depth=6,
                             min_child_weight=10, subsample=0.8, colsample_bytree=0.5, reg_lambda=5.0,
                             tree_method="hist", random_state=42, n_jobs=-1,
                             early_stopping_rounds=None if n_iter else 300)
        if n_iter:
            m.fit(Xa, ya_t)
            best = n_iter
        else:
            m.fit(Xa, ya_t, eval_set=[(Xb, yb_t)], verbose=False)
            best = m.best_iteration + 1
        pred = lambda X: m.predict(X, iteration_range=(0, best))  # noqa: E731
    else:
        m = cb.CatBoostRegressor(iterations=n_iter or 8000, learning_rate=0.03, depth=6, l2_leaf_reg=5,
                                 random_seed=42, verbose=False, thread_count=-1)
        if n_iter:
            m.fit(Xa, ya_t)
            best = n_iter
        else:
            m.fit(Xa, ya_t, eval_set=(Xb, yb_t), early_stopping_rounds=300)
            best = m.get_best_iteration() + 1
        pred = m.predict
    back = inv_logit if tf else (lambda z: z)
    return (lambda X: np.clip(back(pred(X)), 0, 100)), best


def design(a, b, tr_cols):
    ea, eb = target_features(a, b)
    keep = [c for c in ea.columns if c.startswith(TE_KEEP)]
    return (pd.concat([a[tr_cols], ea[keep]], axis=1), pd.concat([b[tr_cols], eb[keep]], axis=1))


def main(names):
    tr, te = load(ROOT / "data")
    tr, te = add_context(tr, te)
    cols = cols_for(tr, use_ctx=True)
    folds = list(GroupKFold(5).split(tr, groups=tr["Zip"]))
    designs = [design(tr.iloc[i], tr.iloc[j], cols) for i, j in folds]  # reused across models
    OUT.mkdir(exist_ok=True)
    for name in names:
        t0, spec = time.time(), MODELS[name]
        oof, iters = np.zeros(len(tr)), []
        for (i_tr, i_va), (Xa, Xb) in zip(folds, designs):
            f, best = fit_one(spec, Xa, tr[TARGET].iloc[i_tr], Xb, tr[TARGET].iloc[i_va])
            oof[i_va] = f(Xb)
            iters.append(best)
        np.save(OUT / f"oof_{name}.npy", oof)
        np.save(OUT / f"iters_{name}.npy", np.array(iters))
        print(f"{name:<12} new-ZIP RMSE {rmse(tr[TARGET], oof):6.3f}  iters {iters}  ({time.time() - t0:.0f}s)",
              flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(MODELS))
