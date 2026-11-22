"""v6+: new ZIPs = NNLS blend of models with v2 features + ZIP-vs-city extras + within-group
percentile ranks (+ rank-space kNN target means); seen ZIPs = June + k x per-state drift.
Provided data only. Needs outputs/oof_p_*.npy / iters_p_*.npy from src/push.py.
Run: python src/train_v6.py <name> <drift_k> <model ...>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).parent))
import push  # noqa: E402
from experiments import cols_for, rmse  # noqa: E402
from features import TARGET, add_context, load  # noqa: E402
from models import MODELS, design, fit_one  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
MORE = push.RANK_COLS + ["RenterPercent", "AvgGenerationsinHousehold", "IncomePerPerson",
                         "ChildrenPerAdult", "AnnualMedianHouseholdIncome"]
EXT = ["OwnLen", "SeniorOwn", "RentShareInc", "IncVsMetro", "RentVsCity", "IncVsCity", "AdultsPerHH"]
SPEC = {"geo_wide": "lgb_reg", "geo_wide_s1": "lgb_reg_s1", "cat_wide": "cat", "lgbl_wide": "lgb_logit",
        "knn": "lgb_reg", "knn_cat": "cat", "knn_logit": "lgb_logit", "knn_s1": "lgb_reg_s1",
        "knn_qm": "lgb_reg", "knn_qm_slow": "lgb_reg", "knn_qm_sd3_slow": "lgb_reg", "knn_logit_qm_sd5_slow": "lgb_logit", "knn_cat_qm_slow": "cat", "knn_logit_qm_slow": "lgb_logit"}


def seen_drift(tr, k):
    """Per-ZIP July drift: state mean May->June change (shrunk to national), scaled by k."""
    p = tr.pivot_table(index=["Zip", "State"], columns="MonthNum", values=TARGET).reset_index()
    p["d"] = p[6] - p[5]
    g = p["d"].mean()
    s = p.groupby("State")["d"].agg(["sum", "count"])
    sd = (s["sum"] + 30 * g) / (s["count"] + 30)
    low = p[6] <= 13  # near-0% ZIPs move about half as much
    d = p["State"].map(sd) * np.where(low, 0.5, 1.0)
    return pd.Series((p[6] + k * d).to_numpy(), index=p["Zip"])


def main(name, k, models):
    tr, te = load(ROOT / "data")
    tr, te = add_context(tr, te)
    y = tr[TARGET].to_numpy()
    old_w = np.array([.303, .155, .394, .149])  # v2/v3 blend; its test preds are cached in new_pred_v3.npy
    old_oof = np.column_stack([np.load(OUT / f"oof_{n}.npy") for n in ["lgb_reg", "lgb_logit", "cat", "cat_logit"]]) @ old_w
    P = np.column_stack([old_oof if m == "v3old" else np.load(OUT / f"oof_p_{m}.npy") for m in models])
    w, _ = nnls(P, y)
    w = w / w.sum()
    print("weights", dict(zip(models, w.round(3))), f"new-ZIP CV {rmse(y, P @ w):.3f}", flush=True)

    Xa, Xt = design(tr, te, cols_for(tr, use_ctx=True))
    allx = push.extra(pd.concat([tr, te]))
    gw = push.geo(pd.concat([tr[tr.MonthNum == 6], te]), ("City", "Metro", "Zip3", "State"), MORE)
    ea = pd.concat([allx.iloc[: len(tr)][EXT].set_index(tr.index), gw.loc[tr["Zip"]].set_index(tr.index)], axis=1)
    et = pd.concat([allx.iloc[len(tr):][EXT].set_index(te.index), gw.loc[te["Zip"]].set_index(te.index)], axis=1)
    Xa, Xt = pd.concat([Xa, ea], axis=1), pd.concat([Xt, et], axis=1)
    if any(m.startswith("knn") for m in models):
        kc = OUT / "knn_full.pkl"
        if not kc.exists():  # own process: sklearn + LightGBM thread pools clash on Windows
            both = pd.concat([tr, te.assign(**{TARGET: np.nan})], ignore_index=True)
            M = gw[[f"State_rk_{c}" for c in MORE]].fillna(0.5)
            kc.write_bytes(push.pickle.dumps(push.knn_feats(both, np.arange(len(tr)), np.arange(len(tr), len(both)), M)))
            print("built full kNN cache; rerun", flush=True)
            return
        ka, kt = push.pickle.loads(kc.read_bytes())
        if any("_qm" in m for m in models):
            qc = OUT / "qm_full.pkl"
            if not qc.exists():
                both = pd.concat([tr, te.assign(**{TARGET: np.nan})], ignore_index=True)
                qc.write_bytes(push.pickle.dumps(push.qmatch_feats(both, np.arange(len(tr)),
                                                                   np.arange(len(tr), len(both)), gw)))
            qa, qt = push.pickle.loads(qc.read_bytes())
            Xa_q = pd.concat([Xa, ka.set_axis(tr.index), qa.set_axis(tr.index)], axis=1)
            Xt_q = pd.concat([Xt, kt.set_axis(te.index), qt.set_axis(te.index)], axis=1)
        Xa_k = pd.concat([Xa, ka.set_axis(tr.index)], axis=1)
        Xt_k = pd.concat([Xt, kt.set_axis(te.index)], axis=1)
    preds = []
    for m, wm in zip(models, w):
        if wm < 1e-3:
            preds.append(np.zeros(len(te)))
            continue
        if m == "v3old":
            preds.append(np.load(OUT / "new_pred_v3.npy"))
            continue
        n_iter = int(np.mean(np.load(OUT / f"iters_p_{m}.npy")) * 1.1)
        A, B = (Xa_q, Xt_q) if "_qm" in m else (Xa_k, Xt_k) if m.startswith("knn") else (Xa, Xt)
        spec = MODELS[SPEC[m]]
        if "_sd" in m:  # push.py seed variant: seed N, feature_fraction .4
            sd = int(m.split("_sd")[1].split("_")[0])
            spec = {**spec, "params": {**spec.get("params", {}), "seed": sd, "ff": 0.4}}
        fast = m.startswith("knn") and not m.endswith("_slow")
        if fast and spec["kind"] == "lgb":  # trained in push.py with FAST lr
            spec = {**spec, "params": {**spec.get("params", {}), "lr": 0.06}}
        if fast and spec["kind"] == "cat":  # push.py FAST CatBoost lr
            import models
            src = models.cb.CatBoostRegressor
            models.cb.CatBoostRegressor = lambda **kw: src(**{**kw, "learning_rate": 0.08})
        f, _ = fit_one(spec, A, tr[TARGET], n_iter=n_iter)
        if fast and spec["kind"] == "cat":
            models.cb.CatBoostRegressor = src
        preds.append(f(B))
        print(f"  fit {m} ({n_iter} rounds)", flush=True)
    new = np.column_stack(preds) @ w
    last = seen_drift(tr, k)
    seen = te["Zip"].isin(last.index).to_numpy()
    pred = np.clip(np.where(seen, te["Zip"].map(last), new), 0, 100)
    np.save(OUT / f"new_pred_{name}.npy", new)

    sub = pd.DataFrame({"Zip": te["Zip"].astype(str).str.zfill(5), "State": te["State"], TARGET: pred})
    ss = pd.read_csv(ROOT / "data" / "sample_submission.csv", dtype=str)
    assert list(sub.columns) == list(ss.columns)
    assert set(zip(sub.Zip, sub.State)) == set(zip(ss.Zip, ss.State))
    assert np.isfinite(sub[TARGET]).all() and len(sub) == len(ss)
    sub.to_csv(OUT / f"{name}.csv", index=False)
    v5 = pd.read_csv(OUT / "v5.csv", dtype={"Zip": str})
    print(f"Wrote {name}.csv | new-ZIP mean abs change vs v5 "
          f"{np.abs(pred - v5[TARGET].to_numpy())[~seen].mean():.2f}")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), sys.argv[3:])
