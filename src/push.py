"""2026-10-30 push: fast variants on cached fold designs. New-ZIP CV (GroupKFold by ZIP).
Run: python src/push.py <variant ...>
"""
import pickle
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from experiments import cols_for, rmse  # noqa: E402
from features import TARGET, add_context, load  # noqa: E402
from models import design, fit_one, MODELS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
CACHE = OUT / "folds_cache.pkl"
import os
FAST = os.environ.get("FAST", "1") == "1"  # lr .06 (LGB) / .08 (CatBoost) when CPU is shared


def extra(d):
    d = d.copy()
    d["OwnLen"] = d["OwnerPercent"] * d["MedianHomeLengthofResidence"]
    d["SeniorOwn"] = d["SeniorAdultinHouseholdPercent"] * d["OwnerPercent"]
    d["RentShareInc"] = d["RenterPercent"] * d["RentBurden"]
    d["IncVsMetro"] = d["LogIncome"] - d["Metro_med_LogIncome"]
    d["RentVsCity"] = d["LogRent"] - d.groupby("City")["LogRent"].transform("median")
    d["IncVsCity"] = d["LogIncome"] - d.groupby("City")["LogIncome"].transform("median")
    d["AdultsPerHH"] = d["AvgNumberofAdults"] / d["AvgHouseholdSize"]
    return d


RANK_COLS = ["LogIncome", "LogRent", "OwnerPercent", "AvgNumberofChildren", "AvgHouseholdSize",
             "MedianHomeLengthofResidence", "RentBurden", "SeniorAdultinHouseholdPercent",
             "YoungAdultinHouseholdPercent", "AvgNumberofAdults", "LogSample"]


def geo(d, groups=("City", "Metro", "Zip3"), cols=None):
    """City context (median, deviation, size) and within-group percentile ranks, on train+test ZIPs."""
    u = d.drop_duplicates("Zip", keep="last").copy()
    out = pd.DataFrame(index=u["Zip"])
    for g in groups:
        grp = u.groupby(g)
        for c in cols or RANK_COLS:
            out[f"{g}_rk_{c}"] = grp[c].rank(pct=True).to_numpy()
        if g == "City":
            for c in RANK_COLS:
                out[f"City_dev_{c}"] = (u[c] - grp[c].transform("median")).to_numpy()
            out["City_n"] = grp["Zip"].transform("size").to_numpy()
    return out



def knn_space(fit_z, fit_y, fit_grp, app_z, app_grp, M, ks=(5, 15, 40), tag="r"):
    """Distance-weighted mean target of the k nearest fit ZIPs in space M, nationally and within group."""
    from sklearn.neighbors import NearestNeighbors
    Xf, Xa = M.loc[fit_z].to_numpy(), M.loc[app_z].to_numpy()
    out = {}
    nn = NearestNeighbors(n_neighbors=max(ks), algorithm="kd_tree").fit(Xf)
    dist, nb = nn.kneighbors(Xa)
    for k in ks:
        w = 1 / (dist[:, :k] + 0.05)
        out[f"kn{tag}_nat{k}"] = (fit_y[nb[:, :k]] * w).sum(1) / w.sum(1)
    out[f"kn{tag}_natd"] = dist[:, :5].mean(1)
    res = {f"kn{tag}_grp{k}": np.full(len(app_z), np.nan) for k in ks[:2]}
    fg = pd.Series(range(len(fit_z))).groupby(np.asarray(fit_grp)).indices
    for g, ia in pd.Series(range(len(app_z))).groupby(np.asarray(app_grp)).indices.items():
        idx = fg.get(g)
        if idx is None or len(idx) < 3:
            continue
        kk = min(max(ks[:2]), len(idx))
        d2, n2 = NearestNeighbors(n_neighbors=kk, algorithm="kd_tree").fit(Xf[idx]).kneighbors(Xa[ia])
        for k in ks[:2]:
            k2 = min(k, kk)
            w = 1 / (d2[:, :k2] + 0.05)
            res[f"kn{tag}_grp{k}"][ia] = (fit_y[idx][n2[:, :k2]] * w).sum(1) / w.sum(1)
    out.update(res)
    return pd.DataFrame(out)


def knn_feats(tr, a, b, M, grp="State", tag="r", seed=7):
    """kNN target features for fold rows: apply rows from all fit ZIPs; fit rows inner-OOF."""
    ta = tr.iloc[a]
    zy = ta.groupby("Zip")[TARGET].mean()
    zg = ta.drop_duplicates("Zip").set_index("Zip")[grp].loc[zy.index]
    fz, fy = zy.index.to_numpy(), zy.to_numpy()
    ea = pd.DataFrame(index=tr.index[a])
    eb = knn_space(fz, fy, zg.to_numpy(), tr["Zip"].iloc[b].to_numpy(), tr[grp].iloc[b].to_numpy(), M, tag=tag)
    eb.index = tr.index[b]
    fold = np.random.default_rng(seed).integers(0, 5, len(fz))
    zf = pd.Series(fold, index=fz)
    rf = ta["Zip"].map(zf).to_numpy()
    parts = []
    for k in range(5):
        m = fold != k
        r = rf == k
        e = knn_space(fz[m], fy[m], zg.to_numpy()[m], ta["Zip"].to_numpy()[r], ta[grp].to_numpy()[r], M, tag=tag)
        e.index = ta.index[r]
        parts.append(e)
    ea = pd.concat(parts).loc[ta.index]
    return ea, eb


QM_COLS = ["ChildrenPerAdult", "RentBurden", "AvgNumberofAdults", "MedianHomeLengthofResidence",
           "IncomePerPerson", "OwnerPercent", "AvgNumberofChildren", "LogIncome"]


def qmatch_space(fit_y, fit_grp, app_grp, app_q):
    """For each group: the training-target quantile at this ZIP's within-group feature percentile (and 1 - it)."""
    out = {}
    fg = pd.Series(fit_y).groupby(np.asarray(fit_grp)).apply(lambda s: np.sort(s.to_numpy()))
    for c in app_q.columns:
        lo, hi = np.full(len(app_q), np.nan), np.full(len(app_q), np.nan)
        q = app_q[c].to_numpy()
        for i, g in enumerate(app_grp):
            ys = fg.get(g)
            if ys is None or len(ys) < 5 or np.isnan(q[i]):
                continue
            lo[i] = np.quantile(ys, q[i])
            hi[i] = np.quantile(ys, 1 - q[i])
        out[f"qm_{c}"], out[f"qmi_{c}"] = lo, hi
    return pd.DataFrame(out)


def qmatch_feats(tr, a, b, gw, groups=("Metro", "Zip3", "State"), seed=11, qcols=None):
    ta = tr.iloc[a]
    zy = ta.groupby("Zip")[TARGET].mean()
    zt = ta.drop_duplicates("Zip").set_index("Zip").loc[zy.index]
    fz, fy = zy.index.to_numpy(), zy.to_numpy()
    fold = np.random.default_rng(seed).integers(0, 5, len(fz))
    rf = ta["Zip"].map(pd.Series(fold, index=fz)).to_numpy()
    ea, eb = [], []
    for g in groups:
        cols = [f"{g}_rk_{c}" for c in (qcols or QM_COLS)]
        tb = tr.iloc[b]
        e = qmatch_space(fy, zt[g].to_numpy(), tb[g].to_numpy(), gw.loc[tb["Zip"], cols].reset_index(drop=True))
        eb.append(e.add_prefix(g + "_").set_axis(tb.index))
        parts = []
        for k in range(5):
            m, r = fold != k, rf == k
            sub = ta[r]
            e = qmatch_space(fy[m], zt[g].to_numpy()[m], sub[g].to_numpy(),
                             gw.loc[sub["Zip"], cols].reset_index(drop=True))
            parts.append(e.add_prefix(g + "_").set_axis(sub.index))
        ea.append(pd.concat(parts).loc[ta.index])
    return pd.concat(ea, axis=1), pd.concat(eb, axis=1)


def get_data():
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    tr, te = load(ROOT / "data")
    tr, te = add_context(tr, te)
    cols = cols_for(tr, use_ctx=True)
    folds = list(GroupKFold(5).split(tr, groups=tr["Zip"]))
    designs = [design(tr.iloc[i], tr.iloc[j], cols) for i, j in folds]
    obj = (tr, te, cols, folds, designs)
    CACHE.write_bytes(pickle.dumps(obj))
    return obj


def cv(name, spec, tr, folds, designs, add=None):
    t0, y = time.time(), tr[TARGET]
    oof, iters = np.zeros(len(tr)), []
    for (a, b), (Xa, Xb) in zip(folds, designs):
        if add is not None:
            Xa = pd.concat([Xa, add.iloc[a]], axis=1)
            Xb = pd.concat([Xb, add.iloc[b]], axis=1)
        f, best = fit_one(spec, Xa, y.iloc[a], Xb, y.iloc[b])
        oof[b] = f(Xb)
        iters.append(best)
    np.save(OUT / f"oof_{name}.npy", oof)
    np.save(OUT / f"iters_{name}.npy", np.array(iters))
    print(f"{name:<14} {rmse(y, oof):6.3f} iters {iters} ({time.time() - t0:.0f}s)", flush=True)
    return oof


def main(variants):
    tr, te, cols, folds, designs = get_data()
    allx = extra(pd.concat([tr, te]))
    ext = allx.iloc[: len(tr)][["OwnLen", "SeniorOwn", "RentShareInc", "IncVsMetro", "RentVsCity",
                                 "IncVsCity", "AdultsPerHH"]].set_index(tr.index)
    g = geo(pd.concat([tr[tr.MonthNum == 6], te]))
    gx = g.loc[tr["Zip"]].set_index(tr.index)
    ext2 = pd.concat([ext, gx], axis=1)
    for v in variants:
        if v == "base":
            cv("p_base", MODELS["lgb_reg"], tr, folds, designs)
        elif v == "extra":
            cv("p_extra", MODELS["lgb_reg"], tr, folds, designs, ext)
        elif v == "huber":
            spec = dict(kind="lgb", params={})
            import models
            orig = models.lgb_model
            models.lgb_model = lambda **k: orig(**k).set_params(objective="huber", alpha=15.0)
            cv("p_huber", spec, tr, folds, designs)
            models.lgb_model = orig
        elif v == "cat8":
            import models
            spec = dict(kind="cat")
            src = models.cb.CatBoostRegressor
            models.cb.CatBoostRegressor = lambda **k: src(**{**k, "depth": 8, "l2_leaf_reg": 10})
            cv("p_cat8", spec, tr, folds, designs)
            models.cb.CatBoostRegressor = src
        elif v == "cat_extra":
            cv("p_cat_extra", MODELS["cat"], tr, folds, designs, ext)
        elif v == "geo":
            cv("p_geo", MODELS["lgb_reg"], tr, folds, designs, ext2)
        elif v in ("geo_wide", "cat_wide", "geo_wide_s1", "lgbl_wide", "geo_fast"):
            more = RANK_COLS + ["RenterPercent", "AvgGenerationsinHousehold", "IncomePerPerson",
                                "ChildrenPerAdult", "AnnualMedianHouseholdIncome"]
            gw = geo(pd.concat([tr[tr.MonthNum == 6], te]), ("City", "Metro", "Zip3", "State"), more)
            ew = pd.concat([ext, gw.loc[tr["Zip"]].set_index(tr.index)], axis=1)
            spec = {"geo_wide": MODELS["lgb_reg"], "cat_wide": MODELS["cat"],
                    "geo_wide_s1": MODELS["lgb_reg_s1"], "lgbl_wide": MODELS["lgb_logit"],
                    "geo_fast": dict(kind="lgb", params=dict(lr=0.06))}[v]
            cv("p_" + v, spec, tr, folds, designs, ew)
        elif v.startswith("knn"):
            more = RANK_COLS + ["RenterPercent", "AvgGenerationsinHousehold", "IncomePerPerson",
                                "ChildrenPerAdult", "AnnualMedianHouseholdIncome"]
            gw = geo(pd.concat([tr[tr.MonthNum == 6], te]), ("City", "Metro", "Zip3", "State"), more)
            ew = pd.concat([ext, gw.loc[tr["Zip"]].set_index(tr.index)], axis=1)
            mspace = "M" in v.split("_")[0]  # knnM*: metro+state rank space, neighbours within metro
            M = gw[[f"{g}_rk_{c}" for g in (("Metro", "State") if mspace else ("State",)) for c in more]].fillna(0.5)
            kc = OUT / ("knnM_folds.pkl" if mspace else "knn_folds.pkl")
            if not kc.exists():  # build in its own process: sklearn + LightGBM thread pools clash on Windows
                kc.write_bytes(pickle.dumps([knn_feats(tr, a, b, M, grp="Metro" if mspace else "State",
                                                       tag="m" if mspace else "r") for a, b in folds]))
                print("built kNN cache; rerun to train", flush=True)
                return
            extra_parts = pickle.loads(kc.read_bytes())
            if "qm" in v:
                qm2 = "qm2" in v  # + City level, all rank columns
                qc = OUT / ("qm2_folds.pkl" if qm2 else "qm_folds.pkl")
                if not qc.exists():
                    kw = dict(groups=("City", "Metro", "Zip3", "State"), qcols=more) if qm2 else {}
                    qc.write_bytes(pickle.dumps([qmatch_feats(tr, a, b, gw, **kw) for a, b in folds]))
                qm = pickle.loads(qc.read_bytes())
                extra_parts = [(pd.concat([a1, a2], axis=1), pd.concat([b1, b2], axis=1))
                               for (a1, b1), (a2, b2) in zip(extra_parts, qm)]
            if mspace and (OUT / "knn_folds.pkl").exists():  # keep the state-rank kNN too
                base = pickle.loads((OUT / "knn_folds.pkl").read_bytes())
                extra_parts = [(pd.concat([a1, a2], axis=1), pd.concat([b1, b2], axis=1))
                               for (a1, b1), (a2, b2) in zip(base, extra_parts)]
            spec = MODELS["cat"] if "cat" in v else MODELS["lgb_logit"] if "logit" in v else MODELS["lgb_reg"]
            if "_sd" in v:  # seed variant, e.g. knn_qm_sd3
                sd = int(v.split("_sd")[1])
                spec = {**spec, "params": {**spec.get("params", {}), "seed": sd, "ff": 0.4}}
            if FAST and spec["kind"] == "cat":
                import models
                src = models.cb.CatBoostRegressor
                models.cb.CatBoostRegressor = lambda **k: src(**{**k, "learning_rate": 0.08})
            if FAST and spec["kind"] == "lgb":
                spec = {**spec, "params": {**spec.get("params", {}), "lr": 0.06}}
            y = tr[TARGET]
            t0, oof, iters = time.time(), np.zeros(len(tr)), []
            for (a, b), (Xa, Xb), (ka, kb) in zip(folds, designs, extra_parts):
                Xa = pd.concat([Xa, ew.iloc[a], ka], axis=1)
                Xb = pd.concat([Xb, ew.iloc[b], kb], axis=1)
                f, best = fit_one(spec, Xa, y.iloc[a], Xb, y.iloc[b])
                oof[b] = f(Xb)
                iters.append(best)
            np.save(OUT / f"oof_p_{v}{"" if FAST else "_slow"}.npy", oof)
            np.save(OUT / f"iters_p_{v}{"" if FAST else "_slow"}.npy", np.array(iters))
            print(f"p_{v:<12} {rmse(y, oof):6.3f} iters {iters} ({time.time() - t0:.0f}s)", flush=True)
        elif v == "geo_dev":
            cv("p_geo_dev", MODELS["lgb_reg"], tr, folds, designs,
               ext2[[c for c in ext2.columns if "_rk_" not in c]])


if __name__ == "__main__":
    main(sys.argv[1:])
