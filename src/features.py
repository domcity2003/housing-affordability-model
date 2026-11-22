"""Feature building for the new-ZIP model. Uses only the provided dataset (no outside data).

Three feature families:
  - row features: raw columns + ratios (rent vs 30% of median monthly income, etc.)
  - context features: metro/state/city/ZIP3 medians of *features*, computed over train+test
    ZIPs (no target involved), and each ZIP's deviation from its metro
  - target features: smoothed group means and "similar ZIPs in the same metro" means,
    always learned from a fit set that excludes the rows they are applied to
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

TARGET = "AffordabilityPercentageTrue"
LOC_COLS = ["State", "Metro", "City", "Zip3"]
SMOOTH = 20
CTX_COLS = ["LogIncome", "LogRent", "OwnerPercent", "AvgNumberofChildren", "AvgHouseholdSize",
            "MedianHomeLengthofResidence", "RentBurden", "SeniorAdultinHouseholdPercent"]
KNN_COLS = ["LogIncome", "RentBurden", "OwnerPercent", "AvgNumberofChildren", "AvgHouseholdSize",
            "MedianHomeLengthofResidence", "YoungAdultinHouseholdPercent", "SeniorAdultinHouseholdPercent"]
KNN_K = (5, 15)


SHIFT_COLS = ["SampleSize", "RentIndex", "OwnerPercent", "RenterPercent", "MedianHomeLengthofResidence",
              "AvgNumberofChildren", "AvgNumberofAdults", "AvgHouseholdSize", "AvgGenerationsinHousehold",
              "YoungAdultinHouseholdPercent", "SeniorAdultinHouseholdPercent"]


def align_july(tr, te):
    """July features step up (adults, household size, young/senior share +0.2-0.4 sd vs June) while
    May->June barely moved. Subtract the mean June->July change, measured on seen ZIPs, from every
    test row so July rows sit on the training months' scale; test MonthNum is set to June."""
    jun = tr[tr["Month"].str.startswith("6")].set_index("Zip")[SHIFT_COLS]
    jul = te[te["Zip"].isin(jun.index)].set_index("Zip")[SHIFT_COLS]
    shift = (jul - jun.loc[jul.index]).mean()
    te = te.copy()
    te[SHIFT_COLS] = te[SHIFT_COLS] - shift
    return te


def load(data_dir, align=True):
    tr = pd.read_csv(data_dir / "train.csv")
    te = pd.read_csv(data_dir / "test.csv")
    if align:
        te = align_july(tr, te)
    for df in (tr, te):
        df["Zip3"] = df["Zip"] // 100
        df["City"] = df["State"] + "|" + df["City"].astype(str)
        df["Metro"] = df["Metro"].fillna("NONE|" + df["State"])
        df["MonthNum"] = pd.to_datetime(df["Month"], format="%m/%d/%Y").dt.month
        df["LogIncome"] = np.log(df["AnnualMedianHouseholdIncome"])
        df["LogRent"] = np.log(df["RentIndex"])
        # rent as a share of 30% of median monthly income: >1 means a typical renter is burdened
        df["RentBurden"] = df["RentIndex"] / (0.025 * df["AnnualMedianHouseholdIncome"])
        df["IncomePerPerson"] = df["AnnualMedianHouseholdIncome"] / df["AvgHouseholdSize"]
        df["LogSample"] = np.log1p(df["SampleSize"])
        df["ChildrenPerAdult"] = df["AvgNumberofChildren"] / df["AvgNumberofAdults"]
    if align:
        te["MonthNum"] = 6
    return tr, te


def add_context(tr, te):
    """Group medians of features over every ZIP we have features for (latest month per ZIP)."""
    allz = pd.concat([tr[tr.MonthNum == 6], te]).drop_duplicates("Zip", keep="last")
    out = []
    for df in (tr, te):
        d = df.copy()
        for g in ["Metro", "State", "Zip3"]:
            med = allz.groupby(g)[CTX_COLS].median()
            cnt = allz.groupby(g).size()
            for c in CTX_COLS:
                d[f"{g}_med_{c}"] = d[g].map(med[c])
                d[f"{g}_dev_{c}"] = d[c] - d[f"{g}_med_{c}"]
            d[f"{g}_nzips"] = d[g].map(cnt)
        # income vs metro rent: proxy for what local housing costs vs this ZIP's income
        d["MetroRentBurden"] = np.exp(d["Metro_med_LogRent"]) / (0.025 * d["AnnualMedianHouseholdIncome"])
        out.append(d)
    return out


def zip_table(df):
    """One row per ZIP: mean target over its months, plus latest features."""
    last = df.sort_values("MonthNum").drop_duplicates("Zip", keep="last").set_index("Zip")
    last["y"] = df.groupby("Zip")[TARGET].mean()
    return last


def _knn_feats(fit, apply_df, scale):
    """Mean target of the k most similar fit ZIPs in the same metro (falls back to state)."""
    res = {f"knn{k}": np.full(len(apply_df), np.nan) for k in KNN_K}
    res["knn_dist"] = np.full(len(apply_df), np.nan)
    Xf = ((fit[KNN_COLS] - scale[0]) / scale[1]).fillna(0).to_numpy()
    Xa = ((apply_df[KNN_COLS] - scale[0]) / scale[1]).fillna(0).to_numpy()
    yf = fit["y"].to_numpy()
    for level in ["State", "Metro"]:  # metro overwrites state where available
        fit_groups = fit.groupby(level).indices
        for g, ia in apply_df.groupby(level).indices.items():
            idx = fit_groups.get(g)
            if idx is None or len(idx) < 3:
                continue
            kmax = min(max(KNN_K), len(idx))
            nn = NearestNeighbors(n_neighbors=kmax).fit(Xf[idx])
            dist, nb = nn.kneighbors(Xa[ia])
            ys = yf[idx][nb]
            for k in KNN_K:
                kk = min(k, kmax)
                w = 1 / (dist[:, :kk] + 0.5)
                res[f"knn{k}"][ia] = (ys[:, :kk] * w).sum(1) / w.sum(1)
            res["knn_dist"][ia] = dist[:, : min(5, kmax)].mean(1)
    return pd.DataFrame(res, index=apply_df.index)


def encode(fit, apply_df, scale):
    """Target features for apply_df learned only from the fit ZIP table."""
    g = fit["y"].mean()
    out = pd.DataFrame(index=apply_df.index)
    for c in LOC_COLS:
        s = fit.groupby(c)["y"].agg(["sum", "count"])
        out[f"te_{c}"] = apply_df[c].map((s["sum"] + SMOOTH * g) / (s["count"] + SMOOTH)).fillna(g)
        out[f"n_{c}"] = apply_df[c].map(s["count"]).fillna(0)
    out = out.join(_knn_feats(fit, apply_df, scale))
    for k in KNN_K:
        out[f"knn{k}"] = out[f"knn{k}"].fillna(out["te_State"])
    return out


def target_features(train_rows, apply_rows, seed=42, n_inner=5):
    """Encodings for apply_rows from all train ZIPs; for train_rows, out-of-fold inside
    train so a row never sees its own ZIP's target."""
    zt = zip_table(train_rows)
    scale = (zt[KNN_COLS].mean(), zt[KNN_COLS].std())
    apply_enc = encode(zt, apply_rows, scale)
    train_enc = pd.DataFrame(index=train_rows.index, columns=apply_enc.columns, dtype=float)
    rng = np.random.default_rng(seed)
    fold_of = pd.Series(rng.integers(0, n_inner, len(zt)), index=zt.index)
    f = train_rows["Zip"].map(fold_of)
    for k in range(n_inner):
        m = (f == k).to_numpy()
        train_enc.loc[m] = encode(zt[fold_of != k], train_rows[m], scale).to_numpy()
    return train_enc, apply_enc
