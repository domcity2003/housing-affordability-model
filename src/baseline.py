"""Baseline for Predicting Housing Affordability in the U.S.

Two-part model, matching the metric (0.2 * RMSE_seen + 0.8 * RMSE_new):
  - Seen ZIPs: carry the ZIP's June target forward, plus the average May->June drift.
  - New ZIPs: LightGBM on ZIP features + out-of-fold target encodings of location
    (State, Metro, City, 3-digit ZIP prefix). Validated with GroupKFold by ZIP so a
    ZIP never sits in both train and validation.

Run: python src/baseline.py   (from the project folder)
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "outputs"
TARGET = "AffordabilityPercentageTrue"
SEED = 42
LOC_COLS = ["State", "Metro", "City", "Zip3"]
SMOOTH = 20  # pull small groups toward the global mean


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def load():
    tr = pd.read_csv(DATA / "train.csv")
    te = pd.read_csv(DATA / "test.csv")
    for df in (tr, te):
        df["Zip3"] = df["Zip"] // 100
        df["City"] = df["State"] + "|" + df["City"].astype(str)
        df["Metro"] = df["Metro"].fillna("NONE")
        df["MonthNum"] = pd.to_datetime(df["Month"], format="%m/%d/%Y").dt.month
        # rent-to-income: yearly rent as a share of median income
        df["RentToIncome"] = df["RentIndex"] * 12 / df["AnnualMedianHouseholdIncome"]
        df["IncomePerPerson"] = df["AnnualMedianHouseholdIncome"] / df["AvgHouseholdSize"]
        df["LogSample"] = np.log1p(df["SampleSize"])
    return tr, te


def zip_table(df):
    """One row per ZIP: mean target over its months (avoids double-counting ZIPs)."""
    return df.groupby("Zip").agg(y=(TARGET, "mean"), **{c: (c, "first") for c in LOC_COLS})


def encode(fit_zips, apply_df):
    """Smoothed mean target per location group, learned from fit_zips only."""
    g = fit_zips["y"].mean()
    out = pd.DataFrame(index=apply_df.index)
    for c in LOC_COLS:
        s = fit_zips.groupby(c)["y"].agg(["sum", "count"])
        te_map = (s["sum"] + SMOOTH * g) / (s["count"] + SMOOTH)
        out[f"te_{c}"] = apply_df[c].map(te_map).fillna(g)
        out[f"n_{c}"] = apply_df[c].map(s["count"]).fillna(0)
    return out


def add_encodings(train_rows, apply_rows, n_inner=5):
    """Encodings for apply_rows from all train ZIPs; for train_rows, out-of-fold
    inside train so a row never sees its own target."""
    zt = zip_table(train_rows)
    apply_enc = encode(zt, apply_rows)
    train_enc = pd.DataFrame(index=train_rows.index, columns=apply_enc.columns, dtype=float)
    zips = zt.index.to_numpy()
    rng = np.random.default_rng(SEED)
    fold_of = dict(zip(zips, rng.integers(0, n_inner, len(zips))))
    f = train_rows["Zip"].map(fold_of)
    for k in range(n_inner):
        fit = zt[zt.index.map(fold_of) != k]
        m = f == k
        train_enc.loc[m] = encode(fit, train_rows[m]).values
    return train_enc, apply_enc


BASE_FEATS = [
    "SampleSize", "LogSample", "RentIndex", "AnnualMedianHouseholdIncome", "OwnerPercent",
    "RenterPercent", "MedianHomeLengthofResidence", "AvgNumberofChildren", "AvgNumberofAdults",
    "AvgHouseholdSize", "AvgGenerationsinHousehold", "YoungAdultinHouseholdPercent",
    "SeniorAdultinHouseholdPercent", "RentToIncome", "IncomePerPerson", "Zip", "MonthNum",
]
PARAMS = dict(
    objective="regression", learning_rate=0.03, num_leaves=31, min_child_samples=20,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    n_estimators=3000, random_state=SEED, verbose=-1,
)


def fit_predict(train_rows, apply_rows, n_estimators=None):
    tr_enc, ap_enc = add_encodings(train_rows, apply_rows)
    Xtr = pd.concat([train_rows[BASE_FEATS], tr_enc], axis=1)
    Xap = pd.concat([apply_rows[BASE_FEATS], ap_enc], axis=1)
    params = dict(PARAMS, n_estimators=n_estimators or PARAMS["n_estimators"])
    model = lgb.LGBMRegressor(**params)
    return model, Xtr, Xap


def main():
    tr, te = load()
    OUT.mkdir(exist_ok=True)

    # ---- Seen ZIPs: validate "carry last month forward" by predicting June from May
    p = tr.pivot_table(index="Zip", columns="MonthNum", values=TARGET)
    drift = (p[6] - p[5]).mean()
    seen_cv = rmse(p[6], p[5] + drift)
    print(f"Seen ZIPs  | May->June persistence RMSE: {seen_cv:.3f} (raw, no drift: {rmse(p[6], p[5]):.3f}), drift {drift:+.3f}")

    # ---- New ZIPs: GroupKFold by ZIP
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(tr))
    best_iters = []
    for k, (i_tr, i_va) in enumerate(gkf.split(tr, groups=tr["Zip"])):
        a, b = tr.iloc[i_tr], tr.iloc[i_va]
        model, Xa, Xb = fit_predict(a, b)
        model.fit(Xa, a[TARGET], eval_set=[(Xb, b[TARGET])],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[i_va] = model.predict(Xb, num_iteration=model.best_iteration_)
        best_iters.append(model.best_iteration_)
        print(f"  fold {k}: RMSE {rmse(b[TARGET], oof[i_va]):.3f}  iters {model.best_iteration_}")
    oof = np.clip(oof, 0, 100)
    new_cv = rmse(tr[TARGET], oof)
    june = tr["MonthNum"] == 6
    print(f"New ZIPs   | GroupKFold RMSE: {new_cv:.3f} (June rows only: {rmse(tr.loc[june, TARGET], oof[june]):.3f})")
    print(f"Mean-only reference RMSE: {rmse(tr[TARGET], np.full(len(tr), tr[TARGET].mean())):.3f}")
    est = 0.2 * seen_cv + 0.8 * new_cv
    print(f"Estimated final score: 0.2*{seen_cv:.3f} + 0.8*{new_cv:.3f} = {est:.3f}")

    imp = pd.Series(model.feature_importances_, index=Xa.columns).sort_values(ascending=False)
    print("Top features (last fold, split count):", imp.head(10).to_dict())

    # ---- Fit on all training, predict test
    n_iter = int(np.mean(best_iters) * 1.1)
    model, Xa, Xt = fit_predict(tr, te, n_estimators=n_iter)
    model.fit(Xa, tr[TARGET])
    model_pred = np.clip(model.predict(Xt), 0, 100)

    last = p[6] + drift
    seen = te["Zip"].isin(last.index)
    pred = np.where(seen, te["Zip"].map(last), model_pred)
    pred = np.clip(pred, 0, 100)

    sub = pd.DataFrame({
        "Zip": te["Zip"].astype(int).astype(str).str.zfill(5),
        "State": te["State"],
        TARGET: pred,
    })
    ss = pd.read_csv(DATA / "sample_submission.csv", dtype=str)
    assert list(sub.columns) == list(ss.columns), (sub.columns, ss.columns)
    assert set(zip(sub.Zip, sub.State)) == set(zip(ss.Zip, ss.State)), "ZIP/State pairs differ from sample"
    assert sub[TARGET].notna().all() and np.isfinite(sub[TARGET]).all()
    path = OUT / "v1_baseline.csv"
    sub.to_csv(path, index=False)
    print(f"Wrote {path} | seen {seen.sum()} rows, new {(~seen).sum()} rows")


if __name__ == "__main__":
    main()
