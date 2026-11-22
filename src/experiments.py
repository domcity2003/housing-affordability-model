"""Ablation harness for the new-ZIP model: GroupKFold by ZIP, one line per feature set.

Run: python src/experiments.py
"""
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET, add_context, load, target_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ["SampleSize", "LogSample", "RentIndex", "AnnualMedianHouseholdIncome", "OwnerPercent",
       "RenterPercent", "MedianHomeLengthofResidence", "AvgNumberofChildren", "AvgNumberofAdults",
       "AvgHouseholdSize", "AvgGenerationsinHousehold", "YoungAdultinHouseholdPercent",
       "SeniorAdultinHouseholdPercent", "RentBurden", "IncomePerPerson", "ChildrenPerAdult",
       "Zip", "MonthNum"]
PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_child_samples=20,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              n_estimators=4000, random_state=42, verbose=-1)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def cols_for(df, use_ctx):
    cols = list(RAW)
    if use_ctx:
        cols += [c for c in df.columns if "_med_" in c or "_dev_" in c or c.endswith("_nzips")]
        cols += ["MetroRentBurden"]
    return cols


def run(tr, name, use_ctx, te_keep, n_splits=5):
    t0 = time.time()
    oof = np.zeros(len(tr))
    for i_tr, i_va in GroupKFold(n_splits).split(tr, groups=tr["Zip"]):
        a, b = tr.iloc[i_tr], tr.iloc[i_va]
        ea, eb = target_features(a, b)
        keep = [c for c in ea.columns if any(c.startswith(p) for p in te_keep)]
        Xa = pd.concat([a[cols_for(tr, use_ctx)], ea[keep]], axis=1)
        Xb = pd.concat([b[cols_for(tr, use_ctx)], eb[keep]], axis=1)
        m = lgb.LGBMRegressor(**PARAMS)
        m.fit(Xa, a[TARGET], eval_set=[(Xb, b[TARGET])], callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[i_va] = m.predict(Xb, num_iteration=m.best_iteration_)
    oof = np.clip(oof, 0, 100)
    print(f"{name:<38} new-ZIP RMSE {rmse(tr[TARGET], oof):6.3f}   ({time.time() - t0:.0f}s)", flush=True)
    return oof


def main():
    tr, te = load(ROOT / "data")
    tr, te = add_context(tr, te)
    TE = ["te_", "n_"]
    KNN = ["knn"]
    run(tr, "A  raw + group target enc (v1-like)", False, TE)
    run(tr, "B  A + feature context (train+test)", True, TE)
    run(tr, "C  A + similar-ZIP knn", False, TE + KNN)
    run(tr, "D  all", True, TE + KNN)


if __name__ == "__main__":
    main()
