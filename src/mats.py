"""Save fold design matrices for the best feature set (v2 features + extras + ranks + rank-kNN +
quantile matching) to outputs/mats.pkl, so tuning / NN / stacking scripts load them directly.
Run: python src/mats.py
"""
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import push  # noqa: E402
from features import TARGET  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs"
MORE = push.RANK_COLS + ["RenterPercent", "AvgGenerationsinHousehold", "IncomePerPerson",
                         "ChildrenPerAdult", "AnnualMedianHouseholdIncome"]
EXT = ["OwnLen", "SeniorOwn", "RentShareInc", "IncVsMetro", "RentVsCity", "IncVsCity", "AdultsPerHH"]


def main():
    tr, te, cols, folds, designs = push.get_data()
    allx = push.extra(pd.concat([tr, te]))
    gw = push.geo(pd.concat([tr[tr.MonthNum == 6], te]), ("City", "Metro", "Zip3", "State"), MORE)
    ew = pd.concat([allx.iloc[: len(tr)][EXT].set_index(tr.index), gw.loc[tr["Zip"]].set_index(tr.index)], axis=1)
    knn = pickle.loads((OUT / "knn_folds.pkl").read_bytes())
    qm = pickle.loads((OUT / "qm_folds.pkl").read_bytes())
    mats = []
    for (a, b), (Xa, Xb), (ka, kb), (qa, qb) in zip(folds, designs, knn, qm):
        mats.append((pd.concat([Xa, ew.iloc[a], ka, qa], axis=1), pd.concat([Xb, ew.iloc[b], kb, qb], axis=1)))
    (OUT / "mats.pkl").write_bytes(pickle.dumps((tr[TARGET].to_numpy(), folds, mats, tr["Zip"].to_numpy())))
    print("features", mats[0][0].shape[1])


if __name__ == "__main__":
    main()
