"""Full-data design for the final refit: all training rows -> test rows, same columns as mats.pkl.
Needs outputs/knn_full.pkl and qm_full.pkl (built by train_v6.py). Saves outputs/full_mats.pkl.
"""
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import push  # noqa: E402
from experiments import cols_for  # noqa: E402
from features import TARGET, add_context, load  # noqa: E402
from mats import EXT, MORE  # noqa: E402
from models import design  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def main():
    tr, te = load(ROOT / "data")
    tr, te = add_context(tr, te)
    Xa, Xt = design(tr, te, cols_for(tr, use_ctx=True))
    allx = push.extra(pd.concat([tr, te]))
    gw = push.geo(pd.concat([tr[tr.MonthNum == 6], te]), ("City", "Metro", "Zip3", "State"), MORE)
    ea = pd.concat([allx.iloc[: len(tr)][EXT].set_index(tr.index), gw.loc[tr["Zip"]].set_index(tr.index)], axis=1)
    et = pd.concat([allx.iloc[len(tr):][EXT].set_index(te.index), gw.loc[te["Zip"]].set_index(te.index)], axis=1)
    ka, kt = pickle.loads((OUT / "knn_full.pkl").read_bytes())
    qa, qt = pickle.loads((OUT / "qm_full.pkl").read_bytes())
    A = pd.concat([Xa, ea, ka.set_axis(tr.index), qa.set_axis(tr.index)], axis=1)
    T = pd.concat([Xt, et, kt.set_axis(te.index), qt.set_axis(te.index)], axis=1)
    cv_cols = pickle.loads((OUT / "mats.pkl").read_bytes())[2][0][0].columns
    assert list(A.columns) == list(cv_cols), "column mismatch vs mats.pkl"
    (OUT / "full_mats.pkl").write_bytes(pickle.dumps((tr[TARGET].to_numpy(), A, T)))
    print("full mats", A.shape, T.shape)


if __name__ == "__main__":
    main()
