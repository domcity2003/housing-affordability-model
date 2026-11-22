"""Final prediction file from saved components: NNLS weights on OOF (oof_p_<c>.npy), applied to test preds
(new_pred_<c>.npy; 'v10blend' uses new_pred_v10.npy). Seen ZIPs: June + k x per-state drift.
Run: python src/combine.py <name> <drift_k> <component ...>
"""
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET, load  # noqa: E402
from train_v6 import seen_drift  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TEST = {"v10blend": os.environ.get("BASE", "new_pred_v10.npy")}  # BASE=new_pred_v13base.npy after July alignment


def main(name, k, comps):
    y, folds, _, _ = pickle.loads((OUT / "mats.pkl").read_bytes())
    P = np.column_stack([np.load(OUT / f"oof_p_{c}.npy") for c in comps])
    nested = np.zeros(len(y))
    for a, b in folds:
        wf, _ = nnls(P[a], y[a])
        nested[b] = P[b] @ wf
    w, _ = nnls(P, y)
    print(dict(zip(comps, (w / w.sum()).round(3))), f"nested new-ZIP CV {np.sqrt(np.mean((y - nested) ** 2)):.3f}")
    T = np.column_stack([np.load(OUT / TEST.get(c, f"new_pred_{c}.npy")) for c in comps])
    tr, te = load(ROOT / "data")
    last = seen_drift(tr, k)
    seen = te["Zip"].isin(last.index).to_numpy()
    # July calibration: models learned May/June levels. On seen ZIPs the July value is known to about
    # +-1.2 (June + drift), so each component's mean gap there is its July bias; remove it everywhere.
    july = te["Zip"][seen].map(last).to_numpy()
    bias = T[seen].mean(0) - july.mean()
    print("July bias per component", dict(zip(comps, bias.round(2))))
    if os.environ.get("CALIB", "1") == "1":
        T = T - bias
    new = T @ w
    pred = np.clip(np.where(seen, te["Zip"].map(last), new), 0, 100)
    np.save(OUT / f"new_pred_{name}.npy", new)
    sub = pd.DataFrame({"Zip": te["Zip"].astype(str).str.zfill(5), "State": te["State"], TARGET: pred})
    ss = pd.read_csv(ROOT / "data" / "sample_submission.csv", dtype=str)
    assert list(sub.columns) == list(ss.columns)
    assert set(zip(sub.Zip, sub.State)) == set(zip(ss.Zip, ss.State))
    assert np.isfinite(sub[TARGET]).all() and len(sub) == len(ss)
    sub.to_csv(OUT / f"{name}.csv", index=False)
    prev = pd.read_csv(OUT / "v10.csv")[TARGET].to_numpy()
    print(f"Wrote {name}.csv | new-ZIP mean abs change vs v10 {np.abs(pred - prev)[~seen].mean():.2f}")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), sys.argv[3:])
