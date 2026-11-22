"""v2: seen ZIPs = June + drift; new ZIPs = NNLS blend of LightGBM/CatBoost (raw and logit
targets) on row + feature-context + out-of-fold target features. Provided data only.

Needs outputs/oof_*.npy and iters_*.npy from src/models.py.
Run: python src/train_v2.py [out_name model ...]   (default: v2_blend + BLEND)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).parent))
from experiments import cols_for, rmse  # noqa: E402
from features import TARGET, add_context, load  # noqa: E402
from models import MODELS, design, fit_one  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
BLEND = ["lgb_reg", "lgb_logit", "cat", "cat_logit"]


def main(out_name="v2_blend", blend=BLEND):
    tr, te = load(ROOT / "data")
    tr, te = add_context(tr, te)
    y = tr[TARGET].to_numpy()

    # blend weights from out-of-fold predictions
    P = np.column_stack([np.load(OUT / f"oof_{n}.npy") for n in blend])
    w, _ = nnls(P, y)
    w = w / w.sum()
    new_cv = rmse(y, P @ w)
    print("weights", dict(zip(blend, w.round(3))), f"new-ZIP CV {new_cv:.3f}")

    # seen ZIPs
    p = tr.pivot_table(index="Zip", columns="MonthNum", values=TARGET)
    drift = (p[6] - p[5]).mean()
    seen_cv = rmse(p[6], p[5] + drift)
    print(f"seen CV {seen_cv:.3f} -> estimated score {0.2 * seen_cv + 0.8 * new_cv:.3f}")

    # new ZIPs: refit every model on all training rows
    Xa, Xt = design(tr, te, cols_for(tr, use_ctx=True))
    preds = []
    for n, wn in zip(blend, w):
        if wn < 1e-3:
            preds.append(np.zeros(len(te)))
            continue
        n_iter = int(np.mean(np.load(OUT / f"iters_{n}.npy")) * 1.1)
        f, _ = fit_one(MODELS[n], Xa, tr[TARGET], n_iter=n_iter)
        preds.append(f(Xt))
        print(f"  fit {n} ({n_iter} rounds)", flush=True)
    model_pred = np.column_stack(preds) @ w

    last = p[6] + drift
    seen = te["Zip"].isin(last.index).to_numpy()
    pred = np.clip(np.where(seen, te["Zip"].map(last), model_pred), 0, 100)

    sub = pd.DataFrame({"Zip": te["Zip"].astype(str).str.zfill(5), "State": te["State"], TARGET: pred})
    ss = pd.read_csv(ROOT / "data" / "sample_submission.csv", dtype=str)
    assert list(sub.columns) == list(ss.columns)
    assert set(zip(sub.Zip, sub.State)) == set(zip(ss.Zip, ss.State)), "ZIP/State pairs differ from sample"
    assert np.isfinite(sub[TARGET]).all() and len(sub) == len(ss)
    path = OUT / f"{out_name}.csv"
    sub.to_csv(path, index=False)
    v1 = pd.read_csv(OUT / "v1_baseline.csv", dtype={"Zip": str})
    diff = (sub[TARGET].to_numpy() - v1[TARGET].to_numpy())[~seen]
    print(f"Wrote {path} | new-ZIP preds vs v1: mean abs change {np.abs(diff).mean():.2f}")


if __name__ == "__main__":
    if len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2:])
    else:
        main()
