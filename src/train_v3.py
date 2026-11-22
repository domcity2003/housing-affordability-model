"""v3-v5: seed-averaged blend for new ZIPs (fit once, cached to outputs/new_pred_v3.npy);
seen ZIPs = June + k * (May->June drift). Writes outputs/<name>.csv.

Run: python src/train_v3.py <name> <drift_multiplier>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from experiments import cols_for  # noqa: E402
from features import TARGET, add_context, load  # noqa: E402
from models import MODELS, design, fit_one  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
# family weights from the NNLS blend; members of a family are averaged equally
FAMILIES = {("lgb_reg", "lgb_reg_s1", "lgb_reg_s2"): 0.34, ("lgb_logit", "lgb_logit_s1"): 0.14,
            ("cat",): 0.37, ("cat_logit",): 0.15}


def new_zip_pred(tr, te):
    cache = OUT / "new_pred_v3.npy"
    if cache.exists():
        return np.load(cache)
    Xa, Xt = design(tr, te, cols_for(tr, use_ctx=True))
    total = np.zeros(len(te))
    for fam, w in FAMILIES.items():
        for n in fam:
            n_iter = int(np.mean(np.load(OUT / f"iters_{n}.npy")) * 1.1)
            f, _ = fit_one(MODELS[n], Xa, tr[TARGET], n_iter=n_iter)
            total += w / len(fam) * f(Xt)
            print(f"  fit {n} ({n_iter} rounds)", flush=True)
    np.save(cache, total)
    return total


def main(name, k):
    tr, te = load(ROOT / "data")
    tr, te = add_context(tr, te)
    model_pred = new_zip_pred(tr, te)
    p = tr.pivot_table(index="Zip", columns="MonthNum", values=TARGET)
    drift = (p[6] - p[5]).mean()
    last = p[6] + k * drift
    seen = te["Zip"].isin(last.index).to_numpy()
    pred = np.clip(np.where(seen, te["Zip"].map(last), model_pred), 0, 100)
    sub = pd.DataFrame({"Zip": te["Zip"].astype(str).str.zfill(5), "State": te["State"], TARGET: pred})
    ss = pd.read_csv(ROOT / "data" / "sample_submission.csv", dtype=str)
    assert list(sub.columns) == list(ss.columns) and len(sub) == len(ss)
    assert set(zip(sub.Zip, sub.State)) == set(zip(ss.Zip, ss.State))
    assert np.isfinite(sub[TARGET]).all()
    sub.to_csv(OUT / f"{name}.csv", index=False)
    print(f"Wrote {name}.csv | seen drift {k * drift:+.3f}")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]))
