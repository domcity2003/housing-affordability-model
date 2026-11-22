"""Seen-ZIP July variants on top of an existing submission (new-ZIP rows are kept as they are).
Drift = state May->June change (shrunk to national) + metro deviation (shrunk to state, weight LAM),
times a level factor (near-0% and 90%+ ZIPs move less), times k.
Run: python src/seen.py <base.csv> <out_name> <k> <metro:0|1> <levels:0|1>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
Y = "AffordabilityPercentageTrue"
LAM = 15
BINS = [0, 5, 13, 30, 90, 101]


def drift(k, metro, levels):
    tr = pd.read_csv(ROOT / "data" / "train.csv")
    tr["Metro"] = tr["Metro"].fillna("NONE|" + tr["State"])
    p = tr.pivot_table(index=["Zip", "State", "Metro"], columns="Month", values=Y).reset_index()
    p["lvl"], p["d"] = p["6/1/2026"], p["6/1/2026"] - p["5/1/2026"]
    g = p["d"].mean()
    s = p.groupby("State")["d"].agg(["sum", "count"])
    sd = p["State"].map((s["sum"] + 30 * g) / (s["count"] + 30))
    if metro:
        m = p.groupby("Metro")["d"].agg(["sum", "count"])
        p["sd"] = sd
        base = p.groupby("Metro")["sd"].mean()
        d = p["Metro"].map((m["sum"] + LAM * base) / (m["count"] + LAM))
    else:
        d = sd
    if levels:
        b = pd.cut(p["lvl"], BINS, right=False)
        fac = p.groupby(b, observed=True)["d"].mean() / p.loc[p["lvl"].between(30, 90), "d"].mean()
        d = d * b.map(fac).astype(float)
    else:
        d = d * np.where(p["lvl"] <= 13, 0.5, 1.0)
    return pd.Series((p["lvl"] + k * d).to_numpy(), index=p["Zip"])


def main(base, name, k, metro, levels):
    sub = pd.read_csv(OUT / base, dtype={"Zip": str})
    last = drift(k, metro, levels)
    z = sub["Zip"].astype(int)
    seen = z.isin(last.index)
    sub.loc[seen, Y] = np.clip(z[seen].map(last), 0, 100)
    assert np.isfinite(sub[Y]).all()
    sub.to_csv(OUT / f"{name}.csv", index=False)
    old = pd.read_csv(OUT / base)[Y]
    print(f"Wrote {name}.csv | seen rows changed, mean abs {np.abs(sub[Y] - old)[seen].mean():.3f}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], float(a[2]), a[3] == "1", a[4] == "1")
