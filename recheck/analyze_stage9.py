"""
Stage-9 analysis, implementing prereg_stage9_reverse_indep.md. Deterministic (no resampling).
Usage: python analyze_stage9.py [cells.jsonl] [earlier_reverse_cells.jsonl] [outdir]
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

R = Path("p9_recheck/results")
_RD = Path(__file__).resolve().parent
CELLS = Path(sys.argv[1]) if len(sys.argv) > 1 else R / "cells_stage9_D_xjtu_rev.jsonl"
OLD = Path(sys.argv[2]) if len(sys.argv) > 2 else next((p for p in [R / "cells_stage3_exp2.jsonl", _RD / "stage3/cells_stage3_exp2.jsonl"] if p.exists()), R / "cells_stage3_exp2.jsonl")
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else R
BAND = ["awgn@+0dB", "awgn@-2dB", "awgn@-6dB"]; RULES = ["final", "oracle"]
ARMS = ["s4d", "bm3_frozen", "frozen_nogate", "frozen_clti"]


def load(p): return pd.DataFrame([json.loads(l) for l in open(p)])
def vals(d, a, c, r="final"):
    s = d[(d.arm == a) & (d.condition == c)].set_index("seed").sort_index()[r] * 100
    assert len(s) == 5, (a, c, len(s)); return s.values
def paired(d, a, b, c, r):
    x = vals(d, a, c, r) - vals(d, b, c, r); h = stats.t.ppf(.975, 4) * x.std(ddof=1) / np.sqrt(5)
    return dict(mean=float(x.mean()), lo=float(x.mean() - h), hi=float(x.mean() + h), n_pos=int((x > 0).sum()))
def label(p):
    pos = sum(q["mean"] > 0 and q["lo"] > 0 for q in p); rev = sum(q["mean"] < 0 and q["hi"] < 0 for q in p)
    return "REVERSED" if rev else {0: "DOES NOT HOLD", 1: "WEAKENED"}.get(pos, "HOLDS")


d = load(CELLS); d = d[~d.get("smoke", False).astype(bool)] if "smoke" in d else d
assert len(d) == 60 and set(d.arm) == set(ARMS)
old = load(OLD)
means, cons, dec = [], [], {}
for r in RULES:
    for a in ARMS:
        for c in BAND:
            x = vals(d, a, c, r); means.append(dict(rule=r, arm=a, condition=c, mean=round(x.mean(), 3), sd=round(x.std(ddof=1), 3)))
for key, a, b in [("L1f_rev", "frozen_clti", "s4d"), ("L1p_rev", "bm3_frozen", "s4d"), ("L2_rev", "bm3_frozen", "frozen_nogate"),
                  ("nogate_minus_s4d", "frozen_nogate", "s4d")]:
    out = {}
    for r in RULES:
        out[r] = [paired(d, a, b, c, r) for c in BAND]
        for c, p in zip(BAND, out[r]): cons.append(dict(key=key, rule=r, contrast=f"{a} - {b}", condition=c, **p))
    dec[key] = dict(label=label(out["final"]) if key != "nogate_minus_s4d" else "descriptive", final=out["final"], best=out["oracle"])
dec["indep_minus_earlier_descriptive"] = {a: [float(vals(d, a, c).mean() - vals(old, a, c).mean()) for c in BAND] for a in ARMS}
dec["any_reversed"] = any(dec[k]["label"] == "REVERSED" for k in ("L1f_rev", "L1p_rev", "L2_rev"))
OUT.mkdir(parents=True, exist_ok=True)
pd.DataFrame(means).to_csv(OUT / "stage9_means.csv", index=False)
pd.DataFrame(cons).round(4).to_csv(OUT / "stage9_contrasts.csv", index=False)
json.dump(dec, open(OUT / "stage9_decisions.json", "w"), indent=1, default=float)
print(json.dumps({k: v["label"] for k, v in dec.items() if isinstance(v, dict) and "label" in v} | {"any_reversed": dec["any_reversed"]}, indent=1))
