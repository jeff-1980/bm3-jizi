"""
Stage-8 analysis, implementing prereg_stage8_indep_noise.md.
Usage: python analyze_stage8.py [A.jsonl] [C.jsonl] [B.jsonl] [old_xjtu_dir_files...] -- defaults below.
Writes stage8_means.csv, stage8_contrasts.csv, stage8_ratios.csv, stage8_decisions.json. Deterministic.
"""
import json, sys, itertools
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

R = Path("p9_recheck/results")
A_ = Path(sys.argv[1]) if len(sys.argv) > 1 else R / "cells_stage8_A_xjtu.jsonl"
C_ = Path(sys.argv[2]) if len(sys.argv) > 2 else R / "cells_stage8_C_xjtu_clti.jsonl"
B_ = Path(sys.argv[3]) if len(sys.argv) > 3 else R / "cells_stage8_B_pu_shared.jsonl"
OUT = Path(sys.argv[4]) if len(sys.argv) > 4 else R
_RD = Path(__file__).resolve().parent          # repo layout: recheck/stageN/...; workspace layout: results/
def _old(*names):
    out = []
    for n in names:
        hits = [p for p in [R / n.split("/")[-1], _RD / n] if p.exists()]
        out.append(hits[0] if hits else R / n.split("/")[-1])
    return out
OLD_X = _old("stage1/cells_recheck.jsonl", "stage2/cells_stage2.jsonl", "stage4/cells_stage4_exp3.jsonl")
OLD_P = _old("stage7/cells_stage7_pu_shared.jsonl")
BAND = ["awgn@+0dB", "awgn@-2dB", "awgn@-6dB"]
RULES = ["final", "oracle"]
MIN_DEN = 2.0
TUPLES = [np.array(t) for t in itertools.product(range(5), repeat=5)]


def load(*ps):
    rows = [json.loads(l) for p in ps if Path(p).exists() for l in open(p)]
    d = pd.DataFrame(rows)
    return d.drop_duplicates(["arm", "condition", "seed"], keep="first") if len(d) else d


def vals(d, arm, c, rule="final"):
    s = d[(d.arm == arm) & (d.condition == c)].set_index("seed").sort_index()[rule] * 100
    return s.values if len(s) == 5 else None


def complete(d, arms):
    return len(d) > 0 and all(vals(d, a, c) is not None for a in arms for c in BAND)


def paired(d, a, b, c, rule="final"):
    x = vals(d, a, c, rule) - vals(d, b, c, rule); h = stats.t.ppf(.975, 4) * x.std(ddof=1) / np.sqrt(5)
    return dict(mean=float(x.mean()), lo=float(x.mean() - h), hi=float(x.mean() + h), n_pos=int((x > 0).sum()))


def label(pcs):
    pos = sum(p["mean"] > 0 and p["lo"] > 0 for p in pcs); rev = sum(p["mean"] < 0 and p["hi"] < 0 for p in pcs)
    if rev: return "REVERSED"
    return {0: "DOES NOT HOLD", 1: "WEAKENED"}.get(pos, "HOLDS")


def ratio(V, num, den, base, idx):
    rs = []
    for c in BAND:
        dn = V[den, c][idx].mean() - V[base, c][idx].mean()
        if dn < MIN_DEN: continue
        rs.append((V[num, c][idx].mean() - V[base, c][idx].mean()) / dn)
    return float(np.mean(rs)) if rs else np.nan


def ratio_stats(d, num, den, base, rule="final"):
    V = {(a, c): vals(d, a, c, rule) for a in (num, den, base) for c in BAND}
    pt = ratio(V, num, den, base, np.arange(5))
    per = {c: ((V[num, c].mean() - V[base, c].mean()) / (V[den, c].mean() - V[base, c].mean())) for c in BAND}
    bs = np.array([ratio(V, num, den, base, t) for t in TUPLES]); ok = bs[~np.isnan(bs)]
    ci = [float(x) for x in np.percentile(ok, [2.5, 97.5])] if len(ok) else [np.nan, np.nan]
    # independent-cell bootstrap (earlier stages), for comparison
    rng = np.random.default_rng(20261009); ib = []
    for _ in range(20000):
        W = {k: rng.choice(v, 5, replace=True) for k, v in sorted(V.items())}
        ib.append(ratio({k: v for k, v in W.items()}, num, den, base, np.arange(5)))
    ib = np.array(ib); ib = ib[~np.isnan(ib)]
    return dict(point=pt, per_level=per, paired_ci=ci, paired_frac_dropped=float(np.isnan(bs).mean()),
                indep_ci=[float(x) for x in np.percentile(ib, [2.5, 97.5])] if len(ib) else [np.nan, np.nan])


def rescue_label(r, l2):
    if l2 != "HOLDS" or not np.isfinite(r["point"]): return "not assessable"
    if r["point"] < 0.3 and r["paired_ci"][1] < 0.5: return "NOT RESCUED"
    if r["point"] >= 0.7 and r["paired_ci"][0] > 0.5: return "RESCUED"
    return "INCONCLUSIVE"


A, Cg, Bg = load(A_), load(C_), load(B_)
AC = pd.concat([A, Cg]) if len(Cg) else A
oldX, oldP = load(*OLD_X), load(*OLD_P)
means, cons, ratios, dec = [], [], [], {}
for tag, d in (("A_xjtu", A), ("C_xjtu_clti", Cg), ("B_pu_shared", Bg)):
    if not len(d): continue
    for rule in RULES:
        for (arm, c), g in d[d.condition.isin(BAND)].groupby(["arm", "condition"]):
            x = g.set_index("seed")[rule] * 100
            means.append(dict(grid=tag, rule=rule, arm=arm, condition=c, n=len(x), mean=round(x.mean(), 3),
                              sd=round(x.std(ddof=1), 3), n_params=int(g.n_params.iloc[0])))


def contrast(key, d, a, b):
    out = {}
    for rule in RULES:
        pcs = [paired(d, a, b, c, rule) for c in BAND]
        for c, p in zip(BAND, pcs): cons.append(dict(key=key, rule=rule, contrast=f"{a} - {b}", condition=c, **p))
        out[rule] = pcs
    return dict(label=label(out["final"]), final=out["final"], best=out["oracle"])


if complete(A, ["bm3_frozen", "s4d", "frozen_nogate", "frozen_clti", "bm3_frozen_add"]):
    dec["L1p"] = contrast("L1p", A, "bm3_frozen", "s4d")
    dec["L1f"] = contrast("L1f", A, "frozen_clti", "s4d")
    dec["L2"] = contrast("L2", A, "bm3_frozen", "frozen_nogate")
    dec["L3b_pointwise"] = contrast("add-nogate", A, "bm3_frozen_add", "frozen_nogate")
    r = ratio_stats(A, "bm3_frozen_add", "bm3_frozen", "frozen_nogate"); r["label"] = rescue_label(r, dec["L2"]["label"])
    dec["L3b"] = r; ratios.append(dict(stat="rho_native", **{k: v for k, v in r.items() if k != "per_level"}))
else:
    dec["A"] = "incomplete"
if complete(AC, ["frozen_clti", "frozen_clti_nogate", "frozen_clti_add", "s4d"]):
    dec["L2c"] = contrast("L2c", AC, "frozen_clti", "frozen_clti_nogate")
    dec["L3c_pointwise"] = contrast("clti_add-clti_nogate", AC, "frozen_clti_add", "frozen_clti_nogate")
    dec["clti_nogate_minus_s4d"] = contrast("clti_nogate-s4d", AC, "frozen_clti_nogate", "s4d")
    r = ratio_stats(AC, "frozen_clti_add", "frozen_clti", "frozen_clti_nogate"); r["label"] = rescue_label(r, dec["L2c"]["label"])
    dec["L3c"] = r; ratios.append(dict(stat="rho_clti", **{k: v for k, v in r.items() if k != "per_level"}))
    dec["combined_attribution_allowed"] = dec["L2c"]["label"] == "HOLDS" and r["label"] == "NOT RESCUED"
else:
    dec["C"] = "incomplete"
if complete(Bg, ["bm3_frozen", "s4d", "frozen_nogate"]):
    dec["L1pp"] = contrast("L1pp", Bg, "bm3_frozen", "s4d")
    dec["L2pp"] = contrast("L2pp", Bg, "bm3_frozen", "frozen_nogate")
else:
    dec["B"] = "incomplete"
# descriptive: independent minus shared noise
desc = {}
for tag, new, old, arms in (("xjtu", AC, oldX, ["bm3_frozen", "s4d", "frozen_nogate", "frozen_clti", "bm3_frozen_add"]),
                            ("pu_shared", Bg, oldP, ["bm3_frozen", "s4d", "frozen_nogate"])):
    for a in arms:
        if not len(new) or not len(old) or vals(new, a, BAND[0]) is None or vals(old, a, BAND[0]) is None: continue
        desc[f"{tag}|{a}"] = [float(vals(new, a, c).mean() - vals(old, a, c).mean()) for c in BAND]
dec["indep_minus_shared_descriptive"] = desc
heads = [k for k in ("L1p", "L1f", "L2", "L1pp", "L2pp") if k in dec]
dec["any_core_downgrade"] = any(dec[k]["label"] != "HOLDS" for k in heads)
OUT.mkdir(parents=True, exist_ok=True)
pd.DataFrame(means).to_csv(OUT / "stage8_means.csv", index=False)
pd.DataFrame(cons).round(4).to_csv(OUT / "stage8_contrasts.csv", index=False)
pd.DataFrame(ratios).to_csv(OUT / "stage8_ratios.csv", index=False)
json.dump(dec, open(OUT / "stage8_decisions.json", "w"), indent=1, default=float)
short = {k: (v["label"] if isinstance(v, dict) and "label" in v else v) for k, v in dec.items() if k != "indep_minus_shared_descriptive"}
print(json.dumps(short, indent=1, default=str))
