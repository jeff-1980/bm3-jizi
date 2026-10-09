"""
Stage-7 analysis (PU shared-bearing diagnostic), implementing prereg_stage7_pu_shared.md.
Usage: python analyze_stage7.py [cells_stage7_pu_shared.jsonl] [cells_stage6_pu.jsonl] [outdir]
Writes stage7_means.csv, stage7_contrasts.csv, stage7_decisions.json (deterministic; no resampling).
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

A = sys.argv[1:] + [None] * 3
CELLS = A[0] or "p9_recheck/results/cells_stage7_pu_shared.jsonl"
S6 = A[1] or "p9_recheck/results/cells_stage6_pu.jsonl"
OUT = Path(A[2] or "p9_recheck/results")
BAND = ["awgn@+0dB", "awgn@-2dB", "awgn@-6dB"]
RULES = ["final", "oracle"]
ARMS = ["s4d", "bm3_frozen", "frozen_nogate"]
FLOOR_MEAN, FLOOR_LB = 65.0, 60.0


def load(*ps):
    rows = [json.loads(l) for p in ps if Path(p).exists() for l in open(p)]
    d = pd.DataFrame(rows)
    return d.drop_duplicates(["arm", "condition", "seed"], keep="first") if len(d) else d


def vals(d, arm, cond, rule):
    s = d[(d.arm == arm) & (d.condition == cond)].set_index("seed").sort_index()
    return (s[rule] * 100) if len(s) else pd.Series(dtype=float)


def complete(d, arms):
    return all(len(vals(d, a, c, "final")) == 5 for a in arms for c in BAND)


def paired(d, a, b, cond, rule):
    x, y = vals(d, a, cond, rule), vals(d, b, cond, rule)
    s = x.index.intersection(y.index)
    dd = (x[s] - y[s]).values
    h = stats.t.ppf(.975, len(dd) - 1) * dd.std(ddof=1) / np.sqrt(len(dd))
    return dict(n=len(dd), mean=dd.mean(), lo=dd.mean() - h, hi=dd.mean() + h, n_pos=int((dd > 0).sum()))


def label(pcs):
    pos = sum(1 for p in pcs if p["mean"] > 0 and p["lo"] > 0)
    rev = sum(1 for p in pcs if p["mean"] < 0 and p["hi"] < 0)
    return {0: "NOT REPLICATED", 1: "PARTIAL"}.get(pos, "REPLICATED"), pos, rev


d = load(CELLS); s6 = load(S6)
means, cons, dec = [], [], {}
for tag, dd in (("shared", d), ("disjoint_stage6", s6)):
    for rule in RULES:
        for (arm, c), g in dd[dd.condition.isin(BAND) & dd.arm.isin(ARMS)].groupby(["arm", "condition"]):
            x = g.set_index("seed")[rule] * 100
            means.append(dict(setting=tag, rule=rule, arm=arm, condition=c, n=len(x), mean=round(x.mean(), 2),
                              sd=round(x.std(ddof=1), 2)))
if not complete(d, ARMS):
    dec["status"] = "incomplete"
else:
    # D1
    gate = {}
    for arm in ARMS:
        lv = []
        for c in BAND:
            x = vals(d, arm, c, "final").values
            lb = x.mean() - stats.t.ppf(.95, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
            lv.append(dict(condition=c, mean=float(x.mean()), lb95_one_sided=float(lb),
                           off_floor=bool(x.mean() >= FLOOR_MEAN and lb > FLOOR_LB)))
        gate[arm] = dict(levels=lv, n_off=sum(l["off_floor"] for l in lv))
    d1 = any(g["n_off"] >= 2 for g in gate.values())
    dec["D1"] = dict(passed=d1, label="PASS" if d1 else "PLATFORM-LIMITED", arms=gate)
    # D2 (computed always, adjudicated only if D1 passes)
    fin = {}
    for k, (a, b) in {"L1pp": ("bm3_frozen", "s4d"), "L2pp": ("bm3_frozen", "frozen_nogate"),
                      "nogate_minus_s4d": ("frozen_nogate", "s4d")}.items():
        for rule in RULES:
            pcs = [paired(d, a, b, c, rule) for c in BAND]
            for c, p in zip(BAND, pcs): cons.append(dict(rule=rule, contrast=f"{a} - {b}", key=k, condition=c, **p))
            if rule == "final": fin[k] = pcs
    for k in ["L1pp", "L2pp"]:
        lab, pos, rev = label(fin[k])
        dec[k] = dict(label=lab if d1 else f"(not adjudicated: D1 failed) {lab}", levels_positive_excl0=pos,
                      levels_reversed=rev, mean=[p["mean"] for p in fin[k]], ci=[[p["lo"], p["hi"]] for p in fin[k]],
                      n_pos=[p["n_pos"] for p in fin[k]])
    rev_any = dec["L1pp"]["levels_reversed"] + dec["L2pp"]["levels_reversed"] > 0
    labs = [label(fin[k])[0] for k in ["L1pp", "L2pp"]]
    if not d1:
        br = "F (PLATFORM-LIMITED)"
    elif rev_any:
        br = "F (reversal)"
    elif "REPLICATED" in labs:
        br = "R"
    elif labs == ["NOT REPLICATED", "NOT REPLICATED"]:
        br = "F (both not replicated)"
    else:
        br = "INCONCLUSIVE (written at F tier, returned to author)"
    dec["branch"] = br
    # descriptive: shared minus disjoint per arm (final), independent seeds -> Welch interval
    sd = {}
    for arm in ARMS:
        sd[arm] = []
        for c in BAND:
            x, y = vals(d, arm, c, "final").values, vals(s6, arm, c, "final").values
            r = stats.ttest_ind(x, y, equal_var=False); ci = r.confidence_interval(0.95)
            sd[arm].append(dict(condition=c, diff=float(x.mean() - y.mean()), lo=float(ci.low), hi=float(ci.high)))
    dec["shared_minus_disjoint_descriptive"] = sd
OUT.mkdir(parents=True, exist_ok=True)
pd.DataFrame(means).to_csv(OUT / "stage7_means.csv", index=False)
pd.DataFrame(cons).round(3).to_csv(OUT / "stage7_contrasts.csv", index=False)
json.dump(dec, open(OUT / "stage7_decisions.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in dec.items() if k != "shared_minus_disjoint_descriptive"}, indent=1, default=lambda x: round(float(x), 2)))
