"""
Stage-6 analysis (PU replication), implementing prereg_stage6_pu.md.
Usage: python analyze_stage6.py [cells_stage6_pu.jsonl] [outdir]
Writes stage6_means.csv, stage6_contrasts.csv, stage6_ratios.csv, stage6_decisions.json.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

A = sys.argv[1:] + [None] * 2
CELLS = A[0] or "p9_recheck/results/cells_stage6_pu.jsonl"
OUT = Path(A[1] or "p9_recheck/results")
BAND = ["awgn@+0dB", "awgn@-2dB", "awgn@-6dB"]
RULES = ["final", "oracle"]
B, SEED, MIN_DEN = 20000, 20261009, 2.0


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


def ratio_boot(d, nums, den, base, rule):
    """mean-over-levels (num - base)/(den - base) for each num; shared resamples; sorted arm order."""
    rng = np.random.default_rng(SEED)
    order = sorted(set(nums) | {den, base})
    V = {(a, c): vals(d, a, c, rule).values for a in order for c in BAND}
    def R(m, a, c):
        dn = m[(den, c)] - m[(base, c)]
        return np.nan if dn < MIN_DEN else (m[(a, c)] - m[(base, c)]) / dn
    pm = {k: v.mean() for k, v in V.items()}
    point = {a: {c: R(pm, a, c) for c in BAND} for a in nums}
    reps = {a: [] for a in nums}; diff = []; dropped = 0
    for _ in range(B):
        m = {k: rng.choice(V[k], len(V[k]), replace=True).mean() for k in sorted(V)}
        per = {a: [R(m, a, c) for c in BAND] for a in nums}
        dropped += any(np.isnan(x) for x in per[nums[0]])
        r = {a: np.nanmean(per[a]) for a in nums}
        for a in nums: reps[a].append(r[a])
        if len(nums) == 2: diff.append(r[nums[0]] - r[nums[1]])
    out = {a: dict(per_level=point[a], mean=float(np.nanmean(list(point[a].values()))),
                   ci=[float(x) for x in np.nanpercentile(reps[a], [2.5, 97.5])]) for a in nums}
    out["frac_replicates_with_level_dropped"] = dropped / B
    if diff:
        out["diff"] = dict(mean=out[nums[0]]["mean"] - out[nums[1]]["mean"],
                           ci=[float(x) for x in np.percentile(diff, [2.5, 97.5])])
    return out


def label(pcs):
    pos = sum(1 for p in pcs if p["mean"] > 0 and p["lo"] > 0)
    neg = sum(1 for p in pcs if p["mean"] < 0 and p["hi"] < 0)
    if neg >= 2: return "REVERSED", pos, neg
    return {0: "NOT REPLICATED", 1: "PARTIAL"}.get(pos, "REPLICATED"), pos, neg


d = load(CELLS)
d = d[~d.get("smoke", False).astype(bool)] if "smoke" in d else d
ARMS = ["bm3_frozen", "frozen_clti", "s4d", "frozen_nogate", "bm3_frozen_add"]
means, cons, ratios, dec = [], [], [], {}
for rule in RULES:
    for (arm, c), g in d[d.condition.isin(BAND)].groupby(["arm", "condition"]):
        x = g.set_index("seed")[rule] * 100
        means.append(dict(rule=rule, arm=arm, condition=c, n=len(x), mean=round(x.mean(), 2),
                          sd=round(x.std(ddof=1), 2), n_params=int(g.n_params.iloc[0])))
if not complete(d, ARMS):
    dec["status"] = "incomplete"
else:
    CON = {"L1_full": ("frozen_clti", "s4d"), "L1_partial": ("bm3_frozen", "s4d"),
           "L2": ("bm3_frozen", "frozen_nogate"), "add_minus_nogate": ("bm3_frozen_add", "frozen_nogate"),
           "frozen_minus_add": ("bm3_frozen", "bm3_frozen_add"), "nogate_minus_s4d": ("frozen_nogate", "s4d")}
    fin = {}
    for k, (a, b) in CON.items():
        for rule in RULES:
            pcs = [paired(d, a, b, c, rule) for c in BAND]
            for c, p in zip(BAND, pcs):
                cons.append(dict(rule=rule, contrast=f"{a} - {b}", key=k, condition=c, **p))
            if rule == "final": fin[k] = pcs
    for k in ["L1_full", "L1_partial", "L2"]:
        lab, pos, neg = label(fin[k])
        dec[k] = dict(label=lab, levels_positive_excl0=pos, levels_negative_excl0=neg,
                      mean=[p["mean"] for p in fin[k]], ci=[[p["lo"], p["hi"]] for p in fin[k]],
                      n_pos=[p["n_pos"] for p in fin[k]])
    dec["L2"]["levels_ge_8pp"] = sum(1 for p in fin["L2"] if p["mean"] >= 8)
    rr = {rule: ratio_boot(d, ["bm3_frozen_add"], "bm3_frozen", "frozen_nogate", rule) for rule in RULES}
    for rule in RULES:
        r = rr[rule]["bm3_frozen_add"]
        ratios.append(dict(rule=rule, stat="rho_PU", arm="bm3_frozen_add", **r["per_level"], mean=r["mean"],
                           ci_lo=r["ci"][0], ci_hi=r["ci"][1], frac_dropped=rr[rule]["frac_replicates_with_level_dropped"]))
    r = rr["final"]["bm3_frozen_add"]; rho, (lo, hi) = r["mean"], r["ci"]
    if dec["L2"]["label"] in ("NOT REPLICATED", "REVERSED") or not np.isfinite(rho):
        v3 = "not assessable"
    elif rho < 0.3 and hi < 0.5:
        v3 = "NOT RESCUED"
    elif rho >= 0.7 and lo > 0.5:
        v3 = "RESCUED"
    else:
        v3 = "INCONCLUSIVE"
    dec["L3b"] = dict(label=v3, rho=rho, ci=[lo, hi], per_level=r["per_level"],
                      frac_dropped=rr["final"]["frac_replicates_with_level_dropped"],
                      rho_best_epoch=rr["oracle"]["bm3_frozen_add"]["mean"],
                      add_minus_nogate=[[p["mean"], p["lo"], p["hi"], p["n_pos"]] for p in fin["add_minus_nogate"]])
    fav = (dec["L1_full"]["label"] == "REPLICATED" and dec["L2"]["label"] == "REPLICATED" and v3 == "NOT RESCUED")
    unfav = any(dec[k]["label"] in ("NOT REPLICATED", "REVERSED") for k in ["L1_full", "L2"]) or v3 == "not assessable"
    dec["writing_branch"] = "full replication" if fav else ("unfavourable: return to author" if unfav else "partial: label level by level")
OUT.mkdir(parents=True, exist_ok=True)
pd.DataFrame(means).to_csv(OUT / "stage6_means.csv", index=False)
pd.DataFrame(cons).round(3).to_csv(OUT / "stage6_contrasts.csv", index=False)
pd.DataFrame(ratios).round(4).to_csv(OUT / "stage6_ratios.csv", index=False)
json.dump(dec, open(OUT / "stage6_decisions.json", "w"), indent=1, default=float)
print(json.dumps(dec, indent=1, default=lambda x: round(float(x), 3)))
