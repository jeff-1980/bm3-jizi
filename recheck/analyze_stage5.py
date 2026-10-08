"""
Stage-5 analysis, implementing prereg_stage5_metric_reverse.md.
Usage: python analyze_stage5.py [exp2.jsonl] [exp3.jsonl] [stage1.jsonl] [stage3_exp2.jsonl] [outdir]
Writes stage5_means.csv, stage5_contrasts.csv, stage5_ratios.csv, stage5_decisions.json.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

A = sys.argv[1:] + [None] * 5
E2 = A[0] or "p9_recheck/results/cells_stage5_exp2.jsonl"
E3 = A[1] or "p9_recheck/results/cells_stage5_exp3.jsonl"
S1 = A[2] or "p9_recheck/results/cells_recheck.jsonl"
S3 = A[3] or "p9_recheck/results/cells_stage3_exp2.jsonl"
OUT = Path(A[4] or "p9_recheck/results")
BAND = ["awgn@+0dB", "awgn@-2dB", "awgn@-6dB"]
RULES = ["final", "oracle"]
B, SEED, MIN_DEN = 20000, 20261008, 2.0
# prereg section 2: manuscript reference directions (best-epoch, original grids), frozen - arm / wide - s4d
REF = {"frozen_noconv": [-6.6, -2.7, 7.9], "frozen_As4d": [-2.4, -2.2, -4.2], "s4d_wide": [1.8, -1.7, 0.5]}


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


p1 = load(E2, S1)
p2 = load(E3, S3)
means, cons, ratios, dec = [], [], [], {}
for d, pair in ((p1, 1), (p2, 2)):
    for rule in RULES:
        for (arm, c), g in d[d.condition.isin(BAND)].groupby(["arm", "condition"]):
            x = g.set_index("seed")[rule] * 100
            means.append(dict(pair=pair, rule=rule, arm=arm, condition=c, n=len(x), mean=round(x.mean(), 2),
                              sd=round(x.std(ddof=1), 2), n_params=int(g.n_params.iloc[0])))

# ---- experiment 3: reverse native additive substitution (decisive) ----
if complete(p2, ["bm3_frozen_add", "bm3_frozen", "frozen_nogate"]):
    for rule in RULES:
        for a, b in [("bm3_frozen", "bm3_frozen_add"), ("bm3_frozen_add", "frozen_nogate"), ("bm3_frozen", "frozen_nogate")]:
            for c in BAND:
                cons.append(dict(pair=2, rule=rule, contrast=f"{a} - {b}", condition=c, **paired(p2, a, b, c, rule)))
    rr = {rule: ratio_boot(p2, ["bm3_frozen_add"], "bm3_frozen", "frozen_nogate", rule) for rule in RULES}
    for rule in RULES:
        r = rr[rule]["bm3_frozen_add"]
        ratios.append(dict(pair=2, rule=rule, stat="rho_rev", arm="bm3_frozen_add", **r["per_level"],
                           mean=r["mean"], ci_lo=r["ci"][0], ci_hi=r["ci"][1]))
    r = rr["final"]["bm3_frozen_add"]; rho, (lo, hi) = r["mean"], r["ci"]
    if rho < 0.3 and hi < 0.5:
        verdict, branch = "NOT RESCUED", "A: Necessary in Its Native Block"
    elif rho >= 0.7 and lo > 0.5:
        verdict, branch = "RESCUED", "B: Gating Benefits Depend on the Backbone"
    else:
        verdict, branch = "PARTIAL/INCONCLUSIVE", "B: Gating Benefits Depend on the Backbone"
    dec["exp3"] = dict(rho_rev=rho, rho_rev_ci=[lo, hi], rho_rev_per_level=r["per_level"], verdict=verdict,
                       title_branch=branch, rho_rev_best_epoch=rr["oracle"]["bm3_frozen_add"]["mean"],
                       rho_rev_best_epoch_ci=rr["oracle"]["bm3_frozen_add"]["ci"],
                       frac_replicates_with_level_dropped=rr["final"]["frac_replicates_with_level_dropped"])
else:
    dec["exp3"] = "incomplete"

# ---- experiment 2: final-epoch re-assessment ----
e2 = {}
for arm in ["frozen_noconv", "frozen_As4d"]:
    if not complete(p1, [arm, "bm3_frozen"]):
        e2[arm] = "incomplete"; continue
    fin = []
    for rule in RULES:
        for c in BAND:
            pc = paired(p1, "bm3_frozen", arm, c, rule)
            cons.append(dict(pair=1, rule=rule, contrast=f"bm3_frozen - {arm}", condition=c, **pc))
            if rule == "final": fin.append(pc)
    flips = [c for c, ref, f in zip(BAND, REF[arm], fin)
             if abs(ref) >= 1.0 and abs(f["mean"]) >= 1.0 and np.sign(ref) != np.sign(f["mean"])]
    e2[arm] = dict(reference_best=REF[arm], final=[f["mean"] for f in fin],
                   final_ci=[[f["lo"], f["hi"]] for f in fin], n_pos=[f["n_pos"] for f in fin],
                   flips=flips, decision="holds" if not flips else "downgrade")
if complete(p1, ["s4d_wide", "bm3_frozen", "s4d"]):
    fz, ws = [], []
    for rule in RULES:
        for c in BAND:
            a = paired(p1, "bm3_frozen", "s4d_wide", c, rule); b = paired(p1, "s4d_wide", "s4d", c, rule)
            cons.append(dict(pair=1, rule=rule, contrast="bm3_frozen - s4d_wide", condition=c, **a))
            cons.append(dict(pair=1, rule=rule, contrast="s4d_wide - s4d", condition=c, **b))
            if rule == "final": fz.append(a); ws.append(b)
    k = sum(1 for a in fz if a["lo"] > 0)
    e2["s4d_wide"] = dict(frozen_minus_wide=[a["mean"] for a in fz], frozen_minus_wide_ci=[[a["lo"], a["hi"]] for a in fz],
                          levels_excluding_zero=k, wide_minus_s4d=[b["mean"] for b in ws],
                          wide_minus_s4d_ci=[[b["lo"], b["hi"]] for b in ws], reference_best_wide_minus_s4d=REF["s4d_wide"],
                          decision="holds" if k >= 2 else "downgrade")
else:
    e2["s4d_wide"] = "incomplete"
dec["exp2"] = e2
dec["exp2_all_hold"] = all(isinstance(v, dict) and v["decision"] == "holds" for v in e2.values())

OUT.mkdir(parents=True, exist_ok=True)
pd.DataFrame(means).to_csv(OUT / "stage5_means.csv", index=False)
pd.DataFrame(cons).round(3).to_csv(OUT / "stage5_contrasts.csv", index=False)
pd.DataFrame(ratios).round(4).to_csv(OUT / "stage5_ratios.csv", index=False)
json.dump(dec, open(OUT / "stage5_decisions.json", "w"), indent=1, default=float)
print(json.dumps(dec, indent=1, default=lambda x: round(float(x), 4)))
