"""
Stage-4 analysis, implementing prereg_native_additive.md.
Usage: python analyze_stage4.py [exp3.jsonl] [exp4.jsonl] [stage1.jsonl] [stage3_exp2.jsonl] [outdir]
Writes stage4_means.csv, stage4_contrasts.csv, stage4_ratios.csv, stage4_decisions.json.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

A = sys.argv[1:] + [None] * 5
E3 = A[0] or "p9_recheck/results/cells_stage4_exp3.jsonl"
E4 = A[1] or "p9_recheck/results/cells_stage4_exp4.jsonl"
S1 = A[2] or "p9_recheck/results/cells_recheck.jsonl"
S3 = A[3] or "p9_recheck/results/cells_stage3_exp2.jsonl"
OUT = Path(A[4] or "p9_recheck/results")
BAND = ["awgn@+0dB", "awgn@-2dB", "awgn@-6dB"]
RULES = ["final", "oracle"]
B, SEED, MIN_DEN = 20000, 20261005, 2.0


def load(*ps):
    rows = [json.loads(l) for p in ps if Path(p).exists() for l in open(p)]
    d = pd.DataFrame(rows)
    return d.drop_duplicates(["arm", "condition", "seed"], keep="first") if len(d) else d


p1 = load(E3, S1)
p2 = load(E4, S3)


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


means, cons, ratios, dec = [], [], [], {}
for d, pair in ((p1, 1), (p2, 2)):
    for rule in RULES:
        for (arm, c), g in d[d.condition.isin(BAND)].groupby(["arm", "condition"]):
            x = g.set_index("seed")[rule] * 100
            means.append(dict(pair=pair, rule=rule, arm=arm, condition=c, n=len(x), mean=round(x.mean(), 2),
                              sd=round(x.std(ddof=1), 2), n_params=int(g.n_params.iloc[0])))

if complete(p1, ["bm3_frozen_add", "bm3_frozen", "frozen_nogate"]):
    for rule in RULES:
        for a, b in [("bm3_frozen", "bm3_frozen_add"), ("bm3_frozen_add", "frozen_nogate"), ("bm3_frozen", "frozen_nogate")]:
            for c in BAND:
                cons.append(dict(pair=1, rule=rule, contrast=f"{a} - {b}", condition=c, **paired(p1, a, b, c, rule)))
    rr = {rule: ratio_boot(p1, ["bm3_frozen_add"], "bm3_frozen", "frozen_nogate", rule) for rule in RULES}
    for rule in RULES:
        r = rr[rule]["bm3_frozen_add"]
        ratios.append(dict(pair=1, rule=rule, stat="rho", arm="bm3_frozen_add", **r["per_level"],
                           mean=r["mean"], ci_lo=r["ci"][0], ci_hi=r["ci"][1]))
    r = rr["final"]["bm3_frozen_add"]; rho, (lo, hi) = r["mean"], r["ci"]
    if rho < 0.3 and hi < 0.5:
        verdict, tier = "NOT RESCUED", "NOT RESCUED"
    elif rho >= 0.7 and lo > 0.5:
        verdict, tier = "RESCUED", "RESCUED"
    else:
        verdict = "PARTIAL/INCONCLUSIVE"
        tier = "RESCUED" if hi >= 0.7 else "PARTIAL"
    dec["exp3"] = dict(rho=rho, rho_ci=[lo, hi], rho_per_level=r["per_level"], verdict=verdict,
                       manuscript_tier=tier, rho_best_epoch=rr["oracle"]["bm3_frozen_add"]["mean"],
                       frac_replicates_with_level_dropped=rr["final"]["frac_replicates_with_level_dropped"])
else:
    dec["exp3"] = "incomplete"

if complete(p2, ["s4d_plus_branch", "s4d_plus_gate_wm", "s4d", "bm3_frozen"]):
    for rule in RULES:
        for a, b in [("s4d_plus_branch", "s4d"), ("s4d_plus_gate_wm", "s4d_plus_branch")]:
            for c in BAND:
                cons.append(dict(pair=2, rule=rule, contrast=f"{a} - {b}", condition=c, **paired(p2, a, b, c, rule)))
    rr = {rule: ratio_boot(p2, ["s4d_plus_gate_wm", "s4d_plus_branch"], "bm3_frozen", "s4d", rule) for rule in RULES}
    for rule in RULES:
        for a in ["s4d_plus_gate_wm", "s4d_plus_branch"]:
            r = rr[rule][a]
            ratios.append(dict(pair=2, rule=rule, stat="R2", arm=a, **r["per_level"], mean=r["mean"], ci_lo=r["ci"][0], ci_hi=r["ci"][1]))
        ratios.append(dict(pair=2, rule=rule, stat="dR2", arm="gate_wm - branch", mean=rr[rule]["diff"]["mean"],
                           ci_lo=rr[rule]["diff"]["ci"][0], ci_hi=rr[rule]["diff"]["ci"][1]))
    f = rr["final"]; dR, (dlo, dhi) = f["diff"]["mean"], f["diff"]["ci"]
    label = ("branch ~ gate" if (dlo <= 0 <= dhi and abs(dR) < 0.2) else
             "branch << gate" if (dR > 0.2 and dlo > 0) else "unresolved")
    dec["exp4"] = dict(R2_gate=f["s4d_plus_gate_wm"]["mean"], R2_gate_ci=f["s4d_plus_gate_wm"]["ci"],
                       R2_branch=f["s4d_plus_branch"]["mean"], R2_branch_ci=f["s4d_plus_branch"]["ci"],
                       R2_branch_per_level=f["s4d_plus_branch"]["per_level"],
                       dR2=dR, dR2_ci=[dlo, dhi], label=label)
else:
    dec["exp4"] = "incomplete"

OUT.mkdir(parents=True, exist_ok=True)
pd.DataFrame(means).to_csv(OUT / "stage4_means.csv", index=False)
pd.DataFrame(cons).round(3).to_csv(OUT / "stage4_contrasts.csv", index=False)
pd.DataFrame(ratios).round(4).to_csv(OUT / "stage4_ratios.csv", index=False)
json.dump(dec, open(OUT / "stage4_decisions.json", "w"), indent=1, default=float)
print(json.dumps(dec, indent=1, default=lambda x: round(float(x), 4)))
