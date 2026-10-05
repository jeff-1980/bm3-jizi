"""
Stage-3 analysis, implementing prereg_graft_control_pair2.md section by section.
Usage: python analyze_stage3.py [exp1.jsonl] [exp2.jsonl] [stage1.jsonl] [stage2.jsonl] [outdir]
Writes stage3_means.csv, stage3_contrasts.csv, stage3_R.csv and stage3_decisions.json.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

A = sys.argv[1:] + [None] * 5
E1 = A[0] or "p9_recheck/results/cells_stage3_exp1.jsonl"
E2 = A[1] or "p9_recheck/results/cells_stage3_exp2.jsonl"
S1 = A[2] or "p9_recheck/results/cells_recheck.jsonl"
S2 = A[3] or "p9_recheck/results/cells_stage2.jsonl"
OUT = Path(A[4] or "p9_recheck/results")
BAND = ["awgn@+0dB", "awgn@-2dB", "awgn@-6dB"]
RULES = ["final", "oracle"]          # final = primary; oracle = best-epoch, reported only
SEEDS = [0, 1, 2, 3, 4]


def load(p, pair):
    if not Path(p).exists():
        return pd.DataFrame()
    d = pd.DataFrame([json.loads(l) for l in open(p)])
    d["pair"] = pair
    return d


p1 = pd.concat([load(E1, 1), load(S1, 1), load(S2, 1)], ignore_index=True)
p2 = load(E2, 2)
for d in (p1, p2):
    if len(d):
        d.drop_duplicates(["arm", "condition", "seed"], keep="first", inplace=True)


def vals(d, arm, cond, rule):
    s = d[(d.arm == arm) & (d.condition == cond)].set_index("seed").sort_index()
    return (s[rule] * 100) if len(s) else pd.Series(dtype=float)


def paired(d, a, b, cond, rule):
    x, y = vals(d, a, cond, rule), vals(d, b, cond, rule)
    s = x.index.intersection(y.index)
    if len(s) < 3:
        return None
    dd = (x[s] - y[s]).values
    h = stats.t.ppf(.975, len(dd) - 1) * dd.std(ddof=1) / np.sqrt(len(dd))
    return dict(n=len(dd), mean=dd.mean(), lo=dd.mean() - h, hi=dd.mean() + h, n_pos=int((dd > 0).sum()))


def complete(d, arms):
    return all(len(vals(d, a, c, "final")) == 5 for a in arms for c in BAND)


def R_boot(d, num_arms, den, base="s4d", rule="final", B=20000, seed=20261004, min_den=None):
    """Mean-over-levels R for each arm in num_arms (shared resamples), plus dR if two arms."""
    rng = np.random.default_rng(seed)
    # deterministic arm order: a set's iteration order varies with string-hash randomisation across
    # processes, which would make the seeded bootstrap non-reproducible
    order = sorted(set(num_arms) | {den, base})
    V = {(a, c): vals(d, a, c, rule).values for a in order for c in BAND}
    def R_of(m, a, c):
        dn = m[(den, c)] - m[(base, c)]
        if min_den is not None and dn < min_den:
            return np.nan
        return (m[(a, c)] - m[(base, c)]) / dn
    point_m = {k: v.mean() for k, v in V.items()}
    point = {a: {c: R_of(point_m, a, c) for c in BAND} for a in num_arms}
    reps = {a: [] for a in num_arms}; dR = []
    for _ in range(B):
        m = {k: rng.choice(v, len(v), replace=True).mean() for k, v in V.items()}
        r = {a: np.nanmean([R_of(m, a, c) for c in BAND]) for a in num_arms}
        for a in num_arms: reps[a].append(r[a])
        if len(num_arms) == 2: dR.append(r[num_arms[0]] - r[num_arms[1]])
    res = {}
    for a in num_arms:
        res[a] = dict(per_level=point[a], mean=float(np.nanmean(list(point[a].values()))),
                      ci=[float(x) for x in np.nanpercentile(reps[a], [2.5, 97.5])])
    if dR:
        res["dR"] = dict(mean=res[num_arms[0]]["mean"] - res[num_arms[1]]["mean"],
                         ci=[float(x) for x in np.percentile(dR, [2.5, 97.5])])
    return res


def tier(r):
    return "dead" if r >= 0.7 else ("partial" if r >= 0.3 else "survives")


TIER_ORDER = ["survives", "partial", "dead"]

means, cons, Rrows, dec = [], [], [], {}
for d, pair in ((p1, 1), (p2, 2)):
    if not len(d): continue
    for rule in RULES:
        for (arm, c), g in d[d.condition.isin(BAND)].groupby(["arm", "condition"]):
            x = g.set_index("seed")[rule] * 100
            means.append(dict(pair=pair, rule=rule, arm=arm, condition=c, n=len(x), mean=round(x.mean(), 2),
                              sd=round(x.std(ddof=1), 2) if len(x) > 1 else np.nan, n_params=int(g.n_params.iloc[0])))

# ---------------- experiment 1 ----------------
E1_ARMS = ["s4d_plus_gate_wm", "s4d_plus_branch"]
if complete(p1, E1_ARMS + ["s4d", "bm3_frozen"]):
    for rule in RULES:
        for a, b in [("s4d_plus_gate_wm", "s4d"), ("s4d_plus_branch", "s4d"), ("s4d_plus_gate_wm", "s4d_plus_branch"),
                     ("s4d_plus_gate_wm", "s4d_plus_gate"), ("bm3_frozen", "s4d")]:
            for c in BAND:
                r = paired(p1, a, b, c, rule)
                if r: cons.append(dict(pair=1, rule=rule, contrast=f"{a} - {b}", condition=c, **r))
    out = {}
    for rule in RULES:
        rb = R_boot(p1, E1_ARMS, "bm3_frozen", rule=rule)
        rb2 = R_boot(p1, E1_ARMS, "frozen_clti", rule=rule)
        out[rule] = dict(R=rb, R_prime=rb2)
        for nm, rr in (("R", rb), ("R_prime", rb2)):
            for a in E1_ARMS:
                Rrows.append(dict(pair=1, rule=rule, stat=nm, arm=a, **{c: rr[a]["per_level"][c] for c in BAND},
                                  mean=rr[a]["mean"], ci_lo=rr[a]["ci"][0], ci_hi=rr[a]["ci"][1]))
            Rrows.append(dict(pair=1, rule=rule, stat=nm + "_dR", arm="gate_wm - branch",
                              mean=rr["dR"]["mean"], ci_lo=rr["dR"]["ci"][0], ci_hi=rr["dR"]["ci"][1]))
    f = out["final"]["R"]; Rw = f["s4d_plus_gate_wm"]; lo, hi = Rw["ci"]
    touched = sorted({tier(lo), tier(Rw["mean"]), tier(hi)}, key=TIER_ORDER.index)
    inconclusive = (lo < 0.3 <= hi) or (lo < 0.7 <= hi)
    write_tier = touched[-1] if inconclusive else tier(Rw["mean"])
    dR = f["dR"]; specific = dR["mean"] > 0.1 and dR["ci"][0] > 0
    dec["exp1"] = dict(
        R_gate_wm=Rw["mean"], R_gate_wm_ci=Rw["ci"], R_gate_wm_per_level=Rw["per_level"],
        R_branch=f["s4d_plus_branch"]["mean"], R_branch_ci=f["s4d_plus_branch"]["ci"],
        dR=dR["mean"], dR_ci=dR["ci"],
        point_tier=tier(Rw["mean"]), inconclusive=bool(inconclusive), tiers_touched=touched,
        manuscript_tier=write_tier, gate_specific=bool(specific),
        rule4_block_shape_statement=bool(Rw["mean"] >= 0.3 and not specific),
        R_prime_gate_wm=out["final"]["R_prime"]["s4d_plus_gate_wm"]["mean"],
        R_prime_gate_wm_ci=out["final"]["R_prime"]["s4d_plus_gate_wm"]["ci"],
        best_epoch_R_gate_wm=out["oracle"]["R"]["s4d_plus_gate_wm"]["mean"],
        best_epoch_R_branch=out["oracle"]["R"]["s4d_plus_branch"]["mean"])
else:
    dec["exp1"] = "incomplete"

# ---------------- experiment 2 ----------------
E2_ARMS = ["frozen_clti", "s4d", "frozen_nogate", "s4d_plus_gate_wm", "bm3_frozen"]
if len(p2) and complete(p2, E2_ARMS):
    for rule in RULES:
        for a, b in [("frozen_clti", "s4d"), ("bm3_frozen", "s4d"), ("bm3_frozen", "frozen_nogate"),
                     ("frozen_clti", "frozen_nogate"), ("frozen_nogate", "s4d"), ("s4d_plus_gate_wm", "s4d"),
                     ("bm3_frozen", "frozen_clti")]:
            for c in BAND:
                r = paired(p2, a, b, c, rule)
                if r: cons.append(dict(pair=2, rule=rule, contrast=f"{a} - {b}", condition=c, **r))
    L1 = [paired(p2, "frozen_clti", "s4d", c, "final") for c in BAND]
    k = sum(1 for r in L1 if r["mean"] > 0 and r["lo"] > 0)
    L2 = [paired(p2, "bm3_frozen", "frozen_nogate", c, "final") for c in BAND]
    out2 = {}
    for rule in RULES:
        rb = R_boot(p2, ["s4d_plus_gate_wm"], "bm3_frozen", rule=rule, min_den=2.0)
        rb2 = R_boot(p2, ["s4d_plus_gate_wm"], "frozen_clti", rule=rule, min_den=2.0)
        out2[rule] = (rb, rb2)
        for nm, rr in (("R2", rb), ("R2_prime", rb2)):
            a = "s4d_plus_gate_wm"
            Rrows.append(dict(pair=2, rule=rule, stat=nm, arm=a, **{c: rr[a]["per_level"][c] for c in BAND},
                              mean=rr[a]["mean"], ci_lo=rr[a]["ci"][0], ci_hi=rr[a]["ci"][1]))
    dec["exp2"] = dict(
        L1_levels_positive_ci=k,
        L1_verdict="replicated on the reverse transition" if k >= 2 else ("partial" if k == 1 else "not replicated: narrow to original transition"),
        L1_contrasts={c: r for c, r in zip(BAND, L1)},
        L2_levels_ge_8pp=sum(1 for r in L2 if r["mean"] >= 8.0),
        L2_contrasts={c: r for c, r in zip(BAND, L2)},
        L3_R2=out2["final"][0]["s4d_plus_gate_wm"], L3_R2_prime=out2["final"][1]["s4d_plus_gate_wm"])
else:
    dec["exp2"] = "incomplete"

OUT.mkdir(parents=True, exist_ok=True)
pd.DataFrame(means).to_csv(OUT / "stage3_means.csv", index=False)
pd.DataFrame(cons).round(3).to_csv(OUT / "stage3_contrasts.csv", index=False)
pd.DataFrame(Rrows).round(4).to_csv(OUT / "stage3_R.csv", index=False)
json.dump(dec, open(OUT / "stage3_decisions.json", "w"), indent=1, default=float)
print(json.dumps(dec, indent=1, default=lambda x: round(float(x), 4)))
