"""
Stage-2 aggregation. Reads the stage-2 grid (layered-freeze arms + bm3_kin) and, where the
contrast needs it, the stage-1 grid (bm3_frozen, s4d, frozen_nogate, s4d_plus_gate), and
reports every quantity under both reporting rules:
    oracle = max over epochs on the evaluation condition (the harness's own rule)
    final  = last epoch (cosine LR annealed to 0; uses no evaluation data to pick the epoch)
Only cells present in both grids are paired; arms with fewer than 3 seeds are skipped.
"""
import json, sys
import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import wilcoxon

S2 = sys.argv[1] if len(sys.argv) > 1 else "p9_recheck/results/cells_stage2.jsonl"
S1 = sys.argv[2] if len(sys.argv) > 2 else "p9_recheck/results/cells_recheck.jsonl"
BAND = ["awgn@+0dB", "awgn@-2dB", "awgn@-6dB"]


def load(p, tag):
    return pd.DataFrame([dict(json.loads(l), grid=tag) for l in open(p)])


d2 = load(S2, "stage2")
try:
    d1 = load(S1, "stage1")
except FileNotFoundError:
    d1 = pd.DataFrame(columns=d2.columns)
df = pd.concat([d2, d1], ignore_index=True)
print(f"stage2 cells {len(d2)}  stage1 cells {len(d1)}")
print(d2.groupby(["arm", "condition"]).seed.count().unstack(fill_value=0).to_string())


def v(arm, cond, rule):
    s = df[(df.arm == arm) & (df.condition == cond)].sort_values("seed")
    return s.set_index("seed")[rule] * 100


rows = []
for rule in ["oracle", "final"]:
    for (arm, cond), g in df.groupby(["arm", "condition"]):
        x = g[rule] * 100
        rows.append(dict(rule=rule, arm=arm, condition=cond, n=len(x),
                         mean=round(x.mean(), 2), sd=round(x.std(ddof=1), 2) if len(x) > 1 else np.nan,
                         n_params=int(g.n_params.iloc[0])))
means = pd.DataFrame(rows).sort_values(["rule", "arm", "condition"])
means.to_csv("p9_recheck/results/stage2_means.csv", index=False)
print("\n== means (macro-F1 %) ==")
print(means[means.condition.isin(BAND)].pivot_table(index=["condition", "arm"], columns="rule",
      values="mean").round(2).to_string())

CONTRASTS = [("bm3_kin", "bm3_frozen", "BM3 - frozen"),
             ("bm3_frozen", "frozen_ctrap", "frozen - frozen_ctrap (freeze trap)"),
             ("bm3_frozen", "frozen_cangle", "frozen - frozen_cangle (freeze angles)"),
             ("bm3_frozen", "frozen_clti", "frozen - frozen_clti (freeze both)"),
             ("frozen_clti", "s4d", "frozen_clti - S4D"),
             ("bm3_kin", "frozen_clti", "BM3 - frozen_clti")]
out = []
for rule in ["oracle", "final"]:
    for a, b, name in CONTRASTS:
        for c in BAND:
            x, y = v(a, c, rule), v(b, c, rule)
            s = x.index.intersection(y.index)
            if len(s) < 3:
                continue
            d = (x[s] - y[s]).values
            h = stats.t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            out.append(dict(rule=rule, contrast=name, condition=c, n=len(d),
                            mean_diff_pp=round(d.mean(), 2), ci_lo=round(d.mean() - h, 2),
                            ci_hi=round(d.mean() + h, 2), n_pos=int((d > 0).sum()),
                            wilcoxon_p=round(wilcoxon(d, alternative="greater").pvalue, 4)))
con = pd.DataFrame(out)
con.to_csv("p9_recheck/results/stage2_contrasts.csv", index=False)
print("\n== paired contrasts (pp) ==")
print(con.to_string(index=False))
