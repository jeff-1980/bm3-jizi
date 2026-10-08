"""Manuscript numbers for v8, computed directly from per-cell values (no re-rounding of aggregates)."""
import json, numpy as np
from scipy import stats
R = "p9_recheck/results/"
L = lambda f: [json.loads(l) for l in open(R + f)]
fwd = L("cells_recheck.jsonl") + L("cells_stage2.jsonl") + L("cells_stage5_exp2.jsonl")
rev = L("cells_stage3_exp2.jsonl") + L("cells_stage5_exp3.jsonl")
B = ["awgn@+0dB", "awgn@-2dB", "awgn@-6dB"]
def v(cells, arm, c, rule):
    x = {k["seed"]: k[rule] * 100 for k in cells if k["arm"] == arm and k["condition"] == c}
    assert sorted(x) == [0, 1, 2, 3, 4], (arm, c, sorted(x))
    return np.array([x[s] for s in range(5)])
def mean(cells, arm, rule): return [float(v(cells, arm, c, rule).mean()) for c in B]
def pc(cells, a, b, rule="final"):
    out = []
    for c in B:
        d = v(cells, a, c, rule) - v(cells, b, c, rule); h = stats.t.ppf(.975, 4) * d.std(ddof=1) / np.sqrt(5)
        out.append([float(d.mean()), float(d.mean() - h), float(d.mean() + h), int((d > 0).sum())])
    return out
arms_f = ["bm3_kin", "bm3_frozen", "s4d", "s4d_wide", "frozen_nogate", "frozen_noconv", "frozen_As4d", "s4d_plus_gate", "frozen_clti"]
arms_r = ["bm3_frozen", "bm3_frozen_add", "frozen_nogate", "s4d", "frozen_clti", "s4d_plus_gate_wm"]
N = {"means_fwd": {a: {r: mean(fwd, a, r) for r in ["final", "oracle"]} for a in arms_f},
     "means_rev": {a: {r: mean(rev, a, r) for r in ["final", "oracle"]} for a in arms_r},
     "params": {a: next(k["n_params"] for k in fwd if k["arm"] == a) for a in arms_f}}
C = {}
for nm, a, b in [("bm3-frozen", "bm3_kin", "bm3_frozen"), ("frozen-s4d", "bm3_frozen", "s4d"),
                 ("frozen-nogate", "bm3_frozen", "frozen_nogate"), ("frozen-noconv", "bm3_frozen", "frozen_noconv"),
                 ("frozen-As4d", "bm3_frozen", "frozen_As4d"), ("frozen-wide", "bm3_frozen", "s4d_wide"),
                 ("wide-s4d", "s4d_wide", "s4d"), ("gate-s4d", "s4d_plus_gate", "s4d"), ("nogate-s4d", "frozen_nogate", "s4d")]:
    C[nm] = {r: pc(fwd, a, b, r) for r in ["final", "oracle"]}
for nm, a, b in [("rev frozen-add", "bm3_frozen", "bm3_frozen_add"), ("rev add-nogate", "bm3_frozen_add", "frozen_nogate"),
                 ("rev frozen-nogate", "bm3_frozen", "frozen_nogate")]:
    C[nm] = {r: pc(rev, a, b, r) for r in ["final", "oracle"]}
N["contrasts"] = C
m = N["means_fwd"]
N["R_final_per_level"] = [(g - s) / (f - s) for g, s, f in zip(m["s4d_plus_gate"]["final"], m["s4d"]["final"], m["bm3_frozen"]["final"])]
N["R_final"] = float(np.mean(N["R_final_per_level"]))
N["shift_final_minus_best"] = {a: [f - o for f, o in zip(m[a]["final"], m[a]["oracle"])] for a in arms_f if a != "frozen_clti"}
d5 = json.load(open(R + "stage5_decisions.json")); d4 = json.load(open(R + "stage4_decisions.json")) if __import__("os").path.exists(R + "stage4_decisions.json") else None
N["dec5"] = d5
json.dump(N, open(R + "stage5_numbers.json", "w"), indent=1)
print("R_final", [round(x, 3) for x in N["R_final_per_level"]], round(N["R_final"], 3))
print("shift range", round(min(min(x) for x in N["shift_final_minus_best"].values()), 1), round(max(max(x) for x in N["shift_final_minus_best"].values()), 1))
for k in ["bm3-frozen", "frozen-s4d", "frozen-nogate", "frozen-noconv", "frozen-As4d", "frozen-wide", "wide-s4d", "gate-s4d", "nogate-s4d"]:
    print(k, [[round(x, 1) for x in t[:3]] + [t[3]] for t in C[k]["final"]], "| best", [round(t[0], 1) for t in C[k]["oracle"]])
for k in ["rev frozen-add", "rev add-nogate"]: print(k, [[round(x, 1) for x in t[:3]] + [t[3]] for t in C[k]["final"]])
for a in arms_f: print(a, N["params"][a], [round(x, 1) for x in m[a]["final"]], [round(x, 1) for x in m[a]["oracle"]])
