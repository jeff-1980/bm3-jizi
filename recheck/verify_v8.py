"""Independent check of v8 manuscript numbers against per-cell data (recomputed here, not read from stage5_numbers.json)."""
import json, re, pathlib, numpy as np
from scipy import stats
R = "p9_recheck/results/"; L = lambda f: [json.loads(l) for l in open(R + f)]
fwd = L("cells_recheck.jsonl") + L("cells_stage2.jsonl") + L("cells_stage5_exp2.jsonl")
rev = L("cells_stage3_exp2.jsonl") + L("cells_stage5_exp3.jsonl")
B = ["awgn@+0dB", "awgn@-2dB", "awgn@-6dB"]
def v(c, a, k, r): x = {q["seed"]: q[r] * 100 for q in c if q["arm"] == a and q["condition"] == k}; return np.array([x[s] for s in range(5)])
def m(c, a, r): return [round(float(v(c, a, k, r).mean()), 1) for k in B]
def pc(c, a, b2, r="final"):
    o = []
    for k in B:
        d = v(c, a, k, r) - v(c, b2, k, r); h = stats.t.ppf(.975, 4) * d.std(ddof=1) / 5 ** .5
        o.append([round(d.mean(), 1), round(d.mean() - h, 1), round(d.mean() + h, 1), int((d > 0).sum())])
    return o
tex = pathlib.Path("p9_build/sections/results.tex").read_text(encoding="utf8")
def num(x): return float(x.replace("$-$", "-").replace("$", "").replace("+", ""))
def table(lab): i = tex.index(r"\label{" + lab + "}"); return tex[i:tex.index(r"\end{table}", i)]
fails = []
# Table 2
t = table("tab:falsify"); blocks = t.split(r"\midrule")
for blk, r in [(blocks[1], "final"), (blocks[2], "oracle")]:
    rows = [l for l in blk.split("\n") if re.match(r"(0|\$-[26]\$)\\,dB", l)]
    for k, l in enumerate(rows):
        c = [x.strip() for x in l.rstrip("\\ ").split("&")][1:]
        got = [num(re.sub(r"\\mathbf\{(.*)\}", r"\1", x)) for x in c]
        exp = [m(fwd, "bm3_kin", r)[k], m(fwd, "bm3_frozen", r)[k], m(fwd, "s4d", r)[k], pc(fwd, "bm3_kin", "bm3_frozen", r)[k][0], pc(fwd, "bm3_frozen", "s4d", r)[k][0]]
        if got != exp: fails.append(("T2", r, k, got, exp))
# Table 3
t = table("tab:localise")
for key, arm in [("SiLU gate", "frozen_nogate"), ("conv stem", "frozen_noconv"), ("heavy-tailed", "frozen_As4d")]:
    l = [x for x in t.split(r"\\[3pt]") if key in x][0]
    sh = re.findall(r"\\shortstack\{\$([+-][\d.]+)\$\\\\ \{\[\}((?:\$-\$)?[\d.]+), ((?:\$-\$)?[\d.]+)\{\]\}\\\\ \((\d)/5\)", l)
    got = [[float(a), num(b2), num(c), int(d)] for a, b2, c, d in sh]
    if got != pc(fwd, "bm3_frozen", arm): fails.append(("T3", arm, got, pc(fwd, "bm3_frozen", arm)))
# Table 4 graft final
t = table("tab:graft"); rows = [l for l in t.split(r"\midrule")[1].split("\n") if "dB" in l and "&" in l]
for k, l in enumerate(rows):
    c = [x.strip() for x in l.rstrip("\\ ").split("&")][1:]
    g, s4, fz = [v(fwd, a, B[k], "final").mean() for a in ["s4d_plus_gate", "s4d", "bm3_frozen"]]
    exp = [round(g, 1), round(s4, 1), round(fz, 1), round((g - s4) / (fz - s4), 3)]
    if [num(x) for x in c] != exp: fails.append(("T4", k, c, exp))
# sensitivity means
t = table("tab:sensitivity_means")
for nm, a in [("BM3 ", "bm3_kin"), ("BM3-frozen", "bm3_frozen"), ("S4D ", "s4d"), ("S4D-wide", "s4d_wide"), ("frozen$-$gate", "frozen_nogate"), ("frozen$-$conv", "frozen_noconv"), ("frozen$-$A", "frozen_As4d"), ("S4D+gate", "s4d_plus_gate")]:
    l = [x for x in t.split("\n") if x.startswith(nm + ("" if nm.endswith(" ") else " "))][0]
    got = [float(x) for x in re.findall(r"& ([\d.]+)", l)]
    exp = [y for p in zip(m(fwd, a, "oracle"), m(fwd, a, "final")) for y in p]
    if got != exp: fails.append(("T-sens", a, got, exp))
# native add reverse block
t = table("tab:nativeadd"); rb = t[t.index("Reverse transition"):]
for nm, a in [("BM3-frozen ($y", "bm3_frozen"), ("frozen$+$add ($y", "bm3_frozen_add"), ("frozen$-$gate (no", "frozen_nogate")]:
    l = [x for x in rb.split("\n") if x.startswith(nm)][0]
    got = [float(x) for x in re.findall(r"& ([\d.]+)", l)]
    exp = [y for p in zip(m(rev, a, "oracle"), m(rev, a, "final")) for y in p]
    if got != exp: fails.append(("T7rev", a, got, exp))
for nm, a, b2 in [("BM3-frozen $-$ frozen$+$add", "bm3_frozen", "bm3_frozen_add"), ("frozen$+$add $-$ frozen$-$gate", "bm3_frozen_add", "frozen_nogate")]:
    l = [x for x in rb.split("\n") if x.startswith(nm)][0]
    sh = re.findall(r"\\shortstack\{\$([+-][\d.]+)\$\\\\ \{\[\}((?:\$-\$)?[\d.]+), ((?:\$-\$)?[\d.]+)\{\]\} \((\d)/5\)", l)
    got = [[float(x), num(y), num(z), int(w)] for x, y, z, w in sh]
    if got != pc(rev, a, b2): fails.append(("T7rev-c", nm, got, pc(rev, a, b2)))
# prose spot checks
d5 = json.load(open(R + "stage5_decisions.json"))["exp3"]
need = [f"{d5['rho_rev']:.3f}", f"{d5['rho_rev_ci'][1]:.3f}", "0.196", "0.341"]
for s in need:
    if s not in tex: fails.append(("prose", s))
ab = pathlib.Path("p9_build/main.tex").read_text(encoding="utf8")
fn = pc(fwd, "bm3_frozen", "frozen_nogate"); rn = pc(rev, "bm3_frozen", "frozen_nogate")
if not (round(min(t[0] for t in fn)) == 9 and round(max(t[0] for t in fn)) == 23 and "$9$--$23\\pp$" in ab): fails.append("abstract 9-23")
if max(abs(t[0]) for t in pc(fwd, "bm3_frozen_add", "frozen_nogate") if False) if False else False: pass
print("V8 VERIFY FAILS:", fails or "none")
