#!/usr/bin/env python3
# cmp_ablation.py — attribute the noise advantage to L_kin vs Mamba backbone.
# Overlays bm3_nokin (ablation run) against bm3_kin / s4d (authoritative grid)
# across the AWGN column, then prints a verdict.
import json, sys, statistics as st
from collections import defaultdict

FULL = "results/fullgrid_fixed_20260628-2248/cells.jsonl"   # bm3_kin, s4d
ABL  = sys.argv[1] if len(sys.argv) > 1 else None             # ablation run-dir cells.jsonl

def load(path, want_arms):
    d = defaultdict(lambda: defaultdict(list))  # cond -> arm -> [f1]
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            if r["arm"] in want_arms:
                d[r["condition"]][r["arm"]].append(float(r["best_macro_f1"]) * 100.0)
    return d

full = load(FULL, {"bm3_kin", "s4d"})
abl  = load(ABL,  {"bm3_nokin"})

AWGN = ["clean","awgn@+10dB","awgn@+6dB","awgn@+0dB","awgn@-2dB","awgn@-6dB","awgn@-10dB"]

def m(d, cond, arm):
    v = d[cond].get(arm, [])
    return sum(v)/len(v) if v else float('nan')

print(f"ablation cells: {sum(len(v) for c in abl.values() for v in c.values())}/35\n")
print("=== AWGN curve: L_kin attribution ===")
print(f"{'cond':<12} {'bm3_kin':>8} {'bm3_nokin':>10} {'s4d':>8} | {'kin-nokin':>9} {'nokin-s4d':>9}")
for c in AWGN:
    bk, bn, s = m(full,c,'bm3_kin'), m(abl,c,'bm3_nokin'), m(full,c,'s4d')
    print(f"{c:<12} {bk:8.1f} {bn:10.1f} {s:8.1f} | {bk-bn:+9.1f} {bn-s:+9.1f}")

# Attribution summary over the noise rows (exclude clean)
noise = AWGN[1:]
kin_minus_nokin = [m(full,c,'bm3_kin') - m(abl,c,'bm3_nokin') for c in noise]
nokin_minus_s4d = [m(abl,c,'bm3_nokin') - m(full,c,'s4d')   for c in noise]
kmn = st.mean([x for x in kin_minus_nokin if x==x])
nms = st.mean([x for x in nokin_minus_s4d if x==x])
print(f"\nmean over noise rows:  L_kin lift (kin-nokin) = {kmn:+.1f}pp  |  backbone lift (nokin-s4d) = {nms:+.1f}pp")
if any(x!=x for x in kin_minus_nokin):
    print("VERDICT: ablation incomplete — rerun when all 35 cells done.")
elif kmn >= 5.0 and nms < kmn:
    print("VERDICT: L_kin is the primary driver — C-01 kinematic claim SURVIVES.")
elif kmn < 2.0:
    print("VERDICT: backbone dominates, L_kin negligible — C-01 kinematic claim COLLAPSES.")
else:
    print("VERDICT: mixed — L_kin gives incremental robustness; narrow C-01 wording.")
