#!/usr/bin/env python3
# cmp_capctrl.py — capacity control: is BM3>S4D advantage capacity or architecture?
# Overlays s4d_wide (~178K, matches BM3) against bm3_kin / s4d (~102K) on AWGN.
import json, sys, statistics as st
from collections import defaultdict

FULL = "results/fullgrid_fixed_20260628-2248/cells.jsonl"   # bm3_kin, s4d
CAP  = sys.argv[1] if len(sys.argv) > 1 else None            # capctrl run-dir cells.jsonl

def load(path, want):
    d = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            if r["arm"] in want:
                d[r["condition"]][r["arm"]].append(float(r["best_macro_f1"]) * 100.0)
    return d

full = load(FULL, {"bm3_kin", "s4d"})
cap  = load(CAP,  {"s4d_wide"})

AWGN = ["clean","awgn@+10dB","awgn@+6dB","awgn@+0dB","awgn@-2dB","awgn@-6dB","awgn@-10dB"]
def m(d, c, a):
    v = d[c].get(a, [])
    return sum(v)/len(v) if v else float('nan')

nc = sum(len(v) for c in cap.values() for v in c.values())
print(f"capctrl cells: {nc}/35\n")
print("=== AWGN: capacity vs architecture ===")
print(f"{'cond':<12} {'bm3(178K)':>9} {'s4dwide(178K)':>13} {'s4d(102K)':>9} | {'bm3-wide':>8} {'wide-s4d':>8}")
for c in AWGN:
    bk, sw, s = m(full,c,'bm3_kin'), m(cap,c,'s4d_wide'), m(full,c,'s4d')
    print(f"{c:<12} {bk:9.1f} {sw:13.1f} {s:9.1f} | {bk-sw:+8.1f} {sw-s:+8.1f}")

noise = AWGN[1:]
bm3_minus_wide = [m(full,c,'bm3_kin') - m(cap,c,'s4d_wide') for c in noise]
wide_minus_s4d = [m(cap,c,'s4d_wide') - m(full,c,'s4d')     for c in noise]
bmw = st.mean([x for x in bm3_minus_wide if x==x])
wms = st.mean([x for x in wide_minus_s4d if x==x])
print(f"\nmean over noise rows:  bm3 - s4d_wide = {bmw:+.1f}pp  |  s4d_wide - s4d = {wms:+.1f}pp")
if any(x!=x for x in bm3_minus_wide):
    print("VERDICT: capctrl incomplete — rerun when all 35 cells done.")
elif bmw >= 8.0 and wms < 4.0:
    print("VERDICT: capacity does NOT explain it — advantage is ARCHITECTURAL (selectivity). Claim holds.")
elif wms >= 8.0:
    print("VERDICT: widening S4D closes most of the gap — advantage was CAPACITY, not selectivity. Narrow claim.")
else:
    print("VERDICT: mixed — capacity explains part; selectivity explains the rest.")
