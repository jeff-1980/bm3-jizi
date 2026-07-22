#!/usr/bin/env python3
# diag_yellow1.py — explain "F1 higher under noise than clean".
# (a) per-arm SNR curve (AWGN) — is non-monotonicity BM3-specific or general?
# (b) per-seed clean — is clean's low mean/high std driven by bad seeds?
import json, statistics as st
from collections import defaultdict

path = "results/fullgrid_fixed_20260628-2248/cells.jsonl"
data = defaultdict(dict)  # (cond) -> arm -> {seed: f1}
percond = defaultdict(lambda: defaultdict(list))
with open(path) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        d = json.loads(line)
        f1 = float(d["best_macro_f1"]) * 100.0
        data[d["condition"]][(d["arm"], d["seed"])] = f1
        percond[d["condition"]][d["arm"]].append(f1)

ARMS = ["bm3_kin", "s4d", "cnnlstm", "tcn", "cnn1d"]
AWGN = ["clean","awgn@+10dB","awgn@+6dB","awgn@+0dB","awgn@-2dB","awgn@-6dB","awgn@-10dB"]

def mean(cond, arm):
    v = percond[cond].get(arm, [])
    return sum(v)/len(v) if v else float('nan')

print("=== (a) AWGN SNR curve per arm (mean F1) — who improves under noise? ===")
print(f"{'arm':<10} " + " ".join(f"{c.replace('awgn@',''):>8}" for c in AWGN))
for a in ARMS:
    row = " ".join(f"{mean(c,a):8.1f}" for c in AWGN)
    lift = max(mean(c,a) for c in AWGN[1:]) - mean('clean',a)   # best-noise minus clean
    print(f"{a:<10} {row}   noise_lift={lift:+.1f}")

print("\n=== (b) per-seed CLEAN F1 — is clean unstable / dragged by bad seeds? ===")
print(f"{'arm':<10} " + " ".join(f"seed{s:>2}" for s in range(5)) + "    mean   std")
for a in ARMS:
    vals = [data['clean'].get((a,s), float('nan')) for s in range(5)]
    m = st.mean([v for v in vals if v==v]); s_ = st.pstdev([v for v in vals if v==v])
    print(f"{a:<10} " + " ".join(f"{v:6.1f}" for v in vals) + f"   {m:6.1f} {s_:5.1f}")

print("\n=== (c) clean vs awgn@-2dB per seed (does EACH seed improve, or just mean?) ===")
print(f"{'arm':<10} " + " ".join(f"s{s}:cln->n2" for s in range(5)))
for a in ARMS:
    cells = []
    for s in range(5):
        c = data['clean'].get((a,s), float('nan'))
        n = data['awgn@-2dB'].get((a,s), float('nan'))
        cells.append(f"{c:4.0f}->{n:4.0f}")
    print(f"{a:<10} " + " ".join(cells))
