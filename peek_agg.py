#!/usr/bin/env python3
# peek_agg.py — early aggregate of completed cells (mean+-std macro_F1 over seeds)
import json, sys, statistics as st
from collections import defaultdict

path = "results/fullgrid_fixed_20260628-2248/cells.jsonl"
rows = defaultdict(lambda: defaultdict(list))  # cond -> arm -> [f1,...]
order = []
with open(path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        cond = d.get("condition") or d.get("label") or "?"
        arm  = d.get("arm") or d.get("model") or "?"
        f1   = d.get("best_macro_f1", d.get("macro_f1", d.get("macro_F1", d.get("f1"))))
        if f1 is None:
            continue
        f1 = float(f1)
        if f1 <= 1.5:
            f1 *= 100.0
        if cond not in order:
            order.append(cond)
        rows[cond][arm].append(f1)

ARMS = ["bm3_kin", "s4d", "cnnlstm", "tcn", "cnn1d"]
def cell(vals):
    if not vals:
        return "   -   "
    m = sum(vals)/len(vals)
    s = st.pstdev(vals) if len(vals) > 1 else 0.0
    return f"{m:5.1f}±{s:4.1f}"

hdr = f"{'condition':<16} " + " ".join(f"{a:>11}" for a in ARMS) + "   n  Δ(bm3-s4d)"
print(hdr)
print("-"*len(hdr))
for cond in order:
    nseed = max((len(rows[cond][a]) for a in ARMS), default=0)
    line = f"{cond:<16} " + " ".join(f"{cell(rows[cond][a]):>11}" for a in ARMS)
    b = rows[cond].get("bm3_kin", []); s = rows[cond].get("s4d", [])
    delta = (sum(b)/len(b) - sum(s)/len(s)) if (b and s) else float('nan')
    line += f"   {nseed}  {delta:+5.2f}"
    print(line)
