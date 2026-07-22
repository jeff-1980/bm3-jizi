#!/usr/bin/env python3
# check_fair.py — verify byte-identical fairness fingerprints across arms within each (condition, seed)
import json
from collections import defaultdict

path = "results/fullgrid_fixed_20260628-2248/cells.jsonl"
g = defaultdict(dict)  # (cond,seed) -> arm -> (eval_sha, noise_sha)
with open(path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        key = (d["condition"], d["seed"])
        fp = d.get("fairness", {})
        g[key][d["arm"]] = (fp.get("eval_sha256"), fp.get("noise_sha256"))

bad = 0
checked = 0
for key, arms in sorted(g.items()):
    evals = {a: v[0] for a, v in arms.items()}
    noises = {a: v[1] for a, v in arms.items()}
    checked += 1
    if len(set(evals.values())) != 1:
        bad += 1; print(f"FAIL eval mismatch {key}: {evals}")
    if len(set(noises.values())) != 1:
        bad += 1; print(f"FAIL noise mismatch {key}: {noises}")

print(f"checked {checked} (condition,seed) groups; arms/group="
      f"{len(next(iter(g.values())))}; fairness FAILS={bad}")
print("ALL PASS — byte-identical across arms" if bad == 0 else "FAIRNESS BROKEN")
