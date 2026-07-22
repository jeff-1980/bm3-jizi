#!/home/jeffwork/论文8/venv/bin/python3
"""
frozensel_regen_curve_fixed.py — one-off regeneration of frozensel_curve_table.csv
and frozensel_curve.png using the corrected snr_for_plot_map() in
frozensel_aggregate.py (bug: build_conditions()'s own "clean" entry, snr_db=None,
overwrote the hardcoded clean->15.0 default, silently dropping the clean
condition from the curve table/plot).

Does NOT touch results/frozensel_20260702-1827/frozensel_curve_table.csv or
frozensel_curve.png (already written, chmod 444, additive-only guardrail).
Writes the corrected pair into a NEW timestamped directory instead.

Usage:
  python frozensel_regen_curve_fixed.py --run-dir results/frozensel_20260702-1827 \
      --prereg results/frozensel_prereg_20260702-1826/frozensel_prereg.json \
      --out-dir results/frozensel_curvefix_<TS>
"""
import argparse
import json
from pathlib import Path

import frozensel_aggregate as FA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    prereg = json.loads(Path(args.prereg).read_text())
    design = prereg["design"]

    cells = FA.load_cells(run_dir)
    agg = FA.aggregate(cells, prereg)

    csv_path = FA.write_curve_table(out_dir, agg, design)
    png_path = FA.write_curve_figure(out_dir, agg, design)
    print(f"[OK] corrected curve table -> {csv_path}")
    print(f"[OK] corrected curve figure -> {png_path}")


if __name__ == "__main__":
    main()
