"""
Portable loader for the unmodified training harness (xjtu_noisy_harness.py).

The harness file is byte-identical to the one that produced every result and is deliberately not edited.
It hard-codes three absolute paths near its top (BM3_ROOT, EXT_DIR, DATA_ROOT) and puts the first two at the
front of sys.path. This loader reads the harness source, replaces exactly those three assignments with
environment-controlled values, and executes the result as module "h" (the name the drivers use):

  P9_BM3_ROOT   directory containing the `bearmamba3` and `baselines` packages  (default: harness directory)
  P9_EXT_DIR    directory containing models_extended.py and noise_utils.py       (default: harness directory)
  P9_XJTU_ROOT  XJTU-SY dataset root (contains XJTU-SY_Bearing_Datasets' contents) (default: original path)

With no environment variables the repository's own copies are imported (they are byte-identical to the files
the original grids imported; checked with cmp). load() asserts where bearmamba3 and models_extended came from.
"""
import os, sys, types
from pathlib import Path

_ORIG_DATA = "/home/jeffwork/data_xjtu/XJTU-SY_Bearing_Datasets"
_REPL = [
    ('BM3_ROOT    = Path("/home/jeffwork/论文8")',
     'BM3_ROOT    = Path(__import__("os").environ.get("P9_BM3_ROOT") or HARNESS_DIR)'),
    ('EXT_DIR     = Path("/home/jeffwork/exp/bm3-defense/extended_baselines_noisy_20260626-2235")',
     'EXT_DIR     = Path(__import__("os").environ.get("P9_EXT_DIR") or HARNESS_DIR)'),
    ('DATA_ROOT   = "' + _ORIG_DATA + '"',
     'DATA_ROOT   = __import__("os").environ.get("P9_XJTU_ROOT") or "' + _ORIG_DATA + '"'),
]


def load(harness_dir):
    path = Path(harness_dir) / "xjtu_noisy_harness.py"
    src = path.read_text(encoding="utf8")
    for old, new in _REPL:
        assert src.count(old) == 1, f"harness path line not found exactly once: {old[:40]}"
        src = src.replace(old, new)
    mod = types.ModuleType("h"); mod.__file__ = str(path); sys.modules["h"] = mod
    exec(compile(src, str(path), "exec"), mod.__dict__)
    import bearmamba3, models_extended
    root = Path(mod.BM3_ROOT).resolve(); ext = Path(mod.EXT_DIR).resolve()
    assert Path(bearmamba3.__file__).resolve().is_relative_to(root), (bearmamba3.__file__, root)
    assert Path(models_extended.__file__).resolve().is_relative_to(ext), (models_extended.__file__, ext)
    print(f"[harness_loader] bearmamba3 from {bearmamba3.__file__}; models_extended from {models_extended.__file__}; "
          f"DATA_ROOT {mod.DATA_ROOT}", flush=True)
    return mod
