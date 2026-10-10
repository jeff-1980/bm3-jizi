"""
Paper-9 stage-6: Paderborn (PU, KAt) binary OR/IR dataset with the interface the XJTU harness uses
(`_windows`, `_labels`, `_rpms`), so that xjtu_noisy_harness.XJTUDatasetNoisy, its noise injection,
z-scoring, sampler and evaluation run unchanged.

Design (fixed in prereg_stage6_pu.md before any training cell):
  - real-damage bearings only (accelerated lifetime tests); OR = 0, IR = 1 (as XJTU LABEL_MAP).
  - channel `vibration_1` (64 kHz) resampled by scipy.signal.resample_poly(up=2, down=5) to 25.6 kHz,
    so a 2048-sample window spans 80 ms exactly as in XJTU-SY (25.6 kHz, 2048 samples).
  - non-overlapping windows of 2048 samples; all 20 recordings of each bearing x condition, in
    recording order 1..20; no onset detection (every recording is post-damage).
  - files are located by name anywhere under DATA_ROOT; duplicates are resolved by content hash and
    must be identical (asserted).
"""
import glob, hashlib, os, re
import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly

DATA_ROOT = __import__("os").environ.get("P9_PU_ROOT", "/home/jeffwork/data_pu")  # Paderborn .mat files, one sub-directory per bearing code
LABEL = {"KA": 0, "KI": 1}                    # outer race 0, inner race 1
COND_RPM = {"N15_M07_F10": 1500.0, "N09_M07_F10": 900.0, "N15_M01_F10": 1500.0, "N15_M07_F04": 1500.0}
WINDOW = 2048
UP, DOWN = 2, 5                               # 64 kHz -> 25.6 kHz
TRAIN_BEARINGS = ["KA04", "KA16", "KI04", "KI14"]
TEST_BEARINGS = ["KA15", "KA22", "KI17", "KI18", "KI21"]
TRAIN_COND, TEST_COND = "N15_M07_F10", "N09_M07_F10"


def _files(bearing, cond, root=DATA_ROOT):
    out = []
    for n in range(1, 21):
        hits = glob.glob(os.path.join(root, "**", f"{cond}_{bearing}_{n}.mat"), recursive=True)
        assert hits, (bearing, cond, n)
        digests = {hashlib.sha256(open(p, "rb").read()).hexdigest() for p in hits}
        assert len(digests) == 1, f"non-identical duplicates for {cond}_{bearing}_{n}: {hits}"
        out.append(sorted(hits)[0])
    return out


def _vibration(path):
    m = sio.loadmat(path)
    key = [k for k in m if not k.startswith("__")][0]
    Y = m[key]["Y"][0, 0]
    names = [str(Y[0, i]["Name"][0]) for i in range(Y.shape[1])]
    return Y[0, names.index("vibration_1")]["Data"].ravel().astype(np.float64)


class PUDatasetBinary:
    def __init__(self, bearings, cond, root=DATA_ROOT, cache_dir=None):
        self.bearings, self.cond = list(bearings), cond
        self._windows, self._labels, self._rpms, self._bearing = [], [], [], []
        for b in self.bearings:
            cache = os.path.join(cache_dir, f"{cond}_{b}.npy") if cache_dir else None
            if cache and os.path.exists(cache):
                W = np.load(cache)
            else:
                ws = []
                for p in _files(b, cond, root):
                    x = resample_poly(_vibration(p), UP, DOWN).astype(np.float32)
                    k = len(x) // WINDOW
                    ws.append(x[: k * WINDOW].reshape(k, WINDOW))
                W = np.concatenate(ws)
                if cache:
                    os.makedirs(cache_dir, exist_ok=True); np.save(cache, W)
            for w in W:
                self._windows.append(w)
                self._labels.append(LABEL[b[:2]])
                self._rpms.append(COND_RPM[cond])
                self._bearing.append(b)

    def __len__(self):
        return len(self._labels)

    def counts(self):
        from collections import Counter
        return dict(Counter(self._bearing)), dict(Counter(self._labels))


def content_fingerprint(ds):
    """sha256 over window bytes and labels (stronger than the harness's label-only fingerprint)."""
    h = hashlib.sha256()
    h.update(np.stack(ds._windows).astype(np.float32).tobytes())
    h.update(np.asarray(ds._labels, dtype=np.int64).tobytes())
    return h.hexdigest()
