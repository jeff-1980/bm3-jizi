"""
bearmamba3/data_xjtu.py

XJTU-SY bearing run-to-failure dataset pipeline.

Dataset: LDK UER204, Z=8, d=7.92mm, D=34.55mm, α=0°, fs=25600Hz
Source: Lei Y. et al., XJTU-SY_Bearing_Datasets (IEEE DataPort)

Label protocol (D23-a):
  Official failure-position labels (OR=0, IR=1).
  Cage / Mixed bearings are excluded entirely.
  Only fault-phase windows are used (kurtosis-based onset detection).

Supported splits (D23-b):
  - within-condition LOBO: train on k-1 bearings, test on 1 (k folds)
  - cross-condition: train on Cond2 (2250rpm), test on Cond3 (2400rpm)

Health boundary (kurtosis onset detection):
  For each bearing, compute per-file kurtosis on the horizontal channel.
  The fault onset is defined as the first file index where the rolling 5-file
  mean kurtosis exceeds `kurtosis_threshold` (default 5.0).
  Only files from the onset onwards are included as labeled windows.
  Rationale: healthy bearing kurtosis ≈ 3 (Gaussian); incipient fault > 4-5;
  clear fault > 6. Threshold 5.0 conservatively captures early fault signature
  while excluding the long healthy run-in period.
  The exact threshold is recorded in the dataset stats dict for reproducibility.
"""

from __future__ import annotations
import os
import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# ── Official failure labels (from Introduction PDF, Table 2) ──────────────────
# None = excluded (Cage / Mixed / complex failure)
BEARING_FAILURE = {
    # Condition 1: 35Hz/12kN → 2100 rpm
    'Bearing1_1': 'OR',
    'Bearing1_2': 'OR',
    'Bearing1_3': 'OR',
    'Bearing1_4': None,   # Cage
    'Bearing1_5': None,   # Inner race + Outer race (mixed)
    # Condition 2: 37.5Hz/11kN → 2250 rpm
    'Bearing2_1': 'IR',
    'Bearing2_2': 'OR',
    'Bearing2_3': None,   # Cage
    'Bearing2_4': 'OR',
    'Bearing2_5': 'OR',
    # Condition 3: 40Hz/10kN → 2400 rpm
    'Bearing3_1': 'OR',
    'Bearing3_2': None,   # Inner race + ball + cage + outer race (complex)
    'Bearing3_3': 'IR',
    'Bearing3_4': 'IR',
    'Bearing3_5': 'OR',
}

CONDITION_RPM: Dict[str, float] = {
    '35Hz12kN':   2100.0,
    '37.5Hz11kN': 2250.0,
    '40Hz10kN':   2400.0,
}

# Bearing name → condition folder name
BEARING_CONDITION: Dict[str, str] = {
    **{f'Bearing1_{i}': '35Hz12kN'   for i in range(1, 6)},
    **{f'Bearing2_{i}': '37.5Hz11kN' for i in range(1, 6)},
    **{f'Bearing3_{i}': '40Hz10kN'   for i in range(1, 6)},
}

LABEL_MAP = {'OR': 0, 'IR': 1}
FS = 25_600  # Hz
WINDOW_SIZE = 2048

# ── Public helpers ────────────────────────────────────────────────────────────

def valid_bearings(condition: str) -> List[str]:
    """Return OR/IR bearings for a condition name (Cage/Mixed excluded)."""
    cond_prefix = {'35Hz12kN': '1', '37.5Hz11kN': '2', '40Hz10kN': '3'}[condition]
    return [
        f'Bearing{cond_prefix}_{i}'
        for i in range(1, 6)
        if BEARING_FAILURE.get(f'Bearing{cond_prefix}_{i}') in ('OR', 'IR')
    ]


def make_lobo_folds(bearings: List[str]) -> List[Tuple[List[str], List[str]]]:
    """Leave-one-bearing-out folds: [(train_list, [test_bearing]), ...]"""
    return [
        ([b for b in bearings if b != leave_out], [leave_out])
        for leave_out in bearings
    ]


def make_cross_condition_split(
    train_cond: str = '37.5Hz11kN',
    test_cond: str  = '40Hz10kN',
) -> Tuple[List[str], List[str]]:
    """Cross-condition split per D23-b / D24: Cond2→Cond3, OR+IR only."""
    return valid_bearings(train_cond), valid_bearings(test_cond)


def compute_fault_onset(
    data_root: str,
    bearing_name: str,
    kurtosis_threshold: float = 5.0,
    rms_multiplier: float = 2.0,
    rolling_window: int = 5,
    baseline_files: int = 20,
) -> Tuple[int, np.ndarray]:
    """
    Find fault onset using dual-criterion (kurtosis OR RMS), take the earlier trigger.

    Criterion 1 — kurtosis: first file where rolling(K, k=5) > kurtosis_threshold (default 5.0).
    Criterion 2 — RMS: first file where rolling(std, k=5) > rms_multiplier × baseline_std,
        where baseline_std = mean std of first `baseline_files` files.
    Onset = earliest of the two criteria. Fallback: last 15% of files (more conservative
    than the former 30% fallback, to avoid labelling healthy data as fault).

    Returns (onset_idx, per_file_kurtosis_array).
    Caller can inspect returned kurtosis array to validate onset detection.
    """
    cond = BEARING_CONDITION[bearing_name]
    folder = os.path.join(data_root, cond, bearing_name)
    csv_files = sorted(
        glob.glob(os.path.join(folder, '*.csv')),
        key=lambda f: int(os.path.splitext(os.path.basename(f))[0])
    )
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {folder}")

    kurtosises = []
    stds = []
    for f in csv_files:
        data = np.loadtxt(f, delimiter=',', skiprows=1, usecols=0)  # horizontal
        mu = data.mean()
        sigma = float(data.std())
        stds.append(sigma)
        if sigma < 1e-10:
            kurtosises.append(3.0)  # Gaussian default
        else:
            kurtosises.append(float(np.mean(((data - mu) / sigma) ** 4)))

    kurtosises = np.array(kurtosises)
    stds       = np.array(stds)

    # Criterion 1: kurtosis rolling mean > threshold
    k_roll = np.convolve(kurtosises, np.ones(rolling_window) / rolling_window, mode='valid')
    k_exceeded = np.where(k_roll > kurtosis_threshold)[0]
    kurt_onset = int(k_exceeded[0]) if len(k_exceeded) > 0 else None

    # Criterion 2: RMS rolling mean > rms_multiplier × baseline
    n_base = min(baseline_files, len(csv_files) // 5)
    baseline_std = float(stds[:n_base].mean()) if n_base > 0 else 0.5
    rms_roll = np.convolve(stds, np.ones(rolling_window) / rolling_window, mode='valid')
    rms_exceeded = np.where(rms_roll > rms_multiplier * baseline_std)[0]
    rms_onset = int(rms_exceeded[0]) if len(rms_exceeded) > 0 else None

    # Take earliest trigger; fallback to last 15%
    candidates = [o for o in (kurt_onset, rms_onset) if o is not None]
    if candidates:
        onset_idx = min(candidates)
    else:
        onset_idx = int(len(csv_files) * 0.85)

    return onset_idx, kurtosises


def bearing_stats(
    data_root: str,
    kurtosis_threshold: float = 5.0,
    rms_multiplier: float = 2.0,
) -> None:
    """Print per-bearing fault onset info with trigger reason for inspection."""
    for bname, label in sorted(BEARING_FAILURE.items()):
        if label is None:
            continue
        try:
            onset, kurt = compute_fault_onset(
                data_root, bname, kurtosis_threshold, rms_multiplier
            )
            cond = BEARING_CONDITION[bname]
            all_csvs = sorted(
                glob.glob(os.path.join(data_root, cond, bname, '*.csv')),
                key=lambda f: int(os.path.splitext(os.path.basename(f))[0])
            )
            total = len(all_csvs)
            n_fault = total - onset
            n_wins = (n_fault * 32768) // WINDOW_SIZE
            # Determine trigger reason
            k_roll = np.convolve(kurt, np.ones(5)/5, mode='valid')
            k_trig = np.where(k_roll > kurtosis_threshold)[0]
            stds = np.array([np.loadtxt(f, delimiter=',', skiprows=1, usecols=0).std()
                             for f in all_csvs])
            n_base = min(20, len(all_csvs) // 5)
            baseline_std = float(stds[:max(n_base, 1)].mean())
            rms_roll = np.convolve(stds, np.ones(5)/5, mode='valid')
            rms_trig = np.where(rms_roll > rms_multiplier * baseline_std)[0]
            if len(k_trig) > 0 and len(rms_trig) > 0:
                reason = 'min(K,RMS)'
            elif len(k_trig) > 0:
                reason = 'kurtosis  '
            elif len(rms_trig) > 0:
                reason = 'RMS       '
            else:
                reason = 'fallback  '
            print(f"{bname} ({label:2s}) onset={onset:4d}/{total:4d}  "
                  f"fault_files={n_fault:4d}  windows≈{n_wins:5d}  [{reason}]")
        except Exception as e:
            print(f"{bname}: ERROR — {e}")


# ── Dataset ───────────────────────────────────────────────────────────────────

class XJTUDataset(Dataset):
    """
    XJTU-SY bearing dataset. Returns (signal, label, rpm) tuples.

    signal: float32 tensor [n_sensors, window_size]
            n_sensors=1: horizontal vibration only
            n_sensors=2: [horizontal, vertical] (B4-5 dual-channel)
    label:  int64 scalar  (0=OR, 1=IR)
    rpm:    float32 scalar (nominal RPM for this bearing's condition)

    Args:
        data_root:          path to XJTU-SY_Bearing_Datasets/
        bearings:           list of bearing names to include
        n_sensors:          1 (horizontal only) or 2 (horizontal+vertical)
        noise_snr_db:       if not None, add AWGN at this SNR (dB), per channel
        window_size:        samples per window (default 2048)
        kurtosis_threshold: threshold for fault onset detection (default 5.0)
        stride:             window stride; if None, uses window_size (non-overlapping)
    """

    def __init__(
        self,
        data_root: str,
        bearings: List[str],
        n_sensors: int = 1,
        noise_snr_db: Optional[float] = None,
        window_size: int = WINDOW_SIZE,
        kurtosis_threshold: float = 5.0,
        stride: Optional[int] = None,
    ) -> None:
        super().__init__()
        if n_sensors not in (1, 2):
            raise ValueError(f"n_sensors must be 1 or 2, got {n_sensors}")
        self.n_sensors    = n_sensors
        self.noise_snr_db = noise_snr_db
        self.window_size  = window_size
        self.stride = stride if stride is not None else window_size

        # Validate bearing labels
        for b in bearings:
            if BEARING_FAILURE.get(b) not in ('OR', 'IR'):
                raise ValueError(
                    f"{b} is not a valid OR/IR bearing "
                    f"(label={BEARING_FAILURE.get(b)!r}). "
                    "Cage/Mixed bearings must be excluded before calling XJTUDataset."
                )

        # _windows elements: (window_size,) for n_sensors=1, (2, window_size) for n_sensors=2
        self._windows: List[np.ndarray] = []
        self._labels:  List[int] = []
        self._rpms:    List[float] = []

        usecols = 0 if n_sensors == 1 else [0, 1]

        for bname in bearings:
            label = LABEL_MAP[BEARING_FAILURE[bname]]
            cond  = BEARING_CONDITION[bname]
            rpm   = CONDITION_RPM[cond]

            onset_idx, _ = compute_fault_onset(
                data_root, bname, kurtosis_threshold
            )

            # Load all CSV files from onset onwards
            folder = os.path.join(data_root, cond, bname)
            all_csvs = sorted(
                glob.glob(os.path.join(folder, '*.csv')),
                key=lambda f: int(os.path.splitext(os.path.basename(f))[0])
            )
            fault_csvs = all_csvs[onset_idx:]

            for csv_path in fault_csvs:
                raw = np.loadtxt(csv_path, delimiter=',', skiprows=1,
                                 usecols=usecols, dtype=np.float32)
                if n_sensors == 1:
                    # raw shape: (N,)
                    n = len(raw)
                    pos = 0
                    while pos + window_size <= n:
                        self._windows.append(raw[pos: pos + window_size])
                        self._labels.append(label)
                        self._rpms.append(rpm)
                        pos += self.stride
                else:
                    # raw shape: (N, 2) → transpose to (2, N)
                    data2 = raw.T  # (2, N)
                    n = data2.shape[1]
                    pos = 0
                    while pos + window_size <= n:
                        self._windows.append(data2[:, pos: pos + window_size].copy())
                        self._labels.append(label)
                        self._rpms.append(rpm)
                        pos += self.stride

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int):
        x = self._windows[idx].copy()

        if self.n_sensors == 1:
            # x: (window_size,)
            mu, sigma = x.mean(), x.std()
            if sigma > 1e-8:
                x = (x - mu) / sigma
            if self.noise_snr_db is not None:
                sig_power = np.mean(x ** 2)
                noise_power = sig_power / (10 ** (self.noise_snr_db / 10))
                x = x + np.random.randn(len(x)).astype(np.float32) * np.sqrt(noise_power)
            signal = torch.from_numpy(x).unsqueeze(0)   # [1, window_size]
        else:
            # x: (2, window_size) — z-score and noise per channel
            for c in range(self.n_sensors):
                mu, sigma = x[c].mean(), x[c].std()
                if sigma > 1e-8:
                    x[c] = (x[c] - mu) / sigma
            if self.noise_snr_db is not None:
                for c in range(self.n_sensors):
                    sig_power = np.mean(x[c] ** 2)
                    noise_power = sig_power / (10 ** (self.noise_snr_db / 10))
                    x[c] = x[c] + np.random.randn(x.shape[1]).astype(np.float32) * np.sqrt(noise_power)
            signal = torch.from_numpy(x)   # [2, window_size]

        label  = torch.tensor(self._labels[idx], dtype=torch.long)
        rpm    = torch.tensor(self._rpms[idx], dtype=torch.float32)
        return signal, label, rpm
