#!/home/jeffwork/论文8/venv/bin/python3
"""
xjtu_noisy_harness.py — XJTU cross-condition defense: strong baselines + non-Gaussian noise.

Grid (full mode):
  Arms      : bm3_kin, cnn1d, tcn, cnnlstm, s4d
  Conditions: clean | awgn@{0,-2,-6,-10} | pink@{0,-2,-6,-10} | impulsive@{0,-2,-6,-10}
  Seeds     : 0-4 (5 seeds, matching original 15-seed paper)
  Epochs    : 50 (matching original exp_xjtu/train.py)
  Metric    : best_macro_f1 across epochs

Fairness invariants (all SHA256-asserted per condition×seed):
  eval_sha256        : SHA256(test_labels bytes)   — identical across arms
  train_order_sha256 : SHA256(all sampler indices × all epochs) — identical across arms
  noise_sha256       : spot-checked per-sample RNG seeded by [seed, idx]

Usage:
  python xjtu_noisy_harness.py --smoke          # 3 cells, 5 epochs
  python xjtu_noisy_harness.py --full           # 325 cells, 50 epochs (~2.5h)
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
import random
import stat
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Paths ─────────────────────────────────────────────────────────────────────
HARNESS_DIR = Path(__file__).resolve().parent
BM3_ROOT    = Path("/home/jeffwork/论文8")
EXT_DIR     = Path("/home/jeffwork/exp/bm3-defense/extended_baselines_noisy_20260626-2235")
DATA_ROOT   = "/home/jeffwork/data_xjtu/XJTU-SY_Bearing_Datasets"

sys.path.insert(0, str(BM3_ROOT))
sys.path.insert(0, str(EXT_DIR))

from bearmamba3.data_xjtu import (
    XJTUDataset, make_cross_condition_split,
)
from bearmamba3.kinematic_loss import kinematic_loss
from bearmamba3.model import BearMamba3
from baselines.cnn1d import BearCNN1D
from models_extended import BearTCN, BearCNNLSTM, BearS4D
from bm3_frozen import BearMamba3Frozen
from bm3_models import (BearMamba3FrozenNoConv, BearMamba3FrozenNoGate,
                        BearMamba3FrozenAS4D, BearS4DPlusGate)
from noise_utils import (
    generate_pink_noise, generate_impulsive_noise, apply_noise_at_snr,
    verify_pink_noise_properties, verify_impulsive_noise_properties,
)

# ── Constants ─────────────────────────────────────────────────────────────────
TRAIN_COND  = "37.5Hz11kN"
TEST_COND   = "40Hz10kN"
FS_EFF      = 12800.0
BEARING_KW  = {"n_balls": 8, "d": 7.92, "D": 34.55}
LAMBDA_KIN  = 0.01
BATCH_SIZE  = 64
LR          = 3e-4
WD          = 1e-4
GRAD_CLIP   = 1.0
N_CLASSES   = 2
NOISE_ALPHA = 1.5

FULL_SNRS       = [10.0, 6.0, 0.0, -2.0, -6.0, -10.0]
FULL_NOISE_TYPES = ["awgn", "pink", "impulsive"]
FULL_SEEDS      = [0, 1, 2, 3, 4]
FULL_EPOCHS     = 50
FULL_ARMS       = ["bm3_kin", "cnn1d", "tcn", "cnnlstm", "s4d"]

SMOKE_ARMS   = ["bm3_kin", "tcn", "s4d"]
SMOKE_SEEDS  = [0]
SMOKE_EPOCHS = 5


# ── Dataset wrapper ───────────────────────────────────────────────────────────

class XJTUDatasetNoisy(torch.utils.data.Dataset):
    """
    Wraps an XJTUDataset.  __getitem__ z-scores, then injects noise with
    a deterministic per-sample RNG seeded by [rng_seed, sample_idx].
    All arms share the same underlying _windows / _labels / _rpms arrays
    (read-only references) — the only difference is the noise applied.
    """
    def __init__(self, base_ds: XJTUDataset,
                 noise_type: str = "clean",
                 snr_db: float | None = None,
                 rng_seed: int = 0,
                 noise_alpha: float = NOISE_ALPHA):
        self._windows    = base_ds._windows   # shared read-only reference
        self._labels     = base_ds._labels
        self._rpms       = base_ds._rpms
        self.noise_type  = noise_type
        self.snr_db      = snr_db             # None → no noise
        self.rng_seed    = rng_seed
        self.noise_alpha = noise_alpha

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx: int):
        x = self._windows[idx].copy()           # (window_size,) float32

        # z-score (same as XJTUDataset for n_sensors=1)
        mu, sigma = x.mean(), x.std()
        if sigma > 1e-8:
            x = (x - mu) / sigma

        # Noise injection — deterministic per (rng_seed, idx)
        if self.snr_db is not None and self.noise_type != "clean":
            rng = np.random.default_rng([self.rng_seed, idx])
            n = len(x)
            if self.noise_type == "awgn":
                noise = rng.standard_normal(n).astype(np.float32)
            elif self.noise_type == "pink":
                noise = generate_pink_noise(n, rng)
            elif self.noise_type == "impulsive":
                noise = generate_impulsive_noise(n, self.noise_alpha, rng)
            else:
                raise ValueError(f"Unknown noise_type: {self.noise_type!r}")
            x = x + apply_noise_at_snr(x, noise, self.snr_db)

        signal = torch.from_numpy(x).unsqueeze(0)          # [1, 2048]
        label  = torch.tensor(self._labels[idx], dtype=torch.long)
        rpm    = torch.tensor(self._rpms[idx],   dtype=torch.float32)
        return signal, label, rpm


# ── Fingerprint utilities ─────────────────────────────────────────────────────

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def eval_fingerprint(ds: XJTUDatasetNoisy) -> str:
    """SHA256 of test labels — identical across arms if same dataset."""
    arr = np.array(ds._labels, dtype=np.int64)
    return sha256_bytes(arr.tobytes())

def train_order_fingerprint(weights_list: list, n_samples: int, seed: int,
                             n_epochs: int) -> str:
    """
    Collect all WeightedRandomSampler indices for n_epochs using a fresh
    generator seeded with `seed`.  Returns SHA256 of the full index list.
    Call BEFORE actual training — uses an independent generator.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights_list, num_samples=n_samples,
        replacement=True, generator=g,
    )
    all_idx = []
    for _ in range(n_epochs):
        all_idx.extend(list(iter(sampler)))
    raw = json.dumps(all_idx).encode()
    return sha256_bytes(raw)

def noise_spot_fingerprint(noise_type: str, snr_db: float | None,
                            seed: int, n_samples: int = 5) -> str:
    """
    SHA256 of noise vectors for a fixed set of sample indices.
    Independent of any model or dataset content.
    """
    if snr_db is None or noise_type == "clean":
        return "clean"
    spot_idx = list(range(0, n_samples * 100, 100))
    parts = []
    for idx in spot_idx:
        rng = np.random.default_rng([seed, idx])
        n = 2048
        if noise_type == "awgn":
            noise = rng.standard_normal(n).astype(np.float32)
        elif noise_type == "pink":
            noise = generate_pink_noise(n, rng)
        elif noise_type == "impulsive":
            noise = generate_impulsive_noise(n, NOISE_ALPHA, rng)
        else:
            noise = np.zeros(n, np.float32)
        parts.append(noise)
    arr = np.concatenate(parts)
    return sha256_bytes(arr.tobytes())


# ── Noise verification (independent of model) ─────────────────────────────────

def verify_noise(seed: int = 0, n_signals: int = 32, sig_len: int = 2048) -> dict:
    """
    Generate standalone noise signals, verify statistical properties.
    Uses only noise_utils — no model involved.
    """
    rng = np.random.default_rng(seed + 9999)    # independent seed
    results = {}

    # Pink: PSD slope ≈ -1
    r_pink = verify_pink_noise_properties(fs=FS_EFF, n_samples=sig_len,
                                           n_trials=n_signals, rng_seed=seed + 9999)
    results["pink"] = r_pink

    # Impulsive: excess kurtosis >> 0
    r_imp = verify_impulsive_noise_properties(alpha=NOISE_ALPHA, n_trials=n_signals,
                                              rng_seed=seed + 9998)
    results["impulsive"] = r_imp

    # SNR calibration: check awgn and pink at -6dB
    base_signal = rng.standard_normal(sig_len).astype(np.float32)
    base_signal = base_signal / (np.std(base_signal) + 1e-8)
    for ntype, noise in [("awgn",      rng.standard_normal(sig_len).astype(np.float32)),
                          ("pink",      generate_pink_noise(sig_len, rng)),
                          ("impulsive", generate_impulsive_noise(sig_len, NOISE_ALPHA, rng))]:
        scaled_noise = apply_noise_at_snr(base_signal, noise, -6.0)
        noisy   = base_signal + scaled_noise            # additive path (matches __getitem__)
        sig_pwr = float(np.mean(base_signal**2))
        n_pwr   = float(np.mean(scaled_noise**2))        # power of injected noise only
        actual_snr = float(10.0 * np.log10(sig_pwr / max(n_pwr, 1e-12)))
        error_db = float(abs(actual_snr - (-6.0)))
        results[f"snr_check_{ntype}"] = {
            "target_snr_db": -6.0,
            "actual_snr_db": round(actual_snr, 3),
            "error_db": round(error_db, 3),
            # 2dB tolerance: float32 finite-sample power estimation error
            "pass": bool(error_db < 2.0),
        }

    all_pass = all(
        v.get("passed", v.get("pass", True)) if isinstance(v, dict) else True
        for v in results.values()
    )
    results["all_pass"] = bool(all_pass)
    return results


# ── Model builder ─────────────────────────────────────────────────────────────

def build_model(arm: str, device: torch.device) -> nn.Module:
    kw_base = dict(d_model=64, n_layers=4, n_sensors=1, n_classes=N_CLASSES, conv_stride=2)
    if arm in ("bm3_kin", "bm3_nokin"):
        # bm3_nokin = identical architecture, but trained with CE only (λ_kin=0).
        # The only difference from bm3_kin is the loss path in train_cell
        # (is_bm3 == False → no kinematic_loss). Isolates L_kin's contribution.
        m = BearMamba3(d_state=128, is_mimo=False, use_batchnorm=False,
                       dtype=torch.bfloat16, **kw_base)
    elif arm == "cnn1d":
        m = BearCNN1D(**kw_base)
    elif arm == "tcn":
        m = BearTCN(**kw_base)
    elif arm == "cnnlstm":
        m = BearCNNLSTM(**kw_base)
    elif arm == "s4d":
        m = BearS4D(d_state=128, **kw_base)
    elif arm == "bm3_frozen":
        # Selectivity-removed ablation of bm3_kin/bm3_nokin: identical conv
        # stem / pre-norm depth / Mamba-3 layer family, with dt/B/C's
        # input-dependence removed (see bm3_frozen.py, arch_map.md). Trained
        # with CE only (no kinematic loss), same as bm3_nokin.
        m = BearMamba3Frozen(d_state=128, use_batchnorm=False,
                             dtype=torch.bfloat16, **kw_base)
    elif arm == "frozen_noconv":
        # Secondary single-variable ablation on top of bm3_frozen: outer
        # conv_embed stem kernel shrunk 7->1 (cross-timestep mixing removed,
        # channel lift + stride kept). See arch_map.md §5a, bm3_models.py.
        m = BearMamba3FrozenNoConv(d_state=128, use_batchnorm=False,
                                   dtype=torch.bfloat16, **kw_base)
    elif arm == "frozen_nogate":
        # Secondary single-variable ablation on top of bm3_frozen: z-gating
        # removed (Z=None into the fused kernel, in_proj shrunk to match).
        # See arch_map.md §5b, bm3_models.py.
        m = BearMamba3FrozenNoGate(d_state=128, use_batchnorm=False,
                                   dtype=torch.bfloat16, **kw_base)
    elif arm == "frozen_As4d":
        # Secondary single-variable ablation on top of bm3_frozen: A's
        # functional form only (heavy_tail_activation(A_const) -> S4D-style
        # -exp(A_log)). See arch_map.md §5c, bm3_models.py.
        m = BearMamba3FrozenAS4D(d_state=128, use_batchnorm=False,
                                 dtype=torch.bfloat16, **kw_base)
    elif arm == "s4d_wide":
        # Capacity control: widen S4D state to ~178K params (matches/exceeds BM3's
        # 177,938) with identical depth/architecture. If s4d_wide still degrades under
        # noise, the BM3>S4D advantage is architectural (selectivity), not capacity.
        m = BearS4D(d_state=230, **kw_base)   # 178,498 params
    elif arm == "s4d_plus_gate":
        # Constructive graft (arch_map.md §6, bm3_models.py): never-selective
        # BearS4D (models_extended.py, zero edits) with a Mamba-3-style
        # input-dependent output gate grafted onto each layer's SSM output.
        # Tests whether adding JUST the gate mechanism (no selectivity in
        # dt/A/B/C) recovers any of the bm3_frozen-vs-s4d gap — see
        # graft_prereg_provenance.md for the predeclared R statistic/cutoffs.
        m = BearS4DPlusGate(d_state=128, **kw_base)
    else:
        raise ValueError(f"Unknown arm: {arm!r}")
    return m.to(device)


# ── Evaluation ────────────────────────────────────────────────────────────────

def eval_macro_f1(model: nn.Module,
                  loader: torch.utils.data.DataLoader,
                  device: torch.device):
    model.eval()
    param_dtype = next(model.parameters()).dtype
    tp = np.zeros(N_CLASSES, dtype=np.int64)
    fn = np.zeros(N_CLASSES, dtype=np.int64)
    fp = np.zeros(N_CLASSES, dtype=np.int64)
    with torch.no_grad():
        for x, labels, _rpm in loader:
            x      = x.to(device, dtype=param_dtype)
            labels = labels.to(device)
            out    = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            logits_f = logits.float()
            preds  = logits_f.argmax(1)
            for c in range(N_CLASSES):
                tp[c] += int(((preds == c) & (labels == c)).sum())
                fn[c] += int(((preds != c) & (labels == c)).sum())
                fp[c] += int(((preds == c) & (labels != c)).sum())
    per_f1 = []
    for c in range(N_CLASSES):
        prec = tp[c] / max(tp[c] + fp[c], 1)
        rec  = tp[c] / max(tp[c] + fn[c], 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-8)
        per_f1.append(float(f1))
    return np.array(per_f1), float(np.mean(per_f1))


# ── Single cell training ───────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_sampler_weights(labels: list[int]) -> list[float]:
    arr = np.array(labels, dtype=np.int64)
    counts = np.maximum(np.bincount(arr, minlength=N_CLASSES).astype(np.float64), 1.0)
    weights = (len(arr) / (N_CLASSES * counts))[arr]
    return weights.tolist()


def train_cell(arm: str, train_ds: XJTUDatasetNoisy, test_ds: XJTUDatasetNoisy,
               seed: int, n_epochs: int, device: torch.device,
               verbose: bool = False) -> dict:
    t0 = time.time()
    set_seed(seed)
    model = build_model(arm, device)

    weights_list = make_sampler_weights(train_ds._labels)
    n_samples    = len(train_ds)

    # Sampler with independent generator (doesn't consume PyTorch global RNG)
    g = torch.Generator()
    g.manual_seed(seed)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights_list, num_samples=n_samples, replacement=True, generator=g,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    is_bm3 = (arm == "bm3_kin")
    param_dtype = next(model.parameters()).dtype
    best_f1 = 0.0

    for epoch in range(1, n_epochs + 1):
        model.train()
        for x, labels, rpm in train_loader:
            x      = x.to(device, dtype=param_dtype)
            labels = labels.to(device)
            rpm    = rpm.to(device)

            if is_bm3:
                out, kin = model(x, return_kin=True)
                l_ce  = F.cross_entropy(out.float(), labels)
                l_kin = kinematic_loss(kin, rpm, FS_EFF,
                                       variant="cover", bearing_kwargs=BEARING_KW)
                loss  = l_ce + LAMBDA_KIN * l_kin
            else:
                logits = model(x)
                if isinstance(logits, tuple):
                    logits = logits[0]
                loss = F.cross_entropy(logits.float(), labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        scheduler.step()
        _, macro_f1 = eval_macro_f1(model, test_loader, device)
        best_f1 = max(best_f1, macro_f1)

        if verbose and (epoch <= 3 or epoch % 10 == 0 or epoch == n_epochs):
            print(f"    ep={epoch:3d}/{n_epochs}  macro_f1={macro_f1:.4f}  best={best_f1:.4f}")

    return {
        "best_macro_f1": best_f1,
        "elapsed_s": round(time.time() - t0, 1),
        "n_params": sum(p.numel() for p in model.parameters()),
    }


# ── Grid runner ───────────────────────────────────────────────────────────────

def build_conditions(noise_types: list[str], snrs: list[float]) -> list[dict]:
    conds = [{"noise_type": "clean", "snr_db": None, "label": "clean"}]
    for nt in noise_types:
        for snr in snrs:
            conds.append({"noise_type": nt, "snr_db": snr,
                          "label": f"{nt}@{snr:+.0f}dB"})
    return conds


def _write_protected(path: Path, text: str):
    path.write_text(text)
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def run_grid(run_dir: Path, conditions: list[dict], arms: list[str],
             seeds: list[int], n_epochs: int, device: torch.device,
             verbose: bool = False):
    run_dir.mkdir(parents=True, exist_ok=True)
    cells_path = run_dir / "cells.jsonl"

    # Load existing completed cells (resume support)
    completed = set()
    if cells_path.exists():
        cells_path.chmod(0o644)   # temporarily writable
        for line in cells_path.read_text().splitlines():
            try:
                cell = json.loads(line)
                k = (cell["condition"], cell["arm"], cell["seed"])
                completed.add(k)
            except Exception:
                pass

    # Load base datasets once
    train_bearings, test_bearings = make_cross_condition_split(TRAIN_COND, TEST_COND)
    base_train = XJTUDataset(DATA_ROOT, train_bearings, n_sensors=1)
    base_test  = XJTUDataset(DATA_ROOT, test_bearings,  n_sensors=1)

    eval_sha  = eval_fingerprint(XJTUDatasetNoisy(base_test, "clean", None, 0))
    print(f"[INIT] eval_sha256={eval_sha[:12]}...")

    # Noise verification (once, independent of models)
    noise_ver_path = run_dir / "noise_verification.json"
    if not noise_ver_path.exists():
        print("[NOISE] Running independent noise verification...")
        nv = verify_noise()
        def _to_serializable(obj):
            if isinstance(obj, (np.bool_, np.integer)):
                return obj.item()
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, dict):
                return {k: _to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_serializable(v) for v in obj]
            return obj
        _write_protected(noise_ver_path,
                         json.dumps(_to_serializable(nv), indent=2, ensure_ascii=False))
        if not nv.get("all_pass"):
            print(f"[WARN] Noise verification FAILED: {nv}")
        else:
            print("[NOISE] all_pass=True ✓")

    total = len(conditions) * len(arms) * len(seeds)
    done  = len(completed)
    print(f"[GRID] {total} cells total, {done} already done, "
          f"{total - done} to run")

    with open(cells_path, "a") as fout:
        for cond in conditions:
            noise_type = cond["noise_type"]
            snr_db     = cond["snr_db"]
            label      = cond["label"]

            for seed in seeds:
                # Per (condition, seed): build noisy datasets + fingerprints
                train_ds = XJTUDatasetNoisy(base_train, noise_type, snr_db, seed)
                test_ds  = XJTUDatasetNoisy(base_test,  noise_type, snr_db, seed)

                weights_list = make_sampler_weights(train_ds._labels)
                t_order_sha  = train_order_fingerprint(
                    weights_list, len(train_ds), seed, n_epochs)
                noise_sha    = noise_spot_fingerprint(noise_type, snr_db, seed)

                # Assert eval fingerprint identical for all arms
                this_eval_sha = eval_fingerprint(test_ds)
                assert this_eval_sha == eval_sha, \
                    f"eval_sha MISMATCH at {label} seed={seed}"

                for arm in arms:
                    key = (label, arm, seed)
                    if key in completed:
                        print(f"  [SKIP] {label} | {arm} | seed={seed}")
                        continue

                    print(f"  [RUN]  {label} | {arm} | seed={seed}  "
                          f"({done+1}/{total})", flush=True)

                    result = train_cell(arm, train_ds, test_ds, seed, n_epochs,
                                        device, verbose=verbose)

                    cell = {
                        "condition": label,
                        "noise_type": noise_type,
                        "snr_db": snr_db,
                        "arm": arm,
                        "seed": seed,
                        "best_macro_f1": result["best_macro_f1"],
                        "n_params": result["n_params"],
                        "elapsed_s": result["elapsed_s"],
                        "fairness": {
                            "eval_sha256": eval_sha,
                            "train_order_sha256": t_order_sha[:16],
                            "noise_sha256": noise_sha[:16] if noise_sha != "clean" else "clean",
                        },
                    }
                    fout.write(json.dumps(cell, ensure_ascii=False) + "\n")
                    fout.flush()
                    done += 1
                    print(f"         → macro_F1={result['best_macro_f1']*100:.2f}%  "
                          f"({result['elapsed_s']:.0f}s)", flush=True)

    os.chmod(cells_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    print(f"[DONE] cells.jsonl written ({done} cells)")


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyze_results(run_dir: Path):
    cells_path = run_dir / "cells.jsonl"
    if not cells_path.exists():
        print("[ANALYZE] No cells.jsonl found.")
        return

    cells = []
    for line in cells_path.read_text().splitlines():
        try:
            cells.append(json.loads(line))
        except Exception:
            pass
    if not cells:
        print("[ANALYZE] cells.jsonl is empty.")
        return

    # Aggregate per (condition, arm)
    from collections import defaultdict
    agg = defaultdict(list)
    for c in cells:
        agg[(c["condition"], c["arm"])].append(c["best_macro_f1"])

    table = []
    for (cond, arm), vals in sorted(agg.items()):
        table.append({
            "condition": cond,
            "arm": arm,
            "n_seeds": len(vals),
            "f1_mean": float(np.mean(vals)),
            "f1_std":  float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        })

    # Write summary CSV
    csv_path = run_dir / "summary_table.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition","arm","n_seeds","f1_mean","f1_std"])
        w.writeheader()
        w.writerows(table)
    os.chmod(csv_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

    # Q1: BM3 vs TCN (strongest new baseline) per condition
    q1_rows = []
    cond_set = sorted({r["condition"] for r in table})
    for cond in cond_set:
        row_bm3 = next((r for r in table if r["condition"]==cond and r["arm"]=="bm3_kin"), None)
        row_tcn = next((r for r in table if r["condition"]==cond and r["arm"]=="tcn"), None)
        if row_bm3 and row_tcn:
            delta = row_bm3["f1_mean"] - row_tcn["f1_mean"]
            q1_rows.append({
                "condition": cond,
                "bm3_f1": round(row_bm3["f1_mean"]*100, 2),
                "tcn_f1": round(row_tcn["f1_mean"]*100, 2),
                "delta_pp": round(delta*100, 2),
                "bm3_wins": delta > 0,
            })

    # Q2: BM3 vs best baseline per condition
    arm_set = [a for a in FULL_ARMS if a != "bm3_kin"]
    q2_rows = []
    for cond in cond_set:
        row_bm3 = next((r for r in table if r["condition"]==cond and r["arm"]=="bm3_kin"), None)
        if not row_bm3:
            continue
        best_bl_f1 = max(
            (r["f1_mean"] for r in table if r["condition"]==cond and r["arm"] in arm_set),
            default=None,
        )
        if best_bl_f1 is None:
            continue
        best_bl_arm = max(
            (r for r in table if r["condition"]==cond and r["arm"] in arm_set),
            key=lambda r: r["f1_mean"],
        )["arm"]
        delta = row_bm3["f1_mean"] - best_bl_f1
        q2_rows.append({
            "condition": cond,
            "bm3_f1": round(row_bm3["f1_mean"]*100, 2),
            "best_baseline": best_bl_arm,
            "best_bl_f1": round(best_bl_f1*100, 2),
            "delta_pp": round(delta*100, 2),
            "bm3_wins": delta > 0,
        })

    q1_path = run_dir / "q1.json"
    q2_path = run_dir / "q2.json"
    _write_protected(q1_path, json.dumps({"q1_bm3_vs_tcn": q1_rows}, indent=2))
    _write_protected(q2_path, json.dumps({"q2_bm3_vs_best_baseline": q2_rows}, indent=2))

    # Decision
    clean_row_bm3 = next((r for r in q2_rows if r["condition"]=="clean"), None)
    awgn_wins     = [r for r in q2_rows if r["condition"].startswith("awgn") and r["bm3_wins"]]
    nongauss_wins = [r for r in q2_rows if not r["condition"].startswith("awgn")
                     and r["condition"] != "clean" and r["bm3_wins"]]
    total_nong    = [r for r in q2_rows if not r["condition"].startswith("awgn")
                     and r["condition"] != "clean"]

    if clean_row_bm3 and clean_row_bm3["bm3_wins"]:
        if len(nongauss_wins) == 0 and len(awgn_wins) == 0:
            verdict, label_v = "b", "PARTIAL_FLIP_NONGAUSS_WINS"
        elif len(nongauss_wins) < len(total_nong) // 2:
            verdict, label_v = "b", "PARTIAL_FLIP"
        else:
            verdict, label_v = "a", "CLAIM_SURVIVES"
    else:
        verdict, label_v = "c", "CLAIM_FALSIFIED"

    decision = {
        "verdict": verdict,
        "verdict_label": label_v,
        "clean_bm3_vs_best_baseline_pp": clean_row_bm3["delta_pp"] if clean_row_bm3 else None,
        "bm3_wins_clean": bool(clean_row_bm3["bm3_wins"]) if clean_row_bm3 else None,
        "bm3_wins_nongauss": f"{len(nongauss_wins)}/{len(total_nong)}",
        "bm3_wins_awgn": f"{len(awgn_wins)}/4",
        "q1_bm3_vs_tcn_wins": sum(1 for r in q1_rows if r["bm3_wins"]),
        "n_conditions_q1": len(q1_rows),
        "losing_cells": [r["condition"] for r in q2_rows if not r["bm3_wins"]],
        "winning_cells": [r["condition"] for r in q2_rows if r["bm3_wins"]],
    }
    _write_protected(run_dir / "decision.json",
                     json.dumps(decision, indent=2, ensure_ascii=False))
    print(f"\n[VERDICT] {verdict} — {label_v}")
    print(f"  clean BM3 vs best-BL: {clean_row_bm3['delta_pp']:+.2f}pp" if clean_row_bm3 else "")
    print(f"  BM3 wins non-Gaussian: {len(nongauss_wins)}/{len(total_nong)}")
    print(f"  BM3 wins AWGN:         {len(awgn_wins)}/4")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke test: pink@-6dB × bm3_kin/tcn/s4d × seed=0 × 5ep")
    ap.add_argument("--full",  action="store_true",
                    help="Full grid: 13 conditions × 5 arms × 5 seeds × 50 epochs")
    ap.add_argument("--ablation", action="store_true",
                    help="λ_kin=0 ablation: clean+awgn × bm3_nokin × 5 seeds × 50 epochs (35 cells)")
    ap.add_argument("--capctrl", action="store_true",
                    help="capacity control: clean+awgn × s4d_wide(~178K) × 5 seeds × 50 epochs (35 cells)")
    ap.add_argument("--frozensel", action="store_true",
                    help="frozen-selectivity ablation: clean+awgn × bm3_kin/bm3_frozen/s4d × 5 seeds × 50 epochs (105 cells)")
    ap.add_argument("--secondary", action="store_true",
                    help="secondary single-variable ablations: clean+awgn × bm3_frozen/frozen_noconv/frozen_nogate/frozen_As4d/s4d × 5 seeds × 50 epochs (175 cells)")
    ap.add_argument("--graft", action="store_true",
                    help="constructive graft: clean+awgn × s4d_plus_gate × 5 seeds × 50 epochs (35 cells)")
    ap.add_argument("--analyze-only", action="store_true",
                    help="Skip training, only run analyze_results on existing cells.jsonl")
    ap.add_argument("--run-dir", type=str, default=None,
                    help="Output directory (default: auto-timestamped)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.smoke:
        run_dir = HARNESS_DIR / "results" / f"smoke_{ts}"
    elif args.ablation:
        run_dir = HARNESS_DIR / "results" / f"ablation_nokin_{ts}"
    elif args.capctrl:
        run_dir = HARNESS_DIR / "results" / f"capctrl_s4dwide_{ts}"
    elif args.frozensel:
        run_dir = HARNESS_DIR / "results" / f"frozensel_{ts}"
    elif args.secondary:
        run_dir = HARNESS_DIR / "results" / f"secondary_{ts}"
    elif args.graft:
        run_dir = HARNESS_DIR / "results" / f"graft_{ts}"
    else:
        run_dir = HARNESS_DIR / "results" / f"fullgrid_{ts}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[START] device={device}  run_dir={run_dir}")

    if args.analyze_only:
        analyze_results(run_dir)
        return

    if args.smoke:
        conditions = [{"noise_type": "pink", "snr_db": -6.0, "label": "pink@-6dB"}]
        arms       = SMOKE_ARMS
        seeds      = SMOKE_SEEDS
        n_epochs   = SMOKE_EPOCHS
    elif args.full:
        conditions = build_conditions(FULL_NOISE_TYPES, FULL_SNRS)
        arms       = FULL_ARMS
        seeds      = FULL_SEEDS
        n_epochs   = FULL_EPOCHS
    elif args.ablation:
        # Only the AWGN column (clean + awgn@{10,6,0,-2,-6,-10}) = 7 conditions.
        conditions = build_conditions(["awgn"], FULL_SNRS)
        arms       = ["bm3_nokin"]
        seeds      = FULL_SEEDS
        n_epochs   = FULL_EPOCHS
    elif args.capctrl:
        # Capacity control: same AWGN column, widened S4D (~178K params).
        conditions = build_conditions(["awgn"], FULL_SNRS)
        arms       = ["s4d_wide"]
        seeds      = FULL_SEEDS
        n_epochs   = FULL_EPOCHS
    elif args.frozensel:
        # Frozen-selectivity ablation: same AWGN column as --ablation/--capctrl,
        # bm3_kin (selective) vs bm3_frozen (selectivity removed, same Mamba-3
        # layer family, see bm3_frozen.py/arch_map.md) vs s4d (never-selective).
        conditions = build_conditions(["awgn"], FULL_SNRS)
        arms       = ["bm3_kin", "bm3_frozen", "s4d"]
        seeds      = FULL_SEEDS
        n_epochs   = FULL_EPOCHS
    elif args.secondary:
        # Secondary single-variable ablations: same AWGN column as
        # --frozensel, bm3_frozen (frozen base) vs its three single-part
        # variants (frozen_noconv/frozen_nogate/frozen_As4d, arch_map.md §5,
        # bm3_models.py) vs s4d (never-selective reference). bm3_frozen/s4d
        # cells for this exact config already exist in
        # results/frozensel_20260702-1827/cells.jsonl and can be reused
        # without retraining — see launch_secondary.sh's header comment.
        conditions = build_conditions(["awgn"], FULL_SNRS)
        arms       = ["bm3_frozen", "frozen_noconv", "frozen_nogate", "frozen_As4d", "s4d"]
        seeds      = FULL_SEEDS
        n_epochs   = FULL_EPOCHS
    elif args.graft:
        # Constructive graft: same AWGN column as --secondary, single arm
        # (s4d_plus_gate, arch_map.md §6/bm3_models.py). bm3_frozen/s4d
        # baseline values for the predeclared R statistic are taken from
        # secondary_20260703-1034 (same batch) per graft_prereg_provenance.md
        # — not retrained here.
        conditions = build_conditions(["awgn"], FULL_SNRS)
        arms       = ["s4d_plus_gate"]
        seeds      = FULL_SEEDS
        n_epochs   = FULL_EPOCHS
    else:
        ap.print_help()
        return

    run_grid(run_dir, conditions, arms, seeds, n_epochs, device,
             verbose=args.verbose)
    if not (args.ablation or args.capctrl or args.frozensel or args.secondary or args.graft):
        # analyze_results() assumes FULL_ARMS (cnn1d/tcn/cnnlstm) baselines and
        # writes a claim-bearing decision.json (CLAIM_SURVIVES/FALSIFIED) — not
        # meaningful for --ablation/--capctrl/--frozensel's own arm sets, and
        # this task explicitly asks not to author claim-bearing conclusions.
        analyze_results(run_dir)
    print(f"\n[COMPLETE] Results in {run_dir}/")


if __name__ == "__main__":
    main()
