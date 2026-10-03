"""
noise_utils.py — Non-Gaussian noise generators + independent statistical verification.

Noise types:
  "awgn"      : Gaussian white noise (replicates CWRUDataset for completeness).
  "pink"      : 1/f colored noise via spectral shaping (FFT method).
  "impulsive" : Heavy-tailed noise via symmetric α-stable distribution (CMS method).

SNR definition (all types, same as CWRUDataset AWGN):
  SNR_dB = 10 * log10(E[s²] / E[n²])
  Implemented as: noise scaled so sample power = signal_power × 10^(-SNR_dB/10).
  Works for any distribution regardless of whether E[n²] is theoretically finite.

Independent verification (NO circular argument — uses only independent noise
samples, NOT model outputs or training metrics):
  Pink : Welch PSD → log-log OLS slope ≈ −1 (1/f spectral character).
  Impulsive : sample excess kurtosis >> 0 + fraction beyond 5×IQR >> Gaussian baseline.
"""
import numpy as np
import scipy.signal
import scipy.stats


# ── Noise generators ───────────────────────────────────────────────────────────

def generate_pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Spectral-shaping: white → FFT → multiply amp by 1/√f → IFFT → pink noise.

    PSD_pink(f) ∝ 1/f  ↔  slope −1 in log–log space.  DC component zeroed.
    Returns float32 array of length n.
    """
    white = rng.standard_normal(n).astype(np.float32)
    F = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n).astype(np.float32)
    freqs[0] = 1.0          # avoid ÷0 at DC
    F_pink = F / np.sqrt(freqs)
    F_pink[0] = 0.0         # zero DC so mean≈0
    return np.fft.irfft(F_pink, n=n).astype(np.float32)


def generate_impulsive_noise(n: int, alpha: float = 1.5,
                              rng: np.random.Generator | None = None) -> np.ndarray:
    """Symmetric α-stable (SαS) noise via Chambers-Mallows-Stuck (CMS) algorithm.

    For α < 2: power-law tail with exponent −α  (infinite variance → impulsive).
    β = 0 (symmetric), unit scale (scaled externally via apply_noise_at_snr).
    α = 2 → Gaussian; α = 1 → Cauchy; α ∈ (1, 2) → sub-Gaussian stable.
    Returns float32 array of length n.
    """
    if rng is None:
        rng = np.random.default_rng()
    alpha = float(alpha)
    assert 0.0 < alpha <= 2.0, f"alpha must be in (0, 2], got {alpha}"

    U = rng.uniform(-np.pi / 2.0, np.pi / 2.0, n)
    W = rng.exponential(1.0, n)

    if abs(alpha - 2.0) < 1e-6:            # Gaussian special case
        X = np.sqrt(2.0) * np.cos(U) * np.sqrt(-np.log(np.maximum(W, 1e-300)))
    elif abs(alpha - 1.0) < 1e-6:          # Cauchy special case
        X = np.tan(U)
    else:
        # General CMS formula for SαS, β=0:
        #   X = sin(αU) / |cos(U)|^(1/α)  ×  (cos((1−α)U) / W)^((1−α)/α)
        sin_aU = np.sin(alpha * U)
        inv_a = 1.0 / alpha
        cos_U_pow = np.power(np.abs(np.cos(U)) + 1e-300, inv_a)
        # (1−α)/α is negative for α>1; cos((1−α)U) is positive for α∈(1,2), U∈(−π/2, π/2)
        cos_term = np.maximum(np.cos((1.0 - alpha) * U), 1e-300)
        exp_power = (1.0 - alpha) / alpha
        factor2 = np.power(cos_term / np.maximum(W, 1e-300), exp_power)
        X = sin_aU / cos_U_pow * factor2

    # Clip extreme values (keeps heavy-tail character but avoids numerical ±inf)
    X = np.clip(X, -1e6, 1e6)
    return X.astype(np.float32)


def apply_noise_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale noise to achieve target SNR relative to signal.

    Both signal and noise have shape (..., L).  Returns scaled noise (same shape).
    SNR_dB = 10*log10(signal_power / noise_power); solved for noise scale.
    """
    sig_pwr = np.mean(signal ** 2, axis=-1, keepdims=True).clip(min=1e-12)
    noise_pwr = np.mean(noise ** 2, axis=-1, keepdims=True).clip(min=1e-12)
    target_noise_pwr = sig_pwr / (10.0 ** (snr_db / 10.0))
    return noise * np.sqrt(target_noise_pwr / noise_pwr)


# ── Independent statistical verification ──────────────────────────────────────

def verify_pink_noise_properties(fs: float = 12_000, n_samples: int = 12_000,
                                  n_trials: int = 50, rng_seed: int = 7001) -> dict:
    """Verify pink noise has 1/f spectral character.

    Method: Welch PSD on INDEPENDENT noise samples (not from training set),
    log–log OLS regression of PSD vs frequency (10–2000 Hz).
    PASS criterion: mean slope ∈ [−1.5, −0.5]  (ideal: −1.0 ± 0.5).
    Returns result dict; never uses any model output.
    """
    rng = np.random.default_rng(rng_seed)
    slopes, r2s = [], []
    for _ in range(n_trials):
        noise = generate_pink_noise(n_samples, rng)
        noise = (noise - noise.mean()) / (noise.std() + 1e-8)   # unit variance
        f_arr, psd = scipy.signal.welch(noise, fs=fs, nperseg=1024)
        mask = (f_arr >= 10.0) & (f_arr <= 2000.0) & (psd > 0)
        if mask.sum() < 5:
            continue
        slope, _, r_val, _, _ = scipy.stats.linregress(
            np.log10(f_arr[mask]), np.log10(psd[mask]))
        slopes.append(float(slope))
        r2s.append(float(r_val ** 2))

    mean_slope = float(np.mean(slopes))
    std_slope = float(np.std(slopes))
    passed = -1.5 <= mean_slope <= -0.5
    return {
        "noise_type": "pink",
        "property_checked": "1/f log-log PSD slope (independent samples, Welch + OLS)",
        "n_trials": len(slopes),
        "mean_slope": mean_slope,
        "std_slope": std_slope,
        "mean_r2": float(np.mean(r2s)),
        "expected_slope": -1.0,
        "tolerance": [-1.5, -0.5],
        "passed": passed,
    }


def verify_impulsive_noise_properties(alpha: float = 1.5, n_samples: int = 50_000,
                                       n_trials: int = 20, rng_seed: int = 7002) -> dict:
    """Verify impulsive noise has heavy-tailed statistics.

    Method: two independent checks on INDEPENDENT noise samples (not training set):
      (a) Sample excess kurtosis > 5.0  (Gaussian = 0, α-stable α<2 → theoretically ∞)
      (b) Fraction of samples beyond 5×IQR from median > 1%
          (Gaussian beyond 5×IQR ≈ ~0.00003%; heavy-tailed >> 1%)
    PASS: both (a) AND (b) hold (median over n_trials).
    Never uses any model output.
    """
    rng = np.random.default_rng(rng_seed)
    kurtoses, tail_fracs = [], []
    for _ in range(n_trials):
        noise = generate_impulsive_noise(n_samples, alpha=alpha, rng=rng)
        k = float(scipy.stats.kurtosis(noise, fisher=True, nan_policy="omit"))
        kurtoses.append(k)
        q75, q25 = np.percentile(noise, [75, 25])
        iqr = q75 - q25 + 1e-8
        frac = float(np.mean(np.abs((noise - np.median(noise)) / iqr) > 5.0))
        tail_fracs.append(frac)

    med_k = float(np.median(kurtoses))
    med_frac = float(np.median(tail_fracs))
    passed = (med_k > 5.0) and (med_frac > 0.01)
    return {
        "noise_type": "impulsive",
        "property_checked": "excess kurtosis + fraction beyond 5×IQR (independent samples)",
        "alpha": alpha,
        "n_trials": n_trials,
        "median_excess_kurtosis": med_k,
        "gaussian_baseline_kurtosis": 0.0,
        "required_min_kurtosis": 5.0,
        "median_tail_frac_beyond_5IQR": med_frac,
        "required_min_tail_frac": 0.01,
        "gaussian_baseline_tail_frac": 0.00003,
        "passed": passed,
    }


def verify_snr_injection(
    target_snrs_db: list = None,
    n_windows: int = 200,
    win_len: int = 2048,
    noise_alpha: float = 1.5,
    rng_seed: int = 7003,
    tolerance_db: float = 0.5,
) -> dict:
    """Back-calculate achieved signal-side SNR after noise injection.

    Independent verification (no model outputs, no training data):
      1. Generates n_windows synthetic clean windows (sum of 5 random sinusoids,
         float32, 12 kHz, length win_len) representative of bearing vibration.
      2. For each noise_type ∈ {pink, impulsive} and each target_snr:
           - Generates raw noise via generate_{pink,impulsive}_noise
           - Scales via apply_noise_at_snr(signal, noise, target_snr_db)
           - Measures achieved SNR BEFORE z-score normalisation:
               achieved_db = 10 * log10(mean(signal²) / mean(scaled_noise²))
      3. Reports per-(noise_type, target_snr): mean_error, std_error, max_abs_error.
    PASS: max_abs_error_db <= tolerance_db for every (noise_type, target_snr).
    """
    if target_snrs_db is None:
        target_snrs_db = [-10.0, -6.0, 0.0, 6.0]

    rng = np.random.default_rng(rng_seed)

    # Synthetic clean windows: sum of 5 sinusoids at random frequencies/phases/amps
    t = np.arange(win_len, dtype=np.float32) / 12_000.0
    freqs = rng.uniform(10, 5000, (n_windows, 5)).astype(np.float32)
    phases = rng.uniform(0, 2 * np.pi, (n_windows, 5)).astype(np.float32)
    amps = rng.uniform(0.1, 1.0, (n_windows, 5)).astype(np.float32)
    clean = np.sum(
        amps[:, :, None] * np.sin(
            2 * np.pi * freqs[:, :, None] * t[None, None, :] + phases[:, :, None]
        ),
        axis=1,
    ).astype(np.float32)  # (n_windows, win_len)

    results_by_type = {}
    all_passed = True

    for noise_type in ["pink", "impulsive"]:
        per_target = []
        for target_snr in target_snrs_db:
            achieved = []
            for i in range(n_windows):
                sig = clean[i]                                      # (win_len,)
                if noise_type == "pink":
                    raw = generate_pink_noise(win_len, rng)
                else:
                    raw = generate_impulsive_noise(win_len, alpha=noise_alpha, rng=rng)

                scaled = apply_noise_at_snr(sig, raw, float(target_snr))  # (win_len,)

                sig_pwr = float(np.mean(sig ** 2))
                noise_pwr = float(np.mean(scaled ** 2))
                if sig_pwr < 1e-15 or noise_pwr < 1e-15:
                    continue
                achieved.append(10.0 * np.log10(sig_pwr / noise_pwr))

            errs = [a - float(target_snr) for a in achieved]
            max_abs = float(np.max(np.abs(errs))) if errs else float("inf")
            per_target.append({
                "target_snr_db": float(target_snr),
                "n_windows": len(achieved),
                "mean_achieved_snr_db": round(float(np.mean(achieved)), 6) if achieved else None,
                "mean_error_db": round(float(np.mean(errs)), 6) if errs else None,
                "std_error_db": round(float(np.std(errs)), 6) if errs else None,
                "max_abs_error_db": round(max_abs, 6),
                "tolerance_db": tolerance_db,
                "passed": max_abs <= tolerance_db,
            })

        type_passed = all(pt["passed"] for pt in per_target)
        all_passed = all_passed and type_passed
        results_by_type[noise_type] = {
            "noise_type": noise_type,
            "n_windows_per_target": n_windows,
            "per_target_snr": per_target,
            "max_abs_error_db_overall": round(
                max(pt["max_abs_error_db"] for pt in per_target), 6
            ),
            "tolerance_db": tolerance_db,
            "passed": type_passed,
        }

    return {
        "check": "signal_side_snr_injection_back_calculation",
        "method": (
            "synthetic clean windows (5-component sinusoid, float32, 12 kHz) → "
            "apply_noise_at_snr → 10*log10(mean(sig²)/mean(scaled_noise²)) vs target"
        ),
        "n_windows": n_windows,
        "win_len": win_len,
        "noise_alpha_impulsive": noise_alpha,
        "target_snrs_db": [float(s) for s in target_snrs_db],
        "results_by_noise_type": results_by_type,
        "ALL_PASS": all_passed,
    }
