"""
Paper-9 stage-8: noise random stream bound to the split, so training and evaluation noise are independent.

The harness's XJTUDatasetNoisy draws noise from default_rng([rng_seed, idx]); every driver passed the same
seed to the training and the evaluation wrapper, so window idx of the training set and window idx of the
evaluation set received the same base noise vector. IndepNoisyDataset keys the stream on
[rng_seed, SPLIT_ID[split], idx]: training and evaluation streams are independent, every arm evaluated with
a given seed sees the same evaluation noise (a common evaluation noise bank per seed), and noise is
fixed per (seed, split, window) as before. Everything else (z-scoring, SNR scaling, labels) is the
harness's own code path; only the random key changes.
"""
import numpy as np
import torch

SPLIT_ID = {"train": 101, "test": 202}


def make(h):
    class IndepNoisyDataset(h.XJTUDatasetNoisy):
        def __init__(self, base_ds, noise_type="clean", snr_db=None, rng_seed=0, split="train", **kw):
            super().__init__(base_ds, noise_type, snr_db, rng_seed, **kw)
            assert split in SPLIT_ID
            self.split = split

        def noise_vector(self, idx, n):
            rng = np.random.default_rng([self.rng_seed, SPLIT_ID[self.split], idx])
            if self.noise_type == "awgn":
                return rng.standard_normal(n).astype(np.float32)
            if self.noise_type == "pink":
                return h.generate_pink_noise(n, rng)
            if self.noise_type == "impulsive":
                return h.generate_impulsive_noise(n, self.noise_alpha, rng)
            raise ValueError(self.noise_type)

        def __getitem__(self, idx):
            x = self._windows[idx].copy()
            mu, sigma = x.mean(), x.std()
            if sigma > 1e-8:
                x = (x - mu) / sigma
            if self.snr_db is not None and self.noise_type != "clean":
                x = x + h.apply_noise_at_snr(x, self.noise_vector(idx, len(x)), self.snr_db)
            return (torch.from_numpy(x).unsqueeze(0), torch.tensor(self._labels[idx], dtype=torch.long),
                    torch.tensor(self._rpms[idx], dtype=torch.float32))
    return IndepNoisyDataset
