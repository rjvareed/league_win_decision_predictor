"""Probabilistic classification metrics. `p` is always P(blue wins)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

EPS = 1e-7


def per_sample_log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float64), EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    """Equal-mass binned |mean prediction - observed blue winrate|."""
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    which = np.searchsorted(edges[1:-1], p, side="right")  # equal predictions share a bin
    ece = 0.0
    for b in np.unique(which):
        chunk = which == b
        ece += chunk.mean() * abs(p[chunk].mean() - y[chunk].mean())
    return float(ece)


def summarize(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = y.astype(np.float64)
    p = p.astype(np.float64)
    return {
        "log_loss": float(per_sample_log_loss(y, p).mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "auc": float(roc_auc_score(y, p)) if 0 < y.mean() < 1 and np.ptp(p) > 0 else 0.5,
        "accuracy": float(np.mean((p > 0.5) == (y > 0.5))),
        "ece": expected_calibration_error(y, p),
    }


def paired_bootstrap_logloss_diff(y, p_a, p_b, n_boot: int = 2000, seed: int = 0):
    """Mean and 95% CI of log_loss(a) - log_loss(b) on the same games (negative: a is better)."""
    d = per_sample_log_loss(y, p_a) - per_sample_log_loss(y, p_b)
    rng = np.random.default_rng(seed)
    means = np.array([d[rng.integers(0, len(d), size=len(d))].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)
