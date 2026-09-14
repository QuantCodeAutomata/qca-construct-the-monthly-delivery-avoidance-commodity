"""
Pytest tests for src/bootstrap.py.

Uses small, deterministic fixtures (T=30, B=50) for speed.
"""
import numpy as np
import pytest

from src.bootstrap import (
    meb_generate_paths,
    paired_sharpe_test,
    holm_adjust,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _rng_series(T: int = 30, seed: int = 0) -> np.ndarray:
    """Deterministic return series of length T."""
    return np.random.default_rng(seed).normal(0.005, 0.03, T)


# ---------------------------------------------------------------------------
# MEB path generation tests
# ---------------------------------------------------------------------------

def test_meb_generate_paths_shape():
    """meb_generate_paths must return an array of shape (B, T)."""
    T, B = 30, 50
    series = _rng_series(T)
    paths = meb_generate_paths(series, B=B, seed=1)
    assert paths.shape == (B, T), f"Expected ({B}, {T}), got {paths.shape}"


def test_meb_generate_paths_reproducible():
    """Same seed must produce identical bootstrap paths."""
    series = _rng_series(30)
    paths_1 = meb_generate_paths(series, B=20, seed=99)
    paths_2 = meb_generate_paths(series, B=20, seed=99)
    np.testing.assert_array_equal(paths_1, paths_2, err_msg="Paths differ with same seed")


def test_meb_generate_paths_different_seeds():
    """Different seeds must (almost surely) produce different paths."""
    series = _rng_series(30)
    paths_a = meb_generate_paths(series, B=20, seed=1)
    paths_b = meb_generate_paths(series, B=20, seed=2)
    assert not np.allclose(paths_a, paths_b), "Different seeds gave identical paths"


def test_meb_marginal_distribution():
    """
    Bootstrap paths' column means should be close to the sorted-series range.

    Each observation in a path is drawn from the interval around its rank,
    so the paths are bounded by the extended boundary points.
    """
    T, B = 40, 200
    series = _rng_series(T, seed=5)
    paths = meb_generate_paths(series, B=B, seed=5, boundary_reaching=True)

    # Extended lower boundary: sorted[0] - (sorted[1]-sorted[0])/2
    sorted_s = np.sort(series)
    lower = sorted_s[0]  - (sorted_s[1]  - sorted_s[0])  / 2
    upper = sorted_s[-1] + (sorted_s[-1] - sorted_s[-2]) / 2

    assert paths.min() >= lower - 1e-9, "Paths go below lower boundary"
    assert paths.max() <= upper + 1e-9, "Paths exceed upper boundary"


# ---------------------------------------------------------------------------
# Paired Sharpe test
# ---------------------------------------------------------------------------

def test_paired_sharpe_test_identical():
    """When A == B, d_obs = 0 so every bootstrap draw satisfies |d_b*| >= 0: p = 1."""
    T, B = 30, 50
    rets = _rng_series(T, seed=7)
    boot_rets = meb_generate_paths(rets, B=B, seed=7)

    d_obs, p_val, n_exc = paired_sharpe_test(rets, rets, boot_rets, boot_rets)

    assert d_obs == pytest.approx(0.0, abs=1e-12), f"d_obs should be 0, got {d_obs}"
    assert p_val == pytest.approx(1.0, abs=1e-12), f"p_val should be 1.0, got {p_val}"
    assert n_exc == B, f"n_exceed should be B={B}, got {n_exc}"


def test_paired_sharpe_test_pvalue_bounds():
    """p-value must lie in [1/(B+1), 1] by construction."""
    T, B = 30, 50
    rets_a = _rng_series(T, seed=10)
    rets_b = _rng_series(T, seed=20)
    boot_a = meb_generate_paths(rets_a, B=B, seed=10)
    boot_b = meb_generate_paths(rets_b, B=B, seed=20)

    _, p_val, _ = paired_sharpe_test(rets_a, rets_b, boot_a, boot_b)

    p_min = 1.0 / (B + 1)
    assert p_min <= p_val <= 1.0, f"p_val={p_val} outside [{p_min:.4f}, 1.0]"


def test_paired_sharpe_test_returns_tuple():
    """paired_sharpe_test must return a 3-tuple (float, float, int)."""
    T, B = 20, 30
    rets_a = _rng_series(T, seed=1)
    rets_b = _rng_series(T, seed=2)
    boot_a = meb_generate_paths(rets_a, B=B, seed=1)
    boot_b = meb_generate_paths(rets_b, B=B, seed=2)

    result = paired_sharpe_test(rets_a, rets_b, boot_a, boot_b)
    assert len(result) == 3
    d, p, n = result
    assert isinstance(d, float)
    assert isinstance(p, float)
    assert isinstance(n, int)


# ---------------------------------------------------------------------------
# Holm adjustment tests
# ---------------------------------------------------------------------------

def test_holm_adjust_no_rejections():
    """Large p-values must all be non-rejected."""
    p_values = [0.5, 0.8, 0.9, 1.0]
    result = holm_adjust(p_values, alpha=0.05)
    assert result == [False, False, False, False], f"Expected no rejections: {result}"


def test_holm_adjust_all_reject():
    """Tiny p-values must all be rejected."""
    p_values = [1e-10, 2e-10, 3e-10]
    result = holm_adjust(p_values, alpha=0.05)
    assert result == [True, True, True], f"Expected all rejections: {result}"


def test_holm_adjust_partial():
    """
    Holm step-down: reject only the smallest p if the second already exceeds threshold.

    With m=3, alpha=0.05:
      Threshold for rank 0: 0.05/3 ≈ 0.0167 → 0.001 ≤ 0.0167 → REJECT
      Threshold for rank 1: 0.05/2 = 0.025  → 0.10  > 0.025  → STOP
    Original indices: 0→0.001 (reject), 1→0.10 (no), 2→0.50 (no).
    """
    p_values = [0.001, 0.10, 0.50]
    result = holm_adjust(p_values, alpha=0.05)
    assert result[0] is True,  f"Index 0 (p=0.001) should be rejected"
    assert result[1] is False, f"Index 1 (p=0.10) should not be rejected"
    assert result[2] is False, f"Index 2 (p=0.50) should not be rejected"


def test_holm_adjust_single():
    """Single p-value: reject iff p <= alpha."""
    assert holm_adjust([0.03], alpha=0.05) == [True]
    assert holm_adjust([0.06], alpha=0.05) == [False]


def test_holm_adjust_preserves_order():
    """Output list must have the same length as input."""
    p_values = [0.01, 0.2, 0.05, 0.001, 0.3]
    result = holm_adjust(p_values, alpha=0.05)
    assert len(result) == len(p_values)
