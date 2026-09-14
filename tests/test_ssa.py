"""
Pytest tests for src/ssa.py and the vol_normalise_returns utility.

All fixtures use small, deterministic data to keep execution fast.
SSA parameters: L=10, rank=3, N=50 (unless otherwise noted).
"""
import numpy as np
import pandas as pd
import pytest

from src.ssa import (
    SSA_K,
    hankelise,
    diagonal_average,
    ssa_decompose,
    ssa_recurrent_forecast,
    al1_robust_svd_component,
    run_rlssa_decompose,
    run_rlssa_forecast,
    run_ssa_strategy,
)
from src.utils import vol_normalise_returns

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)

def _seasonal_series(N: int = 50, L: int = 10, seed: int = 7) -> np.ndarray:
    """Deterministic seasonal + noise series of length N."""
    rng = np.random.default_rng(seed)
    t = np.arange(N)
    return 0.02 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.01, N)


def _make_panel(n_months: int = 100, n_commodities: int = 3, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic monthly panel with n_months × n_commodities rows.

    Month range starts at 2006-01 so that oos_start='2013-11' aligns with
    trade_idx = 94 ≥ N_window = L + SSA_K - 1 when L=10.
    """
    rng = np.random.default_rng(seed)
    months = pd.date_range('2006-01-01', periods=n_months, freq='MS')
    commodities = [f'C{i:02d}' for i in range(n_commodities)]

    records = []
    for comm in commodities:
        t = np.arange(n_months)
        seasonal = 0.015 * np.sin(2 * np.pi * t / 12)
        noise = rng.normal(0.0, 0.03, n_months)
        log_ret = seasonal + noise
        simple_ret = np.expm1(log_ret)
        for i, m in enumerate(months):
            records.append({
                'month_label':   m,
                'commodity':     comm,
                'log_return':    float(log_ret[i]),
                'simple_return': float(simple_ret[i]),
                'selected_flag': 1,
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_hankelise_shape():
    """Trajectory matrix must have shape (L, N-L+1)."""
    N, L = 50, 10
    series = _seasonal_series(N, L)
    H = hankelise(series, L)
    assert H.shape == (L, N - L + 1), f"Expected ({L}, {N-L+1}), got {H.shape}"


def test_diagonal_average_roundtrip():
    """hankelise then diagonal_average must exactly recover the original series."""
    N, L = 50, 10
    series = _seasonal_series(N, L)
    H = hankelise(series, L)
    recovered = diagonal_average(H, N)
    np.testing.assert_allclose(
        recovered, series, atol=1e-12,
        err_msg="diagonal_average(hankelise(x)) != x",
    )


def test_ssa_decompose_rank():
    """SVD rank truncation must produce correctly-shaped factor matrices."""
    N, L, rank = 50, 10, 3
    series = _seasonal_series(N, L)
    U, S, Vt = ssa_decompose(series, L=L, rank=rank)
    K = N - L + 1
    assert U.shape  == (L, rank),    f"U shape wrong: {U.shape}"
    assert S.shape  == (rank,),      f"S shape wrong: {S.shape}"
    assert Vt.shape == (rank, K),    f"Vt shape wrong: {Vt.shape}"
    assert np.all(S >= 0),           "Singular values must be non-negative"


def test_ssa_recurrent_forecast_finite():
    """Recurrent SSA forecast must return a finite scalar."""
    N, L, rank = 50, 10, 3
    series = _seasonal_series(N, L)
    U, S, Vt = ssa_decompose(series, L=L, rank=rank)
    fc = ssa_recurrent_forecast(series, U, L=L)
    assert np.isfinite(fc), f"Forecast is not finite: {fc}"


def test_al1_robust_svd_basic():
    """AL1 component must converge and return correctly shaped results."""
    rng = np.random.default_rng(99)
    L_h, K_h = 8, 15
    H = rng.normal(0, 1, (L_h, K_h))
    u, v, sigma = al1_robust_svd_component(H, max_iter=5000, tol=1e-8)
    assert u.shape == (L_h,),      f"u shape wrong: {u.shape}"
    assert v.shape == (K_h,),      f"v shape wrong: {v.shape}"
    assert np.isfinite(sigma),     f"sigma not finite: {sigma}"
    assert abs(np.linalg.norm(u) - 1.0) < 1e-6, "u must be unit-norm"
    assert abs(np.linalg.norm(v) - 1.0) < 1e-6, "v must be unit-norm"


def test_al1_robust_svd_convergence_error():
    """AL1 must raise RuntimeError when max_iter is too small to converge."""
    rng = np.random.default_rng(123)
    H = rng.normal(0, 1, (6, 10))
    # Force non-convergence: max_iter=1, tol=0 (impossible to satisfy in 1 iter)
    with pytest.raises(RuntimeError, match="RLSSA AL1 failed to converge"):
        al1_robust_svd_component(H, max_iter=1, tol=0.0)


def test_rlssa_decompose_shape():
    """RLSSA decompose must return arrays with correct shapes."""
    N, L, rank = 50, 10, 3
    series = _seasonal_series(N, L)
    H = hankelise(series, L)
    K = N - L + 1
    U_r, S_r, V_r = run_rlssa_decompose(H, rank=rank, max_iter=5000, tol=1e-8)
    assert U_r.shape == (L, rank), f"U_r shape wrong: {U_r.shape}"
    assert S_r.shape == (rank,),   f"S_r shape wrong: {S_r.shape}"
    assert V_r.shape == (K, rank), f"V_r shape wrong: {V_r.shape}"


def test_rlssa_forecast_finite():
    """RLSSA one-step forecast must return a finite scalar."""
    N, L, rank = 50, 10, 3
    series = _seasonal_series(N, L)
    fc = run_rlssa_forecast(series, L=L, rank=rank, max_iter=5000, tol=1e-8)
    assert np.isfinite(fc), f"RLSSA forecast is not finite: {fc}"


def test_vol_normalise_length():
    """vol_normalise_returns must preserve the length of its input."""
    rng = np.random.default_rng(55)
    for N in (12, 50, 120):
        series = rng.normal(0, 0.03, N)
        normed = vol_normalise_returns(series)
        assert len(normed) == N, f"Length mismatch for N={N}: got {len(normed)}"


def test_ssa_strategy_output_shape():
    """
    run_ssa_strategy must return a DataFrame with standard columns.

    Uses L=10 (N_window = L + SSA_K - 1 = 94) and a 100-month panel so
    that months at trade_idx ≥ 94 are tradeable. OOS: 2013-11 to 2014-04.
    """
    panel = _make_panel(n_months=100, n_commodities=3, seed=1)
    result = run_ssa_strategy(
        panel,
        oos_start='2013-11',
        oos_end='2014-04',
        use_rlssa=False,
        vol_normalise=False,
        L=10,
        rank=3,
    )
    expected_cols = {
        'month', 'long_commodity', 'long_net_return',
        'short_commodity', 'short_net_return', 'n_scored',
    }
    assert isinstance(result, pd.DataFrame), "Result must be a DataFrame"
    assert expected_cols.issubset(result.columns), (
        f"Missing columns: {expected_cols - set(result.columns)}"
    )
    assert len(result) > 0, "Result DataFrame must have at least one row"
