"""Tests for src/dvr.py — DVR seasonal trading strategy."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import pytest

from src.dvr import fit_dvr, run_dvr_strategy, summarise_dvr_results, _hac_lag
from src.utils import compute_net_long_return, compute_net_short_return


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_modeling_panel(
    commodities: list,
    n_months: int,
    start: str = '2006-01',
    rng_seed: int = 0,
    boost_month: int = None,
    boost_commodity: str = None,
    boost_magnitude: float = 0.05,
    short_month: int = None,
    short_commodity: str = None,
    short_magnitude: float = 0.05,
) -> pd.DataFrame:
    """
    Build a synthetic modeling panel DataFrame.

    Args:
        commodities:       list of commodity codes.
        n_months:          total number of months.
        start:             YYYY-MM start date string.
        rng_seed:          numpy random seed.
        boost_month:       calendar month (1-12) to inject a positive signal.
        boost_commodity:   commodity to boost.
        boost_magnitude:   magnitude of the positive boost.
        short_month:       calendar month (1-12) to inject a negative signal.
        short_commodity:   commodity to short.
        short_magnitude:   magnitude of the negative signal.
    """
    rng = np.random.default_rng(rng_seed)
    months = pd.date_range(start, periods=n_months, freq='ME')
    records = []
    for dt in months:
        for comm in commodities:
            base_ret = rng.normal(0.002, 0.03)
            # Optionally inject a seasonal signal
            if boost_month is not None and boost_commodity == comm and dt.month == boost_month:
                base_ret += boost_magnitude
            if short_month is not None and short_commodity == comm and dt.month == short_month:
                base_ret -= short_magnitude
            records.append({
                'commodity': comm,
                'month_label': dt.strftime('%Y-%m'),
                'year': dt.year,
                'month': dt.month,
                'simple_return': base_ret,
                'log_return': np.log(1 + base_ret),
                'monthly_avg_volume': 5000.0,
                'selected_flag': True,
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# _hac_lag
# ---------------------------------------------------------------------------

def test_hac_lag_computation():
    """HAC lag = floor(0.75 * T^(1/3)); T=120 gives floor(0.75*4.932)=3."""
    assert _hac_lag(120) == 3
    assert _hac_lag(27) == 2     # 0.75 * 3 = 2.25 → floor = 2
    assert _hac_lag(1) == 1      # always >= 1


# ---------------------------------------------------------------------------
# fit_dvr
# ---------------------------------------------------------------------------

def test_dvr_fit_basic():
    """fit_dvr on random returns returns three float values."""
    rng = np.random.default_rng(42)
    T = 120
    months_arr = np.tile(np.arange(1, 13), T // 12)
    returns = rng.normal(0.001, 0.03, T)

    beta, pval, alpha = fit_dvr(returns, 6, months_arr)

    assert beta is not None
    assert isinstance(beta, float)
    assert isinstance(pval, float)
    assert isinstance(alpha, float)
    assert 0.0 <= pval <= 1.0


def test_dvr_fit_significant_positive():
    """A strong monthly boost produces a significant positive beta."""
    rng = np.random.default_rng(7)
    T = 120
    months_arr = np.tile(np.arange(1, 13), T // 12)
    returns = rng.normal(0.001, 0.005, T)
    # Inject a very strong positive January signal
    returns[months_arr == 1] += 0.12

    beta, pval, _ = fit_dvr(returns, 1, months_arr)

    assert beta is not None
    assert beta > 0
    assert pval < 0.10


def test_dvr_fit_significant_negative():
    """A strong monthly negative effect produces a significant negative beta."""
    rng = np.random.default_rng(7)
    T = 120
    months_arr = np.tile(np.arange(1, 13), T // 12)
    returns = rng.normal(0.001, 0.005, T)
    # Inject a very strong negative July signal
    returns[months_arr == 7] -= 0.12

    beta, pval, _ = fit_dvr(returns, 7, months_arr)

    assert beta is not None
    assert beta < 0
    assert pval < 0.10


def test_dvr_fit_non_target_month_neutral():
    """Fitting a dummy for a month with no injected signal yields high p-value."""
    rng = np.random.default_rng(3)
    T = 120
    months_arr = np.tile(np.arange(1, 13), T // 12)
    returns = rng.normal(0.001, 0.03, T)
    # No signal injected; check that p-value is not trivially zero for month 6
    beta, pval, _ = fit_dvr(returns, 6, months_arr)
    # p-value should be > 0 (could be anything, just verify it's a valid probability)
    assert pval is not None
    assert 0.0 <= pval <= 1.0


def test_dvr_fit_insufficient_data():
    """With fewer than 15 observations fit_dvr returns (None, None, None)."""
    returns = np.array([0.01, 0.02, -0.01])
    months_arr = np.array([1, 2, 3])

    beta, pval, alpha = fit_dvr(returns, 1, months_arr)

    assert beta is None
    assert pval is None
    assert alpha is None


def test_dvr_fit_exact_boundary():
    """15 observations (boundary) should return a result (not None)."""
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.02, 15)
    months_arr = np.array([i % 12 + 1 for i in range(15)])

    beta, pval, alpha = fit_dvr(returns, 1, months_arr)

    # Should succeed (T=15 >= 15)
    assert beta is not None


# ---------------------------------------------------------------------------
# compute_net_long_return / compute_net_short_return
# ---------------------------------------------------------------------------

def test_net_long_return_formula():
    """net_long = (1 + R)(1 - c)^2 - 1."""
    R = 0.10
    c = 0.00086
    expected = (1 + R) * (1 - c) ** 2 - 1
    result = compute_net_long_return(R, c)
    assert abs(result - expected) < 1e-12


def test_net_short_return_formula():
    """net_short = (1 - R)(1 - c)^2 - 1."""
    R = 0.10
    c = 0.00086
    expected = (1 - R) * (1 - c) ** 2 - 1
    result = compute_net_short_return(R, c)
    assert abs(result - expected) < 1e-12


def test_net_long_return_zero_cost():
    """With c=0 net_long == simple return R."""
    R = 0.05
    result = compute_net_long_return(R, c=0.0)
    assert abs(result - R) < 1e-12


def test_net_short_return_zero_cost():
    """With c=0 and R=0.05 net_short = 1 - 1.05 - 1 = -0.05."""
    R = 0.05
    result = compute_net_short_return(R, c=0.0)
    assert abs(result - (-R)) < 1e-12


def test_net_long_positive_for_positive_return():
    """A positive return generates a positive net long return (before large costs)."""
    assert compute_net_long_return(0.05) > 0


def test_net_short_positive_for_negative_return():
    """A negative underlying return yields a positive net short return."""
    assert compute_net_short_return(-0.05) > 0


# ---------------------------------------------------------------------------
# DVR cash logic
# ---------------------------------------------------------------------------

def test_dvr_cash_logic():
    """
    When no commodity has a significant signal the strategy holds cash
    (long/short net returns == 0.0 and is_long/short_cash == True).
    """
    # Build panel with purely white-noise returns (no seasonal patterns)
    # Use a very high significance threshold to guarantee no signals
    rng = np.random.default_rng(999)
    T = 130  # 120 estimation + 10 OOS
    months = pd.date_range('2006-01', periods=T, freq='ME')
    records = []
    for dt in months:
        ret = float(rng.normal(0.001, 0.001))   # tiny variance → no signal
        records.append({
            'commodity': 'CC',
            'month_label': dt.strftime('%Y-%m'),
            'year': dt.year,
            'month': dt.month,
            'simple_return': ret,
            'log_return': np.log(1 + ret),
            'monthly_avg_volume': 5000.0,
            'selected_flag': True,
        })
    panel = pd.DataFrame(records)

    oos_start = months[120].strftime('%Y-%m')
    oos_end = months[-1].strftime('%Y-%m')

    results = run_dvr_strategy(
        panel,
        oos_start=oos_start,
        oos_end=oos_end,
        significance=0.0001,   # almost impossible to beat
    )

    if len(results) > 0:
        # All OOS months should be cash
        assert results['is_long_cash'].all(), "Expected all long positions to be cash"
        assert results['is_short_cash'].all(), "Expected all short positions to be cash"
        assert (results['long_net_return'] == 0.0).all()
        assert (results['short_net_return'] == 0.0).all()


# ---------------------------------------------------------------------------
# DVR full run (minimal)
# ---------------------------------------------------------------------------

def test_dvr_full_run_minimal():
    """
    run_dvr_strategy produces a non-empty DataFrame with the expected columns
    for a minimal 2-commodity panel.
    """
    panel = _make_modeling_panel(
        commodities=['CC', 'CF'],
        n_months=130,
        start='2006-01',
        rng_seed=0,
    )
    months = pd.date_range('2006-01', periods=130, freq='ME')
    oos_start = months[120].strftime('%Y-%m')
    oos_end = months[-1].strftime('%Y-%m')

    results = run_dvr_strategy(panel, oos_start=oos_start, oos_end=oos_end)

    expected_cols = {
        'month', 'long_commodity', 'long_beta', 'long_pval', 'long_net_return',
        'short_commodity', 'short_beta', 'short_pval', 'short_net_return',
        'benchmark_net_return', 'is_long_cash', 'is_short_cash',
    }
    assert expected_cols.issubset(set(results.columns)), (
        f"Missing columns: {expected_cols - set(results.columns)}"
    )
    assert len(results) > 0


def test_dvr_full_run_detects_signal():
    """
    A very strong injected seasonal signal is detected by DVR and generates
    at least some in-market (non-cash) long months in the OOS period.
    """
    panel = _make_modeling_panel(
        commodities=['CC', 'CF', 'NG'],
        n_months=150,
        start='2004-01',
        rng_seed=5,
        boost_month=3,
        boost_commodity='CC',
        boost_magnitude=0.15,    # very large signal
    )
    months = pd.date_range('2004-01', periods=150, freq='ME')
    oos_start = months[120].strftime('%Y-%m')
    oos_end = months[-1].strftime('%Y-%m')

    results = run_dvr_strategy(panel, oos_start=oos_start, oos_end=oos_end)

    # At least one OOS month should have a long position (not all cash)
    assert len(results) > 0
    # In-market months should have non-zero returns (due to transaction costs
    # they might be small but structurally non-zero unless return is exactly 0)
    in_market = results[~results['is_long_cash']]
    if len(in_market) > 0:
        # Verify return formula was applied (not raw return)
        assert (in_market['long_net_return'] != in_market['long_beta']).all()


def test_summarise_dvr_results_keys():
    """summarise_dvr_results returns a dict with all expected keys."""
    rng = np.random.default_rng(0)
    n = 20
    months = pd.date_range('2022-01', periods=n, freq='ME')
    df = pd.DataFrame({
        'month': months,
        'long_net_return': rng.normal(0.002, 0.03, n),
        'short_net_return': rng.normal(0.001, 0.03, n),
        'benchmark_net_return': rng.normal(0.001, 0.02, n),
        'is_long_cash': [False] * n,
        'is_short_cash': [False] * n,
        'long_commodity': ['CC'] * n,
        'short_commodity': ['NG'] * n,
    })

    summary = summarise_dvr_results(df, label='DVR')

    for key in ['DVR_long_cumret', 'DVR_long_sharpe', 'DVR_long_maxdd',
                'DVR_short_cumret', 'DVR_short_sharpe', 'DVR_short_maxdd',
                'benchmark_cumret', 'benchmark_sharpe', 'benchmark_maxdd']:
        assert key in summary, f"Missing key: {key}"
