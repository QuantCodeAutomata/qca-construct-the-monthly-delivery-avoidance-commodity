"""Tests for src/panel.py — delivery-avoidance monthly panel construction."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import pytest

from src.panel import (
    is_contract_eligible,
    compute_monthly_return,
    compute_monthly_avg_volume,
    build_monthly_panel,
    apply_liquidity_screen,
    validate_universe,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_contract_df(
    commodity: str,
    contract_id: str,
    dates: pd.DatetimeIndex,
    expiry: pd.Timestamp,
    delivery_month: str,
    base_open: float = 2500.0,
    base_close: float = 2550.0,
    volume: int = 5000,
) -> pd.DataFrame:
    """Build a minimal contract DataFrame."""
    n = len(dates)
    return pd.DataFrame({
        'commodity': commodity,
        'contract_id': contract_id,
        'date': dates,
        'open_price': np.full(n, base_open),
        'close_price': np.full(n, base_close),
        'volume': np.full(n, volume, dtype=int),
        'expiry_date': expiry,
        'delivery_month': delivery_month,
    })


def _jan2022_eligible_contract() -> pd.DataFrame:
    """
    Contract CC_2022J: delivery April 2022, expiry 2022-03-31.
    Trades Jan 3 → Mar 31 2022.  Eligible for obs_month=2022-01.
    """
    dates = pd.DatetimeIndex(pd.date_range('2022-01-03', '2022-03-31', freq='B'))
    return _make_contract_df(
        commodity='CC',
        contract_id='CC_2022J',
        dates=dates,
        expiry=pd.Timestamp('2022-03-31'),
        delivery_month='2022-04',
    )


# ---------------------------------------------------------------------------
# is_contract_eligible
# ---------------------------------------------------------------------------

def test_contract_eligibility_basic():
    """Contract with expiry 2 months ahead and coverage should be eligible."""
    expiry = pd.Timestamp('2022-03-31')
    contract_dates = pd.DatetimeIndex(pd.date_range('2022-01-03', '2022-03-31', freq='B'))
    month_end = pd.Timestamp('2022-01-31')

    eligible, gap, passes_expiry, has_coverage = is_contract_eligible(
        expiry, 2022, 1, contract_dates, month_end
    )

    assert eligible is True
    assert gap == 2
    assert passes_expiry is True
    assert has_coverage is True


def test_contract_eligibility_expiry_rule():
    """Contract with expiry only 1 month ahead must fail the expiry rule."""
    expiry = pd.Timestamp('2022-02-28')
    contract_dates = pd.DatetimeIndex(pd.date_range('2022-01-03', '2022-02-28', freq='B'))
    month_end = pd.Timestamp('2022-01-31')

    eligible, gap, passes_expiry, has_coverage = is_contract_eligible(
        expiry, 2022, 1, contract_dates, month_end
    )

    assert eligible is False
    assert gap == 1
    assert passes_expiry is False


def test_contract_eligibility_coverage_rule():
    """
    Contract with gap >= 2 but no dates past 14 days after month-end
    must fail the coverage rule.
    """
    expiry = pd.Timestamp('2022-03-31')
    # Only January dates — nothing past Feb 14
    contract_dates = pd.DatetimeIndex(pd.date_range('2022-01-03', '2022-01-31', freq='B'))
    month_end = pd.Timestamp('2022-01-31')

    eligible, gap, passes_expiry, has_coverage = is_contract_eligible(
        expiry, 2022, 1, contract_dates, month_end
    )

    assert eligible is False
    assert passes_expiry is True    # gap = 2
    assert has_coverage is False


def test_contract_eligibility_exactly_14_days_post():
    """A date exactly 14 calendar days after month-end should satisfy coverage."""
    expiry = pd.Timestamp('2022-03-31')
    month_end = pd.Timestamp('2022-01-31')
    # Feb 14 is exactly 14 days after Jan 31
    feb14 = pd.Timestamp('2022-02-14')
    contract_dates = pd.DatetimeIndex(
        pd.date_range('2022-01-03', '2022-01-31', freq='B').tolist() + [feb14]
    )

    _, _, _, has_coverage = is_contract_eligible(
        expiry, 2022, 1, contract_dates, month_end
    )
    assert has_coverage is True


def test_contract_eligibility_large_gap():
    """Contract 6 months ahead should be eligible (gap = 6 >= 2)."""
    expiry = pd.Timestamp('2022-07-29')
    contract_dates = pd.DatetimeIndex(pd.date_range('2022-01-03', '2022-07-29', freq='B'))
    month_end = pd.Timestamp('2022-01-31')

    eligible, gap, _, _ = is_contract_eligible(
        expiry, 2022, 1, contract_dates, month_end
    )

    assert eligible is True
    assert gap == 6


# ---------------------------------------------------------------------------
# compute_monthly_return
# ---------------------------------------------------------------------------

def test_monthly_return_computation():
    """Simple return = last_close / first_open - 1."""
    dates = pd.DatetimeIndex(pd.date_range('2022-01-03', periods=5, freq='B'))
    df = pd.DataFrame({
        'date': dates,
        'open_price': [100.0, 101.0, 102.0, 103.0, 104.0],
        'close_price': [101.0, 102.0, 103.0, 104.0, 110.0],
        'volume': [5000] * 5,
    })

    simple_ret, log_ret, first_open, last_close, _, _ = compute_monthly_return(df)

    expected_simple = 110.0 / 100.0 - 1
    assert simple_ret is not None
    assert abs(simple_ret - expected_simple) < 1e-10
    assert abs(log_ret - np.log(1 + expected_simple)) < 1e-10
    assert abs(first_open - 100.0) < 1e-10
    assert abs(last_close - 110.0) < 1e-10


def test_monthly_return_missing_open():
    """If first open is NaN, all returns should be None."""
    df = pd.DataFrame({
        'date': pd.date_range('2022-01-03', periods=3, freq='B'),
        'open_price': [np.nan, 101.0, 102.0],
        'close_price': [101.0, 102.0, 103.0],
        'volume': [5000] * 3,
    })

    simple_ret, log_ret, first_open, last_close, _, _ = compute_monthly_return(df)

    assert simple_ret is None
    assert log_ret is None


def test_monthly_return_missing_close():
    """If last close is NaN the returns should be None."""
    df = pd.DataFrame({
        'date': pd.date_range('2022-01-03', periods=3, freq='B'),
        'open_price': [100.0, 101.0, 102.0],
        'close_price': [101.0, 102.0, np.nan],
        'volume': [5000] * 3,
    })

    simple_ret, log_ret, first_open, last_close, _, _ = compute_monthly_return(df)

    assert simple_ret is None
    assert log_ret is None


def test_monthly_return_zero_open():
    """Zero first open should return None (avoid division by zero)."""
    df = pd.DataFrame({
        'date': pd.date_range('2022-01-03', periods=2, freq='B'),
        'open_price': [0.0, 100.0],
        'close_price': [100.0, 105.0],
        'volume': [5000, 5000],
    })

    simple_ret, log_ret, _, _, _, _ = compute_monthly_return(df)
    assert simple_ret is None
    assert log_ret is None


def test_monthly_return_empty_df():
    """Empty DataFrame returns all None."""
    df = pd.DataFrame(columns=['date', 'open_price', 'close_price', 'volume'])
    result = compute_monthly_return(df)
    assert all(v is None for v in result)


# ---------------------------------------------------------------------------
# build_monthly_panel (minimal)
# ---------------------------------------------------------------------------

def _build_two_contract_daily_df() -> pd.DataFrame:
    """
    Two contracts for commodity CC:
      - CC_2022H  delivery 2022-03 (expiry 2022-02-28): gap=1 from Jan → NOT eligible
      - CC_2022J  delivery 2022-04 (expiry 2022-03-31): gap=2 from Jan → eligible
    Both contracts have January 2022 trading data.
    """
    jan_dates = pd.DatetimeIndex(pd.date_range('2022-01-03', '2022-01-31', freq='B'))
    feb_dates = pd.DatetimeIndex(pd.date_range('2022-02-01', '2022-02-28', freq='B'))
    mar_dates = pd.DatetimeIndex(pd.date_range('2022-03-01', '2022-03-31', freq='B'))

    df_h = _make_contract_df(
        'CC', 'CC_2022H',
        pd.DatetimeIndex(jan_dates.tolist() + feb_dates.tolist()),
        pd.Timestamp('2022-02-28'), '2022-03',
        base_open=2480.0, base_close=2500.0,
    )
    df_j = _make_contract_df(
        'CC', 'CC_2022J',
        pd.DatetimeIndex(jan_dates.tolist() + feb_dates.tolist() + mar_dates.tolist()),
        pd.Timestamp('2022-03-31'), '2022-04',
        base_open=2510.0, base_close=2560.0,
    )
    return pd.concat([df_h, df_j], ignore_index=True)


def test_build_monthly_panel_minimal():
    """Panel for Jan 2022 selects the nearest eligible contract (gap=2)."""
    daily_df = _build_two_contract_daily_df()
    panel = build_monthly_panel(daily_df, start_year=2022, start_month=1,
                                end_year=2022, end_month=1)

    assert len(panel) == 1
    row = panel.iloc[0]
    assert row['commodity'] == 'CC'
    assert row['year'] == 2022
    assert row['month'] == 1
    assert row['selected_contract'] == 'CC_2022J'
    assert row['expiry_gap_months'] == 2
    assert bool(row['selected_flag']) is True
    # Return should be last_close / first_open - 1 = 2560/2510 - 1
    expected_ret = 2560.0 / 2510.0 - 1
    assert abs(row['simple_return'] - expected_ret) < 1e-8


def test_build_monthly_panel_no_eligible():
    """When no eligible contract exists the row has selected_flag=False."""
    jan_dates = pd.DatetimeIndex(pd.date_range('2022-01-03', '2022-01-31', freq='B'))
    feb_dates = pd.DatetimeIndex(pd.date_range('2022-02-01', '2022-02-28', freq='B'))
    # Only one contract with gap=1 (ineligible)
    df = _make_contract_df(
        'CC', 'CC_2022H',
        pd.DatetimeIndex(jan_dates.tolist() + feb_dates.tolist()),
        pd.Timestamp('2022-02-28'), '2022-03',
    )
    panel = build_monthly_panel(df, start_year=2022, start_month=1,
                                end_year=2022, end_month=1)
    assert len(panel) == 1
    assert bool(panel.iloc[0]['selected_flag']) is False
    assert pd.isna(panel.iloc[0]['simple_return'])


# ---------------------------------------------------------------------------
# apply_liquidity_screen
# ---------------------------------------------------------------------------

def _build_liquidity_panel(vol_cc: float, vol_pd: float) -> pd.DataFrame:
    """
    Build a synthetic panel for years 2006-2015 with two commodities CC and PD.
    CC gets monthly_avg_volume = vol_cc, PD gets vol_pd (for all months).
    """
    months = pd.date_range('2006-01', '2015-12', freq='ME')
    rows = []
    for dt in months:
        for comm, vol in [('CC', vol_cc), ('PD', vol_pd)]:
            rows.append({
                'commodity': comm,
                'year': dt.year,
                'month': dt.month,
                'month_label': dt.strftime('%Y-%m'),
                'monthly_avg_volume': vol,
                'selected_flag': True,
                'simple_return': 0.01,
                'log_return': 0.01,
            })
    return pd.DataFrame(rows)


def test_liquidity_screen_threshold():
    """CC (vol=5000) passes; PD (vol=500) fails the min-1000 screen."""
    panel = _build_liquidity_panel(vol_cc=5000.0, vol_pd=500.0)
    retained, report = apply_liquidity_screen(panel, min_volume=1000.0)

    assert 'CC' in retained
    assert 'PD' not in retained

    cc_row = report[report['commodity'] == 'CC'].iloc[0]
    pd_row = report[report['commodity'] == 'PD'].iloc[0]
    assert bool(cc_row['passes_screen']) is True
    assert bool(pd_row['passes_screen']) is False


def test_liquidity_screen_exact_threshold():
    """Commodity with volume exactly at threshold is retained."""
    panel = _build_liquidity_panel(vol_cc=1000.0, vol_pd=999.0)
    retained, _ = apply_liquidity_screen(panel, min_volume=1000.0)
    assert 'CC' in retained
    assert 'PD' not in retained


def test_liquidity_screen_empty_volume():
    """Commodity with all-NaN volume in screen window fails the screen."""
    months = pd.date_range('2006-01', '2015-12', freq='ME')
    rows = [{'commodity': 'CC', 'year': dt.year, 'month': dt.month,
              'month_label': dt.strftime('%Y-%m'), 'monthly_avg_volume': np.nan,
              'selected_flag': False, 'simple_return': None, 'log_return': None}
             for dt in months]
    panel = pd.DataFrame(rows)
    retained, report = apply_liquidity_screen(panel)
    assert 'CC' not in retained
    cc_row = report[report['commodity'] == 'CC'].iloc[0]
    assert cc_row['min_monthly_avg_volume_2006_2015'] == 0.0


# ---------------------------------------------------------------------------
# validate_universe
# ---------------------------------------------------------------------------

def test_validate_universe_exact_match():
    """Retained == expected produces match=True with no missing/extra."""
    expected = ['CC', 'CF', 'CO', 'CP', 'CT', 'GD', 'HE', 'HO', 'LE',
                'NG', 'SU', 'SV', 'ZC', 'ZS', 'ZW']
    result = validate_universe(expected, expected)

    assert result['match'] is True
    assert result['missing_from_retained'] == []
    assert result['extra_in_retained'] == []
    assert result['n_retained'] == 15


def test_validate_universe_mismatch():
    """Missing and extra commodities are correctly reported."""
    expected = ['CC', 'CF', 'ZW']
    retained = ['CC', 'CF', 'NG']   # missing ZW, extra NG

    result = validate_universe(retained, expected)

    assert result['match'] is False
    assert 'ZW' in result['missing_from_retained']
    assert 'NG' in result['extra_in_retained']


def test_validate_universe_empty_retained():
    """All-missing retained universe is handled gracefully."""
    expected = ['CC', 'CF']
    result = validate_universe([], expected)

    assert result['match'] is False
    assert set(result['missing_from_retained']) == {'CC', 'CF'}
    assert result['extra_in_retained'] == []


# ---------------------------------------------------------------------------
# No imputation
# ---------------------------------------------------------------------------

def test_panel_no_imputation():
    """
    Months where no eligible contract exists must not be imputed:
    simple_return must remain missing (NaN/None), not forward- or back-filled.
    """
    # Build a panel that has a gap: only March–May 2022 has an eligible contract.
    # January 2022 has no eligible contract (contract has no Jan data).
    mar_dates = pd.DatetimeIndex(pd.date_range('2022-03-01', '2022-05-31', freq='B'))
    df = _make_contract_df(
        'CC', 'CC_2022M',
        mar_dates,
        pd.Timestamp('2022-05-31'), '2022-06',
        base_open=2510.0, base_close=2560.0,
    )
    panel = build_monthly_panel(df, start_year=2022, start_month=1,
                                end_year=2022, end_month=3)

    # January 2022: no contract data in Jan → no eligible contract
    jan_row = panel[(panel['year'] == 2022) & (panel['month'] == 1)]
    assert len(jan_row) == 1
    assert pd.isna(jan_row.iloc[0]['simple_return']), "Jan return must be missing (not imputed)"
    assert bool(jan_row.iloc[0]['selected_flag']) is False

    # March 2022: eligible (expiry May 2022, gap = 2 from Mar 2022)
    mar_row = panel[(panel['year'] == 2022) & (panel['month'] == 3)]
    assert len(mar_row) == 1
    assert bool(mar_row.iloc[0]['selected_flag']) is True
    assert not pd.isna(mar_row.iloc[0]['simple_return'])
