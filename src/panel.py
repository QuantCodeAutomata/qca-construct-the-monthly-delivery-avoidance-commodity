"""
Delivery-avoidance monthly commodity futures panel construction.

Implements the paper's non-standard contract selection rule:
  - A contract is eligible only if it expires >= 2 calendar months after
    the observation month.
  - It must have price coverage through at least 14 calendar days after
    month-end.
  - Among eligible contracts, select the nearest eligible expiry.
"""
import numpy as np
import pandas as pd
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from typing import Optional, Tuple, Dict, List
import warnings


def get_business_days_in_month(
    year: int,
    month: int,
    daily_df: pd.DataFrame,
) -> pd.DatetimeIndex:
    """Get all trading dates present in the data for a given year-month."""
    mask = (daily_df['date'].dt.year == year) & (daily_df['date'].dt.month == month)
    dates = daily_df.loc[mask, 'date'].unique()
    return pd.DatetimeIndex(sorted(dates))


def is_contract_eligible(
    expiry_date: pd.Timestamp,
    obs_year: int,
    obs_month: int,
    contract_dates: pd.DatetimeIndex,
    month_end: pd.Timestamp,
) -> Tuple[bool, int, bool, bool]:
    """Check if a contract is eligible for a given observation month.

    Eligibility criteria:
        (a) Contract expires in a month >= obs_month + 2 calendar months.
        (b) Contract has at least one trading date >= month_end + 14 days.

    Args:
        expiry_date:     last trading date of the contract.
        obs_year:        calendar year of the observation month.
        obs_month:       calendar month (1-12) of the observation month.
        contract_dates:  all trading dates for this contract (full history).
        month_end:       last calendar day of the observation month.

    Returns:
        Tuple of (eligible, expiry_gap_months, passes_expiry_rule, has_post_14d_coverage).
    """
    obs_start = pd.Timestamp(year=obs_year, month=obs_month, day=1)
    expiry_month_start = pd.Timestamp(year=expiry_date.year, month=expiry_date.month, day=1)
    diff = (
        (expiry_month_start.year - obs_start.year) * 12
        + (expiry_month_start.month - obs_start.month)
    )
    passes_expiry_rule = diff >= 2

    coverage_threshold = month_end + timedelta(days=14)
    has_post_14d_coverage = any(d >= coverage_threshold for d in contract_dates)

    eligible = passes_expiry_rule and has_post_14d_coverage
    return eligible, diff, passes_expiry_rule, has_post_14d_coverage


def compute_monthly_return(
    contract_month_data: pd.DataFrame,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[str], Optional[str]]:
    """Compute monthly simple and log return for a selected contract-month.

    Return is computed as: last_close / first_open - 1.

    Returns:
        (simple_return, log_return, first_open, last_close, first_date_str, last_date_str)
        All values are None if data is invalid.
    """
    df = contract_month_data.sort_values('date')
    if df.empty:
        return None, None, None, None, None, None

    first_open = df.iloc[0]['open_price']
    last_close = df.iloc[-1]['close_price']
    first_date = str(df.iloc[0]['date'].date())
    last_date = str(df.iloc[-1]['date'].date())

    if pd.isna(first_open) or pd.isna(last_close) or first_open == 0:
        return None, None, None, None, first_date, last_date

    simple_ret = last_close / first_open - 1
    log_ret = np.log(1 + simple_ret)
    return float(simple_ret), float(log_ret), float(first_open), float(last_close), first_date, last_date


def compute_monthly_avg_volume(contract_month_data: pd.DataFrame) -> Optional[float]:
    """Compute average daily trading volume for a contract within one month."""
    if contract_month_data.empty:
        return None
    vol = contract_month_data['volume'].mean()
    return float(vol) if not pd.isna(vol) else None


def build_monthly_panel(
    daily_df: pd.DataFrame,
    start_year: int = 2001,
    start_month: int = 1,
    end_year: int = 2024,
    end_month: int = 12,
) -> pd.DataFrame:
    """Build the monthly delivery-avoidance commodity futures panel.

    For each commodity and each calendar month, applies the paper's contract
    selection rule to produce a representative monthly return series.

    Args:
        daily_df:    DataFrame with columns: commodity, contract_id, date,
                     open_price, close_price, volume, expiry_date, delivery_month.
        start_year, start_month: inclusive start of the panel period.
        end_year, end_month:     inclusive end of the panel period.

    Returns:
        DataFrame with one row per (commodity, month) containing returns,
        volumes, and diagnostic columns.
    """
    daily_df = daily_df.copy()
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    daily_df['expiry_date'] = pd.to_datetime(daily_df['expiry_date'])

    commodities = sorted(daily_df['commodity'].unique())

    # Generate list of (year, month) tuples for the panel period
    periods: List[Tuple[int, int]] = []
    y, m = start_year, start_month
    while (y < end_year) or (y == end_year and m <= end_month):
        periods.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1

    records = []

    for commodity in commodities:
        comm_df = daily_df[daily_df['commodity'] == commodity].copy()
        comm_df = comm_df.sort_values('date')
        contracts = comm_df['contract_id'].unique()

        # Precompute monthly aggregates: (contract_id, year, month) → (first_open, last_close, avg_vol)
        comm_df['_year'] = comm_df['date'].dt.year
        comm_df['_month'] = comm_df['date'].dt.month
        grp = comm_df.groupby(['contract_id', '_year', '_month'])
        _first_open = grp['open_price'].first()
        _last_close = grp['close_price'].last()
        _avg_vol = grp['volume'].mean()
        # Convert to dicts for O(1) lookup
        first_open_dict = _first_open.to_dict()
        last_close_dict = _last_close.to_dict()
        avg_vol_dict = _avg_vol.to_dict()

        # Pre-build per-contract metadata (O(1) eligibility check at query time)
        contract_meta: Dict[str, dict] = {}
        for cid in contracts:
            c_df = comm_df[comm_df['contract_id'] == cid]
            expiry = c_df['expiry_date'].iloc[0]
            all_dates = pd.DatetimeIndex(sorted(c_df['date'].unique()))
            # Precompute max date for O(1) coverage check
            max_date = all_dates.max()
            # Precompute set of (year, month) tuples for O(1) in-month check
            months_with_data = set(zip(all_dates.year, all_dates.month))
            # Expiry month start for O(1) gap computation
            exp_month_start = pd.Timestamp(
                year=expiry.year, month=expiry.month, day=1
            )
            contract_meta[cid] = {
                'expiry': expiry,
                'dates': all_dates,
                'max_date': max_date,
                'months_with_data': months_with_data,
                'exp_month_start': exp_month_start,
            }

        for year, month in periods:
            month_end = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
            obs_start = pd.Timestamp(year=year, month=month, day=1)
            coverage_threshold = month_end + timedelta(days=14)

            eligible_contracts = []
            for cid, meta in contract_meta.items():
                # O(1): check if this contract has any data in this calendar month
                if (year, month) not in meta['months_with_data']:
                    continue

                # O(1): compute expiry gap in calendar months
                exp_month_start = meta['exp_month_start']
                diff = (
                    (exp_month_start.year - obs_start.year) * 12
                    + (exp_month_start.month - obs_start.month)
                )
                if diff < 2:
                    continue

                # O(1): coverage check using precomputed max_date
                if meta['max_date'] < coverage_threshold:
                    continue

                eligible_contracts.append({
                    'contract_id': cid,
                    'expiry': meta['expiry'],
                    'expiry_gap_months': diff,
                    'has_post_14d_coverage': True,
                })

            rec = {
                'commodity': commodity,
                'year': year,
                'month': month,
                'month_label': f'{year}-{month:02d}',
                'selected_contract': None,
                'expiry': None,
                'expiry_gap_months': None,
                'simple_return': None,
                'log_return': None,
                'monthly_avg_volume': None,
                'first_open': None,
                'last_close': None,
                'first_date': None,
                'last_date': None,
                'has_first_open': False,
                'has_last_close': False,
                'has_post_14d_coverage': False,
                'selected_flag': False,
                'n_eligible_contracts': len(eligible_contracts),
                'no_eligible_reason': None,
            }

            if not eligible_contracts:
                rec['no_eligible_reason'] = 'no_eligible_contract'
                records.append(rec)
                continue

            # Select the nearest eligible expiry (smallest gap >= 2)
            eligible_contracts.sort(key=lambda x: x['expiry_gap_months'])
            selected = eligible_contracts[0]
            cid = selected['contract_id']

            # O(1) lookup for precomputed aggregates
            key = (cid, year, month)
            first_open = first_open_dict.get(key)
            last_close = last_close_dict.get(key)
            avg_vol = avg_vol_dict.get(key)

            if (
                first_open is None or last_close is None
                or (isinstance(first_open, float) and np.isnan(first_open))
                or (isinstance(last_close, float) and np.isnan(last_close))
                or first_open == 0
            ):
                simple_ret = None
                log_ret = None
                first_open = None
                last_close = None
            else:
                simple_ret = float(last_close) / float(first_open) - 1
                log_ret = np.log(1 + simple_ret)
                first_open = float(first_open)
                last_close = float(last_close)

            first_date = None
            last_date = None

            rec.update({
                'selected_contract': cid,
                'expiry': selected['expiry'],
                'expiry_gap_months': selected['expiry_gap_months'],
                'simple_return': simple_ret,
                'log_return': log_ret,
                'monthly_avg_volume': avg_vol,
                'first_open': first_open,
                'last_close': last_close,
                'first_date': first_date,
                'last_date': last_date,
                'has_first_open': first_open is not None,
                'has_last_close': last_close is not None,
                'has_post_14d_coverage': selected['has_post_14d_coverage'],
                'selected_flag': simple_ret is not None,
                'n_eligible_contracts': len(eligible_contracts),
            })
            records.append(rec)

    panel = pd.DataFrame(records)
    panel['month_dt'] = pd.to_datetime(panel['month_label'])
    return panel


def apply_liquidity_screen(
    panel: pd.DataFrame,
    screen_start_year: int = 2006,
    screen_end_year: int = 2015,
    min_volume: float = 1000.0,
) -> Tuple[List[str], pd.DataFrame]:
    """Apply the pre-evaluation liquidity screen.

    Retains a commodity only if its minimum monthly average daily volume
    across all months in [screen_start_year, screen_end_year] is >= min_volume.

    Args:
        panel:             monthly panel DataFrame.
        screen_start_year: first year of the screen window (inclusive).
        screen_end_year:   last year of the screen window (inclusive).
        min_volume:        minimum monthly average daily volume threshold.

    Returns:
        (retained_commodities, liquidity_report_df)
    """
    screen_mask = (
        (panel['year'] >= screen_start_year)
        & (panel['year'] <= screen_end_year)
    )
    screen_panel = panel[screen_mask].copy()

    report_rows = []
    retained = []

    for commodity in sorted(panel['commodity'].unique()):
        c_df = screen_panel[screen_panel['commodity'] == commodity]
        vols = c_df['monthly_avg_volume'].dropna()

        min_vol = float(vols.min()) if len(vols) > 0 else 0.0
        n_valid = len(vols)
        passes = min_vol >= min_volume

        if passes:
            retained.append(commodity)

        report_rows.append({
            'commodity': commodity,
            'min_monthly_avg_volume_2006_2015': min_vol,
            'n_valid_months_in_screen': n_valid,
            'passes_screen': passes,
        })

    report_df = pd.DataFrame(report_rows)
    return retained, report_df


def validate_universe(
    retained: List[str],
    expected: List[str] = None,
) -> dict:
    """Validate the retained universe against the paper's expected 15 commodities.

    Returns a dict summarising matches, missing, and extra commodities.
    """
    if expected is None:
        expected = ['CC', 'CF', 'CO', 'CP', 'CT', 'GD', 'HE', 'HO', 'LE',
                    'NG', 'SU', 'SV', 'ZC', 'ZS', 'ZW']
    retained_set = set(retained)
    expected_set = set(expected)
    return {
        'retained': sorted(retained),
        'expected': sorted(expected),
        'match': retained_set == expected_set,
        'missing_from_retained': sorted(expected_set - retained_set),
        'extra_in_retained': sorted(retained_set - expected_set),
        'n_retained': len(retained),
        'n_expected': len(expected),
    }


def build_modeling_panel(
    panel: pd.DataFrame,
    retained_commodities: List[str],
    model_start_year: int = 2006,
    model_end_year: int = 2024,
) -> pd.DataFrame:
    """Filter the panel to the modeling period and retained commodity universe."""
    mask = (
        panel['commodity'].isin(retained_commodities)
        & (panel['year'] >= model_start_year)
        & (panel['year'] <= model_end_year)
    )
    return panel[mask].copy()
