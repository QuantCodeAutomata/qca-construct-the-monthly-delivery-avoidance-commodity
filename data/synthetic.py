"""
Synthetic commodity futures data generator.

Produces realistic daily contract-level data for 17 commodities
(15 retained + Palladium + Platinum) over 2001-01-01 to 2025-01-31.
"""
import numpy as np
import pandas as pd
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta

# Standard CME/ICE futures month codes
MONTH_CODES = {
    1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
    7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z',
}

# Commodity definitions:
#   base_price        : starting price level
#   daily_vol_pct     : daily log-return volatility (fraction)
#   seasonal_amp      : amplitude of seasonal price component (fraction of base price)
#   seasonal_peak_month: calendar month (1-12) where price peaks
#   base_daily_volume : mean daily volume for the nearby contract
#   delivery_months   : list of calendar months that have listed contracts (None = all 12)
#   is_retained       : True → volume kept well above 1000; False → kept below 1000
COMMODITY_SPECS = {
    'CC': dict(base_price=2500.0, daily_vol_pct=0.014, seasonal_amp=0.08,
               seasonal_peak_month=7,  base_daily_volume=6000,  delivery_months=None, is_retained=True),
    'CF': dict(base_price=135.0,  daily_vol_pct=0.016, seasonal_amp=0.10,
               seasonal_peak_month=2,  base_daily_volume=5500,  delivery_months=None, is_retained=True),
    'CO': dict(base_price=55.0,   daily_vol_pct=0.020, seasonal_amp=0.06,
               seasonal_peak_month=1,  base_daily_volume=18000, delivery_months=None, is_retained=True),
    'CP': dict(base_price=3.2,    daily_vol_pct=0.014, seasonal_amp=0.05,
               seasonal_peak_month=4,  base_daily_volume=9000,  delivery_months=[3,5,7,9,12], is_retained=True),
    'CT': dict(base_price=72.0,   daily_vol_pct=0.013, seasonal_amp=0.09,
               seasonal_peak_month=5,  base_daily_volume=5000,  delivery_months=[3,5,7,10,12], is_retained=True),
    'GD': dict(base_price=450.0,  daily_vol_pct=0.011, seasonal_amp=0.04,
               seasonal_peak_month=9,  base_daily_volume=12000, delivery_months=None, is_retained=True),
    'HE': dict(base_price=68.0,   daily_vol_pct=0.016, seasonal_amp=0.12,
               seasonal_peak_month=5,  base_daily_volume=4000,  delivery_months=[2,4,5,6,7,8,10,12], is_retained=True),
    'HO': dict(base_price=1.8,    daily_vol_pct=0.019, seasonal_amp=0.08,
               seasonal_peak_month=1,  base_daily_volume=7000,  delivery_months=None, is_retained=True),
    'LE': dict(base_price=82.0,   daily_vol_pct=0.012, seasonal_amp=0.07,
               seasonal_peak_month=4,  base_daily_volume=3500,  delivery_months=[2,4,6,8,10,12], is_retained=True),
    'NG': dict(base_price=5.5,    daily_vol_pct=0.025, seasonal_amp=0.15,
               seasonal_peak_month=1,  base_daily_volume=11000, delivery_months=None, is_retained=True),
    'SU': dict(base_price=12.0,   daily_vol_pct=0.016, seasonal_amp=0.10,
               seasonal_peak_month=3,  base_daily_volume=7000,  delivery_months=[3,5,7,10], is_retained=True),
    'SV': dict(base_price=5.5,    daily_vol_pct=0.018, seasonal_amp=0.05,
               seasonal_peak_month=9,  base_daily_volume=8000,  delivery_months=None, is_retained=True),
    'ZC': dict(base_price=250.0,  daily_vol_pct=0.015, seasonal_amp=0.08,
               seasonal_peak_month=6,  base_daily_volume=13000, delivery_months=[3,5,7,9,12], is_retained=True),
    'ZS': dict(base_price=620.0,  daily_vol_pct=0.014, seasonal_amp=0.09,
               seasonal_peak_month=7,  base_daily_volume=10000, delivery_months=[1,3,5,7,8,9,11], is_retained=True),
    'ZW': dict(base_price=320.0,  daily_vol_pct=0.016, seasonal_amp=0.10,
               seasonal_peak_month=5,  base_daily_volume=8000,  delivery_months=[3,5,7,9,12], is_retained=True),
    # Excluded — low liquidity
    'PD': dict(base_price=380.0,  daily_vol_pct=0.014, seasonal_amp=0.04,
               seasonal_peak_month=4,  base_daily_volume=400,   delivery_months=[3,6,9,12], is_retained=False),
    'PT': dict(base_price=1100.0, daily_vol_pct=0.013, seasonal_amp=0.04,
               seasonal_peak_month=4,  base_daily_volume=300,   delivery_months=[1,4,7,10], is_retained=False),
}


def _last_business_day(year: int, month: int, biz_dates_set: set) -> pd.Timestamp:
    """Return the last trading day present in biz_dates_set for the given year-month."""
    # Walk backwards from the last calendar day of the month
    last_cal = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    for delta in range(31):
        candidate = last_cal - pd.Timedelta(days=delta)
        if candidate in biz_dates_set:
            return candidate
    # Fallback: last Friday on or before last_cal
    d = last_cal
    while d.weekday() >= 5:
        d -= pd.Timedelta(days=1)
    return d


def _generate_business_dates(start: str, end: str) -> pd.DatetimeIndex:
    """Generate all weekday dates between start and end (proxy for trading days)."""
    all_days = pd.date_range(start=start, end=end, freq='B')
    return all_days


def _price_path(
    n_days: int,
    start_price: float,
    daily_vol: float,
    drift: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a geometric Brownian motion price path of length n_days."""
    log_returns = rng.normal(drift, daily_vol, size=n_days)
    log_prices = np.log(start_price) + np.cumsum(log_returns)
    return np.exp(log_prices)


def _seasonal_price_adjustment(
    dates: pd.DatetimeIndex,
    base_price: float,
    amp: float,
    peak_month: int,
) -> np.ndarray:
    """Sinusoidal seasonal component centred on peak_month."""
    months = dates.month.values
    radians = 2 * np.pi * (months - peak_month) / 12.0
    return amp * base_price * np.cos(radians)


def _volume_profile(
    days_to_expiry: np.ndarray,
    base_volume: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Volume profile as a function of days to expiry.

    Peaks ~60 days before expiry, then declines sharply in the last 30 days
    as traders roll to the next contract.
    """
    dte = np.maximum(days_to_expiry, 1)
    # Bell-shaped peak around 60 DTE
    peak_dte = 60.0
    sigma_dte = 80.0
    shape = np.exp(-0.5 * ((dte - peak_dte) / sigma_dte) ** 2)
    # Suppress volume in last 15 DTE (roll period)
    roll_suppression = np.where(dte < 15, 0.3, 1.0)
    mean_vol = base_volume * shape * roll_suppression
    # Add log-normal noise
    noise = rng.lognormal(mean=0.0, sigma=0.25, size=len(dte))
    raw = mean_vol * noise
    return np.maximum(raw, 1.0)


def generate_synthetic_daily_data(
    start_date: str = '2001-01-01',
    end_date: str = '2025-01-31',
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic daily contract-level futures data.

    Args:
        start_date: first date to include in the output (YYYY-MM-DD).
        end_date:   last date to include in the output (YYYY-MM-DD).
        seed:       numpy random seed for reproducibility.

    Returns:
        DataFrame with columns:
            commodity, contract_id, date, open_price, close_price,
            volume, expiry_date, delivery_month
    """
    rng = np.random.default_rng(seed)
    biz_dates = _generate_business_dates(start_date, end_date)
    biz_dates_set = set(biz_dates)

    # We need contracts that overlap the requested window.  To give a 120-month
    # rolling window starting in 2016 we must have data from 2001 onward, so
    # generate contracts whose delivery month spans at least 1998-01 to 2025-06.
    gen_start = pd.Timestamp('1999-01-01')
    gen_end = pd.Timestamp('2025-06-30')

    records = []

    for commodity, spec in COMMODITY_SPECS.items():
        base_price = spec['base_price']
        daily_vol = spec['daily_vol_pct']
        amp = spec['seasonal_amp']
        peak_month = spec['seasonal_peak_month']
        base_vol = spec['base_daily_volume']
        delivery_months = spec['delivery_months'] if spec['delivery_months'] else list(range(1, 13))
        is_retained = spec['is_retained']

        # Override volume for non-retained commodities so they stay below 1000
        if not is_retained:
            base_vol = min(base_vol, 450)

        # Enumerate delivery months
        dm_cursor = pd.Timestamp(gen_start.year, gen_start.month, 1)
        dm_end_ts = gen_end

        # Track a drifting "true price level" across contracts for realism
        current_log_price = np.log(base_price)
        # Long-run commodity drift (annualised ~2 %)
        annual_drift = 0.02
        daily_drift = annual_drift / 252

        while dm_cursor <= dm_end_ts:
            if dm_cursor.month not in delivery_months:
                dm_cursor += relativedelta(months=1)
                continue

            delivery_yr = dm_cursor.year
            delivery_mo = dm_cursor.month
            delivery_month_str = f'{delivery_yr}-{delivery_mo:02d}'
            month_code = MONTH_CODES[delivery_mo]
            contract_id = f'{commodity}_{delivery_yr}{month_code}'

            # Expiry = last biz day of the month BEFORE delivery
            expiry_ts = dm_cursor - relativedelta(months=1)
            expiry_date = _last_business_day(expiry_ts.year, expiry_ts.month, biz_dates_set)

            # Contract starts trading ~15 months before expiry
            contract_start_approx = expiry_date - relativedelta(months=15)
            contract_start = max(contract_start_approx, pd.Timestamp(start_date) - pd.Timedelta(days=365))

            # Filter to business days in [contract_start, expiry_date] that fall in
            # the output window [start_date, end_date]
            out_start = pd.Timestamp(start_date)
            out_end = pd.Timestamp(end_date)

            # Get biz dates for this contract
            c_dates = biz_dates[
                (biz_dates >= max(contract_start, out_start)) &
                (biz_dates <= min(expiry_date, out_end))
            ]
            if len(c_dates) == 0:
                dm_cursor += relativedelta(months=1)
                continue

            n = len(c_dates)

            # Starting price: use the last known level with seasonal adjustment
            seasonal_at_start = amp * base_price * np.cos(
                2 * np.pi * (c_dates[0].month - peak_month) / 12.0
            )
            start_lvl = np.exp(current_log_price) + seasonal_at_start
            start_lvl = max(start_lvl, base_price * 0.1)

            # Generate close prices
            closes = _price_path(n, start_lvl, daily_vol, daily_drift, rng)

            # Add seasonal component
            seasonal = _seasonal_price_adjustment(c_dates, base_price, amp, peak_month)
            closes = np.maximum(closes + seasonal * 0.3, base_price * 0.05)

            # Open price = prior close + small gap
            open_gaps = rng.normal(0.0, daily_vol * 0.3, size=n)
            opens = np.empty(n)
            opens[0] = start_lvl * (1 + open_gaps[0])
            opens[1:] = closes[:-1] * (1 + open_gaps[1:])
            opens = np.maximum(opens, base_price * 0.05)

            # Volume profile
            days_to_exp = np.array([(expiry_date - d).days for d in c_dates], dtype=float)
            raw_vols = _volume_profile(days_to_exp, base_vol, rng)

            # For non-retained: ensure volume stays below 1000 in 2006-2015
            if not is_retained:
                screen_mask = (c_dates.year >= 2006) & (c_dates.year <= 2015)
                raw_vols[screen_mask] = np.minimum(raw_vols[screen_mask], 950.0)
                raw_vols[~screen_mask] = np.minimum(raw_vols[~screen_mask], 950.0)

            volumes = np.round(raw_vols).astype(int)

            # Update log price for next contract (continuity)
            current_log_price = np.log(max(closes[-1], base_price * 0.05))

            for i, d in enumerate(c_dates):
                records.append({
                    'commodity': commodity,
                    'contract_id': contract_id,
                    'date': d,
                    'open_price': round(float(opens[i]), 4),
                    'close_price': round(float(closes[i]), 4),
                    'volume': int(volumes[i]),
                    'expiry_date': expiry_date,
                    'delivery_month': delivery_month_str,
                })

            dm_cursor += relativedelta(months=1)

    df = pd.DataFrame(records)
    df = df.sort_values(['commodity', 'contract_id', 'date']).reset_index(drop=True)
    return df


def get_commodity_universe() -> list:
    """Return list of all candidate commodity codes."""
    return ['CC', 'CF', 'CO', 'CP', 'CT', 'GD', 'HE', 'HO', 'LE', 'NG',
            'SU', 'SV', 'ZC', 'ZS', 'ZW', 'PD', 'PT']


def get_retained_universe() -> list:
    """Return the paper's 15 retained commodity codes."""
    return ['CC', 'CF', 'CO', 'CP', 'CT', 'GD', 'HE', 'HO', 'LE', 'NG',
            'SU', 'SV', 'ZC', 'ZS', 'ZW']
