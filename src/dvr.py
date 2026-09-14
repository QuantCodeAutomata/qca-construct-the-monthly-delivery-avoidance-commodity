"""
DVR (Dummy Variable Regression) seasonal trading strategy.

Uses a 120-month rolling window, target-month single-dummy regression,
Newey-West HAC inference (lag=3), Top-1 long/short rules, and transaction costs.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Tuple, Optional, Dict, List

from src.utils import (
    vol_normalise_returns,
    compute_net_long_return,
    compute_net_short_return,
    compute_performance_metrics,
    compute_subperiod_metrics,
)

ESTIMATION_WINDOW = 120   # months
HAC_MAXLAGS = 3           # floor(0.75 * 120^(1/3)) = 3
SIGNIFICANCE_LEVEL = 0.10
TRANSACTION_COST = 0.00086


def _hac_lag(T: int) -> int:
    """Compute Newey-West HAC lag: floor(0.75 * T^(1/3))."""
    return max(1, int(np.floor(0.75 * T ** (1.0 / 3.0))))


def fit_dvr(
    returns: np.ndarray,
    target_month: int,
    months_in_window: np.ndarray,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Fit target-month single-dummy regression with Newey-West HAC.

    Model: r_tau = alpha + beta * D_{m,tau} + epsilon
    where D_{m,tau} = 1 if observation tau is in calendar month m.

    Args:
        returns:          array of T log returns.
        target_month:     calendar month (1-12) for which to fit the dummy.
        months_in_window: array of calendar months (1-12) aligned with returns.

    Returns:
        (beta_hat, p_value, alpha_hat) — all None on failure or insufficient data.
    """
    T = len(returns)
    if T < 15:
        return None, None, None

    D = (months_in_window == target_month).astype(float)
    X = sm.add_constant(D)

    try:
        model = sm.OLS(returns, X)
        res = model.fit()
        hac_lag = _hac_lag(T)
        hac_res = res.get_robustcov_results(
            cov_type='HAC',
            maxlags=hac_lag,
            use_correction=False,
        )
        beta_hat = float(hac_res.params[1])
        p_value = float(hac_res.pvalues[1])
        alpha_hat = float(hac_res.params[0])
        return beta_hat, p_value, alpha_hat
    except Exception:
        return None, None, None


def run_dvr_strategy(
    modeling_panel: pd.DataFrame,
    oos_start: str = '2016-01',
    oos_end: str = '2024-12',
    vol_normalise: bool = False,
    estimation_window: int = ESTIMATION_WINDOW,
    cost: float = TRANSACTION_COST,
    significance: float = SIGNIFICANCE_LEVEL,
) -> pd.DataFrame:
    """Run the DVR Top-1 Long and Short seasonal strategy.

    For each out-of-sample month, the prior ``estimation_window`` months are used
    to fit a single-dummy DVR for each commodity.  The commodity with the largest
    (smallest) significant positive (negative) beta is selected for a long (short)
    trade; cash is held when no significant signal exists.

    Args:
        modeling_panel:     monthly panel DataFrame with columns: commodity,
                            month_label, year, month, simple_return, log_return,
                            selected_flag.
        oos_start:          first OOS month (YYYY-MM).
        oos_end:            last OOS month (YYYY-MM).
        vol_normalise:      if True, apply EWMA vol-normalisation to the returns
                            used in DVR fitting.
        estimation_window:  rolling window size in months (default 120).
        cost:               one-way proportional transaction cost (default 0.00086).
        significance:       Newey-West significance threshold (default 0.10).

    Returns:
        DataFrame with one row per OOS month containing long/short selections,
        beta estimates, p-values, net returns, and the equal-weight benchmark.
    """
    panel = modeling_panel.copy()
    panel['month_dt'] = pd.to_datetime(panel['month_label'])

    all_months = sorted(panel['month_dt'].unique())
    oos_months = [
        m for m in all_months
        if oos_start <= m.strftime('%Y-%m') <= oos_end
    ]

    commodities = sorted(panel['commodity'].unique())

    # Pre-build lookup: month_data[month_dt][commodity] = {log_return, simple_return, ...}
    month_data: Dict[pd.Timestamp, Dict[str, dict]] = {}
    for mt in all_months:
        month_data[mt] = {}
        mt_df = panel[panel['month_dt'] == mt]
        for _, row in mt_df.iterrows():
            avail = bool(row['selected_flag']) and not pd.isna(row['log_return'])
            month_data[mt][row['commodity']] = {
                'log_return': row['log_return'] if avail else np.nan,
                'simple_return': row['simple_return'] if avail else np.nan,
                'available': avail,
                'calendar_month': int(row['month']),
            }

    results = []

    for trade_month in oos_months:
        trade_idx = all_months.index(trade_month)
        if trade_idx < estimation_window:
            continue

        window_months = all_months[trade_idx - estimation_window: trade_idx]
        target_cal_month = trade_month.month

        long_best: Optional[str] = None
        long_best_beta = -np.inf
        short_best: Optional[str] = None
        short_best_beta = np.inf
        long_pvals: Dict[str, float] = {}
        short_pvals: Dict[str, float] = {}

        for commodity in commodities:
            rets: List[float] = []
            cal_months: List[int] = []
            for wm in window_months:
                d = month_data.get(wm, {}).get(commodity)
                if d is not None and d['available']:
                    rets.append(d['log_return'])
                    cal_months.append(d['calendar_month'])

            if len(rets) < 24:
                continue

            rets_arr = np.array(rets)
            months_arr = np.array(cal_months)

            if vol_normalise:
                rets_arr = vol_normalise_returns(rets_arr)

            beta, pval, _ = fit_dvr(rets_arr, target_cal_month, months_arr)
            if beta is None:
                continue

            if beta > 0 and pval <= significance:
                if beta > long_best_beta:
                    long_best_beta = beta
                    long_best = commodity
                long_pvals[commodity] = pval

            if beta < 0 and pval <= significance:
                if beta < short_best_beta:
                    short_best_beta = beta
                    short_best = commodity
                short_pvals[commodity] = pval

        is_long_cash = long_best is None
        is_short_cash = short_best is None
        long_net_ret = 0.0
        short_net_ret = 0.0

        if not is_long_cash:
            d = month_data.get(trade_month, {}).get(long_best, {})
            if d.get('available', False):
                long_net_ret = compute_net_long_return(d['simple_return'], cost)
            else:
                is_long_cash = True
                long_best = None

        if not is_short_cash:
            d = month_data.get(trade_month, {}).get(short_best, {})
            if d.get('available', False):
                short_net_ret = compute_net_short_return(d['simple_return'], cost)
            else:
                is_short_cash = True
                short_best = None

        avail_rets = [
            month_data[trade_month][c]['simple_return']
            for c in commodities
            if c in month_data.get(trade_month, {})
            and month_data[trade_month][c]['available']
        ]
        if avail_rets:
            ew_ret = float(np.mean(avail_rets))
            benchmark_net = (1 + ew_ret) * (1 - cost) ** 2 - 1
        else:
            benchmark_net = 0.0

        results.append({
            'month': trade_month,
            'long_commodity': long_best,
            'long_beta': long_best_beta if not is_long_cash else None,
            'long_pval': long_pvals.get(long_best) if not is_long_cash else None,
            'long_net_return': long_net_ret,
            'short_commodity': short_best,
            'short_beta': short_best_beta if not is_short_cash else None,
            'short_pval': short_pvals.get(short_best) if not is_short_cash else None,
            'short_net_return': short_net_ret,
            'benchmark_net_return': benchmark_net,
            'is_long_cash': is_long_cash,
            'is_short_cash': is_short_cash,
        })

    return pd.DataFrame(results)


def summarise_dvr_results(
    results_df: pd.DataFrame,
    label: str = 'DVR',
) -> dict:
    """Compute a performance summary dict for a DVR result DataFrame."""
    long_rets = results_df['long_net_return'].values
    short_rets = results_df['short_net_return'].values
    bench_rets = results_df['benchmark_net_return'].values

    long_metrics = compute_performance_metrics(long_rets)
    short_metrics = compute_performance_metrics(short_rets)
    bench_metrics = compute_performance_metrics(bench_rets)

    n = len(results_df)
    long_cash_freq = results_df['is_long_cash'].sum() / n if n > 0 else np.nan
    short_cash_freq = results_df['is_short_cash'].sum() / n if n > 0 else np.nan

    return {
        f'{label}_long_cumret': long_metrics['cumret'],
        f'{label}_long_sharpe': long_metrics['sharpe'],
        f'{label}_long_maxdd': long_metrics['maxdd'],
        f'{label}_long_cash_freq': float(long_cash_freq),
        f'{label}_short_cumret': short_metrics['cumret'],
        f'{label}_short_sharpe': short_metrics['sharpe'],
        f'{label}_short_maxdd': short_metrics['maxdd'],
        f'{label}_short_cash_freq': float(short_cash_freq),
        'benchmark_cumret': bench_metrics['cumret'],
        'benchmark_sharpe': bench_metrics['sharpe'],
        'benchmark_maxdd': bench_metrics['maxdd'],
    }
