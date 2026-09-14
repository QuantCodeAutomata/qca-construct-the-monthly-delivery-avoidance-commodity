"""Shared utility functions: EWMA volatility, performance metrics, transaction costs."""
import numpy as np
import pandas as pd
from typing import Tuple


def compute_ewma_variance(
    returns: np.ndarray,
    lam: float = 0.97,
    warm_start_obs: int = 12,
) -> np.ndarray:
    """Compute EWMA standard deviation for volatility normalisation.

    Recursion: sigma^2_t = lam * sigma^2_{t-1} + (1-lam) * r^2_{t-1}
    Warm-started from the mean squared return of the first ``warm_start_obs``
    observations.

    Args:
        returns:        array of T returns.
        lam:            EWMA decay factor (default 0.97).
        warm_start_obs: number of initial observations used for the warm start.

    Returns:
        Array of length T containing EWMA standard deviation estimates.
    """
    T = len(returns)
    sigma2 = np.zeros(T)
    warm = returns[:warm_start_obs]
    sigma2[0] = np.mean(warm ** 2) if len(warm) > 0 else returns[0] ** 2
    for t in range(1, T):
        sigma2[t] = lam * sigma2[t - 1] + (1 - lam) * returns[t - 1] ** 2
    return np.sqrt(np.maximum(sigma2, 1e-12))


def compute_expanding_mean(returns: np.ndarray) -> np.ndarray:
    """Compute expanding mean: rbar_t = mean(r_1, ..., r_t) for each t."""
    cumsum = np.cumsum(returns)
    counts = np.arange(1, len(returns) + 1)
    return cumsum / counts


def vol_normalise_returns(
    returns: np.ndarray,
    lam: float = 0.97,
    warm_start_obs: int = 12,
) -> np.ndarray:
    """Apply EWMA + expanding-mean volatility normalisation.

    tilde_r_t = (r_t - rbar_t) / sigma_t
    where rbar_t is the expanding mean and sigma_t is the EWMA std dev.
    """
    sigma = compute_ewma_variance(returns, lam=lam, warm_start_obs=warm_start_obs)
    rbar = compute_expanding_mean(returns)
    return (returns - rbar) / sigma


def compute_net_long_return(R: float, c: float = 0.00086) -> float:
    """Net return for a long futures position with round-trip transaction cost.

    net = (1 + R)(1 - c)^2 - 1
    """
    return (1 + R) * (1 - c) ** 2 - 1


def compute_net_short_return(R: float, c: float = 0.00086) -> float:
    """Net return for a short futures position with round-trip transaction cost.

    net = (1 - R)(1 - c)^2 - 1
    """
    return (1 - R) * (1 - c) ** 2 - 1


def compute_performance_metrics(
    returns: np.ndarray,
    freq: int = 12,
) -> dict:
    """Compute CumRet, annualised Sharpe (zero risk-free rate), and MaxDrawdown.

    Args:
        returns: array of periodic net returns.
        freq:    annualisation factor (12 for monthly data).

    Returns:
        dict with keys: cumret, sharpe, maxdd.
    """
    wealth = np.cumprod(1 + returns)
    cumret = float(wealth[-1] - 1)
    std = float(np.std(returns, ddof=1))
    ann_sharpe = float(np.mean(returns)) / std * np.sqrt(freq) if std > 0 else 0.0
    running_max = np.maximum.accumulate(wealth)
    drawdowns = wealth / running_max - 1
    maxdd = float(np.min(drawdowns))
    return {'cumret': cumret, 'sharpe': float(ann_sharpe), 'maxdd': maxdd}


def compute_subperiod_metrics(
    returns: pd.Series,
    subperiods: list = None,
) -> pd.DataFrame:
    """Compute performance metrics for a list of subperiods.

    Args:
        returns:    monthly returns Series with DatetimeIndex.
        subperiods: list of (label, start_year, end_year) tuples.
                    Defaults to 2016-2018, 2019-2021, 2022-2024.

    Returns:
        DataFrame indexed by period label with columns cumret, sharpe, maxdd.
    """
    if subperiods is None:
        subperiods = [
            ('2016-2018', 2016, 2018),
            ('2019-2021', 2019, 2021),
            ('2022-2024', 2022, 2024),
        ]
    rows = []
    for label, sy, ey in subperiods:
        mask = (returns.index.year >= sy) & (returns.index.year <= ey)
        sub = returns[mask].values
        if len(sub) == 0:
            rows.append({'period': label, 'cumret': np.nan, 'sharpe': np.nan, 'maxdd': np.nan})
        else:
            m = compute_performance_metrics(sub)
            m['period'] = label
            rows.append(m)
    return pd.DataFrame(rows).set_index('period')
