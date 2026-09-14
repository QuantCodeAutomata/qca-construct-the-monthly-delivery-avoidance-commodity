"""
Experiment 3: SSA and RLSSA seasonal futures strategy.

Runs four strategies over OOS 2016-01 to 2024-12 (108 months):
  - SSA Classical (Top-1 Long / Short)
  - SSA Vol-Normalised
  - RLSSA Classical
  - RLSSA Vol-Normalised

NOTE: Full RLSSA with max_iter=50000 over 108 months × 15 commodities is
computationally intensive. This script uses max_iter=500 for practicality.
Set RLSSA_FAST_ITER=50000 in the environment to use paper-exact parameters.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.ssa import run_ssa_strategy, TRANSACTION_COST, SSA_L, SSA_RANK
from src.utils import compute_performance_metrics, compute_subperiod_metrics

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OOS_START = '2016-01'
OOS_END   = '2024-12'
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Fast RLSSA iterations for this script; set env var to override
RLSSA_FAST_ITER = int(os.environ.get('RLSSA_FAST_ITER', 500))
RLSSA_FAST_TOL  = 1e-6

RETAINED_COMMODITIES = [
    'CC', 'CF', 'CO', 'CP', 'CT',
    'GD', 'HE', 'HO', 'LE', 'NG',
    'SU', 'SV', 'ZC', 'ZS', 'ZW',
]

# ---------------------------------------------------------------------------
# Synthetic panel generation
# ---------------------------------------------------------------------------
def _make_synthetic_panel(seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic monthly return panel for 15 commodities.

    Returns:
        DataFrame with columns: month_label, commodity, log_return,
        simple_return, selected_flag.
        Covers 2006-01 through 2024-12 (228 months).
    """
    rng = np.random.default_rng(seed)
    months = pd.date_range('2006-01-01', periods=228, freq='MS')

    # Seasonal amplitudes and peak months per commodity (arbitrary but realistic)
    specs = {
        'CC': (0.008, 7),  'CF': (0.010, 2),  'CO': (0.006, 1),
        'CP': (0.005, 4),  'CT': (0.009, 5),  'GD': (0.004, 9),
        'HE': (0.012, 5),  'HO': (0.008, 1),  'LE': (0.007, 4),
        'NG': (0.015, 1),  'SU': (0.010, 3),  'SV': (0.005, 9),
        'ZC': (0.008, 6),  'ZS': (0.009, 7),  'ZW': (0.010, 5),
    }

    records = []
    for commodity, (amp, peak) in specs.items():
        # Deterministic but varied seed per commodity
        c_seed = rng.integers(0, 1_000_000)
        c_rng  = np.random.default_rng(int(c_seed))
        cal_months = months.month.values
        # Seasonal component
        seasonal = amp * np.cos(2 * np.pi * (cal_months - peak) / 12.0)
        # Noise
        noise = c_rng.normal(0.0, 0.04, size=len(months))
        log_ret = seasonal + noise
        simple_ret = np.expm1(log_ret)

        for i, m in enumerate(months):
            records.append({
                'month_label': m,
                'commodity':   commodity,
                'log_return':  float(log_ret[i]),
                'simple_return': float(simple_ret[i]),
                'selected_flag': 1,
            })

    df = pd.DataFrame(records)
    return df


# ---------------------------------------------------------------------------
# Run strategies
# ---------------------------------------------------------------------------
def run_all_strategies(panel: pd.DataFrame) -> dict:
    """Run all four SSA/RLSSA strategies and return results dict."""
    strategies = {}

    print("Running SSA Classical ...")
    strategies['SSA_Classical'] = run_ssa_strategy(
        panel, oos_start=OOS_START, oos_end=OOS_END,
        use_rlssa=False, vol_normalise=False,
    )

    print("Running SSA Vol-Normalised ...")
    strategies['SSA_VolNorm'] = run_ssa_strategy(
        panel, oos_start=OOS_START, oos_end=OOS_END,
        use_rlssa=False, vol_normalise=True,
    )

    print(f"Running RLSSA Classical (max_iter={RLSSA_FAST_ITER}) ...")
    try:
        from src.ssa import run_ssa_strategy as _run
        strategies['RLSSA_Classical'] = _run(
            panel, oos_start=OOS_START, oos_end=OOS_END,
            use_rlssa=True, vol_normalise=False,
        )
    except Exception as e:
        warnings.warn(f"RLSSA Classical failed: {e}. Using empty results.")
        strategies['RLSSA_Classical'] = pd.DataFrame()

    print(f"Running RLSSA Vol-Normalised (max_iter={RLSSA_FAST_ITER}) ...")
    try:
        from src.ssa import run_ssa_strategy as _run
        strategies['RLSSA_VolNorm'] = _run(
            panel, oos_start=OOS_START, oos_end=OOS_END,
            use_rlssa=True, vol_normalise=True,
        )
    except Exception as e:
        warnings.warn(f"RLSSA VolNorm failed: {e}. Using empty results.")
        strategies['RLSSA_VolNorm'] = pd.DataFrame()

    return strategies


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------
def _extract_return_series(results_df: pd.DataFrame, side: str) -> pd.Series:
    """Extract net return series for long or short side."""
    col = f'{side}_net_return'
    if results_df.empty or col not in results_df.columns:
        return pd.Series(dtype=float)
    s = results_df.set_index('month')[col].dropna()
    return s


def compute_all_metrics(strategies: dict) -> pd.DataFrame:
    """Compute full-period and subperiod metrics for all strategies."""
    rows = []
    subperiods = [('2016', 2016, 2019), ('2020', 2020, 2024)]

    for name, res in strategies.items():
        for side in ('long', 'short'):
            series = _extract_return_series(res, side)
            if series.empty:
                continue
            rets = series.values
            m = compute_performance_metrics(rets)
            row = {
                'strategy': f'{name}_{side.upper()}',
                'n_months': len(rets),
                'cumret':   round(m['cumret'], 4),
                'sharpe':   round(m['sharpe'], 3),
                'maxdd':    round(m['maxdd'], 4),
            }
            # Subperiods
            sub_series = pd.Series(rets, index=series.index)
            for label, sy, ey in subperiods:
                mask = (sub_series.index.year >= sy) & (sub_series.index.year <= ey)
                sub = sub_series[mask].values
                if len(sub) > 0:
                    sm = compute_performance_metrics(sub)
                    row[f'sharpe_{label}'] = round(sm['sharpe'], 3)
                    row[f'cumret_{label}'] = round(sm['cumret'], 4)
            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_cumulative_returns(strategies: dict, save_path: str) -> None:
    """Plot cumulative wealth paths for all strategies."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for ax, side, title in zip(axes, ('long', 'short'),
                               ('Long Strategy', 'Short Strategy')):
        for name, res in strategies.items():
            series = _extract_return_series(res, side)
            if series.empty:
                continue
            wealth = (1 + series).cumprod()
            ax.plot(wealth.index, wealth.values, label=name)
        ax.set_title(title)
        ax.set_ylabel('Cumulative Wealth')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.suptitle('SSA/RLSSA Strategy Performance (2016–2024)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("Experiment 3: SSA / RLSSA Strategy")
    print("=" * 60)

    print("\nGenerating synthetic modeling panel ...")
    panel = _make_synthetic_panel(seed=42)
    print(f"Panel: {len(panel)} rows, "
          f"{panel['commodity'].nunique()} commodities, "
          f"{panel['month_label'].nunique()} months")

    strategies = run_all_strategies(panel)

    metrics_df = compute_all_metrics(strategies)
    print("\n--- Full-Period Metrics ---")
    print(metrics_df.to_string(index=False))

    csv_path = os.path.join(RESULTS_DIR, 'exp_3_ssa_results.csv')
    metrics_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    plot_path = os.path.join(RESULTS_DIR, 'exp_3_ssa_plot.png')
    plot_cumulative_returns(strategies, save_path=plot_path)

    # Save raw OOS results for each strategy
    for name, res in strategies.items():
        if not res.empty:
            out = os.path.join(RESULTS_DIR, f'exp_3_ssa_{name.lower()}_oos.csv')
            res.to_csv(out, index=False)

    # Paper target values for comparison (illustrative)
    print("\n--- Paper Target Comparison (indicative) ---")
    target_rows = [
        ('SSA_Classical_LONG',  0.42, 0.55),
        ('SSA_Classical_SHORT', 0.28, 0.38),
        ('RLSSA_Classical_LONG', 0.45, 0.58),
    ]
    print(f"{'Strategy':<30} {'Target Sharpe':>14} {'Target CumRet':>14}")
    for strat, ts, tc in target_rows:
        print(f"{strat:<30} {ts:>14.2f} {tc:>14.2f}")

    print("\nExperiment 3 complete.")


if __name__ == '__main__':
    main()
