"""
Experiment 4: Maximum Entropy Bootstrap inference.

Loads SSA strategy results (or recomputes abbreviated versions), then:
  - Runs MEB with B=500, seed=42, trim=10%
  - Computes 18 paired Sharpe-difference tests across 6 families
  - Applies Holm sequential adjustment within each family
  - Produces Top-1 selection stability analysis
  - Saves results to results/exp_4_bootstrap_*.csv
  - Updates results/RESULTS.md
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.bootstrap import (
    meb_generate_paths,
    paired_sharpe_test,
    holm_adjust,
    run_bootstrap_inference,
)
from src.ssa import run_ssa_strategy
from src.utils import compute_performance_metrics

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OOS_START   = '2016-01'
OOS_END     = '2024-12'
MEB_B       = 500
MEB_SEED    = 42
MEB_TRIM    = 0.10
ALPHA       = 0.05

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

RETAINED_COMMODITIES = [
    'CC', 'CF', 'CO', 'CP', 'CT',
    'GD', 'HE', 'HO', 'LE', 'NG',
    'SU', 'SV', 'ZC', 'ZS', 'ZW',
]

# ---------------------------------------------------------------------------
# Synthetic panel (mirrors exp_3)
# ---------------------------------------------------------------------------
def _make_synthetic_panel(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = pd.date_range('2006-01-01', periods=228, freq='MS')
    specs = {
        'CC': (0.008, 7),  'CF': (0.010, 2),  'CO': (0.006, 1),
        'CP': (0.005, 4),  'CT': (0.009, 5),  'GD': (0.004, 9),
        'HE': (0.012, 5),  'HO': (0.008, 1),  'LE': (0.007, 4),
        'NG': (0.015, 1),  'SU': (0.010, 3),  'SV': (0.005, 9),
        'ZC': (0.008, 6),  'ZS': (0.009, 7),  'ZW': (0.010, 5),
    }
    records = []
    for commodity, (amp, peak) in specs.items():
        c_seed = rng.integers(0, 1_000_000)
        c_rng  = np.random.default_rng(int(c_seed))
        cal_months = months.month.values
        seasonal = amp * np.cos(2 * np.pi * (cal_months - peak) / 12.0)
        noise = c_rng.normal(0.0, 0.04, size=len(months))
        log_ret = seasonal + noise
        simple_ret = np.expm1(log_ret)
        for i, m in enumerate(months):
            records.append({
                'month_label':   m,
                'commodity':     commodity,
                'log_return':    float(log_ret[i]),
                'simple_return': float(simple_ret[i]),
                'selected_flag': 1,
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Load or recompute strategy return series
# ---------------------------------------------------------------------------
def _load_or_compute_strategy_oos(panel: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Attempt to load pre-saved OOS results from exp_3; fall back to recompute.

    Returns dict: strategy_key -> pd.Series of monthly net returns (long side).
    """
    strategy_files = {
        'SSA_Classical_LONG':  'exp_3_ssa_ssa_classical_oos.csv',
        'SSA_VolNorm_LONG':    'exp_3_ssa_ssa_volnorm_oos.csv',
    }
    loaded: dict[str, pd.Series] = {}

    for key, fname in strategy_files.items():
        fpath = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath, parse_dates=['month'])
            side = 'long' if 'LONG' in key else 'short'
            col  = f'{side}_net_return'
            if col in df.columns:
                loaded[key] = df.set_index('month')[col].dropna()

    # Always recompute SSA Classical & VolNorm (fast) for both sides
    if not loaded:
        print("  Recomputing SSA Classical strategy ...")
        res_classical = run_ssa_strategy(
            panel, oos_start=OOS_START, oos_end=OOS_END,
            use_rlssa=False, vol_normalise=False,
        )
        print("  Recomputing SSA VolNorm strategy ...")
        res_volnorm = run_ssa_strategy(
            panel, oos_start=OOS_START, oos_end=OOS_END,
            use_rlssa=False, vol_normalise=True,
        )
        strategies_raw = {
            'SSA_Classical': res_classical,
            'SSA_VolNorm':   res_volnorm,
        }
        for name, df in strategies_raw.items():
            if df.empty:
                continue
            for side in ('long', 'short'):
                col = f'{side}_net_return'
                if col in df.columns:
                    loaded[f'{name}_{side.upper()}'] = df.set_index('month')[col].dropna()

    return loaded


# ---------------------------------------------------------------------------
# MEB bootstrap of strategy-level returns
# ---------------------------------------------------------------------------
def _bootstrap_strategy_returns(
    returns: np.ndarray,
    B: int = MEB_B,
    seed: int = MEB_SEED,
    trim: float = MEB_TRIM,
) -> np.ndarray:
    """Generate B MEB paths for a strategy-level return series."""
    return meb_generate_paths(returns, B=B, trim=trim, seed=seed)


# ---------------------------------------------------------------------------
# Build comparison families (6 families × 3 comparisons = 18 total)
# ---------------------------------------------------------------------------
def _build_comparison_families(
    strategy_rets: dict[str, np.ndarray],
    boot_rets: dict[str, np.ndarray],
) -> list[dict]:
    """
    Define 18 pairwise comparisons across 6 families.

    Families:
        1. Long strategies vs benchmark (Sharpe: A - Bench)
        2. Short strategies vs benchmark
        3. Long SSA vs Long RLSSA-variant comparisons
        4. Short SSA vs Short variant comparisons
        5. Normalised vs raw within long
        6. Normalised vs raw within short

    If a strategy key is missing from strategy_rets, the comparison is skipped.
    """
    keys = list(strategy_rets.keys())

    def _safe_pair(a, b, family, label):
        if a in strategy_rets and b in strategy_rets:
            return {'family': family, 'label': label, 'a': a, 'b': b}
        return None

    bench_key = 'BENCHMARK'

    raw_comparisons = [
        # Family 1: long vs benchmark
        _safe_pair('SSA_Classical_LONG', bench_key, 1, 'SSA_L vs Bench'),
        _safe_pair('SSA_VolNorm_LONG',   bench_key, 1, 'SSA_VN_L vs Bench'),
        _safe_pair('RLSSA_Classical_LONG', bench_key, 1, 'RLSSA_L vs Bench'),
        # Family 2: short vs benchmark
        _safe_pair('SSA_Classical_SHORT', bench_key, 2, 'SSA_S vs Bench'),
        _safe_pair('SSA_VolNorm_SHORT',   bench_key, 2, 'SSA_VN_S vs Bench'),
        _safe_pair('RLSSA_Classical_SHORT', bench_key, 2, 'RLSSA_S vs Bench'),
        # Family 3: long strategy pairwise
        _safe_pair('SSA_Classical_LONG', 'SSA_VolNorm_LONG', 3, 'SSA_L vs SSA_VN_L'),
        _safe_pair('SSA_Classical_LONG', 'RLSSA_Classical_LONG', 3, 'SSA_L vs RLSSA_L'),
        _safe_pair('SSA_VolNorm_LONG', 'RLSSA_Classical_LONG', 3, 'SSA_VN_L vs RLSSA_L'),
        # Family 4: short strategy pairwise
        _safe_pair('SSA_Classical_SHORT', 'SSA_VolNorm_SHORT', 4, 'SSA_S vs SSA_VN_S'),
        _safe_pair('SSA_Classical_SHORT', 'RLSSA_Classical_SHORT', 4, 'SSA_S vs RLSSA_S'),
        _safe_pair('SSA_VolNorm_SHORT', 'RLSSA_Classical_SHORT', 4, 'SSA_VN_S vs RLSSA_S'),
        # Family 5: normalised vs raw
        _safe_pair('SSA_Classical_LONG', 'SSA_VolNorm_LONG', 5, 'Raw_L vs VN_L'),
        _safe_pair('RLSSA_Classical_LONG', 'SSA_VolNorm_LONG', 5, 'RLSSA_L vs VN_L'),
        _safe_pair('SSA_Classical_LONG', 'SSA_Classical_SHORT', 5, 'L vs S (SSA)'),
        # Family 6: cross-strategy
        _safe_pair('SSA_Classical_LONG', 'SSA_Classical_SHORT', 6, 'SSA_L vs SSA_S'),
        _safe_pair('SSA_VolNorm_LONG', 'SSA_VolNorm_SHORT', 6, 'SSA_VN_L vs SSA_VN_S'),
        _safe_pair('SSA_Classical_LONG', 'SSA_VolNorm_SHORT', 6, 'SSA_L vs SSA_VN_S'),
    ]

    return [c for c in raw_comparisons if c is not None]


# ---------------------------------------------------------------------------
# Selection stability
# ---------------------------------------------------------------------------
def _selection_stability(strategies: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute Top-1 commodity selection frequency over OOS period.

    Returns DataFrame: commodity × strategy with selection count.
    """
    rows = []
    for name, df in strategies.items():
        if df.empty or 'long_commodity' not in df.columns:
            continue
        counts = df['long_commodity'].value_counts()
        for comm, cnt in counts.items():
            rows.append({'strategy': name, 'commodity': comm, 'n_selected': cnt})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).pivot(index='commodity', columns='strategy', values='n_selected').fillna(0).astype(int)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("Experiment 4: MEB Bootstrap Inference")
    print("=" * 60)

    print("\nGenerating synthetic modeling panel ...")
    panel = _make_synthetic_panel(seed=42)

    print("\nLoading / computing strategy returns ...")
    strategy_series = _load_or_compute_strategy_oos(panel)

    if not strategy_series:
        print("No strategy results available. Run exp_3_ssa.py first.")
        return

    print(f"  Available strategies: {list(strategy_series.keys())}")

    # Build a common time index (intersection of all OOS months)
    common_idx = None
    for s in strategy_series.values():
        common_idx = s.index if common_idx is None else common_idx.intersection(s.index)
    if common_idx is None or len(common_idx) == 0:
        print("No common OOS months found.")
        return

    print(f"  Common OOS period: {common_idx.min().strftime('%Y-%m')} "
          f"to {common_idx.max().strftime('%Y-%m')} ({len(common_idx)} months)")

    # Align returns to common index
    aligned: dict[str, np.ndarray] = {
        k: v.reindex(common_idx).fillna(0.0).values
        for k, v in strategy_series.items()
    }

    # Add benchmark: equal-weight of all OOS returns (simple proxy)
    panel_oos = panel.copy()
    panel_oos['month_dt'] = pd.to_datetime(panel_oos['month_label'])
    bench_oos = (
        panel_oos[panel_oos['month_dt'].isin(common_idx)]
        .groupby('month_dt')['simple_return']
        .mean()
        .reindex(common_idx)
        .fillna(0.0)
    )
    aligned['BENCHMARK'] = bench_oos.values

    # Performance summary
    print("\n--- Performance Summary ---")
    perf_rows = []
    for name, rets in aligned.items():
        m = compute_performance_metrics(rets)
        perf_rows.append({
            'strategy': name,
            'sharpe':   round(m['sharpe'], 3),
            'cumret':   round(m['cumret'], 4),
            'maxdd':    round(m['maxdd'], 4),
        })
    perf_df = pd.DataFrame(perf_rows)
    print(perf_df.to_string(index=False))
    perf_df.to_csv(os.path.join(RESULTS_DIR, 'exp_4_bootstrap_performance.csv'), index=False)

    # MEB bootstrap per strategy
    print(f"\nGenerating MEB paths (B={MEB_B}, seed={MEB_SEED}, trim={MEB_TRIM}) ...")
    boot: dict[str, np.ndarray] = {}
    for name, rets in aligned.items():
        s_seed = MEB_SEED + abs(hash(name)) % 10000
        boot[name] = _bootstrap_strategy_returns(rets, B=MEB_B, seed=s_seed)
        print(f"  {name}: paths shape {boot[name].shape}")

    # Build comparison families
    comparisons = _build_comparison_families(aligned, boot)
    print(f"\nDefined {len(comparisons)} pairwise comparisons across 6 families.")

    # Run paired Sharpe tests
    test_results = []
    for comp in comparisons:
        a_key, b_key = comp['a'], comp['b']
        if a_key not in aligned or b_key not in aligned:
            continue
        d_obs, p_val, n_exc = paired_sharpe_test(
            aligned[a_key], aligned[b_key],
            boot[a_key],    boot[b_key],
        )
        test_results.append({
            'family':    comp['family'],
            'label':     comp['label'],
            'strategy_a': a_key,
            'strategy_b': b_key,
            'd_sharpe':  round(d_obs, 4),
            'p_value':   round(p_val, 4),
            'n_exceed':  n_exc,
        })

    tests_df = pd.DataFrame(test_results)

    # Holm adjustment within each family
    holm_rows = []
    for family_id, grp in tests_df.groupby('family'):
        p_vals = grp['p_value'].tolist()
        rejections = holm_adjust(p_vals, alpha=ALPHA)
        for idx_in_grp, (_, row) in enumerate(grp.iterrows()):
            r = row.to_dict()
            r['holm_reject'] = rejections[idx_in_grp]
            holm_rows.append(r)

    holm_df = pd.DataFrame(holm_rows)
    print("\n--- Paired Sharpe Tests with Holm Adjustment ---")
    print(holm_df[['family', 'label', 'd_sharpe', 'p_value', 'holm_reject']].to_string(index=False))
    holm_df.to_csv(os.path.join(RESULTS_DIR, 'exp_4_bootstrap_tests.csv'), index=False)

    # Selection stability (using raw OOS results if available)
    raw_oos_strategies = {}
    for name in ['SSA_Classical', 'SSA_VolNorm']:
        fpath = os.path.join(RESULTS_DIR, f'exp_3_ssa_{name.lower()}_oos.csv')
        if os.path.exists(fpath):
            raw_oos_strategies[name] = pd.read_csv(fpath, parse_dates=['month'])

    if raw_oos_strategies:
        stab_df = _selection_stability(raw_oos_strategies)
        if not stab_df.empty:
            stab_df.to_csv(os.path.join(RESULTS_DIR, 'exp_4_bootstrap_stability.csv'))
            print("\n--- Top-1 Selection Stability ---")
            print(stab_df.to_string())

    # Update RESULTS.md
    _update_results_md(perf_df, holm_df)
    print("\nExperiment 4 complete.")


def _update_results_md(perf_df: pd.DataFrame, holm_df: pd.DataFrame) -> None:
    results_md = os.path.join(RESULTS_DIR, 'RESULTS.md')
    n_reject = int(holm_df['holm_reject'].sum()) if 'holm_reject' in holm_df.columns else 0
    n_tests  = len(holm_df)

    section = f"""
## Experiment 4: Bootstrap Inference

**MEB parameters**: B=500, seed=42, trim=10%

### Performance Summary
{perf_df.to_markdown(index=False)}

### Holm-Adjusted Tests
Total comparisons: {n_tests} across 6 families
Rejections at α={0.05}: {n_reject}/{n_tests}

{holm_df[['family','label','d_sharpe','p_value','holm_reject']].to_markdown(index=False)}
"""

    with open(results_md, 'r') as f:
        content = f.read()

    marker = '## Experiment 4: Bootstrap Inference'
    if marker in content:
        content = content[:content.index(marker)]

    with open(results_md, 'w') as f:
        f.write(content.rstrip() + '\n' + section)

    print(f"Results summary appended to {results_md}")


if __name__ == '__main__':
    main()
