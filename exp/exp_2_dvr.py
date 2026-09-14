"""
Experiment 2: DVR (Dummy Variable Regression) Seasonal Strategy.

Loads (or regenerates) the monthly panel, runs the classical and
vol-normalised DVR strategies, computes full-period and subperiod
performance metrics, and saves outputs to results/.

Run from project root:
    python exp/exp_2_dvr.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.synthetic import generate_synthetic_daily_data, get_retained_universe
from src.panel import (
    build_monthly_panel,
    apply_liquidity_screen,
    build_modeling_panel,
)
from src.dvr import run_dvr_strategy, summarise_dvr_results
from src.utils import compute_performance_metrics, compute_subperiod_metrics

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
MP_CSV = os.path.join(RESULTS_DIR, 'exp_1_modeling_panel.csv')
OOS_START = '2016-01'
OOS_END = '2024-12'


def load_or_build_modeling_panel() -> pd.DataFrame:
    """Load saved modeling panel or rebuild from scratch."""
    if os.path.exists(MP_CSV):
        print(f"    Loading saved modeling panel from {MP_CSV} …")
        mp = pd.read_csv(MP_CSV)
        mp['month_dt'] = pd.to_datetime(mp['month_label'])
        return mp

    print("    No saved panel found — regenerating …")
    daily_df = generate_synthetic_daily_data(seed=42)
    panel = build_monthly_panel(daily_df, start_year=2001, start_month=1,
                                end_year=2024, end_month=12)
    retained, _ = apply_liquidity_screen(panel)
    return build_modeling_panel(panel, retained)


def _make_return_series(results_df: pd.DataFrame, col: str) -> pd.Series:
    s = results_df.set_index('month')[col]
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("Experiment 2: DVR Strategy (OOS 2016-01 → 2024-12)")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # 1. Load modeling panel
    # ------------------------------------------------------------------ #
    print("\n[1] Loading modeling panel …")
    modeling_panel = load_or_build_modeling_panel()
    n_comm = modeling_panel['commodity'].nunique()
    n_months = modeling_panel['month_label'].nunique()
    print(f"    {n_comm} commodities  |  {n_months} panel months")

    # ------------------------------------------------------------------ #
    # 2. Run DVR strategies
    # ------------------------------------------------------------------ #
    print("\n[2] Running DVR (classical) …")
    dvr_results = run_dvr_strategy(
        modeling_panel,
        oos_start=OOS_START,
        oos_end=OOS_END,
        vol_normalise=False,
    )
    print(f"    OOS months produced: {len(dvr_results)}")

    print("\n[3] Running DVR-VN (vol-normalised) …")
    dvrvn_results = run_dvr_strategy(
        modeling_panel,
        oos_start=OOS_START,
        oos_end=OOS_END,
        vol_normalise=True,
    )

    # ------------------------------------------------------------------ #
    # 4. Full-period performance
    # ------------------------------------------------------------------ #
    print("\n[4] Full-period performance summary:")
    dvr_summary = summarise_dvr_results(dvr_results, label='DVR')
    vn_summary = summarise_dvr_results(dvrvn_results, label='DVR_VN')

    def fmt_pct(v):
        return f"{v*100:+.2f}%" if v is not None and not np.isnan(v) else "n/a"

    rows = [
        ("DVR Long",  dvr_summary["DVR_long_cumret"],  dvr_summary["DVR_long_sharpe"],  dvr_summary["DVR_long_maxdd"]),
        ("DVR Short", dvr_summary["DVR_short_cumret"], dvr_summary["DVR_short_sharpe"], dvr_summary["DVR_short_maxdd"]),
        ("DVR-VN Long",  vn_summary["DVR_VN_long_cumret"],  vn_summary["DVR_VN_long_sharpe"],  vn_summary["DVR_VN_long_maxdd"]),
        ("DVR-VN Short", vn_summary["DVR_VN_short_cumret"], vn_summary["DVR_VN_short_sharpe"], vn_summary["DVR_VN_short_maxdd"]),
        ("EW Benchmark", dvr_summary["benchmark_cumret"], dvr_summary["benchmark_sharpe"], dvr_summary["benchmark_maxdd"]),
    ]
    hdr = f"{'Strategy':<20}  {'CumRet':>10}  {'Sharpe':>8}  {'MaxDD':>10}"
    print("    " + hdr)
    print("    " + "-" * len(hdr))
    for name, cr, sh, md in rows:
        print(f"    {name:<20}  {fmt_pct(cr):>10}  {sh:>8.3f}  {fmt_pct(md):>10}")

    # ------------------------------------------------------------------ #
    # 5. Subperiod performance
    # ------------------------------------------------------------------ #
    print("\n[5] Subperiod performance:")
    subperiods = [('2016-2018', 2016, 2018), ('2019-2021', 2019, 2021), ('2022-2024', 2022, 2024)]
    for label, sr in [('DVR Long', _make_return_series(dvr_results, 'long_net_return')),
                       ('DVR Short', _make_return_series(dvr_results, 'short_net_return')),
                       ('DVR-VN Long', _make_return_series(dvrvn_results, 'long_net_return'))]:
        sub_df = compute_subperiod_metrics(sr, subperiods)
        print(f"\n    {label}:")
        print(sub_df.round(4).to_string())

    # ------------------------------------------------------------------ #
    # 6. Signal frequency analysis
    # ------------------------------------------------------------------ #
    print("\n[6] Signal frequency analysis:")
    if len(dvr_results) > 0:
        long_freq = 1 - dvr_results['is_long_cash'].mean()
        short_freq = 1 - dvr_results['is_short_cash'].mean()
        print(f"    DVR Long  in-market: {long_freq:.1%}  (cash: {1-long_freq:.1%})")
        print(f"    DVR Short in-market: {short_freq:.1%}  (cash: {1-short_freq:.1%})")

        print("\n    Most-selected commodities (long):")
        long_counts = (
            dvr_results['long_commodity']
            .dropna()
            .value_counts()
        )
        print(long_counts.to_string())

        print("\n    Most-selected commodities (short):")
        short_counts = (
            dvr_results['short_commodity']
            .dropna()
            .value_counts()
        )
        print(short_counts.to_string())

    # ------------------------------------------------------------------ #
    # 7. Save CSV
    # ------------------------------------------------------------------ #
    csv_path = os.path.join(RESULTS_DIR, 'exp_2_dvr_results.csv')
    dvr_results.to_csv(csv_path, index=False)
    print(f"\n[7] Saved DVR results → {csv_path}")

    vn_csv_path = os.path.join(RESULTS_DIR, 'exp_2_dvrvn_results.csv')
    dvrvn_results.to_csv(vn_csv_path, index=False)
    print(f"    Saved DVR-VN results → {vn_csv_path}")

    # ------------------------------------------------------------------ #
    # 8. Plot
    # ------------------------------------------------------------------ #
    print("\n[8] Generating performance plot …")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    def wealth_series(df, col):
        s = (1 + df.set_index('month')[col]).cumprod()
        s.index = pd.to_datetime(s.index)
        return s.sort_index()

    dvr_long_w  = wealth_series(dvr_results, 'long_net_return')
    dvr_short_w = wealth_series(dvr_results, 'short_net_return')
    dvrvn_long_w  = wealth_series(dvrvn_results, 'long_net_return')
    dvrvn_short_w = wealth_series(dvrvn_results, 'short_net_return')
    bench_w = wealth_series(dvr_results, 'benchmark_net_return')

    # Top-left: DVR Long vs Benchmark
    ax = axes[0, 0]
    ax.plot(dvr_long_w.index, dvr_long_w.values, label='DVR Long', linewidth=1.5)
    ax.plot(bench_w.index, bench_w.values, label='EW Benchmark', linewidth=1.2, linestyle='--', color='grey')
    ax.set_title('DVR Long vs Equal-Weight Benchmark')
    ax.set_ylabel('Wealth')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Top-right: DVR Short vs Benchmark
    ax = axes[0, 1]
    ax.plot(dvr_short_w.index, dvr_short_w.values, label='DVR Short', linewidth=1.5, color='firebrick')
    ax.plot(bench_w.index, bench_w.values, label='EW Benchmark', linewidth=1.2, linestyle='--', color='grey')
    ax.set_title('DVR Short vs Equal-Weight Benchmark')
    ax.set_ylabel('Wealth')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Bottom-left: DVR-VN Long
    ax = axes[1, 0]
    ax.plot(dvrvn_long_w.index, dvrvn_long_w.values, label='DVR-VN Long', linewidth=1.5, color='darkorange')
    ax.plot(bench_w.index, bench_w.values, label='EW Benchmark', linewidth=1.2, linestyle='--', color='grey')
    ax.set_title('DVR-VN Long vs Equal-Weight Benchmark')
    ax.set_ylabel('Wealth')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Bottom-right: DVR-VN Short
    ax = axes[1, 1]
    ax.plot(dvrvn_short_w.index, dvrvn_short_w.values, label='DVR-VN Short', linewidth=1.5, color='purple')
    ax.plot(bench_w.index, bench_w.values, label='EW Benchmark', linewidth=1.2, linestyle='--', color='grey')
    ax.set_title('DVR-VN Short vs Equal-Weight Benchmark')
    ax.set_ylabel('Wealth')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.suptitle(f'DVR Strategy Performance (OOS: {OOS_START} → {OOS_END})', fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'exp_2_dvr_plot.png')
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"    Saved plot → {plot_path}")

    print("\n[Done] Experiment 2 complete.")


if __name__ == '__main__':
    main()
