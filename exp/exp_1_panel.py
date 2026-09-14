"""
Experiment 1: Delivery-Avoidance Monthly Panel Construction.

Generates synthetic data, builds the monthly panel, applies the liquidity
screen, validates the 15-commodity universe, and saves outputs to results/.

Run from project root:
    python exp/exp_1_panel.py
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
    validate_universe,
    build_modeling_panel,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("Experiment 1: Delivery-Avoidance Monthly Panel Construction")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # 1. Generate synthetic data
    # ------------------------------------------------------------------ #
    print("\n[1] Generating synthetic daily contract data …")
    daily_df = generate_synthetic_daily_data(
        start_date='2001-01-01',
        end_date='2025-01-31',
        seed=42,
    )
    print(f"    Rows: {len(daily_df):,}  |  Commodities: {daily_df['commodity'].nunique()}"
          f"  |  Contracts: {daily_df['contract_id'].nunique():,}")
    print(f"    Date range: {daily_df['date'].min().date()} → {daily_df['date'].max().date()}")

    # ------------------------------------------------------------------ #
    # 2. Build monthly panel
    # ------------------------------------------------------------------ #
    print("\n[2] Building monthly delivery-avoidance panel …")
    panel = build_monthly_panel(
        daily_df,
        start_year=2001, start_month=1,
        end_year=2024,   end_month=12,
    )
    print(f"    Panel rows: {len(panel):,}")
    n_with_return = panel['selected_flag'].sum()
    print(f"    Rows with valid return: {n_with_return:,}"
          f"  ({100*n_with_return/len(panel):.1f} %)")

    # ------------------------------------------------------------------ #
    # 3. Liquidity screen (2006-2015)
    # ------------------------------------------------------------------ #
    print("\n[3] Applying liquidity screen (2006–2015, min vol = 1 000 contracts/day) …")
    retained, liq_report = apply_liquidity_screen(panel)
    print(f"    Retained {len(retained)} commodities: {retained}")
    print(liq_report.to_string(index=False))

    # ------------------------------------------------------------------ #
    # 4. Validate universe
    # ------------------------------------------------------------------ #
    print("\n[4] Validating retained universe …")
    val = validate_universe(retained)
    print(f"    Match paper's 15: {val['match']}")
    if not val['match']:
        print(f"    Missing: {val['missing_from_retained']}")
        print(f"    Extra:   {val['extra_in_retained']}")

    # ------------------------------------------------------------------ #
    # 5. Build modeling panel
    # ------------------------------------------------------------------ #
    print("\n[5] Building modeling panel (2006–2024, retained commodities) …")
    modeling_panel = build_modeling_panel(panel, retained)
    print(f"    Modeling panel rows: {len(modeling_panel):,}")

    # Summary statistics per commodity
    summary = (
        modeling_panel.groupby('commodity')
        .agg(
            n_months=('month_label', 'count'),
            n_valid=('selected_flag', 'sum'),
            mean_ret=('simple_return', 'mean'),
            std_ret=('simple_return', 'std'),
            mean_vol=('monthly_avg_volume', 'mean'),
        )
        .round(5)
    )
    print("\n    Per-commodity summary (modeling period 2006–2024):")
    print(summary.to_string())

    # ------------------------------------------------------------------ #
    # 6. Save outputs
    # ------------------------------------------------------------------ #
    panel_path = os.path.join(RESULTS_DIR, 'exp_1_panel.csv')
    liq_path = os.path.join(RESULTS_DIR, 'exp_1_liquidity_report.csv')
    val_path = os.path.join(RESULTS_DIR, 'exp_1_validation_report.csv')
    mp_path = os.path.join(RESULTS_DIR, 'exp_1_modeling_panel.csv')

    panel.to_csv(panel_path, index=False)
    liq_report.to_csv(liq_path, index=False)
    pd.DataFrame([val]).to_csv(val_path, index=False)
    modeling_panel.to_csv(mp_path, index=False)

    print(f"\n[6] Saved outputs to {RESULTS_DIR}/")

    # ------------------------------------------------------------------ #
    # 7. Cumulative return plot
    # ------------------------------------------------------------------ #
    print("\n[7] Plotting equal-weight cumulative return across retained commodities …")

    # Equal-weight monthly return across all retained commodities
    ew = (
        modeling_panel[modeling_panel['selected_flag']]
        .groupby('month_label')['simple_return']
        .mean()
    )
    ew.index = pd.to_datetime(ew.index)
    ew = ew.sort_index()

    wealth = (1 + ew).cumprod()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(wealth.index, wealth.values, linewidth=1.5, color='steelblue')
    axes[0].axhline(1.0, color='grey', linewidth=0.8, linestyle='--')
    axes[0].set_title('Equal-Weight Long Cumulative Wealth (retained 15 commodities)')
    axes[0].set_ylabel('Wealth (start = 1)')
    axes[0].grid(alpha=0.3)

    # Per-commodity heatmap: mean monthly return by calendar month
    pivot = (
        modeling_panel[modeling_panel['selected_flag']]
        .groupby(['commodity', 'month'])['simple_return']
        .mean()
        .unstack('month')
    )
    im = axes[1].imshow(pivot.values, aspect='auto', cmap='RdYlGn', vmin=-0.05, vmax=0.05)
    axes[1].set_yticks(range(len(pivot.index)))
    axes[1].set_yticklabels(pivot.index, fontsize=8)
    axes[1].set_xticks(range(12))
    axes[1].set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun',
                              'Jul','Aug','Sep','Oct','Nov','Dec'], fontsize=8)
    axes[1].set_title('Mean Monthly Return by Commodity and Calendar Month')
    plt.colorbar(im, ax=axes[1], fraction=0.02)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'exp_1_cumret.png')
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"    Saved plot → {plot_path}")

    # ------------------------------------------------------------------ #
    # 8. Overall summary statistics
    # ------------------------------------------------------------------ #
    from src.utils import compute_performance_metrics
    if len(ew) > 0:
        perf = compute_performance_metrics(ew.values)
        print("\n    Equal-weight benchmark performance (modeling period):")
        print(f"      CumRet : {perf['cumret']:.4f}")
        print(f"      Sharpe : {perf['sharpe']:.4f}")
        print(f"      MaxDD  : {perf['maxdd']:.4f}")

    print("\n[Done] Experiment 1 complete.")


if __name__ == '__main__':
    main()
