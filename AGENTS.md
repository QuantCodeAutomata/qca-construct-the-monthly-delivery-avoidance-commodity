# Commodity Futures Seasonal Analysis — Project Knowledge

## Project Overview
Academic replication project implementing seasonal trading strategies for commodity futures.
Uses SSA (Singular Spectrum Analysis), RLSSA (Robust L1 SSA), and Maximum Entropy Bootstrap.

## Repository Layout
```
data/
  synthetic.py  # generate_synthetic_daily_data(), get_retained_universe(), get_commodity_universe()
src/
  utils.py      # vol_normalise_returns (λ=0.97 EWMA+expanding mean),
                # compute_net_long/short_return, compute_performance_metrics,
                # compute_subperiod_metrics
  panel.py      # build_monthly_panel(), apply_liquidity_screen(), validate_universe(),
                # build_modeling_panel(), is_contract_eligible(), compute_monthly_return()
  dvr.py        # fit_dvr(), run_dvr_strategy(), summarise_dvr_results(), _hac_lag()
  ssa.py        # SSA + RLSSA decomposition & recurrent forecasting strategy
  bootstrap.py  # MEB generation, paired Sharpe tests, Holm adjustment
exp/
  exp_1_panel.py      # Panel construction + liquidity screen (saves to results/)
  exp_2_dvr.py        # DVR + DVR-VN strategies (saves to results/)
  exp_3_ssa.py        # SSA/RLSSA strategy experiment (generates synthetic panel inline)
  exp_4_bootstrap.py  # Bootstrap inference experiment (loads exp_3 results or recomputes)
tests/
  test_panel.py       # 19 tests; uses small deterministic contract fixtures
  test_dvr.py         # 17 tests; uses _make_modeling_panel() helper
  test_ssa.py         # 10 tests; uses L=10, rank=3, N=50 fixtures
results/
  RESULTS.md          # Populated by experiment scripts
```

## Key Parameters
| Constant           | Value   | Meaning                                   |
|--------------------|---------|-------------------------------------------|
| ESTIMATION_WINDOW  | 120     | DVR rolling window (months)               |
| HAC_MAXLAGS        | 3       | Newey-West lag = floor(0.75×120^(1/3))    |
| SIGNIFICANCE_LEVEL | 0.10    | DVR beta significance threshold           |
| MIN_VOLUME         | 1000    | Liquidity screen: min monthly avg vol     |
| SCREEN_YEARS       | 2006-15 | Liquidity screen window                   |
| DELIVERY_AVOID_GAP | 2       | Min months between obs & expiry month     |
| POST_COVERAGE_DAYS | 14      | Min days post-month-end contract coverage |
| TRANSACTION_COST   | 0.00086 | One-way proportional cost                 |
| SSA_L              | 36      | Embedding / window dimension              |
| SSA_K              | 85      | N − L + 1 (trajectory columns)           |
| SSA_RANK           | 12      | Truncation rank                           |
| SSA_N              | 120     | Training window (L + K − 1)              |
| RLSSA_MAX_ITER     | 50000   | AL1 convergence budget                    |
| RLSSA_TOL          | 1e-10   | AL1 convergence tolerance                 |
| MEB_B              | 500     | Bootstrap replications                    |
| MEB_SEED           | 42      | Random seed                               |
| MEB_TRIM           | 0.10    | Trim fraction for MEB                     |

## Algorithm Notes
- **Delivery-avoidance panel**: `build_monthly_panel()` precomputes per-contract metadata
  (`months_with_data` set, `max_date`, `exp_month_start`) and monthly aggregates (groupby dict)
  to achieve O(1) eligibility and O(1) return lookup in the inner loop — critical for performance
  on the full 17-commodity × 288-month dataset (runs in ~14s vs >2min without).
- **DVR**: OLS + HAC `get_robustcov_results(cov_type='HAC', maxlags=3, use_correction=False)`.
  Returns (None, None, None) when T < 15 or statsmodels raises any exception.
- **bool/None tests**: pandas DataFrame columns store Python `None` as `NaN` (float),
  and Python `bool` as numpy `bool_`. Tests must use `pd.isna()` and `bool()` not `is None` / `is True`.
- **SSA forecast**: Hankel trajectory matrix → SVD rank-12 → recurrent linear recurrence.
- **RLSSA forecast**: Hawkins AL1 robust SVD (sequential deflation) → diagonal averaging
  → re-Hankelise → standard SVD basis → recurrent forecast.
- **AL1 convergence**: raises `RuntimeError` if not converged; caught in `run_ssa_strategy`.
- **MEB**: Vinod & López-de-Lacalle (2009) meboot; univariate per commodity.
- **Holm adjustment**: step-down within 6 families; `holm_adjust()` returns bool list.

## Modeling Panel Schema
DataFrame with columns: `month_label` (parseable datetime), `commodity` (str),
`log_return` (float), `simple_return` (float), `selected_flag` (bool/int).

## Running Tests
```bash
cd /workspace/project
python -m pytest tests/ -v --tb=short
```

## Performance Notes
RLSSA with `max_iter=50000` over 108 OOS months × 15 commodities is very slow.
Experiment scripts use `max_iter=500` for practicality. Tests use L=10, rank=3 fixtures.
