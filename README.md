# Commodity Futures Seasonal Analysis

Replication and extension of a commodity futures seasonal trading paper, implementing
the delivery-avoidance monthly panel, Dummy Variable Regression (DVR), Seasonal
Singular Spectrum Analysis (SSA/RLSSA), and Moving-Extremum Bootstrap (MEB) inference.

---

## Overview

Seasonal patterns in commodity futures are well-documented but difficult to exploit
due to contract roll costs, liquidity constraints, and overfitting risk. This project
replicates a paper's rigorous methodology:

1. **Delivery-avoidance panel** (Experiment 1) — constructs a clean monthly return
   series for each commodity by selecting the nearest futures contract that expires
   ≥ 2 calendar months after the observation month and has price coverage through
   at least 14 calendar days after month-end, sidestepping physical delivery pressure.

2. **DVR strategy** (Experiment 2) — a 120-month rolling Dummy Variable Regression
   identifies which commodity exhibits the strongest statistically significant
   (Newey–West HAC, α = 0.10) positive (long) or negative (short) seasonal effect
   for the coming calendar month, then trades the Top-1 signal.

3. **SSA/RLSSA strategy** (Experiment 3) — Singular Spectrum Analysis extracts the
   dominant seasonal component from the rolling window, with a Robust/Localised
   variant (RLSSA) for non-stationary seasonality.

4. **MEB bootstrap inference** (Experiment 4) — Moving-Extremum Bootstrap generates
   the null distribution of the Top-1 strategy Sharpe under the hypothesis of no
   seasonality, providing valid inference despite the data-snooping selection step.

---

## Project Structure

```
/workspace/project/
├── data/
│   ├── __init__.py
│   └── synthetic.py        # synthetic daily contract-level data generator
├── src/
│   ├── __init__.py
│   ├── panel.py            # exp_1: delivery-avoidance panel construction
│   ├── dvr.py              # exp_2: DVR strategy
│   └── utils.py            # shared utilities (EWMA vol, metrics, costs)
├── exp/
│   ├── __init__.py
│   ├── exp_1_panel.py      # run panel construction experiment
│   └── exp_2_dvr.py        # run DVR experiment
├── tests/
│   ├── __init__.py
│   ├── test_panel.py
│   └── test_dvr.py
├── results/
│   └── RESULTS.md
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Commodity Universe

### Candidate pool (17 commodities)
| Code | Name               | Exchange |
|------|--------------------|----------|
| CC   | Cocoa              | ICE      |
| CF   | Coffee             | ICE      |
| CO   | Crude Oil (Brent)  | ICE      |
| CP   | Copper             | CME      |
| CT   | Cotton             | ICE      |
| GD   | Gold               | CME      |
| HE   | Lean Hogs          | CME      |
| HO   | Heating Oil        | CME      |
| LE   | Live Cattle        | CME      |
| NG   | Natural Gas        | CME      |
| SU   | Sugar #11          | ICE      |
| SV   | Silver             | CME      |
| ZC   | Corn               | CBOT     |
| ZS   | Soybeans           | CBOT     |
| ZW   | Wheat              | CBOT     |
| PD   | Palladium          | CME      |
| PT   | Platinum           | CME      |

### Retained universe (15 commodities)
PD and PT are excluded by the pre-evaluation liquidity screen: their minimum monthly
average daily volume falls below 1 000 contracts/day in the 2006–2015 screen window.

---

## Methodology

### Delivery-Avoidance Rule
For observation month *t* and commodity *i*:
- Let *S(i,t)* = set of contracts with data in month *t*
- A contract *k* ∈ *S(i,t)* is **eligible** if:
  - Its expiry month is ≥ 2 calendar months after *t* (avoids delivery pressure)
  - Its price series extends ≥ 14 calendar days past the end of month *t*
    (ensures a clean next-month return can be computed)
- Select the eligible contract with the **nearest expiry** (minimises roll cost)
- Monthly return: `R_{i,t} = P_close(last day of t) / P_open(first day of t) - 1`

### DVR Strategy
For each target calendar month *m* ∈ {1,…,12}:
1. Collect the prior 120 monthly log-returns for each commodity
2. Regress returns on an intercept and a single dummy `D_m` = 1 if the observation
   falls in calendar month *m*
3. Use Newey–West HAC standard errors with lag `⌊0.75 × 120^{1/3}⌋ = 3`
4. **Long signal**: commodity with the largest significant (p ≤ 0.10) positive β̂
5. **Short signal**: commodity with the most negative significant negative β̂
6. If no signal qualifies, hold cash (0 % return)
7. Transaction cost: 0.086 % one-way (round-trip applied via `(1+R)(1-c)² - 1`)

### Volatility Normalisation (optional)
`r̃_t = (r_t − r̄_t) / σ_t`
where r̄_t is the expanding mean and σ_t is the EWMA standard deviation
(λ = 0.97, warm-started from the first 12 observations).

### Performance Metrics
- **CumRet**: `∏(1 + R_t) - 1`
- **Sharpe**: `mean(R) / std(R) × √12` (zero risk-free rate, monthly frequency)
- **MaxDD**: maximum peak-to-trough drawdown of the wealth path

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python ≥ 3.10.

---

## Usage

### Run all experiments
```bash
# From project root
python exp/exp_1_panel.py
python exp/exp_2_dvr.py
```

### Run tests
```bash
pytest tests/ -v
```

### Generate synthetic data only
```python
from data.synthetic import generate_synthetic_daily_data, get_retained_universe
df = generate_synthetic_daily_data()
print(df.head())
```

---

## Key Implementation Notes

- **No data imputation**: missing monthly returns are left as `NaN` and commodities
  with insufficient history (< 24 months in the rolling window) are skipped.
- **Pre-evaluation screen**: the liquidity filter uses only 2006–2015 data, which
  predates the 2016–2024 OOS evaluation window, avoiding look-ahead bias.
- **Absolute imports**: all modules use `src.*` / `data.*` imports and must be run
  from `/workspace/project/` (or with that directory on `PYTHONPATH`).

---

## Results Summary

See `results/RESULTS.md` for populated experiment outputs.
