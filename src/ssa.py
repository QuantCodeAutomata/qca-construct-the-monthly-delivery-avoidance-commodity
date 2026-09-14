"""
SSA and RLSSA seasonal trading strategy implementation.

SSA uses standard SVD-based trajectory matrix decomposition.
RLSSA uses Hawkins et al. (2001) alternating L1 (AL1) robust SVD.
Both use L=36 embedding, rank-12 truncation, recurrent one-step-ahead forecasting.
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
from src.utils import (
    vol_normalise_returns,
    compute_net_long_return,
    compute_net_short_return,
    compute_performance_metrics,
    compute_subperiod_metrics,
)

# Custom — Context7 found no library equivalent for paper's exact SSA+RLSSA+recurrent forecast workflow

SSA_L = 36      # Embedding dimension
SSA_K = 85      # N - L + 1 = 120 - 36 + 1
SSA_RANK = 12   # Number of components
SSA_N = 120     # Training window
RLSSA_MAX_ITER = 50000
RLSSA_TOL = 1e-10
TRANSACTION_COST = 0.00086

def hankelise(series: np.ndarray, L: int) -> np.ndarray:
    """
    Build Hankel (trajectory) matrix from a time series.

    H[i,j] = series[i+j] for i=0..L-1, j=0..K-1
    where K = N - L + 1
    """
    N = len(series)
    K = N - L + 1
    H = np.zeros((L, K))
    for j in range(K):
        H[:, j] = series[j:j+L]
    return H

def diagonal_average(matrix: np.ndarray, N: int) -> np.ndarray:
    """
    Diagonal averaging (anti-diagonal averaging) to reconstruct time series from matrix.

    Given an L x K matrix, reconstruct an N-point series where N = L + K - 1.
    Each anti-diagonal is averaged.
    """
    L, K = matrix.shape
    assert L + K - 1 == N, f"Expected N={N}, got L+K-1={L+K-1}"
    series = np.zeros(N)
    counts = np.zeros(N)
    for i in range(L):
        for j in range(K):
            idx = i + j
            series[idx] += matrix[i, j]
            counts[idx] += 1
    return series / counts

def ssa_decompose(series: np.ndarray, L: int = SSA_L, rank: int = SSA_RANK) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Classical SSA decomposition via SVD.

    Returns:
        (U, S, Vt) of the rank-truncated trajectory matrix
    """
    H = hankelise(series, L)
    U, S, Vt = np.linalg.svd(H, full_matrices=False)
    return U[:, :rank], S[:rank], Vt[:rank, :]

def ssa_reconstruct(U: np.ndarray, S: np.ndarray, Vt: np.ndarray, N: int) -> np.ndarray:
    """
    Reconstruct time series from rank-truncated SVD components.
    Reconstructs the approximated trajectory matrix and applies diagonal averaging.
    """
    H_approx = (U * S) @ Vt
    return diagonal_average(H_approx, N)

def ssa_recurrent_forecast(
    series: np.ndarray,
    U: np.ndarray,
    L: int = SSA_L,
) -> float:
    """
    Recurrent SSA one-step-ahead forecast.

    Standard recurrent SSA forecasting (Golyandina and Korobeynikov 2014).
    Uses the basis U from the SSA decomposition.

    The recurrent formula: x_{N+1} = sum_j (pi_j * P_j) / (1 - nu^2)
    where P_j = U_j[-1] (last element of j-th eigenvector),
    pi_j = U_j[:-1] dot x_{N-L+2:N+1} but implemented via the standard approach.
    """
    N = len(series)
    rank = U.shape[1]

    # Last column of the reconstructed basis (R = last row of U without last element)
    # Standard recurrent formula: forecast using linear recurrence relation
    # R = [U_{1:L-1, :}] last eigenvectors without last row
    R = U[-1, :]           # shape (rank,)
    nu2 = np.dot(R, R)     # = sum of squared last elements

    if nu2 >= 1.0:
        # Degenerate case: use mean of reconstructed series as fallback
        return float(np.mean(series[-L:]))

    # The linear recurrence coefficients
    # a = sum_j R[j] * U[:-1, j]
    A = U[:-1, :] @ R  # shape (L-1,)
    A = A / (1 - nu2)

    # Apply recurrence: x_{n+1} = A dot x_{n-L+2:n+1}
    # The last L-1 elements of the series
    last_vals = series[-(L-1):]
    forecast = float(np.dot(A, last_vals))
    return forecast

def run_classical_ssa_forecast(
    series: np.ndarray,
    L: int = SSA_L,
    rank: int = SSA_RANK,
) -> float:
    """
    Classical SSA one-step-ahead forecast.

    Steps:
    1. Form Hankel trajectory matrix (L x K)
    2. Apply SVD, retain rank components
    3. Reconstruct signal via diagonal averaging
    4. Apply recurrent SSA forecasting
    """
    U, S, Vt = ssa_decompose(series, L, rank)
    forecast = ssa_recurrent_forecast(series, U, L)
    return forecast

def al1_robust_svd_component(
    H: np.ndarray,
    max_iter: int = RLSSA_MAX_ITER,
    tol: float = RLSSA_TOL,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Hawkins et al. (2001) Alternating L1 (AL1) robust SVD for first component.

    Finds u, v minimising sum |H - sigma*u*v'|_1 iteratively.

    Custom — Context7 found no library equivalent (Hawkins et al. 2001 AL1)

    Algorithm:
    1. Initialize v as first right singular vector of H
    2. Alternate: given v, find u = sign(Hv) * |Hv| / ||Hv||_2 (L1 regression)
       given u, find v = sign(H'u) * |H'u| / ||H'u||_2
    3. Converge when change in v < tol
    """
    L, K = H.shape

    # Initialize with first right singular vector
    _, _, Vt = np.linalg.svd(H, full_matrices=False)
    v = Vt[0, :].copy()  # K-vector
    v = v / np.linalg.norm(v)

    sigma_prev = np.inf

    for iteration in range(max_iter):
        # Update u: L1 regression of each row of H onto v
        # u_i = sign(sum_j H_ij * v_j)
        Hv = H @ v  # L-vector
        u = Hv / (np.linalg.norm(Hv) + 1e-300)

        # Update v: L1 regression of each column of H onto u
        Htu = H.T @ u  # K-vector
        v_new = Htu / (np.linalg.norm(Htu) + 1e-300)

        # Convergence check on squared vector criterion
        diff = np.sum((v_new - v) ** 2)
        v = v_new

        if diff < tol:
            break

        if iteration == max_iter - 1:
            raise RuntimeError(
                f"RLSSA AL1 failed to converge after {max_iter} iterations "
                f"(diff={diff:.2e} > tol={tol:.2e})"
            )

    sigma = float(np.dot(u, H @ v))
    return u, v, sigma

def run_rlssa_decompose(
    H: np.ndarray,
    rank: int = SSA_RANK,
    max_iter: int = RLSSA_MAX_ITER,
    tol: float = RLSSA_TOL,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    RLSSA: Hawkins AL1 robust SVD with sequential deflation, rank components.

    Extracts rank components one at a time using AL1 robust SVD.
    After each extraction, deflates the residual matrix.

    Returns:
        (U_robust, S_robust, V_robust) as arrays of shape (L,rank), (rank,), (K,rank)
    """
    L, K = H.shape
    H_residual = H.copy()

    U_list = []
    S_list = []
    V_list = []

    for r in range(rank):
        u, v, sigma = al1_robust_svd_component(H_residual, max_iter=max_iter, tol=tol)
        U_list.append(u)
        S_list.append(sigma)
        V_list.append(v)
        # Deflate: remove rank-1 approximation
        H_residual = H_residual - sigma * np.outer(u, v)

    U_robust = np.column_stack(U_list)   # (L, rank)
    S_robust = np.array(S_list)           # (rank,)
    V_robust = np.column_stack(V_list)   # (K, rank)
    return U_robust, S_robust, V_robust

def run_rlssa_forecast(
    series: np.ndarray,
    L: int = SSA_L,
    rank: int = SSA_RANK,
    max_iter: int = RLSSA_MAX_ITER,
    tol: float = RLSSA_TOL,
) -> float:
    """
    RLSSA one-step-ahead forecast.

    Steps:
    1. Form Hankel trajectory matrix
    2. Apply Hawkins AL1 robust SVD (sequential deflation, rank components)
    3. Build rank-12 trajectory approximation from robust components
    4. Reconstruct univariate signal via diagonal averaging
    5. Re-Hankelise the reconstructed signal
    6. Apply standard SVD to re-Hankelised signal for orthonormal basis
    7. Apply recurrent SSA forecasting with the standard basis
    """
    N = len(series)
    H = hankelise(series, L)

    # Step 1: Robust decomposition
    U_r, S_r, V_r = run_rlssa_decompose(H, rank=rank, max_iter=max_iter, tol=tol)

    # Step 2: Build robust trajectory approximation
    H_robust = (U_r * S_r) @ V_r.T  # (L, K)

    # Step 3: Reconstruct signal via diagonal averaging
    reconstructed = diagonal_average(H_robust, N)

    # Step 4: Re-Hankelise
    H_rehankel = hankelise(reconstructed, L)

    # Step 5: Standard SVD on re-Hankelised for recurrent forecast basis
    U_std, S_std, Vt_std = np.linalg.svd(H_rehankel, full_matrices=False)
    U_std_trunc = U_std[:, :rank]

    # Step 6: Recurrent forecast
    forecast = ssa_recurrent_forecast(reconstructed, U_std_trunc, L)
    return forecast

def run_ssa_strategy(
    modeling_panel: pd.DataFrame,
    oos_start: str = '2016-01',
    oos_end: str = '2024-12',
    use_rlssa: bool = False,
    vol_normalise: bool = False,
    L: int = SSA_L,
    rank: int = SSA_RANK,
    cost: float = TRANSACTION_COST,
) -> pd.DataFrame:
    """
    Run SSA or RLSSA Top-1 Long and Short seasonal strategy.

    Args:
        modeling_panel: monthly panel DataFrame
        oos_start: first OOS month YYYY-MM
        oos_end: last OOS month YYYY-MM
        use_rlssa: if True, use RLSSA robust decomposition
        vol_normalise: if True, apply EWMA+expanding-mean normalisation
        L: SSA embedding dimension
        rank: number of SSA components to retain
        cost: one-way proportional transaction cost

    Returns:
        DataFrame with monthly results
    """
    panel = modeling_panel.copy()
    panel['month_dt'] = pd.to_datetime(panel['month_label'])
    all_months = sorted(panel['month_dt'].unique())
    oos_months = [m for m in all_months if oos_start <= m.strftime('%Y-%m') <= oos_end]

    commodities = sorted(panel['commodity'].unique())
    N_window = L + SSA_K - 1  # Should be 120

    # Pre-build monthly data
    month_data = {}
    for mt in all_months:
        month_data[mt] = {}
        mt_df = panel[panel['month_dt'] == mt]
        for _, row in mt_df.iterrows():
            avail = bool(row['selected_flag']) and not pd.isna(row['log_return'])
            month_data[mt][row['commodity']] = {
                'log_return': row['log_return'] if avail else np.nan,
                'simple_return': row['simple_return'] if avail else np.nan,
                'available': avail,
            }

    results = []

    for trade_month in oos_months:
        trade_idx = all_months.index(trade_month)
        if trade_idx < N_window:
            continue

        window_months = all_months[trade_idx - N_window: trade_idx]
        scores = {}
        warnings_list = []

        for commodity in commodities:
            rets = []
            for wm in window_months:
                d = month_data.get(wm, {}).get(commodity, {})
                if d.get('available', False):
                    rets.append(d['log_return'])
                else:
                    rets.append(np.nan)

            rets_arr = np.array(rets)
            # Only proceed if enough non-NaN values
            valid_mask = ~np.isnan(rets_arr)
            if valid_mask.sum() < N_window * 0.8:
                continue

            # Fill NaN with 0 for SSA (strict policy: ideally no NaN)
            rets_filled = np.where(valid_mask, rets_arr, 0.0)

            if vol_normalise:
                rets_filled = vol_normalise_returns(rets_filled)

            try:
                if use_rlssa:
                    forecast = run_rlssa_forecast(rets_filled, L=L, rank=rank)
                else:
                    forecast = run_classical_ssa_forecast(rets_filled, L=L, rank=rank)

                if np.isfinite(forecast):
                    scores[commodity] = forecast
            except (RuntimeError, np.linalg.LinAlgError) as e:
                warnings_list.append(f"{commodity}: {str(e)[:60]}")
                continue

        # Long: highest score; Short: lowest score (always trade)
        long_best = None
        short_best = None
        if scores:
            long_best = max(scores, key=lambda c: scores[c])
            short_best = min(scores, key=lambda c: scores[c])

        long_net_ret = 0.0
        short_net_ret = 0.0

        if long_best is not None:
            d = month_data.get(trade_month, {}).get(long_best, {})
            if d.get('available', False):
                long_net_ret = compute_net_long_return(d['simple_return'], cost)

        if short_best is not None:
            d = month_data.get(trade_month, {}).get(short_best, {})
            if d.get('available', False):
                short_net_ret = compute_net_short_return(d['simple_return'], cost)

        # Benchmark
        avail_rets = [
            month_data[trade_month][c]['simple_return']
            for c in commodities
            if c in month_data.get(trade_month, {}) and month_data[trade_month][c]['available']
        ]
        benchmark_net = (1 + np.mean(avail_rets)) * (1 - cost) ** 2 - 1 if avail_rets else 0.0

        results.append({
            'month': trade_month,
            'long_commodity': long_best,
            'long_score': scores.get(long_best),
            'long_net_return': long_net_ret,
            'short_commodity': short_best,
            'short_score': scores.get(short_best),
            'short_net_return': short_net_ret,
            'benchmark_net_return': benchmark_net,
            'n_scored': len(scores),
            'warnings': '; '.join(warnings_list) if warnings_list else None,
        })

    return pd.DataFrame(results)
