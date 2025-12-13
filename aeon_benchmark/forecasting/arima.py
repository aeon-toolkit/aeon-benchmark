"""Functions to perform ARIMA benchmarking experiments."""
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import ArmaProcess

# Models used to compare different ARMA implementations.
# Each (p, q) maps to a list of parameterizations (ar_params, ma_params)
stationary_arma_models = {
    (1, 0): [
        ([0.6], []),
        ([0.4], []),  # Stable, low persistence
    ],
    (2, 0): [
        ([0.5, -0.3], []),
        ([1.2, -0.5], []),  # Roots at ~1.15 and 0.72 (unstable), adjust if needed
    ],
    (1, 1): [
        ([0.6], [0.4]),
    ],
    (2, 1): [
        ([0.5, -0.3], [0.4]),
    ],
    (2, 2): [
        ([0.5, -0.3], [0.4, -0.2]),
    ],
    (5, 0): [
        ([0.5, -0.3, 0.2, -0.1, 0.05], []),
        ([0.55, -0.35, 0.25, -0.15, 0.05], []),
    ],
    (0, 5): [
        ([], [-0.2, 0.15, -0.1, 0.07, -0.04]),
        ([], [0.4, -0.2, 0.1, -0.05, 0.03]),
    ],
    (3, 2): [
        ([0.5, -0.3, 0.2], [0.4, -0.2]),
    ],
    (3, 3): [
        ([0.5, -0.3, 0.2], [0.4, -0.2, 0.1]),
    ],
    (4, 1): [
        ([0.5, -0.3, 0.2, -0.1], [0.4]),
    ],
    (4, 2): [
        ([0.5, -0.3, 0.2, -0.1], [0.4, -0.2]),
    ],
    (5, 1): [
        ([0.5, -0.3, 0.2, -0.1, 0.05], [0.4]),
    ],
    (3, 4): [
        ([0.4, -0.25, 0.1], [-0.3, 0.2, -0.1, 0.05]),
    ],
    (1, 2): [
        ([0.5], [-0.4, 0.2]),
    ],
    (2, 3): [
        ([0.6, -0.2], [0.3, -0.2, 0.1]),
    ],
    (3, 1): [
        ([0.6, -0.3, 0.1], [-0.4]),
    ],
}


def get_fixed_stationary_arma_params(para_list=stationary_arma_models, seed=None):
    """
    Return a fixed, stationary ARMA(p, q) parameter set
    randomly chosen from the list para_list.

    Parameters
    ----------
    seed : int or None
        Seed for reproducibility.

    Returns
    -------
    ar_params : list of float
    ma_params : list of float
    p : int
    q : int
    """
    if seed is not None:
        np.random.seed(seed)

    (p, q) = list(para_list.keys())[np.random.randint(len(para_list))]
    ar_params, ma_params = para_list[(p, q)]
    return ar_params, ma_params, p, q



def simulate_arma_data(ar_params, ma_params, n_samples=1000, burnin=50, sigma=1.0,
                       seed=None):
    """
    Simulate ARMA time series data.

    Parameters
    ----------
    ar_params : list or np.ndarray
        AR coefficients (excluding the leading 1).
        E.g., for AR(1): [0.5] corresponds to φ₁ = 0.5.
    ma_params : list or np.ndarray
        MA coefficients (excluding the leading 1).
        E.g., for MA(1): [0.4] corresponds to θ₁ = 0.4.
    n_samples : int
        Number of samples to return (after burn-in).
    burnin : int
        Number of initial samples to discard (to remove start-up bias).
    sigma : float
        Standard deviation of the white noise innovations.
    seed : int or None
        Seed for reproducibility.

    Returns
    -------
    np.ndarray
        Simulated ARMA time series of length `n_samples`.
    """
    if seed is not None:
        np.random.seed(seed)

    # Add leading 1 for AR and MA polynomials
    ar = np.r_[1, -np.array(ar_params)]  # statsmodels uses negative sign for AR
    ma = np.r_[1, np.array(ma_params)]

    arma_process = ArmaProcess(ar, ma)
    simulated = arma_process.generate_sample(nsample=n_samples + burnin, scale=sigma)
    return simulated[burnin:]


def make_statsforecast_df(y):
    return pd.DataFrame({
        "ds": pd.date_range(start="1900-01-01 00:00:00", periods=len(y),freq="H"),
        "y": y,
        "unique_id": "ts"
    })

