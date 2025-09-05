#!/usr/bin/env python
# theta_last_point_bench.py
"""
Last-point one-step forecast benchmark for Theta implementations.

For each n in a list, repeat m times:
  - Generate a random series y of length n
  - Train on y[:-1]
  - Forecast y[n-1]
  - Record mean squared error (MSE) and runtime (fit+forecast)

Also performs a one-off warm-up forecast for each backend before timing.

Plots:
  1) MSE vs n
  2) Average runtime vs n (log scale)
  3) Last generated series
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Correct import for your Theta
from aeon.forecasting.stats import Theta
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsforecast import StatsForecast
from statsforecast.models import Theta as SFTheta

# ===================== Config =====================
LENGTHS = list(range(2000, 20001, 2000))
REPEATS = 50
GENERATOR = "trend"   # "constant", "trend", "rw_drift"
SEED = 42
WARMUP_LEN = 64

os.environ.setdefault("STATSFORECAST_BACKEND", "pandas")

# ===================== Generators =====================
def gen_constant(n, level=10.0, sigma=0.0, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    return level + sigma * rng.standard_normal(n)

def gen_trend(n, a=3.0, b=0.2, sigma=0.5, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    t = np.arange(n, dtype=float)
    return a + b * t + sigma * rng.standard_normal(n)

def gen_rw_drift(n, drift=0.1, sigma=0.5, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    eps = sigma * rng.standard_normal(n)
    x = np.empty(n, dtype=float)
    x[0] = eps[0]
    for i in range(1, n):
        x[i] = x[i - 1] + drift + eps[i]
    return x

GENS = {
    "constant": gen_constant,
    "trend": gen_trend,
    "rw_drift": gen_rw_drift,
}

def mse1(y_true, y_pred):
    e = float(y_true - y_pred)
    return e * e

# ===================== Backend wrappers =====================
def one_step_ours(train: np.ndarray) -> tuple[float, float]:
    start = time.perf_counter()
    model = Theta().fit(train)
    pred = model.forecast_
    elapsed = time.perf_counter() - start
    return pred, elapsed

def one_step_statsmodels(train: np.ndarray) -> tuple[float, float]:
    start = time.perf_counter()
    tm = ThetaModel(train, period=None, deseasonalize=False)
    res = tm.fit()
    pred = float(np.asarray(res.forecast(1), dtype=float)[0])
    elapsed = time.perf_counter() - start
    return pred, elapsed

def one_step_statsforecast(train: np.ndarray) -> tuple[float, float]:
    # For warm-up ONLY
    n = int(train.shape[0])
    df = pd.DataFrame({
        "unique_id": ["series_1"] * n,
        "ds": pd.date_range("2000-01-01", periods=n, freq="D"),
        "y": train.astype(float),
    })
    sf = StatsForecast(models=[SFTheta(season_length=1)], freq="D", n_jobs=1)
    start = time.perf_counter()
    out = sf.forecast(1, df)
    elapsed = time.perf_counter() - start
    out = out.sort_values(["unique_id", "ds"])
    col = [c for c in out.columns if c not in ("unique_id", "ds")][0]
    pred = float(out[col].to_numpy(dtype=float)[0])
    return pred, elapsed

# ===================== Warm-up =====================
def warm_up(gen_fn, warmup_len: int, seed: int = 1234):
    print(f"Warming up backends with length={warmup_len}...", flush=True)
    rng = np.random.default_rng(seed)
    y = gen_fn(warmup_len, rng=rng).astype(float)
    train = y[:-1]
    _ = one_step_ours(train)
    _ = one_step_statsmodels(train)
    _ = one_step_statsforecast(train)

# ===================== Evaluation =====================
def evaluate_last_point(lengths, repeats, gen_name, seed, warmup_len):
    rng = np.random.default_rng(seed)
    gen_fn = GENS[gen_name]
    warm_up(gen_fn, warmup_len)

    libs = {
        "aeon": one_step_ours,
        "statsmodels": one_step_statsmodels,
    }

    mse = {"aeon": [], "statsmodels": [], "statsforecast": []}
    avg_time = {"aeon": [], "statsmodels": [], "statsforecast": []}
    last_series = None

    sf = StatsForecast(models=[SFTheta(season_length=1)], freq="D", n_jobs=1) # set `n_jobs=-1` to use all cores
    
    for n in lengths:
        print(f"\nProcessing series length n={n} ({repeats} repeats)...", flush=True)
        ys = [gen_fn(n, rng=rng).astype(float) for _ in range(repeats)]
        trains = [y[:-1] for y in ys]
        truths = [float(y[-1]) for y in ys]
        last_series = ys[-1]  

        # aeon and statsmodels 
        errs_aeon, times_aeon = [], []
        errs_sm, times_sm = [], []

        for train, true_last in zip(trains, truths):
            pred, t = libs["aeon"](train)
            errs_aeon.append(mse1(true_last, pred))
            times_aeon.append(t)

            pred, t = libs["statsmodels"](train)
            errs_sm.append(mse1(true_last, pred))
            times_sm.append(t)

        # StatsForecast
        ids = [f"series_{i}" for i in range(repeats)]
        df_parts = []
        for uid, train in zip(ids, trains):
            m = train.shape[0]
            df_parts.append(pd.DataFrame({
                "unique_id": [uid] * m,
                "ds": pd.date_range("2000-01-01", periods=m, freq="D"),
                "y": train.astype(float),
            }))
        df_all = pd.concat(df_parts, ignore_index=True)

        start = time.perf_counter()
        out = sf.forecast(1, df_all)  
        total_elapsed = time.perf_counter() - start
        per_series_time = total_elapsed / repeats

        fc_col = [c for c in out.columns if c not in ("unique_id", "ds")][0]
        
        errs_sf = []
        for uid, true_last in zip(ids, truths):
            pred = float(out[out["unique_id"] == uid][fc_col].iloc[0])
            errs_sf.append(mse1(true_last, pred))

        aeon_mse = float(np.mean(errs_aeon))
        sm_mse = float(np.mean(errs_sm))
        sf_mse = float(np.mean(errs_sf))

        aeon_time = float(np.mean(times_aeon))
        sm_time = float(np.mean(times_sm))
        sf_time = float(per_series_time)

        mse["aeon"].append(aeon_mse)
        mse["statsmodels"].append(sm_mse)
        mse["statsforecast"].append(sf_mse)

        avg_time["aeon"].append(aeon_time)
        avg_time["statsmodels"].append(sm_time)
        avg_time["statsforecast"].append(sf_time)

        print(f"{'aeon':13s}  MSE={aeon_mse:.6f}  Avg time={aeon_time:.6e}s")
        print(f"{'statsmodels':13s}  MSE={sm_mse:.6f}  Avg time={sm_time:.6e}s")
        print(f"{'statsforecast':13s}  MSE={sf_mse:.6f}  Avg time={sf_time:.6e}s (batched)")

    return mse, avg_time, last_series

# ===================== Plotting =====================
def plot_results(lengths, mse, avg_time, last_series):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for name, vals in mse.items():
        plt.plot(lengths, vals, marker="o", label=name)
    plt.xlabel("Series length n")
    plt.ylabel("Mean Squared Error")
    plt.title("MSE vs series length")
    plt.legend()

    plt.subplot(1, 2, 2)
    for name, vals in avg_time.items():
        plt.plot(lengths, vals, marker="o", label=name)
    plt.xlabel("Series length n")
    plt.ylabel("Average runtime (s, log scale)")
    plt.yscale("log")
    plt.title("Runtime vs series length")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ===================== Main =====================
if __name__ == "__main__":
    mse, avg_time, last_series = evaluate_last_point(LENGTHS, REPEATS, GENERATOR, SEED, WARMUP_LEN)
    plot_results(LENGTHS, mse, avg_time, last_series)
