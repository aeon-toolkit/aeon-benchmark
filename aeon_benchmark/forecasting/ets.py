import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple

# -----------------------------
# Spec and presets
# -----------------------------

@dataclass
class ETSSpec:
    """Specification of an ETS model for simulation."""
    error: str = "A"          # "A" (additive) or "M" (multiplicative)
    trend: str = "N"          # "N" (none), "A" (additive), "Ad" (additive damped), "M" (multiplicative), "Md" (multiplicative damped)
    seasonal: str = "N"       # "N" (none), "A" (additive), "M" (multiplicative)
    m: int = 1                # seasonal period; set >1 if seasonal != "N"
    alpha: float = 0.2        # smoothing for level
    beta: float = 0.05        # smoothing for trend (if trend != "N")
    gamma: float = 0.05       # smoothing for seasonality (if seasonal != "N")
    phi: float = 0.98         # damping (if trend == "Ad")
    sigma: float = 1.0        # innovation scale (Std. Dev. of e_t)
    l0: Optional[float] = None
    b0: Optional[float] = None
    s0: Optional[np.ndarray] = None  # length m for seasonal initial states

def make_ets_benchmark_presets() -> List[ETSSpec]:
    """A few sensible presets covering common ETS families."""
    return [
        ETSSpec("A", "N",  "N", 1, 0.2, 0.0, 0.0, 1.0, 1.0),              # ETS(A,N,N)
        ETSSpec("A", "A",  "N", 1, 0.2, 0.1, 0.0, 1.0, 1.0),              # ETS(A,A,N)
        ETSSpec("A", "Ad", "N", 1, 0.2, 0.1, 0.0, 0.9, 1.0),              # ETS(A,Ad,N)
        ETSSpec("A", "N",  "A", 12, 0.2, 0.0, 0.1, 1.0, 1.0),             # ETS(A,N,A), m=12
        ETSSpec("A", "A",  "A", 12, 0.2, 0.1, 0.1, 0.95, 1.0),            # ETS(A,A,A), m=12
        ETSSpec("M", "N",  "N", 1, 0.2, 0.0, 0.0, 1.0, 0.05),             # ETS(M,N,N) small sigma
        ETSSpec("A", "N",  "N", 1, 0.7, 0.0, 0.0, 1.0, 1.0),              # ETS(A,N,N)
        ETSSpec("A", "A",  "N", 1, 0.3, 0.5, 0.0, 1.0, 1.0),              # ETS(A,A,N)
        ETSSpec("A", "Ad", "N", 1, 0.1, 0.7, 0.0, 0.9, 1.0),              # ETS(A,Ad,N)
        ETSSpec("A", "N",  "A", 12, 0.6, 0.0, 0.3, 1.0, 1.0),             # ETS(A,N,A), m=12
        ETSSpec("A", "A",  "A", 12, 0.1, 0.4, 0.6, 0.95, 1.0),            # ETS(A,A,A), m=12
        ETSSpec("M", "N",  "N", 1, 0.2, 0.0, 0.0, 1.0, 0.05),             # ETS(M,N,N) small sigma
        # ETSSpec("M", "A",  "N", 1, 0.2, 0.1, 0.0, 1.0, 0.05),             # ETS(M,A,N)
        # ETSSpec("M", "Ad", "M", 12, 0.15, 0.05, 0.1, 0.9, 0.05),          # ETS(M,Ad,M), m=12
        # ETSSpec("M", "M",  "M", 7, 0.2, 0.1, 0.15, 0.95, 0.5),            # ETS(A,A,M), weekly-ish
        # ETSSpec("M", "M",  "M", 12, 0.2, 0.1, 0.15, 0.95, 0.5),            # ETS(A,A,M), weekly-ish
    ]

# -----------------------------
# Simulation core
# -----------------------------

def _draw_errors(n: int, sigma: float, multiplicative: bool, rng: np.random.Generator):
    """
    Draw innovations e_t.
    For multiplicative error models we use Normal(0, sigma), but clip so (1 + e_t) stays > 0.
    """
    e = rng.normal(0.0, sigma, size=n)
    if multiplicative:
        # Avoid non-positive (1 + e_t) factors which break recursion.
        e = np.clip(e, -0.98, None)
    return e

def simulate_ets(
    spec: ETSSpec,
    n: int = 500,
    burnin: int = 200,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate a univariate time series from an ETS model with
    error ∈ {"A","M"}, trend ∈ {"N","A","Ad","M"}, seasonal ∈ {"N","A","M"}.

    Innovation e_t ~ N(0, sigma). Observation:
        if error == "A": y_t = f_t + e_t
        if error == "M": y_t = f_t * (1 + e_t)

    The update equations are written in a unified way with minimal branching:
      - Build pre-seasonal base u_t from (l, b) according to the trend.
      - Combine with seasonal to get forecast f_t.
      - Draw noise and form y_t.
      - Update (l, b, s) using simple scaling rules that match how each component
        enters the forecast (additively vs multiplicatively).
    """
    E = spec.error.upper()
    T = spec.trend.upper()
    S = spec.seasonal.upper()

    if E not in {"A", "M"}:
        raise ValueError("spec.error must be 'A' or 'M'.")
    if T not in {"N", "A", "AD", "M"}:
        raise ValueError("spec.trend must be 'N', 'A', 'Ad', or 'M'.")
    if S not in {"N", "A", "M"}:
        raise ValueError("spec.seasonal must be 'N', 'A', or 'M'.")
    if S != "N" and spec.m <= 1:
        raise ValueError("spec.m must be > 1 when seasonal != 'N'.")

    rng = np.random.default_rng(seed)
    total = n + burnin
    TINY = 1e-12

    # ---- Initialise states ----
    # Level: positive default if any multiplicative structure is present
    any_mult = (E == "M") or (T == "M") or (S == "M")
    l = (1.0 if spec.l0 is None else float(spec.l0))
    if any_mult:
        l = max(l, TINY)  # enforce positivity if needed

    # Trend
    if T == "N":
        b = 0.0
    elif T in {"A", "AD"}:
        b = 0.0 if spec.b0 is None else float(spec.b0)
    else:  # T == "M"
        b = 1.0 if spec.b0 is None else float(spec.b0)
        b = max(b, TINY)

    # Seasonal vector
    if S == "N":
        m = 1
        s = np.array([0.0], dtype=float)
    else:
        m = int(spec.m)
        if spec.s0 is None:
            s = (np.zeros(m, dtype=float) if S == "A" else np.ones(m, dtype=float))
        else:
            s = np.asarray(spec.s0, dtype=float).copy()
            if s.size != m:
                raise ValueError("spec.s0 must have length m.")
        if S == "A":
            s -= s.mean()  # sum-to-zero
        else:  # S == "M"
            # normalise product to one and enforce positivity
            s = np.maximum(s, TINY)
            g = np.exp(np.mean(np.log(s)))  # geometric mean
            s /= max(g, TINY)

    # Aliases
    alpha = float(spec.alpha)
    beta = float(spec.beta)
    gamma = float(spec.gamma)
    phi = float(spec.phi) if T == "AD" else 1.0
    sigma = float(spec.sigma)

    y = np.empty(total, dtype=float)
    idx = 0  # seasonal index

    for t in range(total):
        # ----- One-step forecast -----
        # Trend part (pre-seasonal base)
        if T == "N":
            b_used = 0.0
            u = l
        elif T in {"A", "AD"}:
            b_used = phi * b
            u = l + b_used
        else:  # T == "M"
            b_used = b
            u = l * b_used
            u = max(u, TINY)

        # Seasonal combination
        s_lag = s[idx] if S != "N" else (0.0 if S == "A" else 1.0)
        if S == "N":
            f = u
        elif S == "A":
            f = u + s_lag
        else:  # S == "M"
            s_lag = max(s_lag, TINY)
            f = u * s_lag

        # ----- Draw error and observe -----
        e = rng.normal(0.0, sigma)
        if E == "A":
            y_t = f + e
            err = y_t - f            # == e
            rel = None               # not used
        else:
            y_t = f * (1.0 + e)
            err = None               # not used
            rel = (y_t / max(f, TINY)) - 1.0  # == e

        y[t] = y_t

        # ----- State updates (compact rules) -----
        if E == "A":
            # Additive error: adjust by 'err' with derivatives as simple divisors.
            # Level
            # derivative w.r.t. l:   d f / d l = {1 if T in {N,A,Ad}, b if T=='M'} * {1 if S in {N,A}, s if S=='M'}
            deriv_l = (1.0 if T in {"N", "A", "AD"} else max(b_used, TINY)) * (1.0 if S in {"N", "A"} else max(s_lag, TINY))
            l_new = u + alpha * (err / deriv_l)

            # Trend
            if T in {"A", "AD"}:
                # d f / d b = {1 if S in {N,A}, s if S=='M'}
                deriv_b = (1.0 if S in {"N", "A"} else max(s_lag, TINY))
                b_new = b_used + beta * (err / deriv_b)
                # convert b_used (phi*b) back to b if damped additive
                b = (b_new / phi) if T == "AD" else b_new
            elif T == "M":
                # d f / d b = {l if S in {N,A}, l*s if S=='M'}
                deriv_b = (max(l, TINY) if S in {"N", "A"} else max(l * s_lag, TINY))
                b = b + beta * (err / deriv_b)
                b = max(b, TINY)
            # if T == "N": no trend update

            # Seasonal
            if S == "A":
                s[idx] = s_lag + gamma * err
            elif S == "M":
                # d f / d s = u
                s[idx] = s_lag + gamma * (err / max(u, TINY))
                s[idx] = max(s[idx], TINY)

            l = l_new
            if any_mult:
                l = max(l, TINY)

        else:
            # Multiplicative error: use relative error 'rel'.
            # Two simple patterns:
            #  - With additive season (S='A'): additive updates scaled by f.
            #  - With multiplicative season (S='M') or none: pre-seasonal components use u;
            #    multiplicative components update multiplicatively (1 + coef*rel),
            #    additive components update additively by coef * u * rel.
            if S == "A":
                # Level
                l = u + alpha * f * rel
                # Trend
                if T in {"A", "AD"}:
                    b = (phi * b) + beta * f * rel
                    if T == "AD":
                        b /= phi  # keep b as the undamped state
                elif T == "M":
                    b = b * (1.0 + beta * rel)
                    b = max(b, TINY)
                # Seasonal
                s[idx] = s_lag + gamma * f * rel
                if any_mult is True:  # keep positivity if any multiplicative structure elsewhere
                    l = max(l, TINY)
            else:
                # S in {'N','M'}: operate with pre-seasonal base u
                # Level always driven by u; multiplicative update for robustness
                l = u * (1.0 + alpha * rel)
                l = max(l, TINY)
                # Trend
                if T in {"A", "AD"}:
                    b = (phi * b) + beta * u * rel
                    if T == "AD":
                        b /= phi
                else:  # T == "M"
                    b = b * (1.0 + beta * rel)
                    b = max(b, TINY)
                # Seasonal
                if S == "M":
                    s[idx] = s_lag * (1.0 + gamma * rel)
                    s[idx] = max(s[idx], TINY)

        # advance seasonal index
        if S != "N":
            idx = (idx + 1) % m

    return y[burnin:]

# -----------------------------
# Convenience helpers
# -----------------------------

def get_random_ets_spec(
    seasonal: str = "N",
    trend: str = "N",
    error: str = "A",
    m: int = 12,
    seed: Optional[int] = None,
) -> ETSSpec:
    """Sample a plausible ETS spec with parameters in stable ranges."""
    rng = np.random.default_rng(seed)
    alpha = rng.uniform(0.05, 0.4)
    beta = 0.0 if trend == "N" else rng.uniform(0.01, min(0.3, alpha))
    gamma = 0.0 if seasonal == "N" else rng.uniform(0.05, 0.3)
    phi = 1.0 if trend != "Ad" else rng.uniform(0.85, 0.99)
    sigma = 1.0 if error == "A" else rng.uniform(0.02, 0.08)

    spec = ETSSpec(
        error=error, trend=trend, seasonal=seasonal, m=m,
        alpha=alpha, beta=beta, gamma=gamma, phi=phi, sigma=sigma
    )
    return spec

def simulate_many_ets(
    specs: List[ETSSpec],
    n: int = 500,
    burnin: int = 200,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Simulate multiple ETS series and return a tidy DataFrame:
    columns: ['model', 't', 'y'].
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i, sp in enumerate(specs):
        sseed = None if seed is None else int(rng.integers(0, 2**31-1))
        series = simulate_ets(sp, n=n, burnin=burnin, seed=sseed)
        rows.append(pd.DataFrame({"model": [f"ETS({sp.error},{sp.trend},{sp.seasonal})-m{sp.m}"]*n,
                                  "t": np.arange(1, n+1),
                                  "y": series}))
    return pd.concat(rows, ignore_index=True)

# -----------------------------
# Quick examples
# -----------------------------
if __name__ == "__main__":
    # Single series: ETS(A,Ad,M), m=12
    spec = ETSSpec("A", "Ad", "M", m=12, alpha=0.2, beta=0.05, gamma=0.1, phi=0.9, sigma=0.8)
    y = simulate_ets(spec, n=300, burnin=300, seed=42)

    # Batch with presets
    presets = make_ets_benchmark_presets()
    df = simulate_many_ets(presets, n=200, burnin=200, seed=123)
    print(df.groupby("model")["y"].describe().round(3))
