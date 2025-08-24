"""
option_pricing.py — Single-file module with three pricers for your app:
  1) Black–Scholes (European calls/puts) with analytic Greeks
  2) Binomial CRR tree (European & American, user-selectable)
  3) Advanced American: Longstaff–Schwartz (LSM) Monte Carlo with variance reduction

Conventions
- Time T in years. r and q are continuously-compounded annual rates.
- Volatility sigma is annualized.
- Returns prices and, when available, Greeks. For tree/LSM, Greeks are via finite differences
  with common random numbers (for LSM) to stabilize estimates.

Dependencies: numpy only (no SciPy required).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Literal, Optional, Tuple

import numpy as np

# =========================
# Utilities
# =========================

_EPS = 1e-12
_TWO_PI_SQRT = math.sqrt(2.0 * math.pi)


def _norm_pdf(x: np.ndarray | float) -> np.ndarray | float:
    return math.exp(-0.5 * x * x) / _TWO_PI_SQRT if isinstance(x, (int, float)) else np.exp(-0.5 * x * x) / _TWO_PI_SQRT


def _norm_cdf(x: np.ndarray | float) -> np.ndarray | float:
    # 0.5 * (1 + erf(x/sqrt(2)))
    if isinstance(x, (int, float)):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    return 0.5 * (1.0 + np.erf(x / math.sqrt(2.0)))


def _intrinsic(S: np.ndarray | float, K: float, is_call: bool) -> np.ndarray | float:
    if is_call:
        return np.maximum(S - K, 0.0) if isinstance(S, np.ndarray) else max(S - K, 0.0)
    else:
        return np.maximum(K - S, 0.0) if isinstance(S, np.ndarray) else max(K - S, 0.0)


@dataclass
class OptionSpec:
    S: float        # Spot price
    K: float        # Strike
    T: float        # Time to maturity (years)
    r: float        # Risk-free (cont. comp.)
    q: float        # Dividend yield / convenience yield (cont. comp.)
    sigma: float    # Volatility
    is_call: bool   # True for call, False for put


# =========================
# 1) Black–Scholes (European)
# =========================

def bs_price(spec: OptionSpec) -> float:
    """Black–Scholes price for European call/put with dividend yield q.
    Edge cases handled: T=0, sigma=0.
    """
    S, K, T, r, q, sigma, is_call = spec.S, spec.K, spec.T, spec.r, spec.q, spec.sigma, spec.is_call

    if T <= 0.0 or sigma <= _EPS:
        # Immediate exercise / zero vol limit -> discounted intrinsic for European? Strictly, at T=0 it's intrinsic.
        # For sigma->0 with T>0, BS tends to discounted forward intrinsic. We keep it simple and return discounted expected payoff
        # under risk-neutral drift, but to avoid over-engineering, use intrinsic at T=0 and tiny-vol approach else.
        if T <= 0.0:
            return float(_intrinsic(S, K, is_call))
        # Tiny sigma: approximate with d1 large in magnitude; use standard formulas safely
        sigma = max(sigma, 1e-8)

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)

    if spec.is_call:
        return S * disc_q * _norm_cdf(d1) - K * disc_r * _norm_cdf(d2)
    else:
        return K * disc_r * _norm_cdf(-d2) - S * disc_q * _norm_cdf(-d1)


def bs_greeks(spec: OptionSpec) -> Dict[str, float]:
    """Analytic Greeks for Black–Scholes (per unit change, not per 1%).
    Returns: dict with delta, gamma, vega, theta, rho.
    Theta is per year (same units as r)."""
    S, K, T, r, q, sigma, is_call = spec.S, spec.K, spec.T, spec.r, spec.q, spec.sigma, spec.is_call

    if T <= 0.0:
        # At expiry, Greeks undefined; use limits: delta is 0/1 at-the-money undefined; return zeros except delta approx
        intrinsic = _intrinsic(S, K, is_call)
        return {"price": intrinsic, "delta": float("nan"), "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)

    pdf_d1 = _norm_pdf(d1)

    delta = disc_q * _norm_cdf(d1) if is_call else disc_q * (_norm_cdf(d1) - 1.0)
    gamma = disc_q * pdf_d1 / (S * sigma * math.sqrt(T))
    vega = S * disc_q * pdf_d1 * math.sqrt(T)

    # Theta (per year). Using q for dividends.
    term1 = - (S * disc_q * pdf_d1 * sigma) / (2.0 * math.sqrt(T))
    if is_call:
        theta = term1 - r * K * disc_r * _norm_cdf(d2) + q * S * disc_q * _norm_cdf(d1)
        rho = K * T * disc_r * _norm_cdf(d2)
    else:
        theta = term1 + r * K * disc_r * _norm_cdf(-d2) - q * S * disc_q * _norm_cdf(-d1)
        rho = -K * T * disc_r * _norm_cdf(-d2)

    price = bs_price(spec)
    return {"price": price, "delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


# =========================
# 2) Binomial (CRR) — European & American
# =========================

def binomial_price(
    spec: OptionSpec,
    steps: int = 200,
    american: bool = False,
) -> float:
    """Cox–Ross–Rubinstein binomial tree.

    Args:
        spec: OptionSpec
        steps: number of time steps (>=1)
        american: True for American-style early exercise
    """
    S, K, T, r, q, sigma, is_call = spec.S, spec.K, spec.T, spec.r, spec.q, spec.sigma, spec.is_call

    if T <= 0.0:
        return float(_intrinsic(S, K, is_call))
    if sigma <= _EPS:
        # Degenerate tree; fallback to BS with tiny sigma to avoid division by zero
        return bs_price(OptionSpec(S, K, T, r, q, max(1e-8, sigma), is_call))

    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p = (math.exp((r - q) * dt) - d) / (u - d)

    # Guardrails in extreme params
    p = min(1.0, max(0.0, p))

    # Terminal prices S * u^{j} d^{steps-j}
    j = np.arange(steps + 1)
    S_T = S * (u ** j) * (d ** (steps - j))

    payoffs = _intrinsic(S_T, K, is_call)

    # Backward induction
    for i in range(steps - 1, -1, -1):
        payoffs = disc * (p * payoffs[1:i + 2] + (1.0 - p) * payoffs[0:i + 1])
        if american:
            # Early exercise value at node i
            S_i = S * (u ** np.arange(i + 1)) * (d ** (i - np.arange(i + 1)))
            exercise = _intrinsic(S_i, K, is_call)
            payoffs = np.maximum(payoffs, exercise)

    return float(payoffs[0])


# Generic finite-difference Greeks for any pricer f(spec)->price

def greeks_fd(
    spec: OptionSpec,
    pricer: Callable[[OptionSpec], float],
    bump_rel: Dict[str, float] | None = None,
    common_kwargs: Dict | None = None,
) -> Dict[str, float]:
    """Finite-difference Greeks for a given pricer. Uses central differences when possible.

    bump_rel: relative bump sizes for S, sigma, r, T. Defaults: 1e-4 for S, 1e-4 for sigma, 1e-6 for r, 1e-5 for T.
    common_kwargs: extra kwargs passed to pricer (e.g., steps=, american=, seed=)
    """
    if bump_rel is None:
        bump_rel = {"S": 1e-4, "sigma": 1e-4, "r": 1e-6, "T": 1e-5}
    if common_kwargs is None:
        common_kwargs = {}

    base = pricer(spec, **common_kwargs) if hasattr(pricer, "__call__") else pricer(spec)

    # Delta (dV/dS)
    dS = max(1e-8, spec.S * bump_rel["S"])
    p_up = pricer(OptionSpec(spec.S + dS, spec.K, spec.T, spec.r, spec.q, spec.sigma, spec.is_call), **common_kwargs)
    p_dn = pricer(OptionSpec(spec.S - dS, spec.K, spec.T, spec.r, spec.q, spec.sigma, spec.is_call), **common_kwargs)
    delta = (p_up - p_dn) / (2.0 * dS)

    # Gamma (d2V/dS2)
    gamma = (p_up - 2.0 * base + p_dn) / (dS * dS)

    # Vega (dV/dsigma)
    dsig = max(1e-8, spec.sigma * bump_rel["sigma"]) if spec.sigma > 0 else 1e-4
    p_up = pricer(OptionSpec(spec.S, spec.K, spec.T, spec.r, spec.q, spec.sigma + dsig, spec.is_call), **common_kwargs)
    p_dn = pricer(OptionSpec(spec.S, spec.K, spec.T, spec.r, spec.q, spec.sigma - dsig, spec.is_call), **common_kwargs)
    vega = (p_up - p_dn) / (2.0 * dsig)

    # Rho (dV/dr)
    dr = max(1e-10, abs(spec.r) * bump_rel["r"] + 1e-8)
    p_up = pricer(OptionSpec(spec.S, spec.K, spec.T, spec.r + dr, spec.q, spec.sigma, spec.is_call), **common_kwargs)
    p_dn = pricer(OptionSpec(spec.S, spec.K, spec.T, spec.r - dr, spec.q, spec.sigma, spec.is_call), **common_kwargs)
    rho = (p_up - p_dn) / (2.0 * dr)

    # Theta (dV/dT), per year
    dT = max(1e-8, spec.T * bump_rel["T"]) if spec.T > 0 else 1e-5
    if spec.T > 2.0 * dT:
        p_up = pricer(OptionSpec(spec.S, spec.K, spec.T + dT, spec.r, spec.q, spec.sigma, spec.is_call), **common_kwargs)
        p_dn = pricer(OptionSpec(spec.S, spec.K, spec.T - dT, spec.r, spec.q, spec.sigma, spec.is_call), **common_kwargs)
        theta = (p_up - p_dn) / (2.0 * dT)
    else:
        # Backward difference near expiry
        p_dn = pricer(OptionSpec(spec.S, spec.K, max(0.0, spec.T - dT), spec.r, spec.q, spec.sigma, spec.is_call), **common_kwargs)
        theta = (base - p_dn) / dT

    return {"price": base, "delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


# =========================
# 3) Longstaff–Schwartz (LSM) — Advanced American
# =========================

def _simulate_gbm_paths(
    S0: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    n_paths: int,
    n_steps: int,
    seed: Optional[int] = None,
    antithetic: bool = True,
) -> np.ndarray:
    """Simulate GBM paths under risk-neutral measure. Returns array (n_paths, n_steps+1)."""
    if T <= 0.0:
        return np.full((n_paths, 1), S0, dtype=float)
    if sigma <= _EPS:
        # Deterministic forward under r - q
        dt = T / n_steps
        drift = (r - q) * dt
        path = np.empty((n_paths, n_steps + 1), dtype=float)
        path[:, 0] = S0
        for t in range(n_steps):
            path[:, t + 1] = path[:, t] * math.exp(drift)
        return path

    rng = np.random.default_rng(seed)

    if antithetic:
        # Ensure even number of paths
        half = (n_paths + 1) // 2
        Z = rng.standard_normal((half, n_steps))
        Z = np.vstack([Z, -Z])
        Z = Z[:n_paths]
    else:
        Z = rng.standard_normal((n_paths, n_steps))

    dt = T / n_steps
    nudt = (r - q - 0.5 * sigma * sigma) * dt
    sigsdt = sigma * math.sqrt(dt)

    S = np.empty((n_paths, n_steps + 1), dtype=float)
    S[:, 0] = S0
    for t in range(n_steps):
        S[:, t + 1] = S[:, t] * np.exp(nudt + sigsdt * Z[:, t])
    return S


def lsm_american_price(
    spec: OptionSpec,
    n_paths: int = 100_000,
    n_steps: int = 50,
    basis: Literal["poly", "laguerre"] = "poly",
    degree: int = 3,
    seed: Optional[int] = 12345,
    antithetic: bool = True,
    control_variate: bool = True,
) -> float:
    """Longstaff–Schwartz American option pricer (GBM dynamics).

    - Fits continuation value by regression on in-the-money paths.
    - Basis: polynomial in S (1, S, S^2, ... up to degree) or Laguerre polynomials in S.
    - Variance reduction: antithetic variates; optional European control variate using BS.

    Note: For a non-dividend-paying asset (q=0), an American call equals its European price.
    """
    S, K, T, r, q, sigma, is_call = spec.S, spec.K, spec.T, spec.r, spec.q, spec.sigma, spec.is_call

    if T <= 0.0:
        return float(_intrinsic(S, K, is_call))

    # Simulate paths
    paths = _simulate_gbm_paths(S, r, q, sigma, T, n_paths, n_steps, seed=seed, antithetic=antithetic)
    dt = T / n_steps
    disc = math.exp(-r * dt)

    # Payoff along paths
    pay = _intrinsic(paths, K, is_call)

    # Cashflows and exercise time flags
    cf = pay.copy()
    exercise_time = np.full_like(paths, fill_value=n_steps, dtype=int)  # default: exercise at maturity

    # At maturity t = n_steps, exercise payoff is pay[:, -1]
    cf[:, :-1] = 0.0

    # Helper for basis
    def _features(x: np.ndarray) -> np.ndarray:
        if basis == "poly":
            # [1, x, x^2, ...]
            cols = [np.ones_like(x)]
            for k in range(1, degree + 1):
                cols.append(x ** k)
            return np.column_stack(cols)
        elif basis == "laguerre":
            # First few Laguerre polynomials L0=1, L1=1-x, L2=1-2x+x^2/2, ... on scaled x
            z = x / np.maximum(1.0, np.mean(x))
            L0 = np.ones_like(z)
            L1 = 1.0 - z
            if degree == 1:
                return np.column_stack([L0, L1])
            L2 = 1.0 - 2.0 * z + 0.5 * z * z
            if degree == 2:
                return np.column_stack([L0, L1, L2])
            L3 = 1.0 - 3.0 * z + 1.5 * z * z - (1.0 / 6.0) * z ** 3
            mats = [L0, L1, L2, L3]
            for k in range(4, degree + 1):
                # simple recursion for additional terms (approximate) if degree>3
                mats.append(mats[-1] * (1 - z) / (k))
            return np.column_stack(mats[: degree + 1])
        else:
            raise ValueError("Unknown basis")

    # Backward induction for early exercise
    cashflows = np.zeros_like(paths)
    cashflows[:, -1] = pay[:, -1]
    alive = np.ones(n_paths, dtype=bool)  # paths not yet exercised

    for t in range(n_steps - 1, 0, -1):
        S_t = paths[:, t]
        itm = (pay[:, t] > 0.0) & alive
        if not np.any(itm):
            # No in-the-money alive paths -> just discount future cashflows
            cashflows[alive, t] = 0.0
            cashflows[:, t] += disc * cashflows[:, t + 1]
            continue

        # Discounted future CF for alive paths
        Y = disc * cashflows[:, t + 1]

        # Regression only on ITM & alive
        X = _features(S_t[itm])
        y = Y[itm]
        # Linear regression via least squares: beta = argmin ||X beta - y||^2
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)

        # Predicted continuation for all alive paths (only needed at itm nodes)
        cont = np.zeros(n_paths)
        cont[itm] = _features(S_t[itm]) @ beta

        # Exercise decision
        exercise_val = pay[:, t]
        exercise_now = (exercise_val >= cont) & itm

        # If exercise now, set CF at t and kill future CFs for that path
        cashflows[exercise_now, t] = exercise_val[exercise_now]
        # Ensure future CF for exercised paths are zeroed (they already default to 0 except at maturity)
        # We can mark them as not alive
        alive[exercise_now] = False

        # For remaining alive paths, just carry discounted continuation CF forward implicitly
        cashflows[~exercise_now, t] = 0.0
        cashflows[:, t] += disc * cashflows[:, t + 1]

    # At t=0 we only discount from t=1
    price_mc = math.exp(-r * dt) * float(np.mean(cashflows[:, 1]))

    if control_variate:
        # European control variate using same paths: compute European MC on these paths
        euro_pay = pay[:, -1]
        euro_mc = math.exp(-r * T) * float(np.mean(euro_pay))
        euro_bs = bs_price(OptionSpec(S, K, T, r, q, sigma, is_call))
        # Adjust LSM estimate
        price_mc = price_mc + (euro_bs - euro_mc)

    return price_mc


# Convenience wrappers for Greeks on Binomial and LSM

def binomial_greeks(spec: OptionSpec, steps: int = 200, american: bool = False) -> Dict[str, float]:
    return greeks_fd(spec, binomial_price, common_kwargs={"steps": steps, "american": american})


def lsm_greeks(
    spec: OptionSpec,
    n_paths: int = 100_000,
    n_steps: int = 50,
    basis: Literal["poly", "laguerre"] = "poly",
    degree: int = 3,
    seed: Optional[int] = 12345,
    antithetic: bool = True,
    control_variate: bool = True,
) -> Dict[str, float]:
    common = {
        "n_paths": n_paths,
        "n_steps": n_steps,
        "basis": basis,
        "degree": degree,
        "seed": seed,
        "antithetic": antithetic,
        "control_variate": control_variate,
    }
    return greeks_fd(spec, lsm_american_price, common_kwargs=common)
