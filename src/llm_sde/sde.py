"""Known stochastic-process baselines; not surrogate evidence of LLM convergence."""
from __future__ import annotations
import math
import numpy as np


def _validate(theta: float, sigma: float, dt: float, steps: int):
    if not all(math.isfinite(v) for v in (theta, sigma, dt)) or theta <= 0 or sigma < 0 or dt <= 0 or steps < 1:
        raise ValueError("Need theta>0, sigma>=0, dt>0 and steps>=1")


def ou_process(theta: float = 1.0, mu: float = 0.0, sigma: float = 0.5,
               dt: float = .05, steps: int = 20000, x0: float = 0.0,
               seed: int = 0, scheme: str = "exact") -> np.ndarray:
    """dX = theta(mu-X)dt + sigma dW. Includes x0 at index zero."""
    _validate(theta, sigma, dt, steps)
    if not math.isfinite(mu) or not math.isfinite(x0):
        raise ValueError("mu and x0 must be finite")
    if scheme == "exact":
        a = math.exp(-theta*dt)
        scale = sigma * math.sqrt(-math.expm1(-2*theta*dt)/(2*theta))
    elif scheme == "euler":
        if theta*dt >= 2:
            raise ValueError("OU Euler recursion is not stationary for theta*dt >= 2")
        a, scale = 1-theta*dt, sigma*math.sqrt(dt)
    else:
        raise ValueError("scheme must be exact or euler")
    rng = np.random.default_rng(seed)
    x = np.empty(steps+1)
    x[0] = x0
    for i, z in enumerate(rng.normal(size=steps)):
        x[i+1] = mu + a*(x[i]-mu) + scale*z
    return x


def double_well(sigma: float = .7, dt: float = .005, steps: int = 20000,
                x0: float = -1.0, seed: int = 0, scheme: str = "tamed") -> np.ndarray:
    """dX=(X-X^3)dt+sigma dW. Taming changes the discrete drift, not the SDE.

    A tamed step is used by default because a cubic drift is not globally Lipschitz.
    No claim that a finite-step histogram exactly equals the continuous invariant law.
    """
    _validate(1, sigma, dt, steps)
    if scheme not in ("euler", "tamed") or not math.isfinite(x0):
        raise ValueError("Invalid scheme or x0")
    rng = np.random.default_rng(seed)
    x = np.empty(steps+1)
    x[0] = x0
    for i, z in enumerate(rng.normal(size=steps)):
        value = float(x[i])
        try:
            drift = value - value**3
        except OverflowError as exc:
            raise FloatingPointError("Unstable discretization; reduce dt or use tamed drift") from exc
        if scheme == "tamed":
            drift /= 1+dt*abs(drift)
        x[i+1] = value + drift*dt + sigma*math.sqrt(dt)*z
        if not math.isfinite(x[i+1]) or abs(x[i+1]) > 1e50:
            raise FloatingPointError("Nonfinite/unstable path; reduce dt")
    return x


def double_well_density(grid: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0 or not math.isfinite(sigma):
        raise ValueError("sigma must be positive")
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0):
        raise ValueError("grid must be finite and increasing")
    potential = grid**4/4-grid**2/2
    logp = -2*potential/sigma**2
    p = np.exp(logp-logp.max())
    # NumPy 1.24 compatibility without deprecated np.trapz.
    area = np.sum((p[1:]+p[:-1])*np.diff(grid)/2)
    return p/area


def empirical_diagnostics(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 20 or not np.isfinite(x).all():
        raise ValueError("Need a finite 1-D series with >=20 observations")
    variance = float(np.var(x))
    lag1 = float(np.corrcoef(x[:-1], x[1:])[0,1]) if variance > 1e-15 else None
    return {"mean": float(np.mean(x)), "variance": variance, "lag1_autocorrelation": lag1,
            "first_half_mean": float(np.mean(x[:len(x)//2])),
            "second_half_mean": float(np.mean(x[len(x)//2:])),
            "interpretation": "Empirical diagnostics only; not a Harris drift/minorization proof or certified mixing-time estimate."}
