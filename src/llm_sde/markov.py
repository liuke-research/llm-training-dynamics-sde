"""Finite-state Markov surrogate: actual transition operators and tilted spectra.

This module analyzes the *fitted finite chain*. It does not prove that telemetry
is Markov, stationary or ergodic, and is not used to manufacture LLM risk probabilities.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


def validate_transition(matrix: np.ndarray) -> np.ndarray:
    p = np.asarray(matrix, dtype=float)
    if p.ndim != 2 or p.shape[0] != p.shape[1] or len(p) < 1:
        raise ValueError("Transition matrix must be nonempty and square")
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-10):
        raise ValueError("Transition matrix must be finite, nonnegative and row-stochastic")
    return p


def stationary_distribution(matrix: np.ndarray) -> np.ndarray:
    p = validate_transition(matrix)
    # Reject nonunique invariant vectors instead of returning an arbitrary least-squares mixture.
    eigenvalues = np.linalg.eigvals(p)
    if int(np.isclose(eigenvalues, 1, atol=1e-9).sum()) != 1:
        raise ValueError("Stationary distribution is not unique")
    a = np.vstack((p.T-np.eye(len(p)), np.ones(len(p))))
    b = np.r_[np.zeros(len(p)), 1.0]
    pi = np.linalg.lstsq(a, b, rcond=None)[0]
    if np.min(pi) < -1e-8:
        raise ValueError("Numerically invalid stationary vector")
    pi = np.maximum(pi, 0)
    return pi/pi.sum()


def absolute_spectral_gap(matrix: np.ndarray) -> float:
    p = validate_transition(matrix)
    if len(p) == 1:
        return 1.0  # Algebraic convention, not empirical evidence of mixing.
    values = np.sort(np.abs(np.linalg.eigvals(p)))[::-1]
    return float(np.clip(1-values[1], 0, 1))


def scgf(matrix: np.ndarray, observable: np.ndarray, k: float) -> float:
    """lambda(k) = log rho(P diag(exp(k*f))); f is evaluated at arrival states."""
    p = validate_transition(matrix)
    f = np.asarray(observable, dtype=float)
    if f.shape != (len(p),) or not np.isfinite(f).all() or not math.isfinite(k):
        raise ValueError("Invalid observable or tilt")
    shifts = k*f
    offset = float(shifts.max())
    tilted = p*np.exp(shifts-offset)[None,:]
    rho = float(np.max(np.abs(np.linalg.eigvals(tilted))))
    if rho <= 0:
        raise FloatingPointError("Degenerate tilted operator")
    return math.log(rho)+offset


def rate_function(matrix: np.ndarray, observable: np.ndarray, a: float,
                  tilts: np.ndarray | None = None) -> dict:
    """Finite-grid Legendre approximation for a scalar additive observable, NOT a level-2 DV solver."""
    grid = np.linspace(-12,12,481) if tilts is None else np.asarray(tilts, dtype=float)
    f = np.asarray(observable, dtype=float)
    if grid.ndim != 1 or len(grid) < 3 or not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0):
        raise ValueError("tilts must be a finite strictly increasing grid")
    if not math.isfinite(a) or not f.min() <= a <= f.max():
        raise ValueError("a must lie in the range of the observable")
    values = np.array([k*a-scgf(matrix, f, float(k)) for k in grid])
    index = int(values.argmax())
    return {"a": a, "rate_grid_lower_approximation": max(0.0,float(values[index])),
            "maximizing_tilt": float(grid[index]), "optimum_at_grid_boundary": index in (0,len(grid)-1),
            "interpretation": "Asymptotic scalar-observable rate of the fitted finite chain; not a calibrated LLM failure probability."}


def finite_horizon_tail(matrix: np.ndarray, risky_states: np.ndarray, horizon: int,
                        min_count: int, initial: np.ndarray | None = None) -> float:
    """Exact DP under the supplied chain for P(sum_{t=1}^n 1{X_t in R} >= min_count)."""
    p = validate_transition(matrix)
    mask = np.asarray(risky_states)
    if mask.shape != (len(p),) or not np.isin(mask,[0,1,False,True]).all():
        raise ValueError("risky_states must be a binary vector")
    mask = mask.astype(int)
    if not isinstance(horizon, int) or not isinstance(min_count, int) or not 1 <= horizon <= 2000 or not 0 <= min_count <= horizon:
        raise ValueError("Require 1<=horizon<=2000 and 0<=min_count<=horizon")
    pi = stationary_distribution(p) if initial is None else np.asarray(initial,dtype=float)
    if pi.shape != (len(p),) or not np.isfinite(pi).all() or (pi < 0).any() or not np.isclose(pi.sum(),1):
        raise ValueError("initial must be a probability vector")
    dp = np.zeros((len(p), horizon+1))
    dp[:,0] = pi
    for t in range(horizon):
        arrivals = p.T@dp[:,:t+1]
        nxt = np.zeros_like(dp)
        for state in range(len(p)):
            nxt[state,mask[state]:mask[state]+t+1] = arrivals[state]
        dp = nxt
    return float(np.clip(dp[:,min_count:].sum(),0,1))


@dataclass
class QuantileChain:
    transition: np.ndarray
    edges: np.ndarray
    counts: np.ndarray
    pseudocount: float
    empirical_occupancy: np.ndarray

    @classmethod
    def fit(cls, calibration: np.ndarray, bins: int = 8, pseudocount: float = .01):
        x = np.asarray(calibration,dtype=float)
        if x.ndim != 1 or len(x) < 40 or not np.isfinite(x).all() or bins < 2 or pseudocount <= 0:
            raise ValueError("Need >=40 finite values, bins>=2 and positive pseudocount")
        if float(np.ptp(x)) < 1e-12:
            raise ValueError("Constant observations cannot identify a nontrivial dynamic model")
        edges = np.unique(np.quantile(x,np.linspace(0,1,bins+1)[1:-1]))
        states = np.searchsorted(edges,x,side="right")
        size = len(edges)+1
        counts = np.zeros((size,size))
        np.add.at(counts,(states[:-1],states[1:]),1)
        regularized = counts+pseudocount
        p = regularized/regularized.sum(axis=1,keepdims=True)
        occupancy = np.bincount(states,minlength=size)/len(states)
        return cls(p,edges,counts,pseudocount,occupancy)

    def encode(self, observations: np.ndarray) -> np.ndarray:
        x = np.asarray(observations,dtype=float)
        if not np.isfinite(x).all():
            raise ValueError("Observations must be finite")
        return np.searchsorted(self.edges,x,side="right")

    def diagnostics(self, heldout: np.ndarray | None = None) -> dict:
        pi = stationary_distribution(self.transition)
        result = {"states": len(pi), "transition_matrix": self.transition.tolist(),
                  "stationary_distribution_of_fitted_chain": pi.tolist(),
                  "empirical_occupancy_of_calibration_window": self.empirical_occupancy.tolist(),
                  "absolute_spectral_gap_of_fitted_chain": absolute_spectral_gap(self.transition),
                  "pseudocount": self.pseudocount,
                  "raw_row_transition_counts": self.counts.sum(axis=1).tolist(),
                  "warning": "Positive pseudocount enforces irreducibility of the fitted chain; this is NOT evidence that the underlying process satisfies Harris conditions."}
        if heldout is not None:
            z = self.encode(heldout)
            hist = np.bincount(z,minlength=len(pi))/len(z)
            mid = .5*(hist+self.empirical_occupancy)
            def kl(a,b):
                positive = a>0
                return float(np.sum(a[positive]*np.log(a[positive]/b[positive])))
            result["heldout_vs_calibration_JS_divergence"] = .5*(kl(hist,mid)+kl(self.empirical_occupancy,mid))
            if len(z) > 1:
                result["heldout_transition_logloss"] = float(-np.log(self.transition[z[:-1],z[1:]]).mean())
        return result
