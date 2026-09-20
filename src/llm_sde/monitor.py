"""Streaming, finite-memory, explainable *heuristic* diagnostics. No event labels accepted."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Mapping
import json
import pandas as pd
from .features import CausalFeatures
from .io import coerce_record

METHODS = ("multi_signal", "loss_z", "grad_z", "ewma_loss", "cusum_loss")


@dataclass(frozen=True)
class MonitorConfig:
    window: int = 64
    threshold: float = 0.70
    persistence: int = 2
    cooldown: int = 20  # optimizer-step units, NOT row indexes
    method: str = "multi_signal"

    def __post_init__(self):
        if self.window < 8 or self.persistence < 1 or self.cooldown < 0:
            raise ValueError("Invalid window, persistence or cooldown")
        if not 0 < self.threshold < 1:
            raise ValueError("threshold must be in (0, 1)")
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")

    @classmethod
    def load(cls, path: str | Path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict:
        return asdict(self)


class ScoreState:
    def __init__(self):
        self.ewma = 0.0
        self.cusum = 0.0

    def update(self, features: dict) -> dict[str, Any]:
        f = features
        if f["hard_failure"]:
            return {**{m: 1.0 for m in METHODS}, "factors": ["nonfinite_loss_or_gradient"]}
        if not f["ready"]:
            return {**{m: 0.0 for m in METHODS}, "factors": []}
        self.ewma = 0.8 * self.ewma + 0.2 * max(min(f["loss_z"], 20), -20)
        self.cusum = max(0.0, min(30.0, self.cusum + min(f["loss_z"], 10) - 0.75))
        evidence = {
            "loss_level_or_jump": max(f["loss_z"] / 6, f["loss_diff_z"] / 8, 0),
            "gradient_norm_rise": max(f["grad_z"] / 6, 0),
            "learning_rate_shift": max(f["lr_jump"] / 2, 0),
            "measured_update_ratio_rise": max(f["update_z"] / 6, 0),
            "gradient_volatility_rise": max(f["grad_volatility"] / 3, 0),
        }
        squash = lambda z: float(-math.expm1(-min(max(z, 0), 30)))
        ranked = sorted(evidence, key=evidence.get, reverse=True)
        return {
            "multi_signal": squash(max(evidence.values())),
            "loss_z": squash(max(f["loss_z"], f["loss_diff_z"] * .75, 0) / 6),
            "grad_z": squash(max(f["grad_z"], 0) / 6),
            "ewma_loss": squash(max(self.ewma, 0) / 3),
            "cusum_loss": squash(self.cusum / 12),
            "factors": [k for k in ranked[:3] if evidence[k] >= 0.25],
        }


class AlertGate:
    """One alarm per above-threshold episode; persistence and cooldown are explicit."""
    def __init__(self, config: MonitorConfig):
        self.config = config
        self.streak = 0
        self.active = False
        self.last_alert = -10**18
        self.in_failure = False

    def update(self, step: int, score: float, ready: bool, hard: bool = False) -> bool:
        if hard:
            alert = not self.in_failure
            self.in_failure = True
            self.streak = 0
            self.active = False
            if alert:
                self.last_alert = step
            return alert
        self.in_failure = False
        if not ready or score < self.config.threshold:
            self.streak = 0
            self.active = False
            return False
        self.streak += 1
        if (not self.active and self.streak >= self.config.persistence
                and step - self.last_alert >= self.config.cooldown):
            self.active = True
            self.last_alert = step
            return True
        return False


class OnlineMonitor:
    def __init__(self, config: MonitorConfig | None = None):
        self.config = config or MonitorConfig()
        self.features = CausalFeatures(self.config.window)
        self.scorer = ScoreState()
        self.gate = AlertGate(self.config)
        self.last_step: int | None = None
        self.run_id: str | None = None
        self.phase: str | None = None

    def update(self, record: Mapping[str, Any]) -> dict[str, Any]:
        r = coerce_record(record)
        if self.last_step is not None and r["step"] <= self.last_step:
            raise ValueError("step must increase strictly; create a new monitor for a restarted run")
        if self.run_id is not None and self.run_id != r["run_id"]:
            raise ValueError("run_id changed; use a separate monitor for each run")
        phase_reset = self.phase is not None and r["phase"] != self.phase
        if phase_reset:
            self.features = CausalFeatures(self.config.window)
            self.scorer = ScoreState()
            self.gate = AlertGate(self.config)
        self.phase, self.run_id, self.last_step = r["phase"], r["run_id"], r["step"]
        f = self.features.update(r)
        scores = self.scorer.update(f)
        risk = scores[self.config.method]
        alert = self.gate.update(r["step"], risk, f["ready"], f["hard_failure"])
        status = ("NONFINITE" if f["hard_failure"] else "WARMUP" if not f["ready"]
                  else "ALERT" if alert else "ELEVATED" if risk >= self.config.threshold else "NORMAL")
        return {"step": r["step"], "risk_score": risk, "stability_score": 100 * (1 - risk),
                "alert": alert, "status": status, "phase_reset": phase_reset,
                "factors": scores["factors"], **f,
                **{f"score_{m}": scores[m] for m in METHODS}}

    def replay(self, frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame([self.update(r) for r in frame.to_dict("records")])


def apply_gate(scored: pd.DataFrame, config: MonitorConfig) -> pd.DataFrame:
    """Replay cached causal scores for validation-only threshold selection."""
    gate = AlertGate(config)
    out = scored.copy()
    out["risk_score"] = out[f"score_{config.method}"]
    alarms = []
    for r in out.to_dict("records"):
        if r.get("phase_reset", False):
            gate = AlertGate(config)
        alarms.append(gate.update(int(r["step"]), float(r["risk_score"]), bool(r["ready"]), bool(r["hard_failure"])))
    out["alert"] = alarms
    out["stability_score"] = 100 * (1 - out["risk_score"])
    out["status"] = ["NONFINITE" if r["hard_failure"] else "WARMUP" if not r["ready"] else "ALERT" if r["alert"] else "ELEVATED" if r["risk_score"] >= config.threshold else "NORMAL" for r in out.to_dict("records")]
    return out
