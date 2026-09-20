"""Features at t compare the current observation ONLY with observations before t."""
from __future__ import annotations
from collections import deque
import math
import numpy as np


def robust_z(value: float, history: np.ndarray, floor: float) -> float:
    center = float(np.median(history))
    scale = max(1.4826 * float(np.median(np.abs(history - center))), floor)
    return float(np.clip((value - center) / scale, -100.0, 100.0))


class CausalFeatures:
    def __init__(self, window: int = 64):
        if window < 8:
            raise ValueError("window must be at least 8 observations")
        self.window = window
        self.history: deque[dict] = deque(maxlen=window)

    def update(self, record: dict) -> dict:
        r = record
        hard = not (math.isfinite(r["loss"]) and math.isfinite(r["grad_norm"]))
        result = {"ready": len(self.history) >= self.window, "hard_failure": hard,
                  "loss_z": 0.0, "loss_diff_z": 0.0, "grad_z": 0.0,
                  "lr_jump": 0.0, "update_z": 0.0, "grad_volatility": 0.0,
                  "update_available": False, "step_gap": 0}
        if hard:
            return result  # Do not contaminate future reference windows.
        if self.history:
            result["step_gap"] = r["step"] - self.history[-1]["step"]
        if result["ready"]:
            hist = list(self.history)
            loss = np.array([x["loss"] for x in hist])
            grad = np.log(np.maximum([x["grad_norm"] for x in hist], 1e-12))
            lr = np.array([x["learning_rate"] for x in hist])
            gaps = np.diff([x["step"] for x in hist])
            loss_floor = max(0.005 * abs(float(np.median(loss))), 1e-6)
            result["loss_z"] = robust_z(r["loss"], loss, loss_floor)
            result["loss_diff_z"] = robust_z(
                (r["loss"] - loss[-1]) / result["step_gap"],
                np.diff(loss) / gaps, max(loss_floor / 2, 1e-6))
            result["grad_z"] = robust_z(math.log(max(r["grad_norm"], 1e-12)), grad, 0.05)
            base_lr = float(np.median(lr))
            # A restart from zero LR is capped and explicitly treated as a schedule change.
            ratio = max(r["learning_rate"], 1e-12) / max(base_lr, 1e-12)
            result["lr_jump"] = float(np.clip(math.log2(ratio), -20, 20))
            recent = np.r_[grad[-7:], math.log(max(r["grad_norm"], 1e-12))]
            result["grad_volatility"] = float(max(np.std(recent) / max(np.std(grad), 0.05) - 1, 0))
            pairs = [(x["update_norm"], x["param_norm"]) for x in hist]
            valid = [(u, p) for u, p in pairs if math.isfinite(u) and math.isfinite(p) and p > 0]
            if (math.isfinite(r["update_norm"]) and math.isfinite(r["param_norm"])
                    and r["param_norm"] > 0 and len(valid) >= self.window // 2):
                past = np.log([max(u / p, 1e-12) for u, p in valid])
                current = math.log(max(r["update_norm"] / r["param_norm"], 1e-12))
                result["update_z"] = robust_z(current, past, 0.05)
                result["update_available"] = True
        self.history.append(dict(r))
        return result
