"""Controlled telemetry, NOT measurements from an LLM. Labels remain outside the logs."""
from __future__ import annotations
import numpy as np
import pandas as pd

SCENARIOS = ("normal", "grad_ramp", "lr_ramp", "abrupt_loss", "benign_grad", "benign_lr")


def generate_run(seed: int = 0, scenario: str = "grad_ramp", steps: int = 700):
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    if steps < 300:
        raise ValueError("Synthetic scenarios need at least 300 steps")
    rng = np.random.default_rng(seed)
    t = np.arange(steps)
    loss = 1.0 + 1.2 * np.exp(-t / 350) + rng.normal(0, 0.015, steps)
    noise = rng.normal(0, 0.055, steps)
    for i in range(1, steps):
        noise[i] += .35 * noise[i - 1]
    grad = (1.0 + .4 * np.exp(-t / 500)) * np.exp(noise)
    lr = np.full(steps, 0.001)
    onset = int(rng.integers(int(steps * .55), int(steps * .70)))
    end = onset + 19
    precursor = 35
    if scenario in ("grad_ramp", "benign_grad"):
        grad[onset-precursor:onset] *= np.exp(np.linspace(0, 1.8, precursor))
        grad[onset:onset+20] *= np.exp(np.linspace(1.8, 0, 20))
    if scenario in ("lr_ramp", "benign_lr"):
        lr[onset-precursor:onset] *= np.exp(np.linspace(0, np.log(16), precursor))
        lr[onset:onset+20] *= np.exp(np.linspace(np.log(16), 0, 20))
        grad[onset-precursor:onset+20] *= 1.25
    events = []
    if scenario in ("grad_ramp", "lr_ramp", "abrupt_loss"):
        loss[onset:end+1] += 1.5 * np.exp(-np.arange(20) / 15)
        if scenario == "abrupt_loss":
            grad[onset:end+1] *= 2
        events = [{"onset": onset, "end": end, "kind": "injected_loss_spike"}]
    # Simulated scalar update signal. Only the PyTorch demo measures actual parameter changes.
    update = lr * grad * np.exp(rng.normal(0, .03, steps))
    run_id = f"synthetic-{scenario}-{seed}"
    frame = pd.DataFrame({"step": t, "loss": loss, "grad_norm": grad,
                          "learning_rate": lr, "update_norm": update,
                          "param_norm": np.full(steps, 10.0), "run_id": run_id})
    meta = {"run_id": run_id, "data_kind": "synthetic_telemetry", "scenario": scenario,
            "seed": seed, "steps": steps, "events": events,
            "precursor_start": onset-precursor if scenario not in ("normal", "abrupt_loss") else None,
            "warning": "The generator intentionally includes obvious precursors AND harmless lookalikes; no claim of LLM realism."}
    return frame, meta
