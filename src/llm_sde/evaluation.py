"""Strict, event-level early-warning metrics. Onset is never provided to a monitor."""
from __future__ import annotations
import numpy as np
import pandas as pd


def evaluate(scored: pd.DataFrame, events: list[dict], horizon: int = 40, detection_window: int = 10) -> dict:
    if horizon < 1 or detection_window < 0:
        raise ValueError("Invalid evaluation horizon")
    steps = scored["step"].to_numpy(dtype=int)
    if len(steps) == 0 or np.any(np.diff(steps) <= 0):
        raise ValueError("Evaluation requires a nonempty strictly increasing step axis")
    events = sorted(events, key=lambda x: x["onset"])
    previous_end = -1
    for e in events:
        if not (steps[0] <= e["onset"] <= e["end"] <= steps[-1]) or e["onset"] <= previous_end:
            raise ValueError("Events must be in-range, disjoint and onset <= end")
        previous_end = e["end"]
    alarms = scored.loc[scored["alert"].astype(bool), "step"].astype(int).tolist()
    used = set()
    leads = []
    matches = []
    detected_after = 0
    for e in events:
        onset = int(e["onset"])
        candidates = [a for a in alarms if a not in used and onset-horizon <= a < onset]
        if candidates:
            a = min(candidates)
            used.add(a)
            leads.append(onset - a)
            matches.append({"onset": onset, "alarm_step": a, "lead_steps": onset-a})
        if any(onset <= a <= min(onset+detection_window, e["end"]) for a in alarms):
            detected_after += 1
    # False alarm rate is measured on observed, eligible healthy records, not unseen steps.
    healthy = scored["ready"].to_numpy(dtype=bool).copy()
    for e in events:
        healthy &= ~((steps >= e["onset"]-horizon) & (steps <= e["end"]))
    healthy_steps = set(steps[healthy].tolist())
    false_healthy = sum(a in healthy_steps for a in alarms)
    return {"events": len(events), "early_hits": len(leads), "alarms": len(alarms),
            "post_onset_hits": detected_after, "healthy_records": int(healthy.sum()),
            "false_healthy_alarms": int(false_healthy), "lead_steps": leads,
            "early_event_recall": len(leads)/len(events) if events else None,
            "early_alarm_precision": len(leads)/len(alarms) if alarms else None,
            "post_onset_detection_recall": detected_after/len(events) if events else None,
            "false_alerts_per_1000_healthy_records": 1000*false_healthy/healthy.sum() if healthy.any() else None,
            "median_lead_steps": float(np.median(leads)) if leads else None,
            "matches": matches, "alarm_steps": alarms, "horizon_steps": horizon}


def aggregate(records: list[dict]) -> dict:
    keys = ("events", "early_hits", "alarms", "post_onset_hits", "healthy_records", "false_healthy_alarms")
    totals = {k: sum(r[k] for r in records) for k in keys}
    leads = [v for r in records for v in r["lead_steps"]]
    totals.update({
        "runs": len(records),
        "early_event_recall": totals["early_hits"] / totals["events"] if totals["events"] else None,
        "early_alarm_precision": totals["early_hits"] / totals["alarms"] if totals["alarms"] else None,
        "post_onset_detection_recall": totals["post_onset_hits"] / totals["events"] if totals["events"] else None,
        "false_alerts_per_1000_healthy_records": 1000 * totals["false_healthy_alarms"] / totals["healthy_records"] if totals["healthy_records"] else None,
        "median_lead_steps": float(np.median(leads)) if leads else None,
    })
    return totals


def bootstrap_runs(records: list[dict], repetitions: int = 1000, seed: int = 2026) -> dict:
    if not records or repetitions < 10:
        raise ValueError("Need runs and at least 10 bootstrap repetitions")
    rng = np.random.default_rng(seed)
    metrics = ("early_event_recall", "early_alarm_precision", "false_alerts_per_1000_healthy_records")
    values = {k: [] for k in metrics}
    for _ in range(repetitions):
        summary = aggregate([records[i] for i in rng.integers(0, len(records), len(records))])
        for k in metrics:
            if summary[k] is not None:
                values[k].append(summary[k])
    return {k: np.quantile(v, [.025, .975]).tolist() if v else None for k, v in values.items()}


def observed_loss_events(frame: pd.DataFrame, window: int = 64, min_rise: float = .7,
                         ratio: float = 1.8, persistence: int = 2) -> list[dict]:
    """Fixed causal loss rule, used ONLY as a labeler of measured training outcomes.

    An event begins at the first sample of a persistent excursion. Endpoints may be
    decided retrospectively; these labels never feed the predictor. This is not a
    universally accepted definition of divergence.
    """
    loss = frame["loss"].to_numpy(dtype=float)
    steps = frame["step"].to_numpy(dtype=int)
    if window < 8 or persistence < 1 or ratio <= 1 or min_rise <= 0:
        raise ValueError("Invalid event labeling parameters")
    flags = np.zeros(len(loss), dtype=bool)
    for i in range(window, len(loss)):
        history = loss[i-window:i]
        history = history[np.isfinite(history)]
        if not np.isfinite(loss[i]):
            flags[i] = True
        elif len(history):
            center = float(np.median(history))
            flags[i] = loss[i] > max(center + min_rise, ratio * center)
    spans = []
    i = 0
    while i < len(flags):
        if not flags[i]:
            i += 1
            continue
        j = i
        while j+1 < len(flags) and flags[j+1]:
            j += 1
        if j-i+1 >= persistence:
            spans.append({"onset": int(steps[i]), "end": int(steps[j]), "kind": "observed_loss_excursion"})
        i = j+1
    # Merge nearby excursions to avoid multiplying labels inside one failure episode.
    merged = []
    for e in spans:
        if merged and e["onset"] - merged[-1]["end"] <= 20:
            merged[-1]["end"] = e["end"]
        else:
            merged.append(e)
    return merged


def bootstrap_seed_clusters(records: list[dict], seeds: list[int], repetitions: int = 1000,
                            random_seed: int = 2026) -> dict:
    """Preserve matched scenarios sharing a simulation seed when bootstrapping."""
    if len(records) != len(seeds) or not records or repetitions < 10:
        raise ValueError("Need equally sized records/seeds and at least 10 repetitions")
    groups = {}
    for record, seed in zip(records, seeds):
        groups.setdefault(int(seed), []).append(record)
    keys = list(groups)
    rng = np.random.default_rng(random_seed)
    metrics = ("early_event_recall", "early_alarm_precision", "false_alerts_per_1000_healthy_records")
    values = {k: [] for k in metrics}
    for _ in range(repetitions):
        sample = [record for i in rng.integers(0,len(keys),len(keys)) for record in groups[keys[i]]]
        total = aggregate(sample)
        for key in metrics:
            if total[key] is not None:
                values[key].append(total[key])
    return {"resampling_unit":"seed_cluster_including_all_matched_scenarios", "clusters":len(keys),
            "repetitions":repetitions,
            "percentile_95_intervals":{k:np.quantile(v,[.025,.975]).tolist() if v else None for k,v in values.items()}}
