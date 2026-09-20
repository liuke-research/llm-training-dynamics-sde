"""Predeclared development/held-out seeds; honest baselines and hard negative controls."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import importlib.metadata
import platform
import time
import numpy as np
import pandas as pd
from .synthetic import SCENARIOS, generate_run
from .monitor import METHODS, MonitorConfig, OnlineMonitor, apply_gate
from .evaluation import evaluate, aggregate, bootstrap_seed_clusters
from .io import write_json


def environment() -> dict:
    packages = {}
    for name in ("numpy", "pandas", "scipy", "matplotlib", "torch", "pytest"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def run_benchmark(out: str | Path, *, dev_seeds=range(4), test_seeds=range(100, 108),
                  steps: int = 700, horizon: int = 40, false_budget: float = 5.0) -> dict:
    dev_seeds, test_seeds = list(dev_seeds), list(test_seeds)
    if set(dev_seeds) & set(test_seeds) or not dev_seeds or not test_seeds:
        raise ValueError("Development and held-out seeds must be nonempty and disjoint")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = MonitorConfig()
    plan = {"development_seeds": dev_seeds, "heldout_seeds": test_seeds, "scenarios": list(SCENARIOS),
            "steps_per_run": steps, "horizon_steps": horizon, "false_alarm_budget_per_1000_healthy_records": false_budget,
            "threshold_grid": [.50, .60, .70, .80, .90], "base_config": cfg.to_dict(),
            "selection": "maximize early recall subject to development false-alarm budget; ties precision then threshold; if infeasible minimize false alarms",
            "bootstrap_unit": "seed clusters, keeping the six matched scenarios together",
            "data_kind": "synthetic_telemetry", "independence": "no training weights are fitted; only alert thresholds are selected on development runs"}
    write_json(out / "protocol.json", plan)
    write_json(out / "environment.json", environment())

    def generate_scores(seeds, split):
        cached = []
        for scenario in SCENARIOS:
            for seed in seeds:
                frame, meta = generate_run(seed, scenario, steps)
                scored = OnlineMonitor(cfg).replay(frame)
                cached.append((scored, meta))
                folder = out / "runs" / split / meta["run_id"]
                folder.mkdir(parents=True, exist_ok=True)
                frame.to_csv(folder / "telemetry.csv", index=False)
                scored.to_csv(folder / "scores.csv", index=False)
                write_json(folder / "labels.json", meta)
        return cached

    development = generate_scores(dev_seeds, "development")
    selected = {}
    calibration_rows = []
    for method in METHODS:
        candidates = []
        for threshold in plan["threshold_grid"]:
            conf = replace(cfg, method=method, threshold=threshold)
            metrics = [evaluate(apply_gate(s, conf), m["events"], horizon) for s, m in development]
            total = aggregate(metrics)
            feasible = total["false_alerts_per_1000_healthy_records"] <= false_budget
            row = {"method": method, "threshold": threshold, "feasible": feasible, **total}
            calibration_rows.append(row)
            candidates.append(row)
        feasible_rows = [r for r in candidates if r["feasible"]]
        if feasible_rows:
            best = max(feasible_rows, key=lambda r: (r["early_event_recall"] or 0,
                        r["early_alarm_precision"] or 0, r["threshold"]))
        else:
            best = min(candidates, key=lambda r: (r["false_alerts_per_1000_healthy_records"], -(r["early_event_recall"] or 0)))
        selected[method] = replace(cfg, method=method, threshold=best["threshold"]).to_dict()
    pd.DataFrame(calibration_rows).to_csv(out / "development_threshold_search.csv", index=False)
    write_json(out / "selected_configs.json", selected)
    # Thresholds are written to disk before ANY held-out data are generated/scored.
    heldout = generate_scores(test_seeds, "heldout")
    summary_rows, per_run_rows, strata_rows, intervals = [], [], [], {}
    for method in METHODS:
        conf = MonitorConfig(**selected[method])
        all_metrics = []
        by_scenario = {s: [] for s in SCENARIOS}
        for scored, meta in heldout:
            evaluated = evaluate(apply_gate(scored, conf), meta["events"], horizon)
            all_metrics.append(evaluated)
            by_scenario[meta["scenario"]].append(evaluated)
            per_run_rows.append({"method": method, "run_id": meta["run_id"],
                                 "scenario": meta["scenario"], **evaluated})
        summary_rows.append({"method": method, "threshold": conf.threshold, **aggregate(all_metrics)})
        intervals[method] = bootstrap_seed_clusters(all_metrics, [m["seed"] for _, m in heldout])
        for scenario, values in by_scenario.items():
            strata_rows.append({"method": method, "scenario": scenario, **aggregate(values)})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "heldout_summary.csv", index=False)
    pd.DataFrame(strata_rows).to_csv(out / "heldout_by_scenario.csv", index=False)
    write_json(out / "heldout_per_run.json", per_run_rows)
    write_json(out / "bootstrap_seed_cluster_intervals.json", intervals)
    write_json(out / "summary.json", summary_rows)
    report = "# Synthetic held-out benchmark\n\nNot a production LLM result. Identical score algorithms and a common threshold grid; development-only selection.\n\n"
    report += summary.to_string(index=False) + "\n\n"
    report += "See heldout_by_scenario.csv, especially abrupt_loss and benign_*; inspecting only the easy ramp scenarios would exaggerate performance.\n"
    (out / "RESULTS.md").write_text(report, encoding="utf-8")
    return {"protocol": plan, "summary": summary_rows}


def benchmark_latency(out: str | Path, steps: int = 3000) -> dict:
    frame, _ = generate_run(2026, "normal", max(steps, 300))
    monitor = OnlineMonitor()
    times = []
    for record in frame.to_dict("records"):
        begin = time.perf_counter_ns()
        result = monitor.update(record)
        if result["ready"]:
            times.append((time.perf_counter_ns()-begin)/1e6)
    report = {"workload": "one CPU process, one run, monitor.update only, no file IO or plotting",
              "records": len(frame), "timed_after_warmup": len(times), "window": monitor.config.window,
              "median_ms_per_record": float(np.median(times)), "p95_ms_per_record": float(np.quantile(times,.95)),
              "mean_ms_per_record": float(np.mean(times)), "environment": environment(),
              "warning": "Microbenchmark only; not GPU training overhead or production capacity."}
    write_json(Path(out)/"monitor_latency.json", report)
    return report
