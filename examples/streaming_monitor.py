"""Executable streaming API example, using explicitly SYNTHETIC observations.

No model weights are trained here. The training implementation lives in torch_demo.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_sde import MonitorConfig, OnlineMonitor
from llm_sde.io import JSONLLogger, clean_json, write_json
from llm_sde.synthetic import generate_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/streaming_example"))
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    frame, metadata = generate_run(seed=42, scenario="grad_ramp", steps=args.steps)
    output = args.out
    output.mkdir(parents=True, exist_ok=True)
    telemetry_path = output / "telemetry.jsonl"
    scores_path = output / "scores.jsonl"
    if telemetry_path.exists() or scores_path.exists():
        parser.error("Output logs already exist; use a new --out directory.")
    monitor = OnlineMonitor(MonitorConfig())
    alarms = []
    with JSONLLogger(telemetry_path) as logger, scores_path.open("x", encoding="utf-8") as score_file:
        for record in frame.to_dict("records"):
            # No labels or future values are passed to the monitor.
            logger.write(record)
            result = monitor.update(record)
            score_file.write(json.dumps(clean_json(result), allow_nan=False) + "\n")
            if result["alert"]:
                alarms.append(result["step"])
    summary = {
        "data_kind": "synthetic_telemetry",
        "records": len(frame),
        "alarm_steps": alarms,
        "config": monitor.config.to_dict(),
        "note": "API integration example only; not real LLM or RL training.",
    }
    write_json(output / "metadata.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
