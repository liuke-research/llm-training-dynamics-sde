"""Validated single-run CSV/JSONL telemetry and standards-compliant JSON output."""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import pandas as pd

REQUIRED = ("step", "loss", "grad_norm", "learning_rate")
OPTIONAL = ("update_norm", "param_norm", "clipped_grad_norm")


def clean_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(value), ensure_ascii=False,
                               indent=2, allow_nan=False), encoding="utf-8")


def coerce_record(record: Mapping[str, Any]) -> dict[str, Any]:
    missing = set(REQUIRED) - set(record)
    if missing:
        raise ValueError(f"Missing required telemetry fields: {sorted(missing)}")
    out: dict[str, Any] = {}
    try:
        step = float(record["step"])
    except (TypeError, ValueError) as exc:
        raise ValueError("step must be a nonnegative integer") from exc
    if isinstance(record["step"], bool) or not math.isfinite(step) or step < 0 or not step.is_integer():
        raise ValueError("step must be a nonnegative integer")
    out["step"] = int(step)
    for field in (*REQUIRED[1:], *OPTIONAL):
        value = record.get(field)
        try:
            out[field] = float(value) if value is not None else float("nan")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be numeric or null") from exc
        # Nonfinite loss/gradient are retained as hard failure signals, never filled with zero.
        if field != "loss" and math.isfinite(out[field]) and out[field] < 0:
            raise ValueError(f"{field} cannot be negative")
    if not math.isfinite(out["learning_rate"]):
        raise ValueError("learning_rate must be finite (zero is allowed)")
    for key in ("run_id", "phase"):
        value = record.get(key)
        out[key] = "" if value is None or (isinstance(value, float) and math.isnan(value)) else str(value)
    return out


def validate_log(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("The telemetry file is empty")
    records = [coerce_record(r) for r in frame.to_dict("records")]
    steps = [r["step"] for r in records]
    if any(b <= a for a, b in zip(steps, steps[1:])):
        raise ValueError("step must be strictly increasing; split independent/restarted runs first")
    if len({r["run_id"] for r in records}) > 1:
        raise ValueError("One file must contain one run_id; split runs before analysis")
    return pd.DataFrame(records)


def load_log(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() in (".jsonl", ".ndjson"):
        rows = []
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON at line {number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL line {number} must be an object")
            rows.append(row)
        frame = pd.DataFrame(rows)
    else:
        raise ValueError("Expected .csv, .jsonl or .ndjson")
    return validate_log(frame)


class JSONLLogger:
    """Explicitly owned file; no external service, network or automatic overwrite."""
    def __init__(self, path: str | Path, *, overwrite: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w" if overwrite else "x", encoding="utf-8")

    def write(self, record: Mapping[str, Any]) -> None:
        coerce_record(record)
        self._file.write(json.dumps(clean_json(dict(record)), ensure_ascii=False, allow_nan=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
