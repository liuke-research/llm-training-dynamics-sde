# Evidence index

These are fresh local results from the repository packaging run, not production benchmarks.

- [Core tests and environment](release_checks/)
- [Synthetic development / held-out comparison](benchmark/)
- [Synthetic end-to-end example](synthetic_demo/report.html)
- [Six tiny Transformer runs](torch_suite/runs_summary.json)
- [Three exploratory stress runs](strong_stress/runs_summary.json)
- [All nine measured telemetry streams](all_tiny_training_telemetry.csv)
- [One complete normal-run report](tiny_normal/report.html)
- [Known-process / finite-chain results](math_demo/math_results.json)
- [Monitor-only latency](latency/monitor_latency.json)

The combined telemetry CSV is evidence, not a single-run CLI input. Split by
`evidence_suite` and `evidence_run` before loading an individual run.

The bundle includes summaries and selected full traces. Reproduction commands in
the root README regenerate all per-run logs and checkpoints under `runs/`.
Model checkpoint binaries, virtual environments, caches and legacy archives are not
included. HTML reports embed their own images and can be opened locally.
