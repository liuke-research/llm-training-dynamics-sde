# Evidence index

只保留两个可直接复核的示例报告和一份九条运行的合表；其余逐运行产物没有随仓库保留，
可用根 README 里的命令在本地重新生成。

- [Synthetic end-to-end example](synthetic_demo/report.html)
- [One complete normal-run report](tiny_normal/report.html)
- [All nine measured telemetry streams](all_tiny_training_telemetry.csv)

The combined telemetry CSV is evidence, not a single-run CLI input. Split by
`evidence_suite` and `evidence_run` before loading an individual run.

Reproduction commands in the root README regenerate everything else under `runs/`:
`python -m llm_sde benchmark` (development threshold search + held-out comparison),
`examples/run_torch_suite.py` (six tiny Transformer runs),
`examples/run_stress_suite.py` (three exploratory stress runs),
`python -m llm_sde math-demo` (known-process / finite-chain results),
`python -m llm_sde latency` (monitor-only latency).

Model checkpoint binaries, virtual environments, caches and legacy archives are not
included. HTML reports embed their own images and can be opened locally.
