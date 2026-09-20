from __future__ import annotations
import argparse
import json
from pathlib import Path
from .io import load_log,write_json,clean_json
from .monitor import MonitorConfig,OnlineMonitor
from .synthetic import SCENARIOS,generate_run
from .evaluation import evaluate
from .report import make_report,math_demo


def main(argv=None):
    parser=argparse.ArgumentParser(description="Causal training diagnostics and scoped stochastic research demos")
    sub=parser.add_subparsers(dest="command",required=True)
    demo=sub.add_parser("demo",help="Synthetic telemetry end-to-end demo")
    demo.add_argument("--out",default="runs/synthetic_demo")
    demo.add_argument("--scenario",choices=SCENARIOS,default="grad_ramp")
    demo.add_argument("--seed",type=int,default=42)
    demo.add_argument("--steps",type=int,default=700)
    demo.add_argument("--config")
    analyze=sub.add_parser("analyze",help="Analyze one CSV/JSONL run without fabricated event labels")
    analyze.add_argument("input")
    analyze.add_argument("--out",default="runs/analysis")
    analyze.add_argument("--config")
    analyze.add_argument("--events",help="Optional separate JSON object with an events list")
    math=sub.add_parser("math-demo",help="OU, double well and finite-chain numerical validation")
    math.add_argument("--out",default="runs/math_demo")
    math.add_argument("--seed",type=int,default=2026)
    bench=sub.add_parser("benchmark",help="Development-only calibration and held-out synthetic comparison")
    bench.add_argument("--out",default="runs/benchmark")
    bench.add_argument("--dev-seeds",type=int,default=4)
    bench.add_argument("--test-seeds",type=int,default=8)
    bench.add_argument("--steps",type=int,default=700)
    latency=sub.add_parser("latency",help="CPU monitor-only timing, not training slowdown")
    latency.add_argument("--out",default="runs/latency")
    latency.add_argument("--steps",type=int,default=3000)
    train=sub.add_parser("train-demo",help="Offline CPU tiny Transformer with actual gradients/updates")
    train.add_argument("--out",default="runs/tiny_transformer")
    train.add_argument("--scenario",choices=("normal","lr_fault"),default="normal")
    train.add_argument("--seed",type=int,default=7)
    train.add_argument("--steps",type=int,default=320)
    train.add_argument("--fault-start",type=int,default=190)
    train.add_argument("--corpus",help="Optional local UTF-8 text; absent = algorithmic token data")
    train.add_argument("--config",help="Monitor JSON config; never selected using this training run's outcomes")
    args=parser.parse_args(argv)
    out=Path(args.out)
    cfg=MonitorConfig.load(args.config) if getattr(args,"config",None) else MonitorConfig()
    try:
        if args.command in ("demo","analyze"):
            out.mkdir(parents=True,exist_ok=True)
            if args.command=="demo":
                frame,meta=generate_run(args.seed,args.scenario,args.steps)
                events=meta["events"]
                frame.to_csv(out/"telemetry.csv",index=False)
                write_json(out/"labels.json",meta)
            else:
                frame=load_log(args.input)
                meta={"data_kind":"user_supplied_telemetry","input":str(Path(args.input).name)}
                events=json.loads(Path(args.events).read_text(encoding="utf-8"))["events"] if args.events else []
            scored=OnlineMonitor(cfg).replay(frame)
            scored.to_csv(out/"scores.csv",index=False)
            # Unknown labels != a confirmed absence of events.
            metrics=evaluate(scored,events) if args.command=="demo" or args.events else None
            path=make_report(frame,scored,out,cfg.to_dict(),meta,events,metrics)
            result={"report":str(path),"metrics":metrics}
        elif args.command=="math-demo":
            result=math_demo(out,args.seed)
        elif args.command=="benchmark":
            from .benchmark import run_benchmark
            if not 1<=args.dev_seeds<=100 or not 1<=args.test_seeds<=100:
                raise ValueError("Seed counts must lie in 1..100")
            result=run_benchmark(out,dev_seeds=range(args.dev_seeds),test_seeds=range(100,100+args.test_seeds),steps=args.steps)
        elif args.command=="latency":
            from .benchmark import benchmark_latency
            result=benchmark_latency(out,args.steps)
        else:
            try:
                from .torch_demo import TrainConfig,train as run_train
            except ImportError as exc:
                raise RuntimeError('PyTorch is optional. Install with: python -m pip install -e ".[train]"') from exc
            result=run_train(out,TrainConfig(seed=args.seed,scenario=args.scenario,steps=args.steps,fault_start=args.fault_start),cfg,args.corpus)
            import pandas as pd
            frame=load_log(out/"telemetry.csv")
            scored=pd.read_csv(out/"scores.csv")
            events=json.loads((out/"observed_events.json").read_text())["events"]
            make_report(frame,scored,out,cfg.to_dict(),result,events,result["metrics"])
        print(json.dumps(clean_json(result),ensure_ascii=False,indent=2,allow_nan=False))
        return 0
    except (ValueError,FileNotFoundError,FileExistsError,KeyError,RuntimeError) as exc:
        parser.exit(2,f"Error: {exc}\n")


if __name__=="__main__":
    main()
