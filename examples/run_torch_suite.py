"""Reproduce six independent tiny-model integration runs; do not tune thresholds here."""
from pathlib import Path
import argparse
import pandas as pd
from llm_sde.torch_demo import TrainConfig, train
from llm_sde.monitor import MonitorConfig
from llm_sde.io import write_json, load_log
from llm_sde.evaluation import aggregate
from llm_sde.report import make_report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="runs/torch_suite")
    p.add_argument("--config",default="configs/calibrated_synthetic.json")
    args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    conf=MonitorConfig.load(args.config)
    results=[]
    write_json(out/"protocol.json",{"seeds":[7,11,23],"scenarios":["normal","lr_fault"],
             "monitor_config":conf.to_dict(),"threshold_source":"synthetic development runs ONLY",
             "training_config":TrainConfig().__dict__,
             "limitations":"Same tiny architecture and same algorithmic data distribution in all runs; no natural-language or large-model validation."})
    for scenario in ("normal","lr_fault"):
        for seed in (7,11,23):
            folder=out/f"{scenario}_{seed}"
            r=train(folder,TrainConfig(scenario=scenario,seed=seed),conf)
            import json
            events=json.loads((folder/"observed_events.json").read_text())["events"]
            make_report(load_log(folder/"telemetry.csv"),pd.read_csv(folder/"scores.csv"),folder,
                        conf.to_dict(),r,events,r["metrics"])
            results.append(r)
            print(r["run_id"],r["metrics"],flush=True)
    write_json(out/"runs_summary.json",results)
    combined={"all_runs":aggregate([r["metrics"] for r in results])}
    for scenario in ("normal","lr_fault"):
        combined[scenario]=aggregate([r["metrics"] for r in results if r["scenario"]==scenario])
    write_json(out/"aggregate_metrics.json",combined)
    rows=[]
    for r in results:
        rows.append({"run_id":r["run_id"],"parameters":r["parameters"],"records":r["records"],
                     "initial_validation_loss":r["initial_validation_loss"],
                     "final_validation_loss":r["final_validation_loss"],
                     "training_loop_seconds":r["training_loop_seconds"],
                     **{k:v for k,v in r["metrics"].items() if not isinstance(v,list)}})
    pd.DataFrame(rows).to_csv(out/"runs_summary.csv",index=False)


if __name__=="__main__":
    main()
