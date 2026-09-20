"""Exploratory failure-path test, separately retained from the unsuccessful mild stress.

The 50x LR trial did not produce measured loss events. This follow-up fixes 5000x
before running seeds 31/41/59; it is not a blind production or held-out performance claim.
The monitor threshold stays fixed at the synthetic-development choice.
"""
import argparse
import json
from pathlib import Path
from dataclasses import asdict
import pandas as pd
from llm_sde.torch_demo import TrainConfig,train
from llm_sde.monitor import MonitorConfig
from llm_sde.io import load_log,write_json
from llm_sde.evaluation import aggregate
from llm_sde.report import make_report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="runs/strong_stress")
    p.add_argument("--config",default="configs/calibrated_synthetic.json")
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    conf=MonitorConfig.load(args.config)
    cfg=TrainConfig(scenario="lr_fault",fault_multiplier=5000)
    write_json(out/"protocol.json",{"seeds":[31,41,59],"training_config":asdict(cfg),
        "monitor_config":conf.to_dict(),"study_type":"exploratory_integration_stress_test",
        "selection_disclosure":"Added after 50x LR trials produced no measured loss events. Strong amplitude chosen to exercise the failure path; not a generalization benchmark. No monitor threshold or feature tuning.",
        "label_rule":{"window":64,"min_rise":.7,"ratio":1.8,"persistence":2}})
    results=[]
    for seed in (31,41,59):
        folder=out/f"strong_lr_{seed}"
        r=train(folder,TrainConfig(scenario="lr_fault",fault_multiplier=5000,seed=seed),conf)
        events=json.loads((folder/"observed_events.json").read_text())["events"]
        make_report(load_log(folder/"telemetry.csv"),pd.read_csv(folder/"scores.csv"),folder,
                    conf.to_dict(),r,events,r["metrics"])
        results.append(r)
        print(r["run_id"],r["metrics"],flush=True)
    write_json(out/"runs_summary.json",results)
    write_json(out/"aggregate_metrics.json",aggregate([r["metrics"] for r in results]))
    pd.DataFrame([{"seed":r["seed"],"parameters":r["parameters"],"records":r["records"],
                   "initial_validation_loss":r["initial_validation_loss"],
                   "final_validation_loss":r["final_validation_loss"],
                   **{k:v for k,v in r["metrics"].items() if not isinstance(v,list)}} for r in results]).to_csv(out/"runs_summary.csv",index=False)


if __name__=="__main__":
    main()
