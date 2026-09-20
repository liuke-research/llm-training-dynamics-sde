"""Offline exploratory finite-state model of log gradient norms in a selected phase.

No predicted training failure probability is produced. A high calibration/held-out
occupancy shift is a reason to doubt stationarity, not a proof of instability.
"""
import argparse
from pathlib import Path
import numpy as np
from llm_sde.io import load_log,write_json
from llm_sde.markov import QuantileChain,scgf,rate_function,finite_horizon_tail


def main():
    p=argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("--out",default="runs/markov_proxy")
    p.add_argument("--start-step",type=int,default=64)
    p.add_argument("--end-step",type=int)
    p.add_argument("--bins",type=int,default=4)
    args=p.parse_args()
    log=load_log(args.input)
    data=log[log.step>=args.start_step]
    if args.end_step is not None:data=data[data.step<=args.end_step]
    x=np.log(np.maximum(data.grad_norm.to_numpy(),1e-12))
    if len(x)<100 or not np.isfinite(x).all():
        raise ValueError("Selected phase needs >=100 finite gradient observations")
    cut=int(.6*len(x))
    model=QuantileChain.fit(x[:cut],bins=args.bins)
    f=np.zeros(len(model.transition));f[-1]=1
    result=model.diagnostics(x[cut:])
    result.update({"observable":"log pre-clipping gradient norm; highest calibration bin is the monitored region, not a ground-truth failure label",
                   "calibration_records":cut,"heldout_records":len(x)-cut,
                   "scgf_zero":scgf(model.transition,f,0),
                   "occupation_rate_at_half":rate_function(model.transition,f,.5),
                   "model_probability_at_least_15_top_bin_visits_in_30":finite_horizon_tail(model.transition,f,30,15),
                   "limitation":"The scalar projection need not be Markov. Training is often nonstationary; this diagnostic is not used by the online score and cannot certify LLM ergodicity."})
    write_json(Path(args.out)/"markov_proxy.json",result)
    print(result)


if __name__=="__main__":main()
