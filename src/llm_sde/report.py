"""Portable self-contained HTML report with local, non-interactive evidence plots."""
from __future__ import annotations
import base64
import html
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .io import clean_json, write_json


def _plot(frame: pd.DataFrame, scored: pd.DataFrame, field: str, events: list[dict], threshold: float) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10,3.2))
    if field == "risk_score":
        ax.plot(scored["step"],scored[field],label="Heuristic risk (not probability)",linewidth=1.1)
        ax.axhline(threshold,linestyle="--",label="Alarm threshold")
        alarms = scored[scored["alert"].astype(bool)]
        if len(alarms):
            ax.scatter(alarms["step"],alarms["risk_score"],marker="^",label="Alarm episodes",s=35)
        ax.set_ylim(-.03,1.08)
    else:
        ax.plot(frame["step"],frame[field],label=field,linewidth=1.1)
    for i,event in enumerate(events):
        ax.axvline(event["onset"],linestyle=":",label="Observed/injected event onset" if i == 0 else None)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel(field)
    ax.legend(loc="best",fontsize=8)
    fig.tight_layout()
    buffer=io.BytesIO()
    fig.savefig(buffer,format="png",dpi=140)
    plt.close(fig)
    return buffer.getvalue()


def make_report(frame: pd.DataFrame, scored: pd.DataFrame, out: str | Path,
                config: dict, metadata: dict | None = None, events: list[dict] | None = None,
                metrics: dict | None = None) -> Path:
    out = Path(out)
    out.mkdir(parents=True,exist_ok=True)
    metadata,events=metadata or {},events or []
    pieces=[]
    for field in ("loss","grad_norm","risk_score"):
        png=_plot(frame,scored,field,events,float(config["threshold"]))
        (out/f"{field}.png").write_bytes(png)
        pieces.append(f'<h2>{html.escape(field)}</h2><img alt="{field}" src="data:image/png;base64,{base64.b64encode(png).decode()}">')
    alarms=scored[scored["alert"].astype(bool)][["step","risk_score","status","factors"]]
    details={"config":config,"metadata":metadata,"metrics":metrics,"records":len(frame),
             "alarm_episodes":len(alarms),"warmup_records":int((~scored["ready"].astype(bool)).sum())}
    write_json(out/"report_metadata.json",details)
    content='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Training Stability | Evidence Report</title><style>body{font-family:system-ui,sans-serif;max-width:1060px;margin:36px auto;padding:0 24px;line-height:1.65}h1{font-size:30px}img{width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f5f6;padding:18px;border-radius:8px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}.note{border-left:4px solid #888;padding:10px 16px;background:#fafafa}</style>
<h1>Training Stability — 实验报告</h1><p class="note">这是可解释的训练诊断原型。Risk Score 不是故障概率；当前步及之后的告警不计为提前预测。图中的事件标签只用于离线评估，不输入监控器。无事件标签时不计算准确率。</p>'''
    content+=f'<h2>运行与评估口径</h2><pre>{html.escape(json.dumps(clean_json(details),ensure_ascii=False,indent=2))}</pre>'
    content+=''.join(pieces)
    content+='<h2>告警记录</h2>'+alarms.to_html(index=False,escape=True)
    content+='<p>本地生成；不上传训练日志，不执行自动降学习率、停训或回滚。数学研究模块与在线启发式评分分开验证。</p></html>'
    path=out/"report.html"
    path.write_text(content,encoding="utf-8")
    return path


def math_demo(out: str | Path, seed: int = 2026) -> dict:
    from .sde import ou_process,double_well,empirical_diagnostics,double_well_density
    from .markov import QuantileChain,scgf,rate_function,finite_horizon_tail
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    ou=ou_process(seed=seed,steps=50000)
    dw=double_well(seed=seed,steps=30000)
    steady=ou[5000:]
    chain=QuantileChain.fit(steady[:25000],bins=8)
    f=np.r_[np.zeros(6),np.ones(2)]
    diagnostics=chain.diagnostics(steady[25000:])
    result={"seed":seed,"ou":{"theta":1.0,"sigma":.5,"dt":.05,"scheme":"exact",
            "theoretical_stationary_mean":0.0,"theoretical_stationary_variance":.125,
            "theoretical_lag1_correlation":float(np.exp(-.05)),"empirical":empirical_diagnostics(steady)},
            "double_well":{"sigma":.7,"dt":.005,"scheme":"tamed","empirical":empirical_diagnostics(dw[3000:])},
            "finite_chain":diagnostics,"scgf_at_zero":scgf(chain.transition,f,0),
            "scalar_occupation_rate_at_half":rate_function(chain.transition,f,.5),
            "fitted_chain_probability_at_least_15_risky_visits_in_30_steps":finite_horizon_tail(chain.transition,f,30,15),
            "risk_definition":"Risky here means the highest two of eight OU calibration bins. It does not mean a training failure."}
    write_json(out/"math_results.json",result)
    pd.DataFrame({"step":np.arange(len(ou)),"state":ou}).to_csv(out/"ou_path.csv",index=False)
    for name,path in (("ou_path",ou[:4000]),("double_well_path",dw)):
        fig,ax=plt.subplots(figsize=(10,3.2))
        ax.plot(path,linewidth=.8)
        ax.set(xlabel="Simulation step",ylabel="State",title=name.replace("_"," "))
        fig.tight_layout();fig.savefig(out/f"{name}.png",dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,3.8))
    ax.hist(dw[3000:],bins=60,density=True,alpha=.6,label="Finite-step empirical occupancy")
    grid=np.linspace(-2.5,2.5,400)
    ax.plot(grid,double_well_density(grid,.7),label="Continuous-SDE stationary density")
    ax.set(xlabel="State",ylabel="Density");ax.legend();fig.tight_layout()
    fig.savefig(out/"double_well_density.png",dpi=140);plt.close(fig)
    return result
