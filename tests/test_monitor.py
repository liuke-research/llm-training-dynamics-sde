import json
import numpy as np
import pandas as pd
import pytest
from llm_sde.monitor import MonitorConfig, OnlineMonitor, AlertGate, apply_gate, METHODS
from llm_sde.synthetic import generate_run, SCENARIOS
from llm_sde.io import coerce_record, load_log, write_json, JSONLLogger, validate_log


def record(step=0,**kwargs):
    return dict(step=step,loss=1.0,grad_norm=1.0,learning_rate=.001,**kwargs)


def test_prefix_invariance():
    frame,_=generate_run(1,"grad_ramp")
    first=OnlineMonitor().replay(frame.iloc[:350])
    all_rows=OnlineMonitor().replay(frame)
    pd.testing.assert_frame_equal(first,all_rows.iloc[:350].reset_index(drop=True))


def test_future_perturbations_cannot_change_past():
    frame,_=generate_run(2)
    other=frame.copy()
    other.loc[350:,"loss"]*=1000
    other.loc[350:,"grad_norm"]*=1000
    a=OnlineMonitor().replay(frame)
    b=OnlineMonitor().replay(other)
    pd.testing.assert_frame_equal(a.iloc[:350],b.iloc[:350])


def test_optional_missing_works():
    m=OnlineMonitor(MonitorConfig(window=8))
    for i in range(20):
        result=m.update(record(i))
    assert result["ready"] and not result["update_available"]
    assert result["risk_score"] == 0


@pytest.mark.parametrize("field,value",[("loss",float("nan")),("loss",float("inf")),("grad_norm",float("inf")),("grad_norm",float("nan"))])
def test_nonfinite_hard_alarm_without_warmup(field,value):
    r=record();r[field]=value
    result=OnlineMonitor().update(r)
    assert result["alert"] and result["risk_score"]==1 and result["status"]=="NONFINITE"


def test_failure_does_not_contaminate_history():
    m=OnlineMonitor(MonitorConfig(window=8))
    for i in range(8):m.update(record(i))
    r=record(8);r["loss"]=float("nan")
    m.update(r)
    result=m.update(record(9))
    assert result["ready"] and result["risk_score"]==0 and len(m.features.history)==8


def test_warmup_is_explicit_unknown():
    m=OnlineMonitor(MonitorConfig(window=8))
    for i in range(8):
        result=m.update(record(i))
        assert result["status"]=="WARMUP" and not result["alert"]
    assert m.update(record(8))["ready"]


def test_phase_reset():
    m=OnlineMonitor(MonitorConfig(window=8))
    for i in range(10):m.update(record(i,phase="pretrain"))
    r=m.update(record(10,phase="finetune"))
    assert r["phase_reset"] and not r["ready"]


@pytest.mark.parametrize("step",[0,-1])
def test_nonincreasing_steps_rejected(step):
    m=OnlineMonitor();m.update(record(0))
    with pytest.raises(ValueError):m.update(record(step))


def test_different_run_rejected():
    m=OnlineMonitor();m.update(record(0,run_id="a"))
    with pytest.raises(ValueError):m.update(record(1,run_id="b"))


def test_alert_persistence_and_episode_suppression():
    gate=AlertGate(MonitorConfig(persistence=2,cooldown=5))
    assert not gate.update(10,.9,True)
    assert gate.update(11,.9,True)
    assert not gate.update(12,.9,True)
    assert not gate.update(13,0,True)
    assert not gate.update(14,.9,True)
    assert not gate.update(15,.9,True)
    assert gate.update(16,.9,True)


def test_steps_not_row_positions():
    frame,_=generate_run(4)
    frame["step"]=frame["step"]*10+10000
    results=OnlineMonitor().replay(frame)
    assert list(results["step"])==list(frame["step"])


@pytest.mark.parametrize("method",METHODS)
def test_cached_gates_match_streaming(method):
    frame,_=generate_run(5)
    cfg=MonitorConfig(method=method,threshold=.8)
    streaming=OnlineMonitor(cfg).replay(frame)
    cached=apply_gate(OnlineMonitor().replay(frame),cfg)
    assert np.array_equal(streaming["alert"],cached["alert"])
    assert np.allclose(streaming["risk_score"],cached["risk_score"])


@pytest.mark.parametrize("scenario",SCENARIOS)
def test_generator_deterministic_labels_separate(scenario):
    a,meta=generate_run(6,scenario)
    b,meta2=generate_run(6,scenario)
    pd.testing.assert_frame_equal(a,b)
    assert meta==meta2 and "events" not in a and "precursor_start" not in a


@pytest.mark.parametrize("field,value",[("step",.5),("step",True),("step",-1),("grad_norm",-1),("learning_rate",-1),("learning_rate",float("nan"))])
def test_schema_invalid(field,value):
    r=record();r[field]=value
    with pytest.raises(ValueError):coerce_record(r)


def test_schema_missing_field():
    with pytest.raises(ValueError):coerce_record({"step":0})


def test_jsonl_csv_roundtrip(tmp_path):
    rows=[record(i) for i in range(3)]
    with JSONLLogger(tmp_path/"log.jsonl") as logger:
        for r in rows:logger.write(r)
    pd.DataFrame(rows).to_csv(tmp_path/"log.csv",index=False)
    pd.testing.assert_frame_equal(load_log(tmp_path/"log.jsonl"),load_log(tmp_path/"log.csv"))
    with pytest.raises(FileExistsError):JSONLLogger(tmp_path/"log.jsonl")


def test_nonfinite_json_is_null(tmp_path):
    write_json(tmp_path/"result.json",{"score":float("nan")})
    assert json.loads((tmp_path/"result.json").read_text())["score"] is None


def test_invalid_json_line(tmp_path):
    (tmp_path/"bad.jsonl").write_text('{bad\n')
    with pytest.raises(ValueError,match="line 1"):load_log(tmp_path/"bad.jsonl")


def test_empty_log_rejected():
    with pytest.raises(ValueError):validate_log(pd.DataFrame())


def test_wrong_run_mixing_rejected():
    with pytest.raises(ValueError):validate_log(pd.DataFrame([record(0,run_id="a"),record(1,run_id="b")]))


@pytest.mark.parametrize("kwargs",[{"window":2},{"threshold":0},{"threshold":1},{"persistence":0},{"cooldown":-1},{"method":"fake"}])
def test_config_rejects_bad_values(kwargs):
    with pytest.raises(ValueError):MonitorConfig(**kwargs)


def test_label_columns_cannot_affect_monitor():
    frame,_=generate_run(2,steps=300)
    extra=frame.copy()
    extra["instability_phase"]=np.arange(len(extra))
    extra["spike_step"]=12345
    extra["future_loss"]=1e9
    pd.testing.assert_frame_equal(OnlineMonitor().replay(frame),OnlineMonitor().replay(extra))
