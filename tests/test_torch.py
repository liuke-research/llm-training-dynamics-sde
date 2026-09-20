import json
import pytest
import numpy as np
torch=pytest.importorskip("torch")
from llm_sde.torch_demo import TinyCausalLM,TrainConfig,train,parameter_snapshot,measured_parameter_change,TokenSource
from llm_sde.io import load_log


def test_causal_mask_cannot_see_future_tokens():
    torch.manual_seed(4)
    model=TinyCausalLM(vocab=8,width=16,heads=2,layers=1,context=8).eval()
    a=torch.tensor([[0,1,2,3,4,5,6,7]])
    b=a.clone();b[:,4:]=0
    with torch.no_grad():
        la,lb=model(a),model(b)
    assert torch.allclose(la[:,:4],lb[:,:4],atol=1e-6)


def test_measured_updates_are_actual_differences():
    model=torch.nn.Linear(2,1,bias=False)
    before=parameter_snapshot(model)
    with torch.no_grad():
        for p in model.parameters():p.add_(1)
    update,_=measured_parameter_change(model,before)
    assert update==pytest.approx(np.sqrt(2),rel=1e-6)


def test_real_backward_smoke(tmp_path):
    cfg=TrainConfig(steps=12,batch_size=2,context=8,vocab=8,width=16,heads=2,layers=1)
    result=train(tmp_path/"tiny",cfg)
    log=load_log(tmp_path/"tiny"/"telemetry.jsonl")
    assert len(log)==12 and (log["grad_norm"]>0).all() and (log["update_norm"]>0).all()
    assert result["parameters"]>1000 and result["records"]==12
    assert (tmp_path/"tiny"/"final_state_dict.pt").exists()


def test_local_corpus_path(tmp_path):
    path=tmp_path/"local.txt"
    path.write_text("local corpus for a simple test.\n"*40,encoding="utf-8")
    source=TokenSource(TrainConfig(context=8),path)
    x,y=source.batch()
    assert x.shape==y.shape==(12,8)
    assert source.metadata["data_kind"]=="user_local_utf8_corpus"


def test_context_limit():
    model=TinyCausalLM(vocab=8,width=16,heads=2,layers=1,context=8)
    with pytest.raises(ValueError):model(torch.zeros((1,10),dtype=torch.long))
