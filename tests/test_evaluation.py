import numpy as np
import pandas as pd
import pytest
from llm_sde.evaluation import evaluate,aggregate,observed_loss_events,bootstrap_runs


def scored(alarms):
    steps=np.arange(100)
    return pd.DataFrame({"step":steps,"alert":np.isin(steps,alarms),"ready":steps>=10})


def test_onset_is_detection_not_prediction():
    r=evaluate(scored([50]),[{"onset":50,"end":60}],horizon=10)
    assert r["early_hits"]==0 and r["post_onset_hits"]==1


def test_after_onset_never_negative_lead():
    r=evaluate(scored([51]),[{"onset":50,"end":60}],horizon=10)
    assert r["lead_steps"]==[] and r["median_lead_steps"] is None


def test_horizon_boundary_inclusive():
    r=evaluate(scored([40]),[{"onset":50,"end":60}],horizon=10)
    assert r["lead_steps"]==[10]


def test_outside_horizon_false_alarm():
    r=evaluate(scored([39]),[{"onset":50,"end":60}],horizon=10)
    assert r["early_hits"]==0 and r["false_healthy_alarms"]==1


def test_no_event_metrics_are_undefined_not_perfect():
    r=evaluate(scored([]),[],horizon=10)
    assert r["early_event_recall"] is None and r["early_alarm_precision"] is None


def test_one_to_one_event_matching():
    r=evaluate(scored([44]),[{"onset":50,"end":52},{"onset":55,"end":57}],horizon=20)
    assert r["early_hits"]==1 and r["early_event_recall"]==.5


def test_duplicate_alarms_not_duplicate_event_hits():
    r=evaluate(scored([42,45,48]),[{"onset":50,"end":60}],horizon=10)
    assert r["early_hits"]==1 and r["early_alarm_precision"]==pytest.approx(1/3)


def test_observed_events_from_actual_loss():
    df=pd.DataFrame({"step":np.arange(200),"loss":np.ones(200)})
    assert observed_loss_events(df)==[]
    df.loc[100:110,"loss"]=4
    assert observed_loss_events(df)==[{"onset":100,"end":110,"kind":"observed_loss_excursion"}]


@pytest.mark.parametrize("events",[[{"onset":50,"end":49}],[{"onset":50,"end":101}],[{"onset":50,"end":60},{"onset":59,"end":70}]])
def test_bad_labels_rejected(events):
    with pytest.raises(ValueError):evaluate(scored([]),events)


def test_bootstrap_returns_intervals():
    a=evaluate(scored([40]),[{"onset":50,"end":60}],horizon=10)
    b=evaluate(scored([]),[{"onset":50,"end":60}],horizon=10)
    result=bootstrap_runs([a,b],repetitions=50)
    lo,hi=result["early_event_recall"]
    assert 0<=lo<=hi<=1
    assert aggregate([a,b])["early_event_recall"]==.5


def test_bootstrap_respects_seed_clusters():
    from llm_sde.evaluation import bootstrap_seed_clusters
    hit=evaluate(scored([40]),[{"onset":50,"end":60}],horizon=10)
    miss=evaluate(scored([]),[{"onset":50,"end":60}],horizon=10)
    result=bootstrap_seed_clusters([hit,miss,hit,miss],[1,1,2,2],repetitions=50)
    assert result["clusters"]==2
    assert result["percentile_95_intervals"]["early_event_recall"]==[.5,.5]
