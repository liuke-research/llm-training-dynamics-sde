import json
from llm_sde.cli import main


def test_demo_cli_creates_report(tmp_path,capsys):
    assert main(["demo","--out",str(tmp_path),"--steps","300"])==0
    assert (tmp_path/"report.html").exists()
    assert (tmp_path/"scores.csv").exists()


def test_unknown_labels_do_not_create_fake_accuracy(tmp_path,capsys):
    path=tmp_path/"log.csv"
    path.write_text("step,loss,grad_norm,learning_rate\n0,1,1,.001\n1,.9,1,.001\n")
    assert main(["analyze",str(path),"--out",str(tmp_path/"out")])==0
    result=json.loads((tmp_path/"out"/"report_metadata.json").read_text())
    assert result["metrics"] is None
