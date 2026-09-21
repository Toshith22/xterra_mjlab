import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location("watch",Path(__file__).resolve().parents[1]/"scripts/watch_isl.py")
watch=importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


def test_healthy_run_does_not_alert():
    assert not watch.alerts([dict(run="a",status="running",heartbeat=950,process_alive=True)],1000,0)


def test_completed_run_does_not_need_live_pid():
    assert not watch.alerts([dict(run="a",status="complete",process_alive=False)],1000,0)


def test_dead_stalled_or_failed_run_alerts():
    for row in (dict(status="running",process_alive=False),
                dict(status="running",process_alive=True,heartbeat=0),
                dict(status="stopped_error",error="nonfinite")):
        assert watch.alerts([dict(run="a",**row)],1000,0)


def test_evaluation_allows_longer_heartbeat():
    assert not watch.alerts([dict(run="a",status="evaluating",heartbeat=0,process_alive=True)],1000,0)


def test_custom_remote_root_missing_run(tmp_path):
    assert watch.remote_status(str(tmp_path), ["new_run"]) == [dict(run="new_run", status="starting")]


def test_failed_campaign_is_reported_before_continuation(tmp_path):
    import json
    runs=tmp_path/"runs"
    runs.mkdir()
    (runs/"v7_seed1_campaign.json").write_text(json.dumps(dict(
        status="stopped_error",pid=-1,time=10,error="bootstrap failed")))
    rows=watch.remote_status(str(tmp_path),["v7_seed1_continuous"])
    assert watch.alerts(rows,100,0)
