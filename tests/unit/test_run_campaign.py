"""scripts/run_campaign.py decides what a campaign runs, and in which order."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_campaign.py"


@pytest.fixture(scope="module")
def campaign() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_run_campaign", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_run_campaign"] = module  # a dataclass looks its module up here
    spec.loader.exec_module(module)
    return module


def test_the_plan_goes_level_by_level_with_the_decentralised_arm_first(campaign):
    runs = campaign.plan(campaign.ARMS, [0.25, 0.15], [0.25])

    assert [r.name for r in runs] == [
        "decentralised_rate0.25",
        "centralised_rate0.25",
        "section_rate0.25",
        "decentralised+comm_rate0.25",
        "decentralised_rate0.15",
        "centralised_rate0.15",
        "section_rate0.15",
    ]
    assert [r.communication for r in runs].count(True) == 1


def test_a_comm_rate_must_be_a_trained_rate_and_the_inputs_must_be_positive(campaign, tmp_path):
    out = ["--out", str(tmp_path)]
    for argv in (
        out + ["--rates", "0.25", "--comm-rates", "0.3"],
        out + ["--rates", "0"],
        out + ["--eval-seeds", "0"],
        out + ["--retries", "-1"],
    ):
        with pytest.raises(SystemExit):
            campaign.parse_args(argv)
    args = campaign.parse_args(out)
    assert args.rates == list(campaign.DEFAULT_RATES) and args.comm_rates == [0.25]


def test_the_training_command_carries_the_shared_settings_and_the_communication_flag(
    campaign, tmp_path
):
    args = campaign.parse_args(["--out", str(tmp_path), "--gpu", "--env-runners", "3"])
    run = campaign.Campaign(args)

    plain = run.train_command(campaign.Run("section", 0.3))
    comm = run.train_command(campaign.Run("decentralised", 0.25, communication=True))

    assert plain[1].endswith("03_section_based.py")
    for part in (
        "--horizon=630",
        "--arrival-rate=0.3",
        "--env-runners=3",
        "--eval-seeds=6",
        "--gpu",
    ):
        assert part in plain
    assert "--communication" not in plain and "--communication" in comm


def test_a_dry_run_plans_and_runs_nothing(campaign, tmp_path):
    campaign.main(["--out", str(tmp_path / "out"), "--dry-run"])

    assert not (tmp_path / "out" / "runs").exists()
    assert "10 runs planned" in (tmp_path / "out" / "campaign.log").read_text(encoding="utf-8")


def test_merging_sweeps_keeps_the_planner_rows_of_the_planner_sweep_only(campaign, tmp_path):
    def write(path: Path, kinds: list[str]) -> Path:
        path.parent.mkdir(parents=True)
        rows = [{"condition": k, "kind": k, "arrival_rate": 0.25} for k in kinds]
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        return path

    sweep = tmp_path / "out" / "sweep"
    planners = write(sweep / "planners" / "load_sweep.jsonl", ["planner"])
    level = write(sweep / "rate_0.25" / "load_sweep.jsonl", ["planner", "controller"])
    run = campaign.Campaign(campaign.parse_args(["--out", str(tmp_path / "out")]))

    merged = run.merge_sweeps([planners, level])

    assert merged is not None
    kinds = [json.loads(line)["kind"] for line in merged.read_text(encoding="utf-8").splitlines()]
    assert sorted(kinds) == ["controller", "planner"]


def test_the_preflight_refuses_a_checkout_without_the_model_of_the_thesis(
    campaign, tmp_path, monkeypatch
):
    fake = tmp_path / "repo"
    (fake / "src" / "slap_mapd_coupling" / "training").mkdir(parents=True)
    (fake / "examples").mkdir()
    (fake / "src" / "slap_mapd_coupling" / "training" / "callbacks.py").write_text("x = 1\n")
    (fake / "examples" / "04_decentralised_training.py").write_text("WAIT_COST = 2.0\n")
    monkeypatch.setattr(campaign, "REPO", fake)
    monkeypatch.setattr(campaign, "OUTPUT", fake / "examples" / "output")
    run = campaign.Campaign(campaign.parse_args(["--out", str(tmp_path / "out")]))

    with pytest.raises(SystemExit, match="not the model of the thesis"):
        run.preflight()
    run.args.skip_preflight = True
    run.preflight()  # allowed when asked for


def test_a_second_campaign_on_the_same_checkout_refuses_and_leaves_the_first_alone(
    campaign, tmp_path, monkeypatch
):
    monkeypatch.setattr(campaign.tempfile, "tempdir", str(tmp_path))
    first = campaign.Campaign(campaign.parse_args(["--out", str(tmp_path / "one")]))
    second = campaign.Campaign(campaign.parse_args(["--out", str(tmp_path / "two")]))

    first.lock()
    with pytest.raises(SystemExit, match="Another campaign is already running"):
        second.lock()
    assert not (tmp_path / "two").exists()  # the refused start wrote nothing
    assert "one" in next(tmp_path.glob("slap_mapd_campaign_*.lock")).read_text(encoding="utf-8")


def test_the_lock_is_free_again_once_the_first_campaign_is_gone(campaign, tmp_path, monkeypatch):
    monkeypatch.setattr(campaign.tempfile, "tempdir", str(tmp_path))
    first = campaign.Campaign(campaign.parse_args(["--out", str(tmp_path / "one")]))
    first.lock()
    first._lock_handle.close()  # what the operating system does when the process ends

    campaign.Campaign(campaign.parse_args(["--out", str(tmp_path / "two")])).lock()
