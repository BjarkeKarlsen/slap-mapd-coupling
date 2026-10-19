"""training/evaluate_run.py: the checkpoint is chosen on the validation seeds and scored on the
test seeds. Episodes are replaced by a fake, so these check the choice and the rows, not a run."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

import slap_mapd_coupling.training.evaluate_run as evaluate_run_module
from slap_mapd_coupling.evaluation.metrics import RunMetrics
from slap_mapd_coupling.main import cli
from slap_mapd_coupling.training.config import (
    TEST_SEED_START,
    VALIDATION_SEED_START,
    SeedSplitConfig,
)
from slap_mapd_coupling.training.evaluate_run import checkpoints_of, evaluate_run
from slap_mapd_coupling.training.spec import TrainingSpec, load_spec

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "decentralised.yaml"


class FakeMetrics(SimpleNamespace):
    def model_dump(self, mode="python"):
        return dict(vars(self))


def _run_dir(tmp_path: Path, num_checkpoints: int) -> Path:
    from slap_mapd_coupling.training.spec import load_spec

    run = tmp_path / "run"
    trial = run / "decentralised_fixed_only" / "PPO_env_0"
    for index in range(num_checkpoints):
        (trial / f"checkpoint_{index:06d}").mkdir(parents=True)
    (run / "decentralised_fixed_only_spec.json").write_text(
        load_spec(CONFIG).model_dump_json(), encoding="utf-8"
    )
    return run


@pytest.fixture
def fake_episodes(monkeypatch):
    """Checkpoint i completes completed[i] tasks on every seed, and every played seed is kept."""
    played = []
    split = SeedSplitConfig(
        training_seed=0, train_seed_count=5, validation_seed_count=2, test_seed_count=3
    )
    monkeypatch.setattr(
        evaluate_run_module, "build_study", lambda spec: SimpleNamespace(seed_split=split)
    )

    def install(completed):
        def play(study, spec, checkpoint, seed, storage_mode=None):
            index = int(checkpoint.name.split("_")[-1])
            played.append((checkpoint.name, seed))
            return FakeMetrics(
                seed=seed, num_completed_tasks=completed[index], throughput=completed[index] / 10
            )

        monkeypatch.setattr(evaluate_run_module, "play", play)
        return played

    return install


def test_checkpoints_are_in_the_order_they_were_saved(tmp_path):
    run = _run_dir(tmp_path, 12)
    assert [p.name for p in checkpoints_of(run)][:3] == [
        "checkpoint_000000",
        "checkpoint_000001",
        "checkpoint_000002",
    ]
    assert checkpoints_of(run)[-1].name == "checkpoint_000011"


def test_the_checkpoint_with_most_completed_tasks_on_validation_is_chosen(tmp_path, fake_episodes):
    played = fake_episodes([3, 9, 5])
    result = evaluate_run(_run_dir(tmp_path, 3), tmp_path / "out")
    assert (result.chosen.name, result.last.name) == ("checkpoint_000001", "checkpoint_000002")
    # Every checkpoint plays the validation seeds, and only the chosen and the last the test seeds.
    validation = {seed for name, seed in played[:6]}
    assert validation == {VALIDATION_SEED_START, VALIDATION_SEED_START + 1}
    test = [(name, seed) for name, seed in played[6:]]
    assert {seed for _, seed in test} == {TEST_SEED_START + i for i in range(3)}
    assert {name for name, _ in test} == {"checkpoint_000001", "checkpoint_000002"}
    lines = (tmp_path / "out" / "test.jsonl").read_text().splitlines()
    assert len(lines) == 6 and '"role": "chosen"' in lines[0]
    assert len((tmp_path / "out" / "validation.jsonl").read_text().splitlines()) == 6


def test_a_tie_goes_to_the_later_checkpoint(tmp_path, fake_episodes):
    fake_episodes([7, 7, 2])
    result = evaluate_run(_run_dir(tmp_path, 3), tmp_path / "out")
    assert result.chosen.name == "checkpoint_000001"


def _real_metrics(seed, storage_mode, service_time):
    """A run that keeps up, so the paired comparison has a gain to report."""
    return RunMetrics(
        storage_mode=storage_mode,
        controller="decentralised",
        congestion_sensitive=False,
        communication=False,
        num_agents=3,
        arrival_rate=0.25,
        seed=seed,
        horizon=100,
        observation_depth=5,
        num_completed_tasks=20,
        throughput=0.2,
        keep_up_ratio=0.8,
        keeps_up=True,
        num_waiting_tasks=0,
        num_active_tasks=0,
        backlog=0,
        mean_service_time=service_time,
        mean_wait_for_agent=service_time / 2,
        mean_travel_time=service_time / 4,
        mean_blocked_time=service_time / 4,
        # 20 tasks, so the totals are 20 times the means
        total_service_time=round(20 * service_time),
        total_wait_for_agent=round(10 * service_time),
        total_travel_time=round(5 * service_time),
        total_blocked_time=round(5 * service_time),
        mean_crowding=0.1,
        num_storage_updates=0 if storage_mode == "fixed" else 1,
        mean_relocated_units=0.0 if storage_mode == "fixed" else 4.0,
        mean_decision_runtime_seconds=0.001,
    )


def _spec_with_rule(run, **rule):
    from slap_mapd_coupling.training.spec import load_spec

    path = next(run.glob("*_spec.json"))
    data = load_spec(CONFIG).model_dump()
    if rule:
        data["storage_rule"] = rule
    path.write_text(TrainingSpec.model_validate(data).model_dump_json(), encoding="utf-8")


def test_the_test_seeds_are_played_under_every_mode_and_summarised(
    tmp_path, fake_episodes, monkeypatch
):
    fake_episodes([3, 9, 5])
    modes_played = []

    def play(study, spec, checkpoint, seed, storage_mode=None):
        modes_played.append(storage_mode)
        if storage_mode is None:
            return FakeMetrics(seed=seed, num_completed_tasks=3, throughput=0.3)
        return _real_metrics(
            seed, storage_mode, 10.0 if storage_mode == "fixed" else 8.0 + seed % 3
        )

    monkeypatch.setattr(evaluate_run_module, "play", play)
    run = _run_dir(tmp_path, 3)
    _spec_with_rule(run, epoch_length=100, reassignment_cap=48, congestion_weight=3.0)
    modes = ["fixed", "demand", "congestion"]
    result = evaluate_run(run, tmp_path / "out", modes)
    # 3 seeds, 3 modes, 2 checkpoints. Validation plays without a mode, as the run trained.
    assert len(result.test_rows) == 18
    assert modes_played[:6] == [None] * 6
    assert {(row["role"], row["storage_mode"]) for row in result.test_rows} == {
        (role, mode) for role in ("chosen", "last") for mode in modes
    }
    rows = [json.loads(line) for line in (tmp_path / "out" / "test.jsonl").read_text().splitlines()]
    assert [row["storage_mode"] for row in rows[:9]] == ["fixed"] * 3 + ["demand"] * 3 + [
        "congestion"
    ] * 3
    summary = (tmp_path / "out" / "summary.md").read_text()
    lines = [line for line in summary.splitlines() if line.startswith("| chosen")]
    assert len(lines) == 2 and "demand" in lines[0] and "congestion" in lines[1]
    assert len([line for line in summary.splitlines() if line.startswith("| last")]) == 2


def test_the_default_is_the_training_mode_and_has_no_comparison(tmp_path, fake_episodes):
    fake_episodes([3, 9, 5])
    result = evaluate_run(_run_dir(tmp_path, 3), tmp_path / "out")
    assert {row["storage_mode"] for row in result.test_rows} == {"fixed"}
    assert "No comparison" in (tmp_path / "out" / "summary.md").read_text()


def test_a_mode_without_its_settings_fails_before_any_episode(tmp_path, fake_episodes):
    played = fake_episodes([3, 9, 5])
    run = _run_dir(tmp_path, 3)
    with pytest.raises(ValueError, match="storage_rule section"):
        evaluate_run(run, tmp_path / "out", ["fixed", "demand"])
    assert played == []
    _spec_with_rule(run, epoch_length=100, reassignment_cap=48)
    with pytest.raises(ValueError, match="congestion_weight"):
        evaluate_run(run, tmp_path / "out", ["congestion"])
    assert played == []


def test_the_command_reports_a_missing_setting_as_a_usage_error(tmp_path, fake_episodes):
    fake_episodes([3])
    result = CliRunner().invoke(
        cli, ["evaluate", str(_run_dir(tmp_path, 1)), "--storage-modes", "demand"]
    )
    assert result.exit_code == 2 and "storage_rule section" in result.output


def test_evaluate_rejects_a_folder_without_a_spec(tmp_path):
    result = CliRunner().invoke(cli, ["evaluate", str(tmp_path)])
    assert result.exit_code != 0
    assert "_spec.json" in result.output


def test_evaluation_registers_every_storage_rule_it_can_play():
    # In a fresh process, so rules imported by other tests cannot hide a missing import.
    code = (
        "import slap_mapd_coupling.training.evaluate_run\n"
        "from slap_mapd_coupling.storage.registry import get_storage_rule\n"
        "for mode in ('fixed', 'demand', 'congestion'):\n"
        "    get_storage_rule(mode)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_another_load_plays_the_test_seeds_but_the_choice_stays_at_the_training_load(
    tmp_path, fake_episodes, monkeypatch
):
    rates = []

    def play(study, spec, checkpoint, seed, storage_mode=None):
        rates.append((seed >= TEST_SEED_START, spec.arrival_rate))
        return FakeMetrics(seed=seed, num_completed_tasks=1, throughput=0.1)

    fake_episodes([1, 1])
    monkeypatch.setattr(evaluate_run_module, "play", play)
    run = _run_dir(tmp_path, 2)
    training_rate = load_spec(CONFIG).arrival_rate
    evaluate_run(run, tmp_path / "out", arrival_rate=0.3)
    assert {rate for is_test, rate in rates if not is_test} == {training_rate}
    assert {rate for is_test, rate in rates if is_test} == {0.3}


def test_the_command_names_the_output_after_the_load(tmp_path, fake_episodes):
    fake_episodes([1, 1])
    run = _run_dir(tmp_path, 2)
    result = CliRunner().invoke(cli, ["evaluate", str(run), "--arrival-rate", "0.3"])
    assert result.exit_code == 0, result.output
    assert (run / "evaluation_rate0.3" / "test.jsonl").exists()


ARRIVAL_CONFIG = (
    Path(__file__).resolve().parents[2] / "configs" / "medium_arrival" / "decentralised.yaml"
)


def _arrival_metrics(seed, storage_mode, service_time):
    """An arrival run that keeps up: its put-away and its wait for stock are reported."""
    values = _real_metrics(seed, storage_mode, service_time).model_dump()
    values.update(
        stock_model="arrival",
        num_storage_updates=1,
        mean_relocated_units=0.0,
        mean_stock_wait=1.0 if storage_mode == "fixed" else 0.5,
        mean_putaway_distance=110.0 if storage_mode == "fixed" else 54.0,
        mean_arrived_units=30.0,
        mean_split_skus=0.0 if storage_mode == "fixed" else 4.0,
        mean_order_completion=12.0 if storage_mode == "fixed" else 9.0,
        num_completed_orders=4,
    )
    return RunMetrics(**values)


def _summary_of(tmp_path, monkeypatch, metrics, spec_path):
    def play(study, spec, checkpoint, seed, storage_mode=None):
        if storage_mode is None:
            return FakeMetrics(seed=seed, num_completed_tasks=3, throughput=0.3)
        return metrics(seed, storage_mode, 10.0 if storage_mode == "fixed" else 8.0 + seed % 3)

    monkeypatch.setattr(evaluate_run_module, "play", play)
    run = _run_dir(tmp_path, 2)
    next(run.glob("*_spec.json")).write_text(
        load_spec(spec_path).model_dump_json(), encoding="utf-8"
    )
    evaluate_run(run, tmp_path / "out", ["fixed", "demand"])
    return (tmp_path / "out" / "summary.md").read_text().splitlines()


def _cells(line):
    return [cell.strip() for cell in line.strip("|").split("|")]


def test_a_template_summary_keeps_its_columns(tmp_path, fake_episodes, monkeypatch):
    fake_episodes([3, 9])
    run_spec = Path(__file__).resolve().parents[2] / "configs" / "medium" / "decentralised.yaml"
    lines = _summary_of(tmp_path, monkeypatch, _real_metrics, run_spec)
    header = evaluate_run_module._SUMMARY_HEADER.rstrip("\n").split("\n")
    assert lines[2:4] == header
    rows = [line for line in lines if line.startswith("| chosen") or line.startswith("| last")]
    assert len(rows) == 2
    assert all(len(_cells(row)) == 15 for row in rows)
    # Every F_fix run takes 200 steps in total, 100 of them travel and blocked, so the relative
    # gain is the total gain over 100. The total gain and the relative gain lead.
    chosen = _cells(rows[0])
    assert chosen[6].startswith(f"{float(chosen[4]) / 100:.3f} [")
    assert chosen[8].startswith(f"{float(chosen[4]) / 20:.3f} [")  # the mean pick gain
    assert "putaway" not in "\n".join(lines) and "stock wait" not in "\n".join(lines)


def test_an_arrival_summary_adds_putaway_stock_wait_and_order_completion(
    tmp_path, fake_episodes, monkeypatch
):
    fake_episodes([3, 9])
    lines = _summary_of(tmp_path, monkeypatch, _arrival_metrics, ARRIVAL_CONFIG)
    header = _cells(lines[2])
    assert len(header) == 20 and len(_cells(lines[3])) == 20
    assert header[15:] == [
        "putaway per update F_fix",
        "putaway per update mode",
        "stock wait F_fix",
        "stock wait mode",
        "order completion F_fix / mode",
    ]
    chosen = _cells(next(line for line in lines if line.startswith("| chosen")))
    assert chosen[:2] == ["chosen", "demand"]
    assert chosen[15:] == ["110.000", "54.000", "1.000", "0.500", "12.000 / 9.000"]


def test_the_test_seeds_start_from_the_stock_of_their_load(tmp_path, fake_episodes, monkeypatch):
    """The study is built at the training load for validation and again at the test load."""
    fake_episodes([1, 1])
    split = SeedSplitConfig(
        training_seed=0, train_seed_count=5, validation_seed_count=2, test_seed_count=3
    )
    monkeypatch.setattr(
        evaluate_run_module,
        "build_study",
        lambda spec: SimpleNamespace(seed_split=split, rate=spec.arrival_rate),
    )
    rates = []

    def play(study, spec, checkpoint, seed, storage_mode=None):
        rates.append((seed >= TEST_SEED_START, study.rate, spec.arrival_rate))
        return FakeMetrics(seed=seed, num_completed_tasks=1, throughput=0.1)

    monkeypatch.setattr(evaluate_run_module, "play", play)
    run = _run_dir(tmp_path, 2)
    training_rate = load_spec(CONFIG).arrival_rate
    evaluate_run(run, tmp_path / "out", arrival_rate=0.3)
    assert {(rate, spec_rate) for is_test, rate, spec_rate in rates if not is_test} == {
        (training_rate, training_rate)
    }
    assert {(rate, spec_rate) for is_test, rate, spec_rate in rates if is_test} == {(0.3, 0.3)}
