"""training/evaluate_run.py: the checkpoint is chosen on the validation seeds and scored on the
test seeds. Episodes are replaced by a fake, so these check the choice and the rows, not a run."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

import slap_mapd_coupling.training.evaluate_run as evaluate_run_module
from slap_mapd_coupling.main import cli
from slap_mapd_coupling.training.config import (
    TEST_SEED_START,
    VALIDATION_SEED_START,
    SeedSplitConfig,
)
from slap_mapd_coupling.training.evaluate_run import checkpoints_of, evaluate_run

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
        def play(study, spec, checkpoint, seed):
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


def test_evaluate_rejects_a_folder_without_a_spec(tmp_path):
    result = CliRunner().invoke(cli, ["evaluate", str(tmp_path)])
    assert result.exit_code != 0
    assert "_spec.json" in result.output
