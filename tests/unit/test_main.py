"""Unit tests for slap_mapd_coupling.main (the CLI).

These only cover argument validation -- every command's actual body is
`raise NotImplementedError(...)` (nothing to implement is built yet), so
tests assert that placeholder is what's reached once validation passes,
not any real behaviour.
"""

from click.testing import CliRunner

from slap_mapd_coupling.main import cli


def test_cli_help_exits_zero():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0


def test_simulate_requires_checkpoint_for_decentralised():
    result = CliRunner().invoke(cli, ["simulate", "--controller", "decentralised"])
    assert result.exit_code != 0
    assert "--checkpoint is required" in result.output


def test_simulate_rejects_checkpoint_for_non_decentralised(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_text("x")
    result = CliRunner().invoke(
        cli, ["simulate", "--controller", "centralised", "--checkpoint", str(checkpoint)]
    )
    assert result.exit_code != 0
    assert "--checkpoint only applies" in result.output


def test_simulate_rejects_communication_for_non_decentralised():
    result = CliRunner().invoke(cli, ["simulate", "--controller", "centralised", "--communication"])
    assert result.exit_code != 0
    assert "--communication only applies" in result.output


def test_simulate_reaches_not_implemented_once_validation_passes():
    result = CliRunner().invoke(cli, ["simulate", "--controller", "centralised"])
    assert isinstance(result.exception, NotImplementedError)


def test_train_reaches_not_implemented_once_validation_passes():
    result = CliRunner().invoke(cli, ["train"])
    assert isinstance(result.exception, NotImplementedError)


def test_sweep_requires_config():
    result = CliRunner().invoke(cli, ["sweep"])
    assert result.exit_code != 0


def test_sweep_reaches_not_implemented_once_validation_passes(tmp_path):
    config = tmp_path / "grid.yaml"
    config.write_text("{}")
    result = CliRunner().invoke(cli, ["sweep", "--config", str(config)])
    assert isinstance(result.exception, NotImplementedError)
