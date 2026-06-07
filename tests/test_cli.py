import json
from pathlib import Path

from click.testing import CliRunner

from bestee_chat.cli import cli


def test_help_lists_groups() -> None:
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "resources" in result.output
    assert "wiki" in result.output


def test_resources_group_owns_the_lifecycle() -> None:
    result = CliRunner().invoke(cli, ["resources", "--help"])
    assert result.exit_code == 0
    for command in ("status", "download", "index", "clean", "info"):
        assert command in result.output


def test_top_level_has_learn() -> None:
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "learn" in result.output


def test_resources_status_json(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli,
        ["resources", "status", "--cache-dir", str(tmp_path / "cache"), "--json"],
        env={
            "HF_HOME": str(tmp_path / "hf"),
            "BESTEE_PERSONS_DIR": str(tmp_path / "persons"),
            "BESTEE_ARTICLES_DIR": str(tmp_path / "articles"),
            "BESTEE_OFFLINE": "",
        },
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "hardware" in data
    assert "auto_profile" in data
    kinds = {entry["kind"] for entry in data["resources"]}
    assert kinds == {"model", "dump", "persons", "articles"}


def test_resources_status_text(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli,
        ["resources", "status", "--cache-dir", str(tmp_path / "cache")],
        env={
            "HF_HOME": str(tmp_path / "hf"),
            "BESTEE_PERSONS_DIR": str(tmp_path / "persons"),
            "BESTEE_ARTICLES_DIR": str(tmp_path / "articles"),
        },
    )
    assert result.exit_code == 0, result.output
    assert "hardware" in result.output
    assert "profile" in result.output


def test_learn_has_person_and_article() -> None:
    result = CliRunner().invoke(cli, ["learn", "--help"])
    assert result.exit_code == 0
    assert "person" in result.output
    assert "article" in result.output


def test_learn_person_refuses_when_offline() -> None:
    result = CliRunner().invoke(
        cli,
        ["learn", "person", "https://en.wikipedia.org/wiki/Ada_Lovelace"],
        env={"BESTEE_OFFLINE": "1"},
    )
    assert result.exit_code != 0


def test_learn_article_without_index_errors(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli, ["learn", "article", "anything", "--cache-dir", str(tmp_path)]
    )
    assert result.exit_code != 0


def test_resources_clean_requires_a_selection() -> None:
    result = CliRunner().invoke(cli, ["resources", "clean"])
    assert result.exit_code != 0


def test_resources_download_requires_a_selection() -> None:
    result = CliRunner().invoke(cli, ["resources", "download", "--no-model"])
    assert result.exit_code != 0


def test_resources_clean_dump_runs(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli, ["resources", "clean", "--dump", "--yes", "--cache-dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert "removed" in result.output


def test_resources_index_offline_without_dump_errors(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli,
        ["resources", "index", "--cache-dir", str(tmp_path)],
        env={"BESTEE_OFFLINE": "1"},
    )
    assert result.exit_code != 0


def test_wiki_search_without_index_errors(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli, ["wiki", "--cache-dir", str(tmp_path), "search", "anything"]
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, FileNotFoundError)
