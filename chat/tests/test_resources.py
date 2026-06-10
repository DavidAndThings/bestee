import json
from pathlib import Path

import pytest

from bestee_chat import config
from bestee_chat.resources import (
    DumpResource,
    KnowledgeCacheResource,
    ModelResource,
    Resource,
    ResourceManager,
    ResourceStatus,
)
from bestee_chat.wiki.cache import get_layout
from bestee_chat.wiki.sources import WikiSource


class _FakeResource(Resource):
    kind = "fake"

    def __init__(self, name: str, present: bool = False) -> None:
        self.name = name
        self._present = present
        self.ensured = False
        self.cleaned = False

    def status(self) -> ResourceStatus:
        return ResourceStatus(self.name, self.kind, self._present, "loc", 10, "x")

    def ensure(self, *, force: bool = False, progress: object = None) -> None:
        self.ensured = True
        self._present = True

    def clean(self) -> None:
        self.cleaned = True
        self._present = False


# ---------------------------------------------------------------------------
# ResourceManager
# ---------------------------------------------------------------------------


def test_manager_status_lists_every_resource() -> None:
    manager = ResourceManager([_FakeResource("a"), _FakeResource("b", present=True)])
    statuses = manager.status()
    assert [s.name for s in statuses] == ["a", "b"]
    assert [s.present for s in statuses] == [False, True]


def test_manager_ensure_and_clean_apply_to_all() -> None:
    a, b = _FakeResource("a"), _FakeResource("b")
    manager = ResourceManager([a, b])
    manager.ensure()
    assert a.ensured and b.ensured
    manager.clean()
    assert a.cleaned and b.cleaned


# ---------------------------------------------------------------------------
# config helpers
# ---------------------------------------------------------------------------


def test_cache_root_resolution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("BESTEE_CACHE_DIR", raising=False)
    assert config.cache_root().name == "bestee-chat"
    monkeypatch.setenv("BESTEE_CACHE_DIR", str(tmp_path))
    assert config.cache_root() == tmp_path
    assert config.cache_root("/explicit/here") == Path("/explicit/here")


def test_is_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BESTEE_OFFLINE", raising=False)
    assert config.is_offline() is False
    monkeypatch.setenv("BESTEE_OFFLINE", "yes")
    assert config.is_offline() is True


def test_knowledge_dir_resolution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("BESTEE_PERSONS_DIR", raising=False)
    monkeypatch.delenv("BESTEE_ARTICLES_DIR", raising=False)
    monkeypatch.setenv("BESTEE_CACHE_DIR", str(tmp_path))
    assert config.knowledge_dir("persons") == tmp_path / "persons"
    assert config.persons_dir() == tmp_path / "persons"  # alias
    monkeypatch.setenv("BESTEE_ARTICLES_DIR", str(tmp_path / "a"))
    assert config.knowledge_dir("articles") == tmp_path / "a"


# ---------------------------------------------------------------------------
# KnowledgeCacheResource (no network)
# ---------------------------------------------------------------------------


def test_knowledge_cache_status(tmp_path: Path) -> None:
    directory = tmp_path / "persons"
    resource = KnowledgeCacheResource("persons", directory=directory)
    assert resource.kind == "persons"
    assert resource.status().present is False

    directory.mkdir(parents=True)
    (directory / "0001.json").write_text(
        json.dumps([{"name": "Ada"}, {"name": "Alan"}]), encoding="utf-8"
    )
    status = resource.status()
    assert status.present is True
    assert "2 records" in status.detail


def test_knowledge_cache_ensure_is_a_noop(tmp_path: Path) -> None:
    resource = KnowledgeCacheResource("articles", directory=tmp_path / "articles")
    resource.ensure()  # nothing to download
    assert resource.status().present is False


def test_knowledge_cache_clean(tmp_path: Path) -> None:
    directory = tmp_path / "articles"
    directory.mkdir(parents=True)
    (directory / "0001.json").write_text("[]", encoding="utf-8")
    KnowledgeCacheResource("articles", directory=directory).clean()
    assert not directory.exists()


# ---------------------------------------------------------------------------
# DumpResource (no network)
# ---------------------------------------------------------------------------


def test_dump_resource_status_present_and_missing(tmp_path: Path) -> None:
    source = WikiSource(lang="simple")
    resource = DumpResource(source, cache_dir=str(tmp_path))
    assert resource.status().present is False

    layout = get_layout(source, str(tmp_path))
    layout.dump_path.parent.mkdir(parents=True, exist_ok=True)
    layout.dump_path.write_bytes(b"x" * 123)

    status = resource.status()
    assert status.present is True
    assert status.kind == "dump"
    assert status.size_bytes == 123


def test_dump_resource_clean_removes_source_dir(tmp_path: Path) -> None:
    source = WikiSource(lang="simple")
    layout = get_layout(source, str(tmp_path))
    layout.dump_path.parent.mkdir(parents=True, exist_ok=True)
    layout.dump_path.write_bytes(b"x")

    DumpResource(source, cache_dir=str(tmp_path)).clean()
    assert not layout.source_dir.exists()


def test_offline_ensure_raises_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BESTEE_OFFLINE", "1")
    resource = DumpResource(WikiSource(lang="simple"), cache_dir=str(tmp_path))
    with pytest.raises(RuntimeError):
        resource.ensure()


# ---------------------------------------------------------------------------
# ModelResource (no network: point the HF cache at an empty dir)
# ---------------------------------------------------------------------------


def test_model_resource_reports_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    resource = ModelResource("nonexistent/model-xyz")
    assert resource.kind == "model"
    assert resource.status().present is False

    monkeypatch.setenv("BESTEE_OFFLINE", "1")
    with pytest.raises(RuntimeError):
        resource.ensure()
