import json
from pathlib import Path

from bestee_chat.storage import store_mapping


def _record(name: str, **extra: str) -> dict[str, str]:
    return {"name": name, **extra}


def _names(path: Path) -> list[str]:
    return [r["name"] for r in json.loads(path.read_text(encoding="utf-8"))]


def test_creates_directory_and_stores(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    assert not cache_dir.exists()

    target = store_mapping(_record("Ada"), cache_dir)

    assert cache_dir.is_dir()
    assert target.parent == cache_dir
    assert _names(target) == ["Ada"]


def test_multiple_records_share_one_file(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    for name in ["Ada", "Alan", "Grace"]:
        store_mapping(_record(name), cache_dir)

    files = list(cache_dir.glob("*.json"))
    assert len(files) == 1
    assert _names(files[0]) == ["Ada", "Alan", "Grace"]


def test_stores_into_the_smallest_file(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    big = cache_dir / "0001.json"
    big.write_text(json.dumps([{"name": "Existing", "pad": "y" * 500}]))
    small = cache_dir / "0002.json"
    small.write_text(json.dumps([{"name": "Ada"}]))

    target = store_mapping(_record("Grace"), cache_dir, max_bytes=10_000)

    assert target == small
    assert _names(small) == ["Ada", "Grace"]
    assert _names(big) == ["Existing"]  # untouched


def test_spills_over_when_file_exceeds_max(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"

    first = store_mapping(_record("Ada", bio="x" * 300), cache_dir, max_bytes=200)
    second = store_mapping(_record("Alan"), cache_dir, max_bytes=200)

    assert first != second
    assert first.name == "0001.json"
    assert second.name == "0002.json"
    # The overflowed record spilled into a new file; the first file is intact.
    assert _names(first) == ["Ada"]
    assert _names(second) == ["Alan"]


def test_single_oversized_record_is_kept(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"

    # A lone record bigger than max can't be split, so it stays in one file.
    target = store_mapping(_record("Ada", bio="x" * 500), cache_dir, max_bytes=100)

    assert _names(target) == ["Ada"]
    assert len(list(cache_dir.glob("*.json"))) == 1


def test_corrupt_file_is_treated_as_empty(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "0001.json").write_text("{ not valid json")

    target = store_mapping(_record("Ada"), cache_dir)

    assert _names(target) == ["Ada"]
