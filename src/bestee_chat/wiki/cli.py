"""Command-line interface for the Wikipedia dump tool.

A thin ``argparse`` wrapper over the library functions -- it never reimplements
logic. Subcommands: ``status``, ``download``, ``index``, ``search``, ``info``
and ``clear``.

    bestee-chat-wiki status
    bestee-chat-wiki download --lang simple
    bestee-chat-wiki index --lang simple --profile auto
    bestee-chat-wiki search "List of English kings" --lang simple
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys

from bestee_chat.wiki.builder import build_index
from bestee_chat.wiki.cache import Manifest, get_layout
from bestee_chat.wiki.download import download_dump
from bestee_chat.wiki.hardware import probe_hardware
from bestee_chat.wiki.profiles import resolve_profile
from bestee_chat.wiki.query import search_wiki
from bestee_chat.wiki.sources import WikiSource


def _source_from_args(args: argparse.Namespace) -> WikiSource:
    return WikiSource(
        lang=args.lang,
        multistream=not args.no_multistream,
        dated=args.dated,
    )


def _progress(label: str):
    def report(done: int, total: int | None) -> None:
        if total:
            pct = 100 * done / total
            print(f"\r{label}: {pct:5.1f}%  ({done}/{total})", end="", file=sys.stderr)
        else:
            print(f"\r{label}: {done}", end="", file=sys.stderr)

    return report


def _cmd_status(args: argparse.Namespace) -> int:
    source = _source_from_args(args)
    hw = probe_hardware(args.cache_dir)
    profile = resolve_profile("auto", hw)
    layout = get_layout(source, args.cache_dir)
    manifest = Manifest.load(layout.manifest_path)

    if args.json:
        print(
            json.dumps(
                {
                    "hardware": vars(hw),
                    "auto_profile": profile.name,
                    "source": source.slug,
                    "dump_present": layout.dump_path.exists(),
                    "indexes": sorted(manifest.indexes),
                },
                indent=2,
            )
        )
        return 0

    print(
        f"hardware : {hw.cores} cores, {hw.ram_gb} GB RAM, "
        f"{hw.free_disk_gb} GB free disk, accel={hw.accel}"
    )
    print(f"profile  : auto -> {profile.name}")
    print(f"source   : {source.slug}")
    print(f"dump     : {'present' if layout.dump_path.exists() else 'not downloaded'}")
    built = sorted(manifest.indexes) or ["(none)"]
    print(f"indexes  : {', '.join(built)}")
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    source = _source_from_args(args)
    path = download_dump(
        source,
        cache_dir=args.cache_dir,
        refresh=args.force,
        progress=_progress("downloading"),
    )
    print(f"\nsaved -> {path}", file=sys.stderr)
    return 0


def _cmd_index(args: argparse.Namespace) -> int:
    source = _source_from_args(args)
    result = build_index(
        source,
        args.profile,
        cache_dir=args.cache_dir,
        lexical_progress=_progress("indexing"),
        vector_progress=_progress("embedding"),
    )
    print(
        f"\nbuilt profile={result.profile.name} "
        f"articles={result.stats.articles} "
        f"redirects={result.stats.redirects} "
        f"vectors={'yes' if result.vectors_built else 'no'}",
        file=sys.stderr,
    )
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    source = _source_from_args(args)
    if args.build:
        build_index(source, args.profile, cache_dir=args.cache_dir)
    hits = search_wiki(
        args.query,
        source=source,
        profile=args.profile,
        limit=args.limit,
        cache_dir=args.cache_dir,
    )
    if args.json:
        print(json.dumps([vars(h) for h in hits], indent=2))
        return 0
    if not hits:
        print("no results")
        return 0
    for hit in hits:
        print(f"{hit.rank:>2}. {hit.title}  [{hit.score:.3f}]")
        print(f"    {hit.snippet}")
        print(f"    {hit.url}")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    source = _source_from_args(args)
    layout = get_layout(source, args.cache_dir)
    manifest = Manifest.load(layout.manifest_path)
    print(json.dumps(vars(manifest), indent=2, default=str))
    return 0


def _cmd_clear(args: argparse.Namespace) -> int:
    source = _source_from_args(args)
    layout = get_layout(source, args.cache_dir)
    if args.index:
        for path in layout.source_dir.glob("index.*"):
            shutil.rmtree(path, ignore_errors=True)
        print(f"cleared indexes under {layout.source_dir}")
    else:
        shutil.rmtree(layout.source_dir, ignore_errors=True)
        print(f"cleared {layout.source_dir}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bestee-chat-wiki", description=__doc__)
    parser.add_argument("--lang", default="simple", help="wiki language code")
    parser.add_argument("--dated", default=None, help="dump date YYYYMMDD or latest")
    parser.add_argument(
        "--no-multistream", action="store_true", help="use the single-stream dump"
    )
    parser.add_argument("--cache-dir", default=None, help="override the cache root")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="probe hardware and show cache state")

    p_dl = sub.add_parser("download", help="download and verify the dump")
    p_dl.add_argument("--force", action="store_true", help="re-download if cached")

    p_idx = sub.add_parser("index", help="build/refresh the index")
    p_idx.add_argument("--profile", default="auto", help="quality profile")

    p_search = sub.add_parser("search", help="query the index")
    p_search.add_argument("query", help="free-text query")
    p_search.add_argument("--profile", default="auto", help="quality profile")
    p_search.add_argument("--limit", type=int, default=10, help="number of results")
    p_search.add_argument(
        "--build", action="store_true", help="build the index first if missing"
    )

    sub.add_parser("info", help="print the cache manifest")

    p_clear = sub.add_parser("clear", help="delete cached artifacts")
    p_clear.add_argument(
        "--index", action="store_true", help="only delete indexes, keep the dump"
    )
    return parser


_COMMANDS = {
    "status": _cmd_status,
    "download": _cmd_download,
    "index": _cmd_index,
    "search": _cmd_search,
    "info": _cmd_info,
    "clear": _cmd_clear,
}


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``bestee-chat-wiki`` console script."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    return _COMMANDS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
