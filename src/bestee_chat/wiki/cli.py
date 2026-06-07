"""Click CLI for *searching* a Wikipedia dump.

The resource lifecycle (download / index / clean / status / info) lives under
``bestee-chat resources``; this group is just the query surface, exposed as
``bestee-chat wiki`` and the ``bestee-chat-wiki`` console script::

    bestee-chat wiki search "List of English kings" --lang simple
"""

from __future__ import annotations

import json
from dataclasses import asdict

import click

from bestee_chat import config
from bestee_chat.wiki.builder import build_index
from bestee_chat.wiki.query import search_wiki
from bestee_chat.wiki.sources import WikiSource


@click.group()
@click.option("--lang", default="simple", show_default=True, help="wiki language code")
@click.option("--dated", default=None, help="dump date YYYYMMDD (default: latest)")
@click.option("--no-multistream", is_flag=True, help="use the single-stream dump")
@click.option("--cache-dir", default=None, help="override the cache root")
@click.option("--json", "as_json", is_flag=True, help="machine-readable output")
@click.pass_context
def wiki(ctx, lang, dated, no_multistream, cache_dir, as_json) -> None:
    """Search a downloaded Wikipedia dump."""
    config.apply_offline()
    ctx.obj = {
        "source": WikiSource(lang=lang, multistream=not no_multistream, dated=dated),
        "cache_dir": cache_dir,
        "json": as_json,
    }


@wiki.command()
@click.argument("query")
@click.option("--profile", default="auto", show_default=True, help="quality profile")
@click.option("--limit", default=10, show_default=True, type=int, help="result count")
@click.option("--build", "build_first", is_flag=True, help="build the index first")
@click.pass_obj
def search(obj, query, profile, limit, build_first) -> None:
    """Query the index for QUERY."""
    source = obj["source"]
    cache_dir = obj["cache_dir"]
    if build_first:
        build_index(source, profile, cache_dir=cache_dir)
    hits = search_wiki(
        query, source=source, profile=profile, limit=limit, cache_dir=cache_dir
    )
    if obj["json"]:
        click.echo(json.dumps([asdict(hit) for hit in hits], indent=2))
        return
    if not hits:
        click.echo("no results")
        return
    for hit in hits:
        click.echo(f"{hit.rank:>2}. {hit.title}  [{hit.score:.3f}]")
        click.echo(f"    {hit.snippet}")
        click.echo(f"    {hit.url}")


def main() -> None:
    """Entry point for the ``bestee-chat-wiki`` console script."""
    wiki()


if __name__ == "__main__":
    main()
