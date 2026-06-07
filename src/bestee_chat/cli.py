"""Unified ``bestee-chat`` command line.

Resource management is consolidated under one tool. ``bestee-chat resources``
owns the whole lifecycle of every external resource -- the embedding model and
the Wikipedia dumps + indexes -- so download, index, clean and status all live
in one place:

* ``bestee-chat resources status``   -- hardware, auto profile, what is cached.
* ``bestee-chat resources download`` -- fetch the model and/or a dump.
* ``bestee-chat resources index``    -- build a dump's search index.
* ``bestee-chat resources clean``    -- delete cached resources.
* ``bestee-chat resources info``     -- print a dump's manifest.
* ``bestee-chat learn person``       -- learn a person from a live URL.
* ``bestee-chat learn article``      -- learn an article from a dump search.
* ``bestee-chat wiki search``        -- query a built index.
* ``bestee-chat chat``               -- the (placeholder) chat loop.

``BESTEE_OFFLINE=1`` makes every subcommand avoid the network.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import asdict

import click

from bestee_chat import config
from bestee_chat.resources import (
    DumpResource,
    KnowledgeCacheResource,
    ModelResource,
    Resource,
    default_manager,
)
from bestee_chat.wiki.cache import Manifest, get_layout
from bestee_chat.wiki.cli import wiki
from bestee_chat.wiki.hardware import probe_hardware
from bestee_chat.wiki.profiles import resolve_profile
from bestee_chat.wiki.sources import WikiSource


def _human_size(size_bytes: int | None) -> str:
    """Render a byte count as a short human-readable string."""
    if size_bytes is None:
        return "-"
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _progress(label: str) -> Callable[[int, int | None], None]:
    """Return a progress callback that writes a throttled line to stderr."""

    def report(done: int, total: int | None) -> None:
        if total:
            pct = 100 * done / total
            print(f"\r{label}: {pct:5.1f}%  ({done}/{total})", end="", file=sys.stderr)
        else:
            print(f"\r{label}: {done}", end="", file=sys.stderr)

    return report


def _source_options[F: Callable[..., object]](func: F) -> F:
    """Attach the shared dump-source options to a command."""
    func = click.option("--cache-dir", default=None, help="override the cache root")(
        func
    )
    func = click.option(
        "--no-multistream", is_flag=True, help="use the single-stream dump"
    )(func)
    func = click.option("--dated", default=None, help="dump date YYYYMMDD")(func)
    func = click.option(
        "--lang", default="simple", show_default=True, help="wiki language code"
    )(func)
    return func


def _make_source(lang: str, dated: str | None, no_multistream: bool) -> WikiSource:
    return WikiSource(lang=lang, multistream=not no_multistream, dated=dated)


@click.group()
def cli() -> None:
    """bestee-chat: a small conversational chatbot toolkit."""
    config.apply_offline()


@cli.group()
def resources() -> None:
    """Manage external resources: the embedding model and Wikipedia dumps."""


@resources.command("status")
@_source_options
@click.option("--json", "as_json", is_flag=True, help="machine-readable output")
def resources_status(lang, dated, no_multistream, cache_dir, as_json) -> None:
    """Show hardware, the auto profile, and which resources are present."""
    source = _make_source(lang, dated, no_multistream)
    hardware = probe_hardware(cache_dir)
    profile = resolve_profile("auto", hardware)
    statuses = default_manager(source=source, cache_dir=cache_dir).status()

    if as_json:
        click.echo(
            json.dumps(
                {
                    "hardware": asdict(hardware),
                    "auto_profile": profile.name,
                    "resources": [asdict(s) for s in statuses],
                },
                indent=2,
            )
        )
        return

    click.echo(
        f"hardware : {hardware.cores} cores, {hardware.ram_gb} GB RAM, "
        f"{hardware.free_disk_gb} GB free, accel={hardware.accel}"
    )
    click.echo(f"profile  : auto -> {profile.name}")
    suffix = " (offline)" if config.is_offline() else ""
    click.echo(f"resources{suffix}:")
    for s in statuses:
        mark = "present" if s.present else "missing"
        click.echo(f"  [{mark:>7}] {s.name}  ({_human_size(s.size_bytes)})  {s.detail}")
        click.echo(f"            {s.location}")


@resources.command("download")
@click.option("--model/--no-model", default=True, help="download the embedding model")
@click.option("--dump", "do_dump", is_flag=True, help="also download the wiki dump")
@_source_options
@click.option("--force", is_flag=True, help="re-download even if present")
def resources_download(model, do_dump, lang, dated, no_multistream, cache_dir, force):
    """Download the selected resources for offline use."""
    source = _make_source(lang, dated, no_multistream)
    items: list[Resource] = []
    if model:
        items.append(ModelResource())
    if do_dump:
        items.append(DumpResource(source, cache_dir))
    if not items:
        raise click.UsageError("nothing selected; enable --model and/or --dump")
    for resource in items:
        if resource.status().present and not force:
            click.echo(f"already present: {resource.name}")
            continue
        click.echo(f"downloading {resource.name} ...")
        resource.ensure(force=force, progress=_progress("  "))
        click.echo(f"ready: {resource.name}")


@resources.command("index")
@_source_options
@click.option("--profile", default="auto", show_default=True, help="quality profile")
def resources_index(lang, dated, no_multistream, cache_dir, profile) -> None:
    """Build (or rebuild) the search index for a wiki dump."""
    source = _make_source(lang, dated, no_multistream)
    dump = DumpResource(source, cache_dir)
    result = dump.build_index(
        profile,
        lexical_progress=_progress("indexing"),
        vector_progress=_progress("embedding"),
    )
    click.echo(
        f"\nbuilt profile={result.profile.name} "
        f"articles={result.stats.articles} "
        f"redirects={result.stats.redirects} "
        f"vectors={'yes' if result.vectors_built else 'no'}",
        err=True,
    )


@resources.command("clean")
@click.option("--model", "do_model", is_flag=True, help="remove the embedding model")
@click.option("--dump", "do_dump", is_flag=True, help="remove the wiki dump + indexes")
@click.option("--persons", "do_persons", is_flag=True, help="remove learned people")
@click.option("--articles", "do_articles", is_flag=True, help="remove learned articles")
@click.option("--index-only", is_flag=True, help="with --dump, remove only indexes")
@_source_options
@click.option("--yes", is_flag=True, help="skip the confirmation prompt")
def resources_clean(
    do_model,
    do_dump,
    do_persons,
    do_articles,
    index_only,
    lang,
    dated,
    no_multistream,
    cache_dir,
    yes,
) -> None:
    """Delete cached resources (select with --model/--dump/--persons/--articles)."""
    if not (do_model or do_dump or do_persons or do_articles):
        raise click.UsageError(
            "nothing selected; pass --model, --dump, --persons and/or --articles"
        )

    actions: list[tuple[str, Callable[[], None]]] = []
    if do_model:
        model = ModelResource()
        actions.append((model.name, model.clean))
    if do_dump:
        dump = DumpResource(_make_source(lang, dated, no_multistream), cache_dir)
        if index_only:
            actions.append((f"{dump.name} (indexes)", dump.clean_indexes))
        else:
            actions.append((dump.name, dump.clean))
    if do_persons:
        persons = KnowledgeCacheResource("persons")
        actions.append((persons.name, persons.clean))
    if do_articles:
        articles = KnowledgeCacheResource("articles")
        actions.append((articles.name, articles.clean))

    names = ", ".join(label for label, _ in actions)
    if not yes:
        click.confirm(f"Delete {names}?", abort=True)
    for _, action in actions:
        action()
    click.echo(f"removed: {names}")


@resources.command("info")
@_source_options
def resources_info(lang, dated, no_multistream, cache_dir) -> None:
    """Print the on-disk manifest for a wiki dump."""
    source = _make_source(lang, dated, no_multistream)
    layout = get_layout(source, cache_dir)
    manifest = Manifest.load(layout.manifest_path)
    click.echo(json.dumps(asdict(manifest), indent=2, default=str))


@cli.group()
def learn() -> None:
    """Learn knowledge of different kinds and cache it for the brain."""


@learn.command("person")
@click.argument("target")
@click.option(
    "--dump",
    "from_dump",
    is_flag=True,
    help="read TARGET as an article title from a local dump (offline) "
    "instead of scraping a live URL",
)
@_source_options
def learn_person_cmd(target, from_dump, lang, dated, no_multistream, cache_dir) -> None:
    """Learn a person from a live Wikipedia URL, or a local dump with --dump.

    Without --dump, TARGET is a page URL scraped over HTTP. With --dump, TARGET
    is an article title looked up in a downloaded dump (works offline); the
    --lang/--dated/--no-multistream/--cache-dir options select which dump.
    """
    from bestee_chat.learn import learn_person

    if from_dump:
        source = _make_source(lang, dated, no_multistream)
        try:
            path = learn_person(target, source=source, cache_dir=cache_dir)
        except (FileNotFoundError, LookupError) as exc:
            raise click.UsageError(str(exc)) from exc
    else:
        if config.is_offline():
            raise click.UsageError(
                "cannot scrape a live URL while BESTEE_OFFLINE is set; "
                "use --dump to learn from a local dump instead"
            )
        path = learn_person(target)
    click.echo(f"learned person -> {path}")


@learn.command("article")
@click.argument("query")
@_source_options
@click.option("--profile", default="auto", show_default=True, help="quality profile")
def learn_article_cmd(query, lang, dated, no_multistream, cache_dir, profile) -> None:
    """Learn an article by searching a local Wikipedia dump for QUERY."""
    from bestee_chat.learn import learn_article

    source = _make_source(lang, dated, no_multistream)
    try:
        path = learn_article(query, source=source, profile=profile, cache_dir=cache_dir)
    except (FileNotFoundError, LookupError) as exc:
        raise click.UsageError(str(exc)) from exc
    click.echo(f"learned article -> {path}")


@cli.command()
def chat() -> None:
    """Run the (placeholder) chat loop."""
    from bestee_chat.main import main as run_chat

    run_chat()


cli.add_command(wiki)


def main() -> None:
    """Entry point for the ``bestee-chat`` console script."""
    cli()


if __name__ == "__main__":
    main()
