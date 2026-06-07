"""Catalog of downloadable Wikipedia dumps and their canonical URLs.

A :class:`WikiSource` names *which* dump to work with (language + whether to
use the random-access ``multistream`` variant) and knows how to build the
concrete file URLs on a Wikimedia mirror. It is pure data + string building,
so it is fully unit-testable without any network access.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote

#: Primary Wikimedia dump mirror. Override per-source for a closer/faster one.
DEFAULT_MIRROR = "https://dumps.wikimedia.org"


@dataclass(frozen=True, slots=True)
class WikiSource:
    """Identifies a single Wikipedia dump.

    Parameters
    ----------
    lang:
        Wiki language code, e.g. ``"simple"`` (Simple English) or ``"en"``
        (full English). Defaults to ``"simple"`` because it is ~50x smaller
        and exercises the whole pipeline on a laptop in minutes.
    multistream:
        Use the ``multistream`` dump (plus its index), which supports
        random access to individual pages. Recommended for large wikis.
    dated:
        A specific dump date as ``"YYYYMMDD"``, or ``None`` for ``"latest"``.
        Pinning a date makes a build reproducible.
    mirror:
        Base URL of the Wikimedia mirror to download from.
    """

    lang: str = "simple"
    multistream: bool = True
    dated: str | None = None
    mirror: str = DEFAULT_MIRROR

    @property
    def dbname(self) -> str:
        """The Wikimedia database name, e.g. ``"simplewiki"`` or ``"enwiki"``."""
        return f"{self.lang}wiki"

    @property
    def date_segment(self) -> str:
        """The path/date segment used in dump URLs (``latest`` or a date)."""
        return self.dated or "latest"

    @property
    def base_url(self) -> str:
        """Directory URL containing this dump's files on the mirror."""
        return f"{self.mirror}/{self.dbname}/{self.date_segment}"

    @property
    def data_filename(self) -> str:
        """Filename of the articles dump (``.xml.bz2``)."""
        variant = "-multistream" if self.multistream else ""
        return f"{self.dbname}-{self.date_segment}-pages-articles{variant}.xml.bz2"

    @property
    def index_filename(self) -> str | None:
        """Filename of the multistream offset index, or ``None`` if not used."""
        if not self.multistream:
            return None
        return (
            f"{self.dbname}-{self.date_segment}"
            "-pages-articles-multistream-index.txt.bz2"
        )

    @property
    def checksums_filename(self) -> str:
        """Filename of the published sha1 checksum manifest."""
        return f"{self.dbname}-{self.date_segment}-sha1sums.txt"

    @property
    def data_url(self) -> str:
        """Full URL of the articles dump."""
        return f"{self.base_url}/{self.data_filename}"

    @property
    def index_url(self) -> str | None:
        """Full URL of the multistream index, or ``None`` if not used."""
        if self.index_filename is None:
            return None
        return f"{self.base_url}/{self.index_filename}"

    @property
    def checksums_url(self) -> str:
        """Full URL of the sha1 checksum manifest."""
        return f"{self.base_url}/{self.checksums_filename}"

    @property
    def slug(self) -> str:
        """Stable cache slug for this source, e.g. ``"simplewiki/latest"``."""
        return f"{self.dbname}/{self.date_segment}"

    def page_url(self, title: str) -> str:
        """Return the live Wikipedia URL for an article ``title``."""
        path = quote(title.replace(" ", "_"), safe="/:()-_,'")
        return f"https://{self.lang}.wikipedia.org/wiki/{path}"
