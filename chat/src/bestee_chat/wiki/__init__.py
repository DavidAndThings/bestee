"""Download a Wikipedia content dump and search it locally.

A hardware-adaptive, offline search tool over a full Wikipedia dump. The
lexical (BM25) layer always works with zero ML dependencies; a semantic
reranking layer engages automatically when ``torch`` and the hardware allow.

Public API
----------
* :class:`WikiSource` -- which dump to use.
* :func:`download_dump` -- fetch + verify a dump.
* :func:`build_index` -- build the lexical (and optional vector) index.
* :func:`search_wiki` -- query a built index.
* :class:`WikiKnowledge` -- plug dump search into a chatbot ``Brain``.
* :func:`probe_hardware` / :func:`resolve_profile` -- inspect what ``auto`` picks.
"""

from __future__ import annotations

from bestee_chat.wiki.builder import BuildResult, build_index
from bestee_chat.wiki.download import download_dump
from bestee_chat.wiki.hardware import HardwareProfile, probe_hardware
from bestee_chat.wiki.knowledge import WikiKnowledge
from bestee_chat.wiki.models import RawPage, SearchHit
from bestee_chat.wiki.profiles import Profile, list_profiles, resolve_profile
from bestee_chat.wiki.query import search_wiki
from bestee_chat.wiki.sources import WikiSource

__all__ = [
    "BuildResult",
    "HardwareProfile",
    "Profile",
    "RawPage",
    "SearchHit",
    "WikiKnowledge",
    "WikiSource",
    "build_index",
    "download_dump",
    "list_profiles",
    "probe_hardware",
    "resolve_profile",
    "search_wiki",
]
