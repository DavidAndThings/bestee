# bestee-chat

A small conversational chatbot.

## Setup

```sh
uv sync
```

## Command line

Everything is under the unified `bestee-chat` command:

```sh
uv run bestee-chat --help
```

| Command | Purpose |
| --- | --- |
| `bestee-chat resources status` | hardware, the auto profile, and what is cached |
| `bestee-chat resources download [--dump]` | download the embedding model (and optionally a dump) |
| `bestee-chat resources index` | build a Wikipedia dump's search index |
| `bestee-chat resources clean --model/--dump/--persons/--articles` | delete cached resources |
| `bestee-chat learn person <url>` | learn a person from a live Wikipedia page |
| `bestee-chat learn person --dump <title>` | learn a person from a downloaded dump (offline) |
| `bestee-chat learn article <query>` | learn an article by searching a local dump |
| `bestee-chat wiki search <query>` | query a built dump index |
| `bestee-chat chat` | the (placeholder) chat loop |

Set `BESTEE_OFFLINE=1` to forbid all network access (model and dumps), and
`BESTEE_CACHE_DIR` to relocate every cache (default `~/.cache/bestee-chat`).

## Semantic similarity

A `Knowledge` source scores how close a query is in meaning to its stored
answers using local sentence embeddings (via `sentence-transformers`), and the
`Brain` returns the best `Candidate` across all sources. The model runs entirely
on-device; weights are downloaded once from the Hugging Face Hub and cached
locally.

Pre-download the embedding model (useful before going offline):

```sh
uv run bestee-chat resources download
```

Afterwards a single switch forces fully offline operation (it also sets
`HF_HUB_OFFLINE`):

```sh
BESTEE_OFFLINE=1 uv run bestee-chat wiki search "..."
```

### Hardware adaptivity

The encoder automatically selects the best available device, preferring a
CUDA GPU, then an Apple Silicon (MPS) GPU, then the CPU. Override the defaults
with environment variables:

| Variable                  | Purpose                          | Example                  |
| ------------------------- | -------------------------------- | ------------------------ |
| `BESTEE_DEVICE`           | Force a specific torch device    | `cpu`, `cuda`, `mps`     |
| `BESTEE_EMBEDDING_MODEL`  | Use a different embedding model  | `all-mpnet-base-v2`      |

On more capable hardware, switch to a larger, higher-quality model:

```sh
BESTEE_EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2 uv run bestee-chat
```

## Wikipedia infobox scraping

`scrape_infobox` fetches a Wikipedia person page and returns its infobox as a
plain, JSON-serializable dict:

```python
import json

from bestee_chat.wikipedia import scrape_infobox

data = scrape_infobox("https://en.wikipedia.org/wiki/Ada_Lovelace")
print(json.dumps(data, indent=2))
```

The keys are the infobox row labels (plus a `"name"` key from the title), with
citations and hidden markup stripped out. Returns an empty dict if the page has
no infobox.

The same data can be learned **offline** from a downloaded dump, parsing the
infobox out of the page's wikitext instead of scraping HTML. Pass a
`WikiSource` and an article title (rather than a URL):

```python
from bestee_chat.wikipedia import learn_about_a_person
from bestee_chat.wiki.sources import WikiSource

# Online: scrape a live page.
learn_about_a_person("https://simple.wikipedia.org/wiki/Ada_Lovelace")

# Offline: look the title up in an already-downloaded dump.
learn_about_a_person("Ada Lovelace", source=WikiSource())
```

From the CLI, `bestee-chat learn person --dump "Ada Lovelace"` does the same
(with `--lang`/`--dated`/`--cache-dir` selecting the dump). The dump path
follows redirects and resolves common infobox templates (dates, lists,
marriages) into readable text.

## Wikipedia dump search (`bestee_chat.wiki`)

Download a full Wikipedia content dump and search it locally. The lexical
(BM25) layer always works with zero extra dependencies; a semantic reranking
layer engages automatically when `torch` and the hardware allow. See
`docs/wikipedia_module_plan.md` for the full design.

### CLI

```sh
# What will `auto` pick on this machine, and what's already cached?
uv run bestee-chat resources status

# Download + verify the latest Simple English dump (resumable, sha1-checked).
uv run bestee-chat resources download --dump --lang simple

# Build the index at the auto-selected quality profile.
uv run bestee-chat resources index --lang simple --profile auto

# Query it.
uv run bestee-chat wiki search "List of English kings" --lang simple
```

`wiki search` accepts `--build` to construct the index on first use, and takes
`--json` for scripting. Scale up with `--lang en` (full English Wikipedia); pin
a reproducible dump with `--dated YYYYMMDD`.

### Library

```python
from bestee_chat.wiki import WikiSource, build_index, search_wiki

source = WikiSource(lang="simple")          # ~50x smaller than full English
build_index(source, profile="auto")          # download + index (once)

for hit in search_wiki("List of English kings", source=source, limit=5):
    print(hit.rank, hit.title, f"{hit.score:.3f}")
    print("   ", hit.url)
```

### Hardware-adaptive profiles

`profile="auto"` probes the machine (CPU, RAM, accelerator via the shared
`SemanticEncoder`) and selects the richest feasible preset; better hardware
yields better results from the same commands via a re-index.

| Profile    | Targets                  | Retrieval                          |
| ---------- | ------------------------ | ---------------------------------- |
| `lexical`  | no ML deps               | BM25 only                          |
| `economy`  | ~16 GB, CPU/MPS          | BM25 + MiniLM rerank (lead)        |
| `standard` | 32-64 GB, MPS/GPU        | BM25 + mpnet rerank (article)      |
| `quality`  | GPU + 64 GB              | BM25 + bge-large rerank (chunks)   |
| `max`      | big CUDA GPU + 96 GB+    | exact + bge-large rerank           |

### Plug into the chatbot

```python
from bestee_chat.engine import Brain
from bestee_chat.wiki import WikiKnowledge, WikiSource

brain = Brain()
brain.add_knowledge(WikiKnowledge(WikiSource(lang="simple")))
print(brain.best_candidate(["English", "kings"]).source)
```

## Develop

```sh
uv run pytest        # run tests
uv run ruff check    # lint
uv run ruff format   # format
uv run ty check      # type-check
```
