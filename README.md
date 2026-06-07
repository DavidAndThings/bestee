# bestee-chat

A small conversational chatbot.

## Setup

```sh
uv sync
```

## Run

```sh
uv run bestee-chat
```

## Semantic similarity

A `Knowledge` source scores how close a query is in meaning to its stored
answers using local sentence embeddings (via `sentence-transformers`), and the
`Brain` returns the best `Candidate` across all sources. The model runs entirely
on-device; weights are downloaded once from the Hugging Face Hub and cached
locally.

Pre-download the default model (useful before going offline):

```sh
uv run bestee-chat-download
```

Afterwards you can force fully offline operation:

```sh
HF_HUB_OFFLINE=1 uv run bestee-chat
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

## Wikipedia dump search (`bestee_chat.wiki`)

Download a full Wikipedia content dump and search it locally. The lexical
(BM25) layer always works with zero extra dependencies; a semantic reranking
layer engages automatically when `torch` and the hardware allow. See
`docs/wikipedia_module_plan.md` for the full design.

### CLI

```sh
# What will `auto` pick on this machine?
uv run bestee-chat-wiki status

# Download + verify the latest Simple English dump (resumable, sha1-checked).
uv run bestee-chat-wiki download --lang simple

# Build the index at the auto-selected quality profile.
uv run bestee-chat-wiki index --lang simple --profile auto

# Query it.
uv run bestee-chat-wiki search "List of English kings" --lang simple
```

`search` accepts `--build` to construct the index on first use, and every
command takes `--json` for scripting. Scale up with `--lang en` (full English
Wikipedia); pin a reproducible dump with `--dated YYYYMMDD`.

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
