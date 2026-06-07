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

`Exchange.similarity` scores how close a query is in meaning to a stored
user utterance using local sentence embeddings (via `sentence-transformers`).
The model runs entirely on-device; weights are downloaded once from the
Hugging Face Hub and cached locally.

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

## Develop

```sh
uv run pytest        # run tests
uv run ruff check    # lint
uv run ruff format   # format
uv run ty check      # type-check
```
