# queue

Celery worker that runs the auto-tuning optimizers from `bestee-compute`
(`bestee_compute.workflow.tuning`) as background jobs.

Each task in [`tasks.py`](tasks.py) mirrors one `*Request` model: it validates
the incoming payload, runs the matching `optimize_*` grid search, and returns a
JSON-serializable result.

| Task name | Request model |
| --- | --- |
| `tuning.optimize_clustering` | `ClusteringRequest` |
| `tuning.optimize_regime` | `RegimeRequest` |
| `tuning.optimize_fama_french` | `FamaFrenchRequest` |
| `tuning.optimize_rrg` | `RRGRequest` |

## Configuration

The Redis broker and result backend are read from `DO_REDIS_CONNECTION` (a local
`.env` is loaded if present). Because the optimizers fetch live market data, the
worker also needs `MASSIVE_API_KEY` (and `FRED_API_KEY` for the Fama-French
factors).

## Running

```sh
uv sync
uv run celery -A main worker --loglevel=info
```

Enqueue a job:

```python
from tasks import optimize_clustering

optimize_clustering.delay(
    {
        "tickers": ["AAPL", "MSFT", "NVDA"],
        "start_date": "2022-01-01",
        "end_date": "2024-12-31",
    }
)
```

## Development

```sh
uv run pytest      # offline tests for the task wiring
uv run ruff check  # lint
uvx ty check       # type-check
```
