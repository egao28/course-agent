# course-agent

Retrieval + agent layer for a UIUC course-planning assistant. Data comes from [AlmaBot](../almabot-web)
(React frontend; its `scripts/build_data.py` exports courses and the prerequisite DAG from `course-catalog.csv`).

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python is pinned to **3.12** via `.python-version` and
`requires-python`: one interpreter version keeps `uv.lock`, indexes and reports reproducible across
machines. The stack also resolves on 3.13/3.14 today, but `chromadb`'s `onnxruntime` dependency has
lagged new CPython releases before, so we stay one version behind the system Python (3.14).
`uv sync` downloads the interpreter if it is missing.

```bash
uv sync --locked         # creates .venv (Python 3.12), installs deps + dev group; fails if uv.lock is stale
cp -n .env.example .env  # then fill in OPENAI_API_KEY (-n never overwrites an existing .env)
uv run python -c "import chromadb, rank_bm25, openai, yaml; print('ok')"
```

Optional local-embedding route (BGE-M3 via `sentence-transformers`): `uv sync --locked --extra local`.
`uv sync` is exact by default, so pass `--extra local` on **every** sync or the extra is removed again;
alternatively use `uv sync --inexact` to leave already-installed extras in place.

## Layout

```
course-agent/
├── config.yaml    # embedding model, chunk strategy, BM25/RRF params, seed — every experiment output carries the config hash
├── config.py      # load_config() / config_hash()
├── data/          # courses.json, prereqs.json, requirements.json, chunks.jsonl
├── ingest/        # export → chunk → build indexes (dense / bm25)
├── retrieval/     # dense.py, bm25.py, fusion.py, retrieve(query, mode)
├── agent/         # graph.py, tools/          (Day 2+)
├── eval/          # datasets/, scripts/       (Day 1: smoke set only)
├── report/        # one markdown report per experiment, filename tagged with the config hash
└── indexes/       # Chroma persistence + BM25 pickle (gitignored, rebuildable)
```

## Config & reproducibility

All experiment settings live in `config.yaml`; nothing is hard-coded in the modules. `config.py`
loads it and derives the **config hash**: sha256 of the canonical dump (sorted keys, no whitespace),
first 8 hex chars. Changing any value — including `seed` — changes the hash; reordering keys does
not; list order matters (`chunk.fields` order defines the chunk template).

```python
from config import load_config, config_hash, collection_name, bm25_path, report_path

cfg = load_config()                 # config.yaml at repo root
config_hash(cfg)                    # e.g. "a1b2c3d4"
collection_name(cfg)                # "courses_a1b2c3d4"           Chroma collection
bm25_path(cfg)                      # indexes/bm25_a1b2c3d4.pkl
report_path("day1_smoke", cfg)      # report/day1_smoke_a1b2c3d4.md
```

Every artifact is tagged with the hash so outputs from different settings never mix, and
`retrieve()` results carry a `config_hash` field (#8). Print the current hash with
`uv run python config.py` (add `--canonical` to see the exact string that is hashed).

## Day 1 goal

End-to-end `query → dense + BM25 recall → RRF → top-k`:

```python
from retrieval import retrieve
retrieve("machine learning courses", mode="hybrid")   # mode ∈ {dense, bm25, hybrid}
```

- Each retriever returns top-20; RRF `score(d) = Σ 1/(60 + rank_i(d))`; keep top-5 after fusion
- One interface for all three modes so the Day 4 ablation reuses it directly
- Acceptance: hand-check top-5 for 5 queries

Progress is tracked in GitHub Issues (milestone: Day 1).
