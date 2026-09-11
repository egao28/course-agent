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
uv run pytest -q         # smoke: imports resolve + config hash properties hold
```

Optional local-embedding route (BGE-M3 via `sentence-transformers`): `uv sync --locked --extra local`.
`uv sync` is exact by default, so pass `--extra local` on **every** sync or the extra is removed again;
alternatively use `uv sync --inexact` to leave already-installed extras in place.

## Layout

```
course-agent/
├── config.yaml    # embedding model, chunk strategy, BM25/RRF params, seed — every experiment output carries the config hash
├── config.py      # load_config() -> Config (hash frozen at load), artifact-name helpers
├── data/          # courses.json, prereqs.json, requirements.json (exported), chunks.jsonl
├── ingest/        # export_almabot.py → chunk → build indexes (dense / bm25)
├── retrieval/     # dense.py, bm25.py, fusion.py, retrieve(query, mode)
├── agent/         # graph.py, tools/          (Day 2+)
├── eval/          # datasets/, scripts/       (Day 1: smoke set only)
├── report/        # one markdown report per experiment, filename tagged with the config hash
├── tests/         # pytest suite (fixtures/ holds a synthetic config, not the live one)
└── indexes/       # Chroma persistence + BM25 pickle (contents gitignored, dir kept)
```

## Config & reproducibility

All experiment settings live in `config.yaml`; nothing is hard-coded in the modules. `config.py`
loads it into a read-only `Config` whose **config hash** is computed once, at load time: sha256 of
the canonical dump (sorted keys, no whitespace), first 8 hex chars. Changing any value — including
`seed` and the `data.source_sha256` pin — changes the hash; reordering keys does not; list order
matters (`chunk.fields` order defines the chunk template). The mapping is immutable, so the hash
cannot drift between two calls inside one build. For an ablation, copy and rebuild:
`Config({**cfg.to_dict(), "seed": 43})`.

```python
from config import load_config, collection_name, bm25_path, report_path, rel_to_repo, source_path

cfg = load_config()              # read-only; cfg["fusion"]["rrf_k"] works like a dict
cfg.hash                         # "a1b2c3d4", frozen at load
collection_name(cfg)             # "courses_a1b2c3d4"                   Chroma collection
bm25_path(cfg)                   # <repo>/indexes/bm25_a1b2c3d4.pkl     absolute, dir created on demand
report_path("day1_smoke", cfg)   # <repo>/report/day1_smoke_a1b2c3d4.md absolute, dir created on demand
rel_to_repo(bm25_path(cfg))      # "indexes/bm25_a1b2c3d4.pkl"          for reports/logs, never str(path)
source_path(cfg)                 # data.source with ~ expanded; cfg itself is untouched
```

The naming helpers return **absolute** paths so a build works from any cwd; print them through
`rel_to_repo()` so a local home directory never lands in a git-tracked report.

`data.source_sha256` pins the *content* of the input CSV, not just its path. The exporter calls
`verify_source(cfg)`, which refuses a file that does not match, so re-exporting from new data
forces a new hash instead of silently reusing an index built from the old data. After replacing the
CSV, print the new digest with `uv run python config.py --source-digest` and update the pin.

Print the current hash with `uv run python config.py`, and the exact string being hashed with
`--canonical`.

## Data

`ingest/export_almabot.py` re-exports the catalog CSV into `data/`. AlmaBot's `public/courses.json`
cannot be reused: it truncates `desc` to 240 chars (3,255 of 4,502 courses lose text) and carries no
term, so the exporter re-reads the CSV and keeps only the *logic* from `build_data.py` — section
de-duplication, credit parsing, prerequisite extraction.

```bash
uv run python -m ingest.export_almabot   # -> data/{courses,prereqs,requirements}.json + stats
```

4,502 courses (term `2026-sp`), 2,080 prerequisite edges, 13 Gen Ed categories. The outputs are
committed, so a clone can build indexes without the private CSV; see [data/README.md](data/README.md)
for the schema and the known data quality issues.

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
