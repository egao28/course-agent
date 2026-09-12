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
├── data/          # courses.json, prereqs.json, requirements.json (committed), chunks.jsonl (rebuilt)
├── ingest/        # export_almabot.py → chunk.py → build indexes (dense / bm25)
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

4,502 courses (term `2026-sp`), 2,086 prerequisite and 34 corequisite edges, 13 Gen Ed categories.
The outputs are committed, so a clone can build indexes without the private CSV; see
[data/README.md](data/README.md) for the schema, how the requirements block is parsed, and the
known data quality issues.

## Chunking

One course is one chunk — descriptions are short (p50 111 tokens, max 523, against an 8191-token
embedding limit), so splitting would only scatter a course's facts across several vectors.

```bash
uv run python -m ingest.chunk                # -> data/chunks.jsonl + chunks.meta.json, with stats
uv run python -m ingest.chunk --show CS 374  # print one rendered chunk and exit
```

```
doc_id   2026-sp:CS-374
metadata {"code": "CS 374", "subject": "CS", "number": "374", "credits": 4.0, "term": "2026-sp", "prereq_codes": ["CS 173", "CS 225", "MATH 213"], "gened_ids": []}
chroma   {"code": "CS 374", …, "prereq_codes": "CS 173|CS 225|MATH 213", "gened_ids": ""}
text
CS 374 — Introduction to Algorithms & Models of Computation
Credits: 4
Term: 2026-sp
Description: Analysis of algorithms, major paradigms of algorithm design including recursive algorithms, … Same as ECE 374. Prerequisite: One of CS 173, MATH 213; CS 225.
Prerequisites: One of CS 173, MATH 213; CS 225.
Degree Attributes: None
```

The requirements block appears twice there: the catalog runs it to the end of the description, so
`prereqRaw` is a suffix of `desc` — true of 2,732 courses, and what `v1` was specified to render.
Changing it is a template change, measured in #15 rather than baked into the baseline. The last
line is labelled the way the catalog labels the column, *not* "Gen Ed": it also carries non-Gen-Ed
attributes ("Global Classroom course."), so the authoritative Gen Ed answer is `metadata.gened_ids`,
computed from the rules in `requirements.json`.

The wording of that text is a retrieval parameter, not a detail: it is what gets embedded and
BM25-tokenized. So it is built in `render_chunk(course, template_version)` and nowhere else, keyed
by `chunk.template_version` in `config.yaml`. A template ablation adds a `v2` renderer beside `v1`,
flips that one value, and gets fresh index names for free — the version is part of the config hash.
`chunk.fields` must list exactly what the chosen template renders; the build refuses to run
otherwise, so the hash can never describe a corpus that was not built. It also refuses to write a
chunk over the 8191-token limit, rather than letting the embedding call discover it mid-batch.

`doc_id` is `{term}:{SUBJ}-{NUM}` (`2026-sp:CS-374`), and `metadata` carries `code, subject, number,
credits, term, prereq_codes, gened_ids` for filtering. `chunks.jsonl` is **not** committed: unlike
the exported `data/*.json`, it is rebuilt from them by the command above with no private input.
Its `chunks.meta.json` sidecar records a `chunk_key` — a hash of just the term, strategy, template
version and field list — so an index build can detect chunks left over from another template while
ignoring the `seed` / `bm25` / `fusion` tuning that a Day 4 ablation changes on purpose (the full
`config_hash` of the run is recorded too, but only as a record of who wrote the file). See
[data/README.md](data/README.md#chunksjsonl).

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
