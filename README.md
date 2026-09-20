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

Nothing loads `.env` by itself yet (that is #11), so commands that call the embeddings API are run
as `uv run --env-file .env python -m ...`. Everything else — tests, chunking, `--dry-run` — needs
no key at all.

Optional local-embedding route (BGE-M3 via `sentence-transformers`): `uv sync --locked --extra local`.
`uv sync` is exact by default, so pass `--extra local` on **every** sync or the extra is removed again;
alternatively use `uv sync --inexact` to leave already-installed extras in place.

## Layout

```
course-agent/
├── config.yaml    # embedding model, chunk strategy, BM25/RRF params, seed — every experiment output carries the config hash
├── config.py      # load_config() -> Config (hash frozen at load), artifact-name helpers
├── embedding.py   # text → vector: OpenAI or local BGE-M3, the one encoder both sides share
├── data/          # courses.json, prereqs.json, requirements.json (committed), chunks.jsonl (rebuilt)
├── ingest/        # export_almabot.py → chunk.py → build_dense.py (+ build_bm25.py)
├── retrieval/     # types.py (Hit), dense.py, bm25.py, fusion.py, retrieve(query, mode)
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
collection_name(cfg, chunk_key)  # "courses_094143d8"                   Chroma collection, see below
bm25_path(cfg)                   # <repo>/indexes/bm25_a1b2c3d4.pkl     absolute, dir created on demand
report_path("day1_smoke", cfg)   # <repo>/report/day1_smoke_a1b2c3d4.md absolute, dir created on demand
rel_to_repo(bm25_path(cfg))      # "indexes/bm25_a1b2c3d4.pkl"          for reports/logs, never str(path)
source_path(cfg)                 # data.source with ~ expanded; cfg itself is untouched
```

The naming helpers return **absolute** paths so a build works from any cwd; print them through
`rel_to_repo()` so a local home directory never lands in a git-tracked report.

The Chroma collection is the one artifact *not* named after the full hash. It is keyed on
`dense_key()` — `chunk_key` plus `embedding.provider` and `embedding.model`, i.e. everything that
can change a vector. A Day 4 ablation over `fusion.rrf_k`, `bm25.k1` or `seed` moves `cfg.hash`
without moving a single number in the index, so naming the collection after it would orphan the
index and re-embed (and re-pay for) a byte-identical corpus. `chroma_path()` is also the one path
helper that does **not** create its directory: opening a persistent Chroma client is what writes a
store, so a dry run, a failed query or an error message quoting the path leaves nothing behind.

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

One course is one chunk — descriptions are short (p50 112 tokens, max 526, against an 8191-token
embedding limit), so splitting would only scatter a course's facts across several vectors.

```bash
uv run python -m ingest.chunk                # -> data/chunks.jsonl + chunks.meta.json, with stats
uv run python -m ingest.chunk --show CS 374  # print one rendered chunk and exit
```

```
doc_id   2026-sp:CS-374
metadata {"code": "CS 374", "subject": "CS", "number": "374", "credits_min": 4.0, "credits_max": 4.0, "term": "2026-sp", "prereq_codes": ["CS 173", "CS 225", "MATH 213"], "gened_ids": []}
chroma   {"code": "CS 374", …, "prereq_codes": "CS 173|CS 225|MATH 213", "gened_ids": ""}
text
CS 374 — Introduction to Algorithms & Models of Computation
Credits: 4
Term: 2026-sp
Description: Analysis of algorithms, major paradigms of algorithm design including recursive algorithms, … Same as ECE 374. Prerequisite: One of CS 173, MATH 213; CS 225.
Prerequisites: One of CS 173, MATH 213; CS 225.
Degree Attributes: None
```

That is template `v1`, the default, and the requirements block is in it twice: the catalog runs
the block to the end of the description, so `prereqRaw` is a suffix of `desc`. The last line is
labelled the way the catalog labels the column, *not* "Gen Ed" — it also carries non-Gen-Ed
attributes ("Global Classroom course."), so the authoritative Gen Ed answer is
`metadata.gened_ids`, computed from the rules in `requirements.json`.

The wording of that text is a retrieval parameter, not a detail: it is what gets embedded and
BM25-tokenized. So it is built in `render_chunk(course, template_version, corpus)` and nowhere
else, keyed by `chunk.template_version` in `config.yaml`.

### Templates

Two ship. Which one retrieves better is a Day 4 measurement, not a decision baked into the
baseline, so `v1` renders the catalog fields exactly as #4 specified them, and `v2` makes the two
changes #4 left out:

| | `v1` (default) | `v2` |
|---|---|---|
| The requirements block, which `v1` renders twice (2,732 courses) | as specified | cut from the `Description:` line, label included, and kept once under `Prerequisites:` |
| A cross-reference-only `desc` — `"Same as HIST 283. See HIST 283."` and nothing else (497 courses, 11% of the corpus) | as written, so neither retriever has topical text to match | the `See` clause is followed and the target's description added on a `Description of HIST 283:` line of its own — 496 of the 497 resolve |

```
                    v1                v2
total tokens        522,658           513,445      −50,222 de-duplicated
p50 / p95 / max     111 / 202 / 523   110 / 182 / 446   +41,009 inherited
chunk_key           05d00726          bca89eb0
```

Switching is one line — `chunk.template_version: v2` — then re-run the chunk build and the index
builds, which pick up new names from the moved `chunk_key`. Nothing else in the `chunk` section
moves: both templates read the same seven fields, and `chunk.fields` must list exactly what the
chosen template renders (the build refuses to run otherwise, so the hash can never describe a
corpus that was not built). The build also refuses to write a chunk over the 8191-token limit,
rather than letting the embedding call discover it mid-batch.

`v2` keeps the `Same as ...` sentence it resolved: it is the only place a course's sibling codes
appear (`AAS 357` lists four), so following the reference adds a line instead of replacing one.
The borrowed text is labelled with the course it came from, and has that course's requirements
block stripped as well — 274 of the sources have one, and it describes the source, not the course
doing the borrowing, whose own `metadata.prereq_codes` is empty. Two things `v2` is expected to
cost, both for Day 4 to weigh against the recall it buys: the 496 resolved courses and their 351
sources are 847 chunks whose topical text now appears more than once (up to 8 at a time — `FR 418`
is quoted by seven), and an inherited description is the *source's* answer to "what is this course
about", which is only as true as the catalog's claim that the two are the same course.

`doc_id` is `{term}:{SUBJ}-{NUM}` (`2026-sp:CS-374`), and `metadata` carries `code, subject, number,
credits_min, credits_max, term, prereq_codes, gened_ids` for filtering — credits as a *pair*, because
1,415 courses (31%) are variable-credit and one number could only ever name one end of the range
(#17), so `credits_min <= n <= credits_max` is the filter for "can I take this for n credits". `chunks.jsonl` is **not** committed: unlike
the exported `data/*.json`, it is rebuilt from them by the command above with no private input.
Its `chunks.meta.json` sidecar records a `chunk_key` — a hash of just the term, strategy, template
version and field list — so an index build can detect chunks left over from another template while
ignoring the `seed` / `bm25` / `fusion` tuning that a Day 4 ablation changes on purpose (the full
`config_hash` of the run is recorded too, but only as a record of who wrote the file). See
[data/README.md](data/README.md#chunksjsonl).

## Dense index

`ingest/build_dense.py` embeds every chunk once and puts the vectors in Chroma, in the collection
`courses_{dense_key}` under `indexes/chroma` (gitignored: rebuildable, and never committed).

```bash
uv run python -m ingest.build_dense --dry-run                  # what it would cost; calls no API
uv run --env-file .env python -m ingest.build_dense            # build it
uv run --env-file .env python -m ingest.build_dense --rebuild  # drop the collection, embed again
```

```
config hash 1edbaf46  (chunk key 05d00726)
collection             courses_094143d8  (does not exist yet)
would embed            4502 chunks via openai in batches of 256
tokens billed          522658  (text-embedding-3-small)
cost                   $0.0105  (at $0.02 / 1M tokens)
store                  indexes/chroma
```

The whole 2026-sp corpus is about one cent on `text-embedding-3-small`. After a real run the same
report prints what the API said it *billed*, not an estimate of it. `--dry-run` quotes the cost of
the *next* run: against a collection that is already complete it quotes nothing, since that run
would embed nothing — and it combines with `--rebuild`, which does change the answer. It creates
nothing either, not even the store.

Re-running costs nothing: a collection that already holds exactly as many rows as there are chunks
is left alone. Any other count — an interrupted build — stops and asks for `--rebuild` instead of
topping the collection up, which would leave a corpus that is quietly a mix of two runs. That makes
`count()` the signal everything rests on, so `doc_id`s are checked for duplicates before the build
(Chroma keeps the first of two rows with one id and drops the second silently) and what landed is
counted after it. The chunks are checked against the config's `chunk_key` first, since discovering
afterwards that they came from another template means paying for the build twice. Everything that
can fail without spending money — the config, the corpus, the collection state, the API key — is
checked before the store is opened, so a run that cannot embed leaves no `indexes/chroma` behind.

The collection is created in **cosine** space (Chroma defaults to squared L2) and records how it was
built — `chunk_key`, `config_hash`, `provider`, `model`, `dim`, `built_at` — for the person who opens
an index six weeks later.

### Querying it

```bash
uv run --env-file .env python -m retrieval.dense "machine learning" -k 5
```

```python
from config import load_config
from retrieval.dense import DenseRetriever

DenseRetriever(load_config()).search("machine learning", 5)
```

`search` returns `Hit(doc_id, score, rank, text, metadata)` — the same shape BM25 returns, so RRF can
fuse two lists without knowing which retriever produced either. `score` is `1 - cosine_distance`:
higher is better, and comparable only *within* one list, which is why fusion works on `rank`. Text
and metadata come back out of the index rather than `chunks.jsonl`, with the joined metadata lists
split into lists again so a dense hit and a BM25 hit mean the same thing.

### Embeddings

`embedding.py` is the one place text becomes a vector — build side and query side both go through
it, so a query can never be encoded by a different model than the corpus was. `embedding.provider`
picks the route:

| provider | model | needs |
| --- | --- | --- |
| `openai` | `text-embedding-3-small` | `OPENAI_API_KEY` |
| `local` | `BAAI/bge-m3` | `uv sync --locked --extra local` |

Both are part of `dense_key`, so switching providers names a new collection instead of mixing two
vector spaces in one.

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
