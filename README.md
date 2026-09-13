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
├── ingest/        # export_almabot.py → chunk.py → build_dense.py / tokenize.py → build_bm25.py
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
cfg.hash                         # "7045a614", frozen at load
collection_name(cfg, chunk_key)  # "courses_094143d8"                  Chroma collection, see below
bm25_path(cfg, chunk_key)        # <repo>/indexes/bm25_877a6610.pkl    absolute, names only
report_path("day1_smoke", cfg)   # <repo>/report/day1_smoke_7045a614.md  dir created on demand
rel_to_repo(bm25_path(cfg, ck))  # "indexes/bm25_877a6610.pkl"         for reports/logs, never str(path)
source_path(cfg)                 # data.source with ~ expanded; cfg itself is untouched
```

The naming helpers return **absolute** paths so a build works from any cwd; print them through
`rel_to_repo()` so a local home directory never lands in a git-tracked report.

Only the reports are named after the full hash. **An index is named after its own key** — the
settings that can actually change what is in it:

| artifact | key | moves when |
| --- | --- | --- |
| Chroma collection `courses_{dense_key}` | `chunk_key` + `embedding.provider` + `embedding.model` | the chunk text or the encoder changes |
| BM25 pickle `bm25_{bm25_key}.pkl` | `chunk_key` + the whole `bm25` section | the chunk text, the tokenizer or `k1`/`b`/`code_boost` changes |

A Day 4 ablation over `fusion.rrf_k` or `seed` moves `cfg.hash` without moving a single number in
either index, so naming them after it would orphan both — re-embedding (and re-paying for) a
byte-identical corpus, and re-tokenizing one that nothing about tokenization touched. Tuning
`bm25.k1` renames the BM25 index and leaves the dense one alone, which is the point of keying them
separately. `bm25_key` hashes the `bm25` section whole rather than field by field, so the next knob
added to it cannot be quietly left out of the name of the index it changes.

Neither index helper creates a directory — `chroma_path()` because opening a persistent Chroma
client is what writes a store, `bm25_path()` for the same reason in reverse: both are asked for by
*readers* looking for an index, and a read that creates what it could not find leaves an empty
store behind every time a query fails. The build that writes an index creates its directory.

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

## Dense index

`ingest/build_dense.py` embeds every chunk once and puts the vectors in Chroma, in the collection
`courses_{dense_key}` under `indexes/chroma` (gitignored: rebuildable, and never committed).

```bash
uv run python -m ingest.build_dense --dry-run                  # what it would cost; calls no API
uv run --env-file .env python -m ingest.build_dense            # build it
uv run --env-file .env python -m ingest.build_dense --rebuild  # drop the collection, embed again
```

```
config hash 7045a614  (chunk key 05d00726)
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

## BM25 index

`ingest/build_bm25.py` tokenizes every chunk and pickles an Okapi BM25 index to
`indexes/bm25_{bm25_key}.pkl` (gitignored: rebuildable, and never committed).

```bash
uv run python -m ingest.build_bm25
```

```
config hash 7045a614  (bm25 key 877a6610)
documents indexed      4502
vocabulary             21500 terms
terms emitted          298000  (mean 66.2 per chunk)
k1 / b                 1.5 / 0.75
elapsed                0.1s
wrote                  indexes/bm25_877a6610.pkl  (6063 KB)
```

No API key, no network, no cost — which is the one design difference from the dense build. There is
no `--dry-run` to price a run that is free and no idempotency check to skip one that is cheaper
than deciding whether to skip it: every run rebuilds, and the write is atomic, so an interrupted
one leaves the previous index rather than a truncated file under the name the next query opens.
What it does keep is the dense side's freshness check on the corpus, through the same
`ingest.chunk.check_chunks` — an index built from a stale `chunks.jsonl` is wrong the same way
whichever retriever reads it. The index stores the `doc_id`, text and metadata of every chunk
alongside the postings, so a query returns whole hits without opening `chunks.jsonl`.

It is a **pickle**, so it is local-only: unpickling executes whatever the file says. This one is
written by the command above, ignored by git and never fetched from anywhere — do not load one you
did not build. The rank_bm25 version that wrote it is in the header, and a file that will not load
is answered with the command that rewrites it.

### Tokenizing

`ingest/tokenize.py` is the one place text becomes terms — build side and query side both go
through it, so a query can never be tokenized differently from the corpus. `bm25.tokenizer` picks
the function and is hashed into the index name.

`simple_v1` lowercases, splits on `\w+`, drops a small stopword list, and does two things that
matter for a course catalog:

**Course codes become three terms.** `CS 374` → `cs`, `374`, `cs374`, and so do `CS374` and
`cs 374` — the code is matched as a unit before the text is split into words, so all three
spellings produce the same terms and score identically. Without the glued term, `CS374` (one word)
and `CS 374` (two) would share no term at all. `IS` is both a real subject and a stopword; because
codes are matched first, `IS 407` keeps its `is`.

**A course's own code is a different term from a code it cites.** Every chunk is indexed under a
reserved term for its own code (`code:cs374`, from `metadata.code`), repeated `bm25.code_boost`
times, and a query naming a code asks for it. The corpus is why: `ECE 374`'s entire description is
"Same as CS 374. See CS 374." — one of 497 cross-reference-only descriptions ([#15](../../issues/15))
— so for the query `CS 374` it is a 20-term document matching `cs374` twice while CS 374's own
110-term chunk matches it once. BM25 is right on its own terms and ranks the stub first; it has no
way to know that one of those documents *is* CS 374 and the other points at it. Measured on the
live corpus, `CS 374` ranks the course itself 11th at `code_boost: 0`, 2nd at `1`, and 1st at `2`.
The `:` in the prefix is a character `\w+` cannot produce, so the reserved namespace cannot collide
with a term from the corpus.

### Querying it

```bash
uv run python -m retrieval.bm25 "CS 374" -k 5
uv run python -m retrieval.bm25 "CS 374" --terms     # also print the query's terms
```

```
'CS 374'  ->  5 hits from bm25_877a6610
  1. 29.034  CS 374 — Introduction to Algorithms & Models of Computation
  2. 26.514  ECE 374 — Introduction to Algorithms & Models of Computation
  3. 24.680  CS 477 — Formal Software Development Methods
  4. 23.583  CS 476 — Program Verification
  5. 21.471  CS 473 — Algorithms
```

```python
from config import load_config
from retrieval.bm25 import BM25Retriever

BM25Retriever(load_config()).search("CS 374", 5)
```

`search` returns the same `Hit(doc_id, score, rank, text, metadata)` the dense side returns, with
the metadata lists still lists. `score` is the raw BM25 sum: higher is better, unbounded above, and
comparable only *within* one list — a 29 here says nothing about a cosine 0.9, which is why fusion
works on `rank`. Documents that no query term touched score exactly 0 and are dropped rather than
padded out to `k`: fusion ranks every list it is handed, and an untouched document has not been
retrieved. Ties are broken by corpus order, so a ranking is the same on every run and machine.

This is the half of the hybrid that can find a course by name. Embeddings put `CS 374` near
"algorithms" and "theory of computation"; only the lexical side knows the string is a course code.

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
