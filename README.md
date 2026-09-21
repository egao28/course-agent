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
├── llm.py         # prompt → text: the one chat client the router and the answer node share
├── data/          # courses.json, prereqs.json, requirements.json (committed), chunks.jsonl (rebuilt)
├── ingest/        # export_almabot.py → chunk.py → build_dense.py / tokenize.py → build_bm25.py
├── retrieval/     # types.py (Hit), dense.py, bm25.py, fusion.py, __init__.py = retrieve(query, mode)
├── agent/         # tools/ (get_course, get_prereq_chain, check_requirement …), graph.py   (Day 2)
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
cfg.hash                         # "ade2f799", frozen at load
collection_name(cfg, chunk_key)  # "courses_094143d8"                  Chroma collection, see below
bm25_path(cfg, chunk_key)        # <repo>/indexes/bm25_877a6610.pkl    absolute, names only
report_path("day1_smoke", cfg)   # <repo>/report/day1_smoke_ade2f799.md  dir created on demand
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
metadata {"code": "CS 374", "subject": "CS", "number": 374, "level": 300, "credits": 4.0, "term": "2026-sp", "prereq_codes": ["CS 173", "CS 225", "MATH 213"], "gened_ids": []}
chroma   {"code": "CS 374", …, "prereq_codes": ["CS 173", "CS 225", "MATH 213"]}
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
level, credits, term, prereq_codes, gened_ids` for filtering. The `chroma` line above is the same
record as Chroma stores it: the lists stay lists, so `{"gened_ids": {"$contains": "cultus"}}` and
`{"$and": [{"subject": "CS"}, {"level": 400}]}` are answerable, and an *empty* list is dropped
rather than stored, because that is the one value Chroma refuses. Which field supports which
operator is tabulated in [data/README.md](data/README.md#metadata-what-each-field-is-good-for). `chunks.jsonl` is **not** committed: unlike
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
config hash ade2f799  (chunk key 05d00726)
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
and metadata come back out of the index rather than `chunks.jsonl`, with the list fields Chroma
could not store — the empty ones — restored as `[]`, so a dense hit and a BM25 hit mean the same
thing.

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
config hash ade2f799  (bm25 key 877a6610)
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

## Fusion

`retrieval/fusion.py` combines the two rankings into one, on **rank alone**:

```
score(d) = Σ_i 1 / (k + rank_i(d))        k = fusion.rrf_k (60), ranks 1-based
```

```python
from config import load_config
from retrieval.fusion import fuse, rrf

rankings = {"dense": dense_hits, "bm25": bm25_hits}

rrf(rankings)                     # the whole fused ranking, k = 60
fuse(rankings, load_config())     # ... with the config's k, cut to top_k_final
```

No score crosses the boundary, and none can: a cosine similarity of 0.42 and a BM25 score of 11.3
are not on one scale, and both distributions move with the query, so an `α·dense + (1-α)·bm25`
tunes a constant against a moving target. Rank is the one thing both retrievers report in the same
unit. What the formula buys is **agreement**: a document one retriever put first scores `1/61 =
0.0164`, while one *both* put tenth scores `2/70 = 0.0286` and wins. `k` is the knob on that trade
— it damps the head of each list, so the gap between rank 1 and rank 2 (0.0164 vs 0.0161) stays
small beside the gain from being found twice. At `k = 0` it degenerates to plain reciprocal rank,
where one first place merely ties two second places and the fusion becomes "whoever was most
confident". 60 is Cormack et al.'s constant, and a config value, so Day 4 can ablate it.

Each fused hit reports where it came from, in `sources` — the rank each retriever gave it, `None`
where that retriever did not return it at all. The ranks are what explain the order:

```
Hit(doc_id="2026-sp:CS-374", score=0.02857, rank=1, …, sources={"dense": 10, "bm25": 10})
Hit(doc_id="2026-sp:CS-357", score=0.01639, rank=2, …, sources={"dense": 1, "bm25": None})
```

(`2/70` against `1/61`: agreement at rank ten, against a first place only one half of the hybrid
found. Real top-5s for the Day 1 queries are in the smoke report, once the dense index is built.)

So the Day 4 ablation can say which half of the hybrid produced a result, and a result only one
half ever finds is visible rather than inferred. Ties — two documents found by the same retrievers
at the same ranks, which happens on nearly every query at 20 hits each — are broken by `doc_id`,
not by which retriever the mapping happened to list first; otherwise an ablation would measure
that ordering instead of the change it made.

The module is **pure**: no index, no API key, no config in `rrf` itself. That is why it is not
part of either retriever — the arithmetic that decides the final order is tested against numbers
written out by hand, on rankings built in the test file. `fuse` is the one function that reads the
`fusion` section (`method`, `rrf_k`, `top_k_final`), for `retrieve(query, "hybrid")`.

## retrieve(query, mode)

`retrieval/__init__.py` is the one entry point, and the one the Day 4 ablation calls — so the
signature is frozen now and a mode is a *parameter*, never a different function with a different
return shape:

```python
from retrieval import retrieve

retrieve("machine learning")                   # mode="hybrid", k=fusion.top_k_final (5)
retrieve("CS 374", "bm25", k=10)               # mode ∈ {dense, bm25, hybrid}
retrieve("machine learning", "dense", cfg=my_cfg)
```

```bash
uv run python -m retrieval "CS 374" --mode bm25
uv run --env-file .env python -m retrieval "machine learning" --mode hybrid -k 10
```

```
'CS 374'  ->  5 hits, mode bm25, config ade2f799, 1.9 ms

#  doc_id             score  name                                            sources
1  2026-sp:CS-374   29.0337  Introduction to Algorithms & Models of Comput…  bm25 1
2  2026-sp:ECE-374  26.5137  Introduction to Algorithms & Models of Comput…  bm25 2
3  2026-sp:CS-477   24.6805  Formal Software Development Methods             bm25 3
4  2026-sp:CS-476   23.5827  Program Verification                            bm25 4
5  2026-sp:CS-473   21.4707  Algorithms                                      bm25 5

(plus 79 ms opening the bm25 index, once per process -- not counted in the query time above)
```

| mode | what runs | needs |
| --- | --- | --- |
| `dense` | `DenseRetriever.search(query, k)` | `OPENAI_API_KEY` + the Chroma collection |
| `bm25` | `BM25Retriever.search(query, k)` | the BM25 pickle |
| `hybrid` | both, at `fusion.top_k_each` (20) → `fuse()` → `k` | both of the above |

`k` defaults to `fusion.top_k_final` (5) in **every** mode, so switching modes cannot quietly
change the length of the list a report is comparing. In `hybrid` it cuts the fused ranking, not
either input: each retriever is still asked for the tuned `top_k_each`. Asking for more than that
depth (`k=50`) deepens the recall instead of returning the ≤40 documents two 20-deep lists can hold
between them.

The result carries what a report has to quote next to the hits:

```python
RetrievalResult(query=…, mode="hybrid", config_hash="ade2f799", hits=[Hit, …], latency_ms=128.4)
```

`config_hash` is the **full** hash, not either index key: a ranking is the product of the whole
configuration, including the fusion parameters, which name no index. `hits` are the same `Hit`
whichever mode produced them — in `hybrid` each one's `sources` says which retrievers found it and
where, and in the other two it is `None`, the mode already being the answer. The CLI fills the
`sources` column in from the mode, so all three tables read the same way down the page.

Retrievers are built on first use and cached in-process, keyed by the config hash: opening Chroma
and unpickling 6 MB costs about a second each, and a smoke report or an ablation pays it once
rather than per query. Keying on the full hash is stricter than it needs to be — an `rrf_k`
ablation reopens two indexes whose contents did not change — but a cached retriever can then never
outlive a setting the caller has since changed, which would not raise, it would answer. Nothing is
loaded until a mode asks for it, so `retrieve(q, "bm25")` never opens Chroma, never imports
`chromadb` or `openai`, and runs in a checkout with no API key and no dense index. `clear_cache()`
drops the lot, for a process that rebuilt an index after querying it.

`latency_ms` times the **query**, not the process warm-up: the retrievers are built before the
clock starts, since a number that included a one-off index load on the first query and not on the
next four would not be the latency of anything. The CLI prints the load on its own line when it
happened.

A fourth mode — `keyword`, the `searchCourses()` baseline from AlmaBot's `planner.ts`, which Day 4
may want to compare against — is a builder in `BUILDERS` and an entry in `MODE_SOURCES`, and
nothing else. That room is deliberate; the implementation is out of scope for Day 1.

Real top-5s for the Day 1 queries, in all three modes, are [#9](../../issues/9)'s smoke report,
once the dense index is built.

## Chat client

`llm.py` is to the chat model what `embedding.py` is to the encoder: the one place a prompt becomes
text. Day 2's two model calls — the intent router and the answer node — both take their client from
`make_chat(cfg)`, so neither names a model of its own. Nothing on Day 1's path calls it.

```yaml
llm:
  provider: openai          # the only route implemented; `local` is an embedding-only axis
  model: gpt-4o-mini
  temperature: 0            # a classification and a grounded answer are both deterministic tasks
  max_tokens: 800
```

```python
from config import load_config
from llm import make_chat

chat = make_chat(load_config())
chat.complete("How many credits is CS 374?", system="Reply with one label.")  # -> "course_fact"
```

`Chat` is a two-member protocol (`model`, `complete`), and that is the point: **the client is
injectable everywhere.** `make_chat(cfg, client=…)` swaps the SDK object underneath `OpenAIChat`
for a stub, and any object with a `complete` method is a `Chat`. Not a nicety — the OpenAI account
behind this repo has no credits (every call returns 429 `insufficient_quota`), so each node's logic
is built and tested against a fake, and the live run is the last step of Day 2 instead of a
prerequisite for any of it.

`complete()` returns a plain `str`, so a fake is three lines long, and raises rather than returning
one when a completion arrives with no answer in it: no choices, an empty or refused message, or a
reply cut off at `max_tokens`. The last matters most for the answer node — a truncated answer keeps
its citation and loses the claim that citation was supporting, which is the one failure a
post-generation citation check cannot see.

The section is part of the full config hash and reaches neither index key, so adding it renamed the
reports (`7045a614` → `ade2f799`) and left the dense collection (`094143d8`) and the BM25 pickle
(`877a6610`) exactly where they were.

## Course lookup

`agent/tools/courses.py` is the first of Day 2's three tools. "How many credits is CS 374?" names
the row it wants, so it is a lookup rather than a search — running it through the retriever would
rank five plausible chunks where the catalog holds one exact answer.

```python
from agent.tools.courses import get_course

get_course("cs 374").credits        # -> 4.0
get_course("CS374").code            # -> 'CS 374'      (one normalizer, see below)
get_course("CS 999").found          # -> False         (no exception)
```

**Not found is a value.** `CS 999`, and anything that is not a course code at all, comes back as
`found=False` with a message naming the catalog that was searched — the one term `data/courses.json`
covers, because "no such course" and "not offered this spring" are indistinguishable from inside the
graph. A tool that raised here would take the "I don't know" exit away from the node that called it.

**The code is normalized by `ingest/tokenize.py`'s matcher**, not by a second regex. `CODE_RE`
already decides what a course code looks like — BM25 needs `CS 374` / `cs 374` / `CS374` to collapse
into one term stream — so a tool with its own pattern would eventually accept a spelling the index
cannot find, and nothing would report the divergence.

**The corpus is read once**, lazily, into a `{code: record}` index behind a module-level cache
(`clear_cache()` drops it). 4,502 records and 2.9 MB of JSON, several tool calls per query.

The whole argument has to be one code: `"CS 374 or ECE 374"` and `"what is cs 374 about"` are both
`found=False`. Pulling a code out of a sentence is the router's job ([#35](../../issues/35)).

Both models are Pydantic and `extra="forbid"`: [#33](../../issues/33) registers these functions with
FastMCP and publishes the schemas, while [#36](../../issues/36) imports `get_course` and calls it
in-process — there is no stdio subprocess between the graph and its own tools.

> `CourseDetail.credits` is provisional. A record on this branch carries `credits: 4.0`; open
> [#25](../../issues/25) replaces it with `creditsRaw` / `creditsMin` / `creditsMax`, because 1,415
> of the 4,502 courses have a credit *range*. Whichever merges second updates the model —
> `build_index()` raises on a corpus that has already moved rather than report `credits=None` 4,502
> times.

## Prerequisite chain

`agent/tools/prereqs.py` is the second tool. "What do I need before SE 495?" is neither a lookup
nor a search — the answer is a walk over `data/prereqs.json`, which the export already filtered to
courses this catalog offers.

```python
from agent.tools.prereqs import get_prereq_chain

chain = get_prereq_chain("cs 374")
[[n.code for n in level] for level in chain.levels]
# -> [['CS 173', 'CS 225', 'MATH 213'],                       # level 0 = immediate
#     ['CS 124', 'CS 128', 'CS 413', 'ECE 220', 'MATH 220', 'MATH 314', 'MATH 412', 'MATH 413'],
#     ['ECE 120', 'MATH 112', 'MATH 115', 'MATH 241'], ['MATH 231']]
chain.cycle_detected                       # -> True   (CS 225 -> MATH 412 -> CS 374 -> CS 225)
get_prereq_chain("SE 495").truncated       # -> False  (27 courses, walked to exhaustion)
get_prereq_chain("CS 374", max_depth=1)    # -> the immediate three, truncated=True
```

**Every walk carries a visited set, because the graph is not acyclic.** Building the DAG from
`prereqs` only — as `data/README.md` said — is necessary and not sufficient: the prerequisite edges
alone contain seven strongly-connected components over 19 courses. Sixteen of those 19 are
corequisites stated *inside* the prerequisite sentence ("SE 310; concurrent registration in
SE 312."), where the export's corequisite pass cannot see them; the other three are disjunction
members (`MATH 412` is one alternative in `CS 225`'s "one of …", and lists `CS 374`, which requires
`CS 225`). `data/README.md` is corrected in this branch.

**No depth cap by default.** `dag.ts`'s `allPrereqs(code, depth = 6)` drops a prerequisite outright
for 113 of the 1,135 courses — cosmetic in a diagram, a wrong answer in a sentence. One visited set
across the whole walk expands each course once, which bounds a call by the graph (1,442 courses,
2,086 edges) and in practice by the largest closure in it (47 courses, `IS 557`; median 4), so
exhaustion is affordable. `max_depth` stays as something a caller can ask for, and `truncated`
reports when it bit.

**Level 0 is the immediate prerequisites**, and each course sits at its shortest distance from the
queried one — the contract the answer node needs. `dag.ts::buildSubgraph` layers by longest chain
instead, so the levels here are not a schedule; `edges` carries the full `(prereq, dependent)`
structure, cycle-closing edges included, and both endpoints of every edge are always courses the
result reports.

**Every node carries its raw requirement sentence.** 744 of the 1,135 statements contain a
disjunction and 397 of those resolve to more than one code, so the code list is not a conjunction:
`ABE 228` exports `["MATH 220", "MATH 234"]` from "One of MATH 220, MATH 221, MATH 234." The prompt
quotes the sentence rather than asserting the list; parsing it into an AND/OR tree is Day 3+ work.

`found=False` is delegated whole to `get_course`, so the two tools can never disagree about what is
in the catalog. A course that *is* in it with no prerequisites returns `found=True` and empty levels
— that is an answer — and `chain.raw` still quotes its requirements block, which matters for the
1,608 courses that have one and no chain at all, because every course it names is either not
offered this term or only a corequisite (`PHYS 211`: "PHYS 211 placement test or MATH 221.").

## Requirement check

`agent/tools/requirements.py` is the third tool, and the one with a hole in its data. "What Gen Eds
do I still need?" is a question about a rule set rather than about a course, and
`data/requirements.json` answers half of it: `genEd` holds the 13 campus categories as declarative
rules, `majors` is `[]` and always has been — `course-catalog.csv` lists sections, and a degree
requirement has never been in it ([#32](../../issues/32)).

```python
from agent.tools.requirements import check_requirement

status = check_requirement("gened", ["RHET 105", "MATH 221"])
{c.id: c.satisfied_by for c in status.satisfied}   # -> {'comp1': 'RHET 105'}
status.unknown_courses                             # -> ['MATH 221']   (not offered in 2026-sp)
[c.id for c in status.unsatisfied]                 # -> the other 12

check_requirement("Computer Science, BS", [...]).supported   # -> False, with both lists empty
```

**The unsupported half is a typed refusal, not an empty result.** Anything that is not Gen Ed comes
back `supported=False` with `satisfied` and `unsatisfied` both empty and a message naming what is
missing (a source of degree requirements) and what is available (Gen Ed). A degree audit improvised
from an LLM's parametric memory is the most harmful thing this agent could emit — fluent, specific,
checkable only by the advisor the student was trying not to bother — so the refusal leaves nothing
for the answer node to paraphrase, and [#39](../../issues/39) asserts the query leaves by the
hand-off exit. The refusal is decided before any file is read, so it cannot fail with a missing
corpus either. If `majors` is ever non-empty, `load_requirements()` raises rather than keep
declining a program the data now holds.

**The rules are evaluated from the file, by the exporter's own evaluator** (`matches_attrs`), not
re-transcribed here. They stopped being code in [#3](../../issues/3) so there would be one copy:
`gened.ts`'s `socbeh` matcher is wrong, open [#23](../../issues/23) fixes it in the data (142 → 24)
and deliberately leaves the TS broken. On this branch the pre-#23 rules are what ship, so `ECON 102`
still satisfies `socbeh`; when #23 merges the answer changes with no code change here. The test that
pins this compares the tool against `matches_attrs` on whatever rule is on disk, so it survives
either way.

**A course counts for every category it is designated for**, exactly as the frontend resolved them:
`IS 145` is Quantitative Reasoning II *and* Soc Sci *and* US Minority. 570 of the 4,502 courses
carry at least one designation and 299 carry more than one (244 after #23, whose largest is 3 rather
than 4). Where several completed courses qualify, the first one listed is credited — a transcript is
the student's own ordering, and re-ranking it would be a recommendation dressed up as an audit.

**Codes the catalog does not have come back in `unknown_courses`.** The corpus is one term, so a
real transcript is mostly courses it has never seen — `MATH 221`, in the acceptance query above, is
one of them. An audit that silently dropped them would report "you still need Quantitative
Reasoning I" about a student who has it, which is the one error the reader cannot detect; they are
listed in the field *and* in `message`. Whether a code is in the catalog is decided by `get_course`,
so the three tools cannot disagree about it.


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
