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
├── ingest/        # export_almabot.py → chunk.py → build_dense.py / tokenize.py → build_bm25.py; majors.json = the hand-encoded degrees
├── retrieval/     # types.py (Hit), dense.py, bm25.py, fusion.py, __init__.py = retrieve(query, mode)
├── agent/         # state.py + graph.py = the LangGraph skeleton, nodes/ = one node per state key (route_intent, dispatch, check_evidence, answer, exits), prompts.py = the versioned prompts, tools/ (get_course, get_prereq_chain, check_requirement …), mcp_server.py, __main__.py + trace.py = the CLI and the trace it prints   (Day 2)
├── eval/          # schema.py + validate.py = the eval row and the check over a file of them, GUIDELINES.md = the annotation spec (Day 3), scripts/draft_items.py = the seeded drafting harness → datasets/drafts/, scripts/build_unanswerable.py = the 25 constructed refusals, scripts/build_multihop.py = the 25 multi-hop rows built from the prereq graph, scripts/review.py = the verification pass (three verdicts → datasets/review_log.jsonl → datasets/v1.jsonl), scripts/agreement.py = the second annotator's packet and the agreement report (percent + Cohen's κ → report/day3_agreement_{hash}.md), scripts/split.py = the frozen 50/50 calibration/test split (load_split("calib"|"test")), scripts/day3_report.py = Day 3's closing report over all of the above (→ report/day3_evalset_{hash}.md), datasets/day2_queries.jsonl = ten handwritten queries, scripts/day2_smoke.py = the run that writes report/day2_smoke_{hash}.md
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
cfg.hash                         # "c73b197d", frozen at load
collection_name(cfg, chunk_key)  # "courses_094143d8"                  Chroma collection, see below
bm25_path(cfg, chunk_key)        # <repo>/indexes/bm25_877a6610.pkl    absolute, names only
report_path("day1_smoke", cfg)   # <repo>/report/day1_smoke_c73b197d.md  dir created on demand
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

4,502 courses (term `2026-sp`), 2,086 prerequisite and 34 corequisite edges, 13 Gen Ed categories,
and the 2 degree programs of `ingest/majors.json` (the CSV has no degree requirements in it, so
those are transcribed by hand from the catalog program pages and checked against the export).
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
config hash c73b197d  (chunk key 05d00726)
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
config hash c73b197d  (bm25 key 877a6610)
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
'CS 374'  ->  5 hits, mode bm25, config c73b197d, 1.9 ms

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
RetrievalResult(query=…, mode="hybrid", config_hash="c73b197d", hits=[Hit, …], latency_ms=128.4)
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

`llm.StandInChat` is the one canned client that ships with the module: it answers with the first
line of the evidence it was handed and cites that block's `doc_id`, which is what lets the whole
graph — prompt, citation check, exit — run end to end with nothing behind it. It is reachable only
through [`--fake-model`](#command-line), and it names itself `stand-in` everywhere it is printed.

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
`found=False`. Pulling a code out of a sentence belongs to the dispatch node
([#36](../../issues/36)); the router ([#35](../../issues/35)) only labels the question.

Both models are Pydantic and `extra="forbid"`: the [MCP server](#mcp-server) publishes the schemas,
while [#36](../../issues/36) imports `get_course` and calls it in-process — there is no stdio
subprocess between the graph and its own tools.

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

`agent/tools/requirements.py` is the third tool, and the one whose answer depends most on what is
in the data. "What Gen Eds do I still need?" and "am I on track for my CS degree?" are questions
about a rule set rather than about a course, and `data/requirements.json` carries both rule sets:
`genEd` holds the 13 campus categories as declarative rules, and `majors` holds two degree programs
— Computer Science, BS and Statistics, BSLAS — transcribed by hand in [#32](../../issues/32) from
the catalog program pages into `ingest/majors.json`, because `course-catalog.csv` lists sections and
a degree requirement has never been in it.

```python
from agent.tools.requirements import check_requirement

status = check_requirement("gened", ["RHET 105", "MATH 221"])
{c.id: c.satisfied_by for c in status.satisfied}   # -> {'comp1': 'RHET 105'}
status.unknown_courses                             # -> ['MATH 221']   (not offered in 2026-sp)
[c.id for c in status.unsatisfied]                 # -> the other 12

cs = check_requirement("Computer Science, BS", ["CS 124", "CS 128", "CS 173", "MATH 221"])
cs.catalog_year, cs.source                         # -> '2025-2026', the program page it was read off
[c.id for c in cs.satisfied]                       # -> ['calculus-i']   (MATH 221, not in this term)
next(c for c in cs.unsatisfied if c.id == "cs-core").options
                                                   # -> ['CS 222', 'CS 225', 'CS 233', …]
[c.id for c in cs.not_checked]                     # -> the 8 an advisor decides, with their text

check_requirement("Aerospace Engineering, BS", [...]).supported   # -> False, every list empty
```

**A program with no entry in the file is a typed refusal, not an empty result.** It comes back
`supported=False` with every list empty and a message naming the programs that *are* available. A
degree audit improvised from an LLM's parametric memory is the most harmful thing this agent could
emit — fluent, specific, checkable only by the advisor the student was trying not to bother — so
the refusal leaves nothing for the answer node to paraphrase, and [#39](../../issues/39) asserts the
query leaves by the hand-off exit. Two of roughly 250 programs are encoded: this makes the
`degree_requirement` intent real, it does not make the agent a degree auditor.

**A degree is audited by course code; Gen Ed is audited by catalog record.** A Gen Ed category is a
rule about a course's `Degree Attributes`, so it can only be decided for a course this export holds.
A degree requirement is a list of codes, and the codes outlive the term: `MATH 221` is required by
both programs and is not offered in `2026-sp`. Resolving a degree through the catalog index would
tell a student who has taken Calculus I to take it again — wrong in the one direction they cannot
check — so requirements are matched against the normalized codes, and the catalog is used only to
report the gap (`notOffered`, measured at export time and repeated in `message`).

**What a course list cannot decide is reported, not guessed.** 128 hours, a 2.0 technical GPA, 40
hours of upper-division work, a language placement, an approved science-elective list that is not in
this export: those requirement groups carry `{"kind": "advising"}` in `ingest/majors.json` and come
back in `not_checked` with the catalog's own sentence and the page it is on — 8 of the CS degree's
17 groups. An audit that dropped them would read as "you are done" to a student who is not, and one
that marked them satisfied would be a fabrication. Every group, checked or not, carries the program
page and catalog year it was read off, so an answer can cite the line it is reporting.

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


## MCP server

`agent/mcp_server.py` publishes the three tools above to any MCP client — Claude Desktop, another
agent, a shell script — over stdio, as `FastMCP("course-tools")` with one `@mcp.tool` per function:

```bash
uv run python -m agent.mcp_server            # serve stdio
uv run python -m agent.mcp_server --list     # print what a client sees, then exit
```

```
course-tools serves 3 tools

get_course
  Look up one course in the University of Illinois course catalog by its code.
  …
  arguments:
    course_id (string, required) -- A course code: 'CS 374', 'cs 374' and 'CS374' all name the same course.
  returns: found, code, name, credits, term, desc, attrs, prereq_raw, section_count, message
```

**The tool logic lives in plain functions. This module registers them. The graph imports the
functions directly.** A tool call from a LangGraph node ([#36](../../issues/36)) is an ordinary
Python call: no subprocess, no stdio framing, no event loop threaded through the graph, and no
second failure mode when the transport hiccups. Sending the graph to its own tools over MCP would
buy nothing — the same three functions answer either way — and would cost a process boundary on
every node. The server is the *external* surface, useful on its own; it is not the graph's
transport. Revisit only if a tool ends up somewhere the graph cannot import.

**The Pydantic models are the contract for both callers.** A field added to `CourseDetail` for the
graph shows up in the published output schema with no edit here, and the *input* schemas are built
out of the query models' own field objects — `Annotated[str, CourseQuery.model_fields["course_id"]]`
— so a description, a default, and a constraint like `max_depth >= 1` each have exactly one source.
`tests/test_mcp_server.py` compares every served schema against the model it came from, which is
what fails when a field is added to a model and not to a wrapper.

**The docstring under each `@mcp.tool` is the tool description a client reads**, and it is what
decides whether a model picks the right tool, so those three are written for a model rather than for
a maintainer: what the tool answers, what belongs in each argument, when to call a different one,
and what `found=false` / `supported=false` mean. The reasoning behind each tool stays in
`agent/tools/`.

**No auth, no HTTP mount, no `OPENAI_API_KEY`.** `bloom-mcp` needs the first two because it is
deployed behind a gateway; this is a local stdio server over a published course catalog, and every
tool is read-only (`readOnlyHint`, so a client need not prompt before a call). The three tools read
committed JSON and never embed, classify, or generate — the server starts with no key, no `.env`
and no index built, and nothing it imports pulls in `openai` or `chromadb`.

Verified over a real stdio client, not only in-memory: 3 tools listed, `get_course("cs 374")` →
`CS 374`, 4.0 hours, and `get_prereq_chain("SE 495")` → 5 levels with `cycle_detected=true`. The
in-memory equivalent runs on every commit — `tests/test_mcp_server.py` lists the tools through a
real `fastmcp.Client` and asserts each call returns the same dict the Python call does, so a schema
that will not generate or a model that will not serialize fails in CI rather than in a client.

## Agent graph

`agent/state.py` + `agent/graph.py`: one message in, one of three exits out.

```
route_intent ──► out_of_scope ──────────────────────────────► END
     │
     └────────► dispatch ──► check_evidence ──► answer ─────► END
                                  │              │
                                  └──────────────┴► refuse_handoff ──► END
```

```python
from agent.graph import run

state = run("What do I need before CS 374?")
state["exit"]       # 'answer' | 'refuse_handoff' | 'out_of_scope'
state["answer"]     # the text a person reads, on all three
state["signals"]    # every number the run measured on the way
```

`run(query, cfg=None, nodes=None)` is the whole public surface — the CLI ([#39](../../issues/39))
and every test call that, not the compiled graph, so compilation, the config plumbing and the
final-state check have one home. All six nodes are real as of [#38](../../issues/38) — the router,
the dispatcher, the evidence gate and the answer node each have a section below. `nodes=` replaces
any of them by name, which is how the three exits are exercised with no OpenAI credits and how each
node is tested on its own.

**`docs` is one record shape whatever produced it.** A retrieval `Hit` and a tool result both
become `Evidence(doc_id, text, score, source)` in the dispatch node, so the answer node has a
single input format and its citation check works the same for both. `score=None` marks an exact
catalog lookup and is what lets the evidence gate skip the threshold for it — there is no ranking
to be unsure about. Passing a raw `Hit` through is refused: it has `doc_id`, `text` and `score`, so
everything downstream would work except the one field the gate needs.

**Which exit a run took is a field, not an inference.** `exit` is written by the node that ends the
run, rather than guessed from which of `answer` / `evidence_ok` / `docs` happen to be set — #39's
report counts exits, and inference would make that count a guess.

**`signals` has a declared key set.** Eight keys, prefilled with `None`, listed in one place
(`SIGNAL_KEYS`) and enforced: a node that writes an undeclared signal is refused. Day 5 calibrates
on these across many runs, so the record has to be rectangular — a dict that grows keys ad hoc over
five days gives it a ragged table. It is also the only state key with a reducer: four nodes each
write their own signals, and without one the last update would erase the rest.

**No node reaches into another's state key.** `NODE_WRITES` says who owns what — routing writes
`intent`, dispatch writes `docs`, the gate writes `evidence_ok`, the three exits write `answer` and
`exit`, nobody rewrites `query` — and every node is wrapped in a check that refuses anything else.
LangGraph enforces none of this: an update naming a key the state does not have is **silently
dropped**, and a node writing another node's key simply wins by running later. Both are the same
bug, so both fail at the node that caused them.

**The two refusals are fixed text, not completions.** `out_of_scope` and `refuse_handoff` never
call a model: nothing grounds a generated refusal, and a model asked to write one will argue itself
into a partial answer with a hedge in front of it. The hand-off message says what was asked, where
the run stopped and what would get it answered — the middle line read off `signals`, which is the
second reason those are recorded rather than compared and dropped:

```
I could not answer this from the 2026-sp course catalog, so I am handing it over rather than guessing.

What you asked: What are the prerequisites?
Where I stopped: the question does not name a course, and which one was meant is not something to
guess at -- answering confidently about the wrong course is worse than not answering
What would get it answered: name it by code: 'CS 374', 'cs 374' and 'CS374' all reach the same
catalog record

An academic advisor can also see what this catalog export cannot -- your transcript, program
requirements outside the Gen Eds, and sections as they actually fill.
```

**Until #38 the answer node was a stub that raised**, rather than one that returned a plausible
`answer` string: a stand-in that answered would have let the graph run end to end and #39's smoke
report come out of it — ten queries, three exits, a table of numbers, and not one model call behind
any row. The rule outlived the stub, and it is why the two refusal exits are fixed text and why a
draft whose citations do not check out is discarded rather than shown with a caveat.

## Intent router

`agent/nodes/route_intent.py` + `agent/prompts.py`: the first node, and the only place a sentence
becomes a decision. Five labels, a few-shot prompt, and a validator that trusts none of it.

```python
from agent.nodes.route_intent import route_intent_node

node = route_intent_node(cfg)                          # the real classifier
node = route_intent_node(cfg, chat=fake)               # anything with .complete() — how it is tested
node = route_intent_node(cfg, forced="prereq_chain")   # #39's --intent: no model call at all
```

| label | what it covers | example |
|---|---|---|
| `course_fact` | one course, one attribute | "How many credits is CS 374?" |
| `prereq_chain` | what comes before a course | "What do I need before SE 495?" |
| `degree_requirement` | progress against a program | "Which Gen Eds do I still need?" |
| `compare_recommend` | open-ended, several candidates | "Which intro ML course should I take?" |
| `out_of_scope` | not about this catalog | "What's the weather in Champaign?" |

**The completion is validated, never trusted.** The model's string is compared against the five
literals and nothing else is accepted: a sentence, a label with a period on the end, an invented
sixth category all become `out_of_scope` with `signals["intent_unparsed"] = True` and the raw
string kept in `signals["intent_raw"]`. Free text must never reach the dispatch table — it indexes
a table by this value — and a classification that could not be read has to be **countable**
afterwards rather than absorbed.

**The strictness has a price and the price is recorded.** `Course_fact` and `course_fact.` are
in-scope classifications this parser refuses, and a refused in-scope query reaches the person as
"outside what I can answer". Lowercasing and stripping punctuation would buy those back and hide
how often the model ignored the format — which is the number Day 5 needs in order to decide
whether the prompt or the parser is what should change. So the parser stays exact and `intent_raw`
keeps what came back.

**The router does not decide what the catalog has.** A question about a course that was never
offered is still a question *about a course*, and its refusal belongs to the node that looked and
came back empty — the hand-off exit, not the out-of-scope one. Two of the prompt's four rules say
so, because "I do not do that" and "I looked and could not find it" are different sentences, and
only one of them is honest about a missing course.

**An API failure is not a classification.** A 429, a timeout or a cut-off completion propagates.
Turning those into `out_of_scope` would put a billing problem into #39's report as a routing
decision, and would mix infrastructure failures into the `intent_unparsed` count.

**`intent_raw is None` means no model was asked.** `forced=` skips the call and writes the label it
was given, which is what `--intent` is built from; no other path through the node leaves the raw
signal empty, so a row in #39's report with an empty raw column is a row with no model behind its
label.

**The prompt is one versioned constant**, assembled in `agent/prompts.py` from the label list in
`agent/state.py` — a label added to the state with no gloss and no examples fails at import, so
the prompt and the validator cannot teach different vocabularies. The version is a module constant
rather than a config key: a chunk template decides what is *in* the index, while a router prompt is
built per query and thrown away, and hashing it into the config would rename both index keys and
every report name to record that an example sentence was reworded. A digest test pins the text to
its version, so editing one without the other fails rather than publishing a second "v1".

**v2: two examples, against a mistake the rules already forbade.** The first live run with a real
model (gpt-4o-mini, 2026-09-21) labelled two questions `out_of_scope` that are not:
`What are the prerequisites?` and `Is there a course about quantum origami?`. Both are covered in
words — rule 2 says to classify what the question asks for and not whether the catalog can answer
it, rule 3 says a question that names no course is still `prereq_chain` — and neither rule held,
because every example in the prompt was a question the catalog *can* answer. The cost is not a
mislabelled row: the first hands a person "that is outside what I can answer" when the honest
reply asks which course, and the second never reaches the evidence gate that exists to refuse it.
So `prereq_chain` gained a course-less example and `compare_recommend` an existence question, the
version went to `v2`, and the digest moved with it.

Neither new example is one of the two questions above: they are rows of
`eval/datasets/day2_queries.jsonl`, and a prompt that quotes its own evaluation set stops measuring
anything. They teach the same two shapes on questions the set does not contain — a test asserts
that, because the temptation to paste the failing string in is exactly what makes a smoke run look
green.

The chat client is built on the **first query**, not when the node is built, and `llm` is imported
there — it reaches `openai` and `OPENAI_API_KEY` through `embedding.py`, and `build_graph()` has to
stay callable in a checkout with no key. The `llm` config section *is* validated at build time:
that costs nothing, and a config naming a provider with no client should not survive until the
first query.

## Evidence dispatch

`agent/nodes/dispatch.py`: the second node, and the only place an intent becomes an action.

| intent | where the evidence comes from |
|---|---|
| `course_fact` | `get_course(code)` when the question names a code — otherwise `retrieve()` |
| `prereq_chain` | `get_prereq_chain(code)`; no code is a refusal, never a guessed course |
| `degree_requirement` | `check_requirement(major, taken)`, both read out of the question |
| `compare_recommend` | `retrieve()` — several ranked candidates is the shape of that answer |
| `out_of_scope` | never arrives: the router's own exit ended the run before this node |

```python
from agent.nodes.dispatch import dispatch_node

node = dispatch_node(cfg)                  # the mode comes from config.yaml's agent.retrieval_mode
node = dispatch_node(cfg, mode="hybrid")   # an ablation, or a test that ignores the shipped config
node = dispatch_node(cfg, retrieve=fake)   # how the retrieval rows are tested with no index on disk
```

The alternative — four conditional edges into four near-identical nodes — would put this table in
the graph's wiring, where the rows could not be read side by side.

**Everything becomes `Evidence`.** A retrieval hit keeps its score and names the mode; a tool
result is rendered to text, carries `score=None` and names the tool (`get_course`,
`get_prereq_chain`, `check_requirement`). One record shape is what gives [#38](../../issues/38) a
single input format, and the `None` is what tells [#37](../../issues/37) that an exact catalog
lookup has no ranking to be unsure about. A lookup block is rendered in the **chunk template's own
labels**, so an answer built from `get_course` and one built from a retrieved chunk of the same
course are reading the same shape:

```
CS 374 — what the catalog requires before it
Prerequisite statement: One of CS 173, MATH 213; CS 225.
Immediately before it: CS 173 — Discrete Structures; CS 225 — Data Structures; MATH 213 — Basic Discrete Mathematics
2 steps back: CS 124 — Introduction to Computer Science I; CS 128 — …
16 courses over 4 levels, 26 prerequisite edges.
A level names every course the statements at that depth mention; where a statement says "one of", only one of them is required.
The catalog's own prerequisites form a cycle inside this chain — some of these courses list each other — so the walk stopped rather than looping.
```

That last-but-one line is not decoration. The layers flatten the booleans — CS 374 asks for CS 225
and *one of* CS 173 / MATH 213 — and 744 of the catalog's 1,135 prerequisite statements contain a
disjunction, so the block says what it is rather than letting an answer read three required courses
off it. The catalog's own sentence comes first for the same reason: it is the only part that is
exactly right.

**A tool that came back empty produces no evidence, not evidence about its emptiness.** `docs` is
empty and `signals["tool_error"]` says which of three things happened; `agent/nodes/exits.py` has
the sentence and the remedy for each, and a test asserts it still does.

| `tool_error` | when |
|---|---|
| `no_course_code` | the question needs one course and names none ("What are the prerequisites?") |
| `course_not_found` | it names a course this export does not have (`CS 999`, or `CS 428`, real but not offered in 2026-sp) |
| `no_program` | no program this corpus can audit could be read out of it ("the Physics major") |

Putting "CS 999 is not in the catalog" into `docs` instead would hand the answer node a sentence
about the absence of an answer as though it were a source, and a model given that writes a fluent
refusal with a guess folded into it — which is the whole reason the two refusal exits are fixed
text. It is also why the requirement row resolves the program against `ingest/majors.json` itself:
`check_requirement`'s own `supported=False` refusal is for the MCP surface, where a client names a
program directly.

**Codes come out of free text through the matcher the BM25 index was built with** — `CODE_RE`, then
`normalize_code`, so `CS 374` / `cs 374` / `CS374` reach one catalog record and a query cannot
retrieve under one normalization and look up under another. What this node adds is the catalog's
own subject vocabulary, **187 subjects** in this term: "I have 100 credits" matches the pattern and
there is no subject `HAVE`, so it names no course. The cost is that a code whose *subject* has
nothing at all in the export reads as no code — accepting "have 100" and "top 100" as courses is
the worse trade, and either way the hand-off asks for a course code. A question naming several
codes looks up the first the catalog holds (so a stray match cannot shadow a real course) and falls
back to the first named (so `CS 999` still gets "not in this catalog" rather than "you named no
course"); only the transcript row reads every code in the question.

**Gen Ed wins when a question names both a degree and the Gen Eds**, because a degree's audit is
the one that cannot answer it — every program reports the campus General Education requirements in
`not_checked`, unevaluated. Course codes are stripped before any program name is looked for, since
`stat` is both a Statistics-degree alias and the subject of `STAT 200`, and a transcript is not a
program.

**The mode is a config value**, `agent.retrieval_mode`, not a constant in the node: which retriever
answers is an experiment setting, it belongs in the hash the report is named after, and a checkout
with no dense index has to be able to run. It says `bm25` today because the dense collection is
still empty ([#5](../../issues/5) is blocked on OpenAI credits) and a default nothing can run is a
crash on query one. Adding the section moved the full hash (`ade2f799` → `51939bdf`) and left
`chunk_key` `05d00726`, `dense_key` `094143d8` and `bm25_key` `877a6610` exactly where they were.

**What the retrieval rows are worth today is bm25 alone, and it shows.** "What does the algorithms
course cover?" is a `course_fact` with no code, so it retrieves — and on the live BM25 index the
top five are ECON 300, ERAM 557, HDFS 322, UP 230 and IE 531, with CS 374 nowhere in them. That is
the dense half's job, and the gate ([below](#evidence-gate)) is what refuses it in the meantime:
11.79 against a `bm25` threshold of 15.0, so that row hands the question to a person today. The row
is correct; the index behind it is half-built.

## Evidence gate

`agent/nodes/check_evidence.py` ([#37](../../issues/37)): the third node, and the only place the
run decides there is not enough to answer from. Day 5 replaces it with conformal prediction; Day
2's version is a threshold — one per source, because the scores are not comparable:

```yaml
evidence:
  method: top_score     # day 5 adds "conformal"
  thresholds:
    bm25: 15.0          # Okapi BM25, unbounded
    dense: 0.46         # 1 - cosine distance — measured, band 0.4529–0.6781
    hybrid: 0.025       # RRF — kept, but top_score refuses to gate on it at all
```

Four outcomes, and only the last one consults a threshold:

| in `docs` / `signals` | verdict | why |
|---|---|---|
| `signals["tool_error"]` set | `False` | the tool produced nothing, so there is nothing to be confident *about* |
| `docs` empty | `False` | a retrieval that came back with nothing — the honest bottom of the scale |
| any block with `score=None` | `True` | an exact catalog lookup; there is no ranking to be unsure about |
| otherwise | `score >= thresholds[source]` | the best score in the list, against its own source's number |

**The module holds no number of its own** — a test reads its syntax tree and fails on any float.
A threshold in the code would be a tuning decision outside the config hash, and the hash is what
ties a refusal rate in a report to the threshold that produced it.

**Both numbers are recorded on every run, passing or failing.** `signals["retrieval_top_score"]`
and `signals["evidence_threshold"]`, which is what the hand-off message quotes, and what Day 5
calibrates on: a gate that compared the score and dropped it would make Day 5 start by re-running
everything. The *source* is not written twice — `signals["retrieval_mode"]` is the dispatcher's and
for a scored block it is exactly `Evidence.source`.

**The `bm25` number is a cutoff on this corpus, not a confidence, and the measurements say so.**
Top-1 scores on the live index, against `bm25: 15.0`:

| query | top-1 | what came first | verdict |
|---|---|---|---|
| `How many credits is MATH 241?` | 30.91 | MATH 241 | passes, right |
| `CS 374` | 29.03 | CS 374 | passes, right |
| `Which course covers organic chemistry?` | 18.65 | CHEM 360 (Chemistry of the Environment) | passes, wrong |
| `Which intro machine learning course should I take?` | 18.27 | IS 557 | passes |
| `I want a course on Shakespeare` | 16.22 | ENGL 218 (Introduction to Shakespeare) | passes, right |
| `linear algebra` | 14.97 | MATH 227 (Linear Algebra for Data Science) | refused, right |
| `What does the data structures course cover?` | 13.20 | ERAM 557 (Semiotics) | refused, wrong |
| `machine learning` | 11.91 | IS 557 | refused |
| `What does the algorithms course cover?` | 11.79 | ECON 300 | refused, wrong |
| `Is there a course about quantum origami?` | 11.06 | ECE 305 | refused, wrong |
| `introduction to statistics` | 9.00 | STAT 400 | refused, right |
| `algorithms` | 8.48 | CS 374 | refused, right |
| `asdfghjkl qwertyuiop` | — | no hits at all | refused before any threshold |

Two rows are the whole argument. `algorithms` puts **CS 374 first at 8.48** — exactly right — and
scores *below* `Is there a course about quantum origami?` at 11.06, which puts ECE 305 first and
names a course that does not exist: no threshold on this scale admits the first without admitting
the second. And `machine learning` (11.91) and `Which intro machine learning course should I take?`
(18.27) return the **same document** — BM25 sums a contribution per query term, so the longer
sentence scores 53% higher for the same match, which means part of what this threshold measures is
how many words the question had. It separates the handwritten set correctly and it was chosen from
that set. It is not a probability, and that gap is what Day 5 closes.

**The dense number is a cutoff on thinner evidence than the BM25 one.** Top-1 cosine similarity
over the same twelve queries (collection `courses_094143d8`, measured 2026-09-21) runs from
**0.4529 to 0.6781** — so the `0.35` this section carried while the collection was empty sat below
the entire band and refused nothing, twelve passes out of twelve. `dense: 0.46` refuses exactly one
row, `Is there a course about quantum origami?` at 0.4529, and the next row up (`algorithms`,
0.4619) clears it by 0.002. One negative example and a 2% margin is a floor fitted to a single
query, not a tuned threshold; Day 4's negative set is what earns it. Dense top-1 is also the right
course only 6 times in 12 — `CS 374` returns **CS 473** first, at 0.6200 — so it is no more a
confidence than the BM25 number is.

**An RRF threshold cannot be a gate at all, and the measurement is what says so.** With `rrf_k: 60`
and `top_k_each: 20`, a document one retriever ranked first scores `1/61 = 0.0164` and one both
retrievers returned at the worst rank either reports scores `2/80 = 0.025`, so `0.025` is exactly
"both retrievers returned it" — which is what this section claimed for it while the dense index was
unbuilt. With the index built, top-1 over those twelve queries spans **0.0296 to 0.0328**, and
`Is there a course about quantum origami?` scores **0.0325 — fourth highest of the twelve**, with
six answerable queries *below* it. The top document of a fused list is nearly always one that both
retrievers returned near the top, whatever it is, so the score barely moves with the query and no
cutoff separates the set.

So `top_score` **refuses** a hybrid block instead of comparing it (`UNGATEABLE` in
`check_evidence.py`), and `check_evidence_node` raises when it is built if `agent.retrieval_mode`
is `hybrid` — not on the fifth query, the first one whose evidence carried a score. Passing
everything is worse than handing over: in a report it is indistinguishable from evidence that was
checked and found good. The number stays in the table so it holds one entry per retrieval mode, and
so Day 5's calibrated method has somewhere to land.

**One scored source per run.** A `docs` list carrying two retrievers' raw scores is refused rather
than resolved: "the top score" is only defined inside one source, the two numbers in `signals` can
only describe one comparison, and a `max()` over two scales is not a decision. The dispatcher
produces one source per run by construction, which is what makes that safe to rely on.

The ten handwritten queries through the real graph — real dispatch, real gate, the router forced to
each label so no model is asked (what #39's `--intent` will drive), a stand-in for #38 — on the live
corpus and the live BM25 index:

```
exit            evidence_ok  top score  threshold  tool_error        query
answer          True         —          —          —                 How many credits is MATH 241?
answer          True         —          —          —                 What do I need before CS 374?
answer          True         —          —          —                 Which Gen Eds do I still need if I have taken RHET 105 and MATH 241?
answer          True         18.2746    15.0       —                 Which intro machine learning course should I take?
refuse_handoff  False        11.7914    15.0       —                 What does the algorithms course cover?
answer          True         —          —          —                 What is left for the Statistics major if I have taken STAT 200?
refuse_handoff  False        —          —          no_course_code    What are the prerequisites?
refuse_handoff  False        —          —          course_not_found  How many credits is CS 999?
refuse_handoff  False        11.0561    15.0       —                 Is there a course about quantum origami?
out_of_scope    None         —          —          —                 Who is the head football coach at Illinois?
```

Five answers (four of them exact lookups that never reached a threshold), four hand-offs, one out
of scope. Adding the section moved the full hash (`51939bdf` → `e0d6ff30`) and left `chunk_key`
`05d00726`, `dense_key` `094143d8` and `bm25_key` `877a6610` exactly where they were.

## Answer node

`agent/nodes/answer.py` + the second prompt in `agent/prompts.py` ([#38](../../issues/38)): the
last node, and the only one that puts a model's words in front of a person.

```python
from agent.nodes.answer import answer_node, verify_citations

node = answer_node(cfg)              # the real client, built on the first query
node = answer_node(cfg, chat=fake)   # anything with .complete() — how it is tested
```

Three constraints, which are the whole reason it is a module and not one call:

1. **Answer only from the evidence.** The model knows these courses from pretraining, all of it
   from some other year of the catalog and none of it checkable against this export. The evidence
   blocks are the only course data in the message, and there is no second turn and no tool.
2. **If the evidence does not contain it, say so** — as a normal answer, "the catalog entry for
   CS 374 doesn't list who teaches it", not an apology and not a value that is probably right.
3. **Cite `doc_id`, then verify the citation.** Every id the answer cites must be a `doc_id` this
   run actually retrieved. One that is not discards the whole answer for the hand-off exit: a model
   that invented a citation invented the claim behind it, and an *unchecked* citation is worse than
   none, because it looks like verification.

The evidence is rendered numbered, each block headed by its own id, and neither `score` nor
`source` reaches the prompt — a score means nothing outside the retriever that produced it, and a
model shown one will weigh the blocks by it. Whether there is enough here was already decided, by
the gate:

```
Question: Which intro machine learning course should I take?

Evidence:

Evidence 1 of 5
doc_id: 2026-sp:IS-557
---
IS 557 — Applied Machine Learning: Team Projects
Credits: 4
Term: 2026-sp
Description: A comprehensive exploration of the applied machine learning workflow …
Prerequisites: Students should have demonstrated ability, and must have taken one of …

Evidence 2 of 5
doc_id: 2026-sp:IS-327
---
…

Answer:
```

**The citation check is a set lookup, not a judgement.** A bracketed id matches a block when the
two strings are equal after collapsing whitespace and folding case, and nothing looser:

| the answer says | `docs` holds | outcome |
|---|---|---|
| `[CS 374]`, `[cs 374]` | `CS 374` | verified — an id is an identifier, and case is not a claim |
| `[2026-sp:CS-374]` | `2026-sp:CS-374` | verified |
| `[Computer Science, BS]` | `Computer Science, BS` | verified — a bracket is matched **whole** before it is split on commas |
| `[CS 374, CS 225]` | both | two citations — the split is the fallback, not the rule |
| `[CS 374]` | `2026-sp:CS-374` | **unknown** — never resolved by containment |
| `[1]` | (the first block) | **unknown** — the prompt says cite the id, not the number |
| `[CS 999]` | anything else | **unknown** — the fabrication the node exists to catch |

Resolving `[CS 374]` against a chunk id that contains it is the tempting rule and the wrong one: an
answer about CS 225 that cites the CS 374 chunk would pass it, with a reassuring bracket after the
wrong claim. The near misses strictness costs are recorded as themselves in
`signals["citations_unknown"]`, so Day 5 can count them and decide whether the prompt or the parser
is what should change — the same argument as the router's exact label match, one node later.

**Two ways a draft is discarded, and they are distinguishable afterwards:**

| `citations_verified` | `citations_unknown` | exit | what happened |
|---|---|---|---|
| `["CS 374"]` | `None` | `answer` | every citation checks out; the draft is the answer |
| `["CS 374"]` | `["CS 999"]` | `refuse_handoff` | one invented citation; the whole draft goes |
| `[]` | `None` | `refuse_handoff` | it cited nothing at all — nothing invented, nothing traceable |
| `None` | `None` | (any) | this node never ran |

The uncited case is not pedantry: rule 4 of the prompt asks even "the catalog does not say" to cite
the block it looked in, so every answer the node can accept is able to carry a citation, and a
draft with none is not a modest answer but an untraceable one. Either way **the discarded draft is
dropped, not shown with a warning**, and this node writes no refusal of its own — it claims the
hand-off exit and `agent/nodes/exits.py` mints the sentence, so the message a person reads is
written in one place however the run got there:

```
Where I stopped: I drafted an answer, but it cited CS 999, which is not in the evidence I
retrieved -- a citation I cannot check is not one I will pass on, so I dropped the answer
```

**The prompt is a second versioned constant** (`ANSWER_PROMPT_VERSION`, digest-pinned like the
router's), and there is **one** of it rather than one per intent: the four intents differ in what
their evidence *is*, not in what is to be done with it, and four variants would be four texts that
drift and four things Day 4 has to ablate. It names no term either — the term is in the blocks
themselves, off the catalog record.

The ten handwritten queries through the whole graph — real dispatch, real gate, real answer node,
the router forced to each label (what #39's `--intent` drives), over the live corpus and the live
BM25 index. The **model is a stand-in**: it answers with the first line of block 1 and cites that
block's id, so the wording is canned and everything around it — the prompt sent, the citation
written, the verdict on it — is real:

```
exit            ev_ok  top score    thresh  tool_error         cited              query
answer          True           —         —  —                  MATH 241           How many credits is MATH 241?
answer          True           —         —  —                  CS 374             What do I need before CS 374?
answer          True           —         —  —                  General Education  Which Gen Eds do I still need if I have taken RHET 105 and MATH 241?
answer          True     18.2746      15.0  —                  2026-sp:IS-557     Which intro machine learning course should I take?
refuse_handoff  False    11.7914      15.0  —                  —                  What does the algorithms course cover?
answer          True           —         —  —                  Statistics, BSLAS  What is left for the Statistics major if I have taken STAT 200?
refuse_handoff  False          —         —  no_course_code     —                  What are the prerequisites?
refuse_handoff  False          —         —  course_not_found   —                  How many credits is CS 999?
refuse_handoff  False    11.0561      15.0  —                  —                  Is there a course about quantum origami?
out_of_scope    None           —         —  —                  —                  Who is the head football coach at Illinois?
```

Five answers, four hand-offs, one out of scope — and five model calls, one per run that got past
the gate. All three `doc_id` shapes this repo produces are cited and verified there: a course code
(`MATH 241`), a chunk id (`2026-sp:IS-557`) and a program label with a comma in it
(`Statistics, BSLAS`). Feeding the same first row a fabricated `[CS 999]` flips it to
`refuse_handoff` with `evidence_ok` still `True` — which is exactly how #39's table will tell a
citation downgrade from a gate refusal.

Measured for #39's cost line, on `gpt-4o-mini`'s tokenizer: the answer system prompt is 470 tokens
and the router's 449, and the five answering rows send 5,314 input tokens in total (164 for the
MATH 241 lookup, 1,341 for the five-chunk retrieval row) — so the whole ten-query run is about 10k
input tokens plus at most five completions.

No config value moved there: the prompt version is a module constant, so #38 left the hash where it found it
and `chunk_key` / `dense_key` / `bm25_key` are untouched.

## Command line

`python -m agent` is the agent as a person uses it: one question in, one of three exits out, and
the whole run printed.

```bash
uv run --env-file .env python -m agent "What do I need before CS 374?"
uv run python -m agent "..." --intent prereq_chain --fake-model   # no model, no key
```

```
query     What do I need before CS 374?
intent    prereq_chain       classified
evidence  get_prereq_chain, 1 block

  #  score  doc_id  first line
  1      —  CS 374  CS 374 — what the catalog requires before it

gate      passed -- an exact catalog lookup carries no score, so no threshold was consulted
exit      answer             cited CS 374
answer    generated by gpt-4o-mini from the evidence above

Before CS 374, you need "one of CS 173, MATH 213; CS 225" [CS 374].

(config c73b197d, prompts router v2 / answer v1, 2.2 s including any index load)
```

| flag | what it does |
| --- | --- |
| `--intent` | skip the classifier and force one of the five labels ([#35](../../issues/35)). Everything downstream still runs — the dispatch table, the three tools, the retriever, the gate — and `signals["intent_raw"]` stays empty, which is what the trace prints as "no model was asked". |
| `--fake-model` | answer with `llm.StandInChat`: it replies with the first line of the first evidence block and cites that block's `doc_id`. Requires `--intent`, because a stand-in cannot classify. Nothing is sent and no key is read, which is how the graph was built and tested before the account had credits — and still the way to exercise it for free. |
| `--config` | a config file other than `config.yaml`. |

**It prints a trace, not an answer.** Which label was chosen, which row of the dispatch table ran,
what the gate compared and which exit was taken are the things that go wrong; the answer alone
shows none of them, and on the two refusal exits it is fixed text that reads the same however the
run got there. `agent/trace.py` renders it from the final state and nothing else, so the smoke
report below can print the same facts per query without a second formatter to drift.

**`--fake-model` is loud everywhere it touches.** The client names itself `stand-in` rather than
the configured model, the trace says the text is canned, and the report says so in its header, in
a section of its own and beside every answer. What it buys is that the prompt, the citation parse,
the check against `docs` and the exit the node claims are all real, which is the most a run can
honestly show with nothing behind it. What it does not buy is any evidence about *wording* —
whether a real model stays inside its evidence is the one thing a stand-in cannot demonstrate,
and it is what the live run below is for.

## Day 2 smoke report

`eval/scripts/day2_smoke.py` runs the ten handwritten queries in
`eval/datasets/day2_queries.jsonl` end to end and writes
[`report/day2_smoke_7980f15f.md`](report/day2_smoke_7980f15f.md):

```bash
uv run --env-file .env python -m eval.scripts.day2_smoke   # the committed report
uv run python -m eval.scripts.day2_smoke --fake-model      # the same run with no model, no key
```

```
config hash 7980f15f  (mode bm25)
queries                10  (eval/datasets/day2_queries.jsonl)
answers written by     gpt-4o-mini
exits                  answer 4, refuse_handoff 5, out_of_scope 1
agreement              intent 10/10, exit 9/10
wrote                  report/day2_smoke_7980f15f.md  (17 KB)
```

The report keeps its `7980f15f` name: an artifact is named after the config that produced it, and
[#61](../../issues/61) has since added `eval:` to `config.yaml` and moved the hash to `c73b197d`
without touching anything this run depended on (see [Frozen split](#frozen-split)).

Each row is `{query, expected_intent, expected_exit, note}`: all five intents are covered, and all
three exits are expected by at least one row. The set is **handwritten, not sampled** — it exists
to prove the wiring, and Day 4's ablation set is a different artifact with a different job.

| exit | expected | actual |
| --- | --- | --- |
| `answer` | 5 | 4 |
| `refuse_handoff` | 4 | 5 |
| `out_of_scope` | 1 | 1 |

**`expected_intent` and `expected_exit` are what a row should do, not what it does today**, so a
row that does something else is reported as a disagreement rather than re-labelled in the dataset.
One row still is:

- **`What does the algorithms course cover?`** — a `course_fact` with no code, so it retrieves,
  and BM25 alone puts ECON 300 first at 11.79 against a threshold of 15.0. The refusal is right
  for that evidence; the row stays as written, and `agent.retrieval_mode` is what has to move.

The other two disagreements this set had are gone, and they are worth keeping in the record: under
router prompt `v1`, `gpt-4o-mini` sent **"What are the prerequisites?"** and **"Is there a course
about quantum origami?"** to `out_of_scope`, so a person asking either got "that is outside what
I can answer" where the honest answers are "name the course by code" and "I looked and this
catalog does not have it". The prompt's rules already said both belong elsewhere; what was missing
was an example of each shape, since every example above them named a course the catalog has.
[Prompt `v2`](#intent-router) adds one apiece — deliberately not the eval set's own strings — and
on this run both land where they belong, the origami query refusing at the gate (11.06 against
15.0) rather than at the router. Intent agreement went 8/10 → **10/10**, and
`intent_unparsed` was `false` on every row both times, so #35's strict validator has still never
cost this set anything.

**The acceptance is a check, not a sentence in a report.** The runner writes the report and then
exits non-zero if any of the three exits never fired, so a regression is a failed command and not
a table nobody re-read.

**A live run is not reproducible, and `temperature: 0` does not make it one.** The report carries
no timestamp and no latency, so nothing in it can move unless the run did — under `--fake-model`
a re-run rewrites a byte-identical file. Across four live runs, one row has moved: "Which intro
machine learning course should I take?" was answered three times ("The catalog does not carry a
recommendation …", citing IS 557) and once drafted with **no citation at all**, which
[#38](../../issues/38)'s check discarded for the hand-off exit. That is the uncited-draft path
firing on its own, and it is also the sharpest thing these runs say about the *answer* prompt: for
`compare_recommend`, rule 2 ("say plainly when the evidence does not carry it") is winning against
a row whose whole shape is several ranked candidates, and the model declines to recommend at all.
That one is open as [#53](../../issues/53); Day 4 ablates the prompt, and this is the row to
ablate it on.

Every path in the file goes through `rel_to_repo()`, so no home directory lands in a tracked file,
and the caveats are generated from the run rather than pasted in — the stand-in report says "no
model was called anywhere in this run" and the live one does not.

**What the live run still does not measure:** `agent.retrieval_mode` is `bm25`, so the retrieval
rows were answered lexically and the hybrid path is plumbing-verified, not result-verified. Which
mode runs is a config value, so switching it is a new hash and a new report rather than a footnote
to this one — and on this corpus that switch is [not a free one](#evidence-gate): the fused score
of a top-1 document barely moves between a good match and a bad one, which is why `top_score`
refuses a hybrid block instead of comparing it.

## Eval item schema

Day 3's deliverable is a dataset, and a dataset without a schema is a text file. The contract and
its check land **before** the first question is drafted: `eval/schema.py` is the row and the bucket
table, `eval/validate.py` is the runnable check over a file of them
([#54](../../issues/54)).

```bash
uv run python -m eval.validate eval/datasets/v1.jsonl
```

```json
{
  "id": "v1-0001",
  "bucket": "fact",
  "question": "How many credits is MATH 241?",
  "gold_answer": "4 credit hours.",
  "gold_doc_ids": ["2026-sp:MATH-241"],
  "should_abstain": false
}
```

Six fields, all required, `extra="forbid"` — a drafting run ([#56](../../issues/56)) that invents a
`difficulty` field fails loudly instead of writing a column nothing reads. `id` is minted by
`schema.item_id()` in pipeline order and never chosen by a model: ids are how the agreement sample
([#60](../../issues/60)) and the frozen split ([#61](../../issues/61)) name rows, so a renumbering
after either of those lands is a corruption that still loads.

| bucket | n | `should_abstain` | `gold_doc_ids` | Day 2 exit it should reach |
| --- | ---: | --- | --- | --- |
| `fact` | 50 | `false` | ≥ 1 | `answer` |
| `multihop` | 25 | `false` | ≥ 2 | `answer` |
| `unanswerable` | 25 | `true` | empty | `out_of_scope` or `refuse_handoff` |
| `ambiguous` | 20 | `true` | empty or ≥ 1 | `refuse_handoff` (clarify) |

**`should_abstain` is redundant with `bucket`, on purpose.** The bucket fixes it, so the two can
disagree — and a row where they do is a mislabel that one field alone would have hidden. It stays a
single bit ("must not emit a confident answer") because the two abstaining buckets do not want the
same behaviour: an `unanswerable` row should end at *"this catalog does not have that"*, an
`ambiguous` one at *"which course do you mean?"*. That distinction is carried by `bucket`, every
metric conditions on it, and for `ambiguous` rows `gold_answer` holds **the clarifying question
that should have been asked** rather than a factual answer.

**The counts warn; everything else fails.** Deleting a bad item during verification
([#59](../../issues/59)) is the whole point of verification, and a check that fails on 49 facts is
a check that argues for keeping a bad row.

What it checks, all at once and one message per problem:

- every `gold_doc_id` is really in the corpus — and a miss gets the id it was probably meant to be,
  because `2026-sp:MATH 241` for `2026-sp:MATH-241` is the likeliest error in a hand-typed dataset
  (`metadata.code` has the space; only the doc id has the dash);
- no id repeated inside one row, which is how a one-hop row wears two `multihop` ids;
- `should_abstain` agrees with the bucket, and the gold-id count is inside the bucket's bounds;
- ids unique, and no question asked twice once whitespace and case stop counting;
- every row is six fields with non-empty prose, `bucket` one of the four, `id` a `v1-NNNN`.

```
$ uv run python -m eval.validate tests/fixtures/eval_bad.jsonl
tests/fixtures/eval_bad.jsonl:1  v1-0001 gold_doc_ids.0  '2026-sp:MATH 241' is not in the corpus; did you mean '2026-sp:MATH-241'?
tests/fixtures/eval_bad.jsonl:3  v1-0003 gold_doc_ids  1 id; an unanswerable row carries no gold ids at all -- this catalog cannot answer it, so there is nothing to cite
tests/fixtures/eval_bad.jsonl:4  v1-0004 question  repeats line 1 (v1-0001) once whitespace and case are normalised
tests/fixtures/eval_bad.jsonl:6  v1-0006 gold_doc_ids.1  repeats '2026-sp:CS-173' from position 0; a document cited twice is still one document, and in a multihop row it is one hop wearing two ids
tests/fixtures/eval_bad.jsonl:9  v1-0009 difficulty  Extra inputs are not permitted; the row has exactly six fields
warning: tests/fixtures/eval_bad.jsonl  fact  5 rows, short of the planned 50
tests/fixtures/eval_bad.jsonl: FAILED -- 9 error(s), 5 warning(s)
```

(Four of the nine errors and four of the five warnings are elided above; the run prints every one.)
That fixture holds **one problem per row**, so "prints every violation rather than the first" is a
count in `tests/test_eval_validate.py` and not a judgement call.

**The corpus it checks against.** `data/chunks.jsonl` when it has been built, and otherwise the
same 4,502 ids derived from the committed `data/courses.json` through `ingest.chunk.doc_id` — the
function that minted them in the first place, so the two sets are equal by construction and a test
says so. The chunk file is gitignored and rebuildable, and a check that skips on a fresh clone is a
check that does not run on the machine it is for.

`eval/datasets/day2_queries.jsonl` is **not** retrofitted into this shape. It is
`{query, expected_intent, expected_exit, note}`, it exists to prove the wiring, and a test asserts
that it still fails this validator — two artifacts with two jobs, not one artifact with a migration.

## Annotation guidelines

`eval/GUIDELINES.md` is the annotation spec, and it is written **before** the first question is
drafted ([#55](../../issues/55)): the drafting prompt ([#56](../../issues/56)) is derived from it,
the verification pass ([#59](../../issues/59)) judges against it, and the second annotator
([#60](../../issues/60)) is handed that file and nothing else — an agreement number computed
against a spec written afterwards measures nothing.

One page, settling four things a second person has to be able to apply without asking: what the
knowledge base is (`data/chunks.jsonl` at term `2026-sp`, the prereq graph, the Gen Ed set, and
nothing else), when a `gold_answer` is correct, when a refusal is correct, and where the bucket
boundaries run. Two rules carry most of the disagreements:

- **A refusal is correct when the knowledge base does not hold the answer, not when retrieval
  missed it.** That gap is the thing Day 4 measures, so relabelling a hard-to-retrieve question
  `unanswerable` raises the score by deleting the measurement. Labels are read off the chunk —
  never off a run of the agent, which is the system under test.
- **`fact` vs `multihop` counts records, not reasoning steps.** CS 374's own chunk prints *"One of
  CS 173, MATH 213; CS 225."*, so its prerequisite question is one hop however many codes that
  sentence names; ECE 374's chunk is *"Same as CS 374. See CS 374."* with `Prerequisites: None
  listed`, so the same question there is two records and a one-id row would certify *"none"*.

Eight boundary cases, two per bucket, carry the rest. `tests/test_eval_guidelines.py` reads the
document back against the corpus so its examples cannot rot: every gold id it cites resolves,
every course it names is in the catalog — except CS 428, whose absence *is* the term-scope rule,
and which the test asserts is still absent.

## Drafting harness

`eval/scripts/draft_items.py` is the machine half of the eval set: it feeds a model **ten real
course records at a time** and asks for questions in one bucket, with an answer and the ids that
support it ([#56](../../issues/56)). It writes **drafts, never dataset rows** — everything lands
in `eval/datasets/drafts/`, and only the verification pass ([#59](../../issues/59)) promotes
anything into `v1.jsonl`.

```bash
uv run --env-file .env python -m eval.scripts.draft_items --bucket fact --batches 20
uv run python -m eval.scripts.draft_items --bucket fact --batches 3 --fake-model   # no credits
```

```
config hash c73b197d  bucket fact
batches                3 run, 0 already done, 0 failed  (plan 1-3, 10 records each)
drafted by             stand-in, prompt v1
kept                   9 this run, 9 in the file (v1-0001 .. v1-0009)
dropped                0
overlap                n 9  min 0.049  p50 0.066  p90 0.098  max 0.098   flagged 1 (>= 0.098)
credit questions       9  (GUIDELINES rule 2 deletes the ones about a variable-credit course …)
wrote                  eval/datasets/drafts/fact_c73b197d.standin.jsonl, sidecar … .meta.json
```

A draft row is a dataset row plus `provenance`, and the extra field is what makes it
re-generatable and sliceable:

```json
{
  "id": "v1-0001", "bucket": "fact",
  "question": "how many hours is accy 290?",
  "gold_answer": "ACCY 290 is 3 credit hours.",
  "gold_doc_ids": ["2026-sp:ACCY-290"], "should_abstain": false,
  "provenance": {
    "batch": 1, "batch_doc_ids": ["2026-sp:ACCY-290", "2026-sp:EDUC-499", "…eight more"],
    "cited_doc_ids": ["2026-sp:ACCY-290"], "subjects": ["ACCY"], "levels": [200],
    "model": "gpt-4o-mini", "prompt_version": "v1", "config_hash": "c73b197d", "seed": 42,
    "overlap": 0.0656, "flagged": false, "credit_question": true
  }
}
```

**The sample is seeded, stratified and a prefix.** The order is a function of the corpus and
`seed` alone — shuffled inside each subject with a private `random.Random(cfg["seed"])`, then
dealt one course per subject, round-robin — so `--batches 3` and a later `--batches 5` agree on
batches 1–3 and the resume re-rolls nothing. 4,502 records make 450 batches of ten, each of them
ten *different* subjects: MUSC's 123 courses get no more of the eval set than a department with
three.

**Resumable, and the sidecar is why.** `fact_{hash}.meta.json` records which batches ran, what
each kept and what it dropped. A file of rows cannot say that batch 7 ran and produced nothing,
and that is exactly the batch a resume must not pay for twice. Rows are appended before the
sidecar is updated, so a crash between them costs a re-draft whose rows are then dropped as
duplicates — never a row that was paid for.

**The gold ids are constrained to the batch, and checked anyway.** The prompt hands the model the
ten `doc_id`s; a row citing anything else is **dropped with a counted reason, never repaired** —
separately for an id that is not in the corpus at all and one that is real but was not among the
ten. Shape and the bucket's rules go through the same `eval/schema.py` and `eval.validate.check_row`
that #59's output is held to, duplicates go through `normalize_question`, and the file is
re-validated from disk when the run ends.

**The hazard it measures.** A question drafted from a chunk reuses that chunk's rare vocabulary,
BM25 finds it at rank 1, and the set reports retrieval recall the system does not have. So the
prompt forbids copying distinctive phrases and asks for the question a student would type; every
`fact`/`multihop` row records the Jaccard overlap between its question and its gold chunk over
`ingest/tokenize.py`'s own tokenizer; and the top decile is marked `flagged` for rewriting during
verification. The absolute numbers are small — a six-term question against a 110-term record
cannot be otherwise — so what is acted on is the distribution and its top decile, not a threshold.

**One limitation cannot be fixed here.** The drafter and the agent under test are the same model
family, and shared blind spots inflate every number downstream; no human verification detects that
a question is one this family finds easy. `--model` drafts with a different model where one is
available **without touching `config.yaml`**, every row records who wrote it, and the rest is a
limitation for #62 to report.

**The run that drafted them** (2026-09-23, once the account had credits). Both buckets were
drafted in two halves under two models, which is the mitigation above taken rather than noted:

| bucket | `gpt-4o-mini` | `gpt-4.1-mini` | total | batches | dropped |
| --- | --- | --- | --- | --- | --- |
| `fact` | 30 | 24 | 54 | 18 | 0 |
| `ambiguous` | 17 | 31 | 48 | 40 | 166 |

Two things came out of it that the plan did not predict. **`ambiguous` is expensive to draft and
the two models are not close.** The bucket needs a question ambiguous between *two* records, and
a batch is ten courses drawn at random, so most batches contain no ambiguous pair at all: 30
batches under `gpt-4o-mini` at six questions each kept 17 rows, a 9% keep rate, every drop for
the same reason — one cited record, which is a `fact` question. Ten batches under `gpt-4.1-mini`
kept 31, a **52%** keep rate on the identical prompt and the same kind of batch. That is a
five-fold difference in following one instruction, and it is the clearest evidence in this repo
for why the drafting model belongs in provenance.

**So `ambiguous` is over-drafted: 48 candidates for 20 slots.** They are kept rather than
trimmed. A draft is a candidate and the verification pass is where rows are chosen; 2.4
candidates per slot in the bucket whose boundary is a judgement call (**A2** against **U1**) is a
better set than 20 rows with no alternative, and the cost is the reviewer's, who deletes the
surplus with a reason like any other row. `fact` is 54 for 50 for the same reason.

**The `fact` bucket skews to credit questions.** 23 of 54 rows ask how many credits a course is
— the failure mode `DEFAULT_PER_BATCH` exists to limit, still visible at three questions per
batch. Every one is flagged (`provenance.credit_question`), `GUIDELINES.md` rule 2 deletes the
ones about a variable-credit course, and thinning the rest is a judgement for the pass: a `fact`
bucket that is half one question shape measures one lookup and calls it a bucket.

`--fake-model` runs the whole harness with `StandInDrafter` and no network: real sampling, real
parse, real checks, canned questions. Those rows go to a separate `.standin.jsonl` file that
nothing downstream reads — a canned row in `fact_{hash}.jsonl` would be indistinguishable from a
drafted one by the time anybody looked.

## Unanswerable bucket

The other three buckets are drafted from documents; this one cannot be
([#57](../../issues/57)). A model shown ten course records writes questions those records answer,
which is the opposite of the requirement, so the 25 rows are a **table in
`eval/scripts/build_unanswerable.py`** and the script's job is to check that table against the
corpus, score every row, and write it out in the same draft shape #56 produces.

```bash
uv run python -m eval.scripts.build_unanswerable
uv run python -m eval.scripts.build_unanswerable --no-retrieval   # no BM25 index needed
```

**Four failure shapes, and a different falsifiable claim for each.** A row that does not support
its shape's claim stops the run — nothing is written, since the table is in the repository either
way.

| shape | rows | what every row's codes must be |
|---|---|---|
| `invented` | 6 | absent from the export **and named by no description in it** — a code the catalog talks about is a real course. May name no code at all (an invented *subject*), and then its question may name none either |
| `out_of_term` | 7 | absent, plus `named_by`: a course that **is** offered whose description names it |
| `field_absent` | 6 | present — and `field` outside both the chunk template and `CourseDetail` |
| `not_course` | 6 | none |

**`out_of_term` is checkable rather than remembered.** #57 records that every prereq code in
`data/prereqs.json` resolves inside the catalog, so "a prereq pointing at a missing course" is not
an available shape — but that is true because the exporter *filters* the flattened list. CS 225's
raw sentence names MATH 347 and CS 126 and its `prereqs` array lists neither; CS 527, whose
prerequisites are only absent courses, has no row in `prereqs.json` at all. The raw text keeps
what the list drops, so scanning descriptions finds **636 codes the 2026-sp catalog names and does
not offer**, and seven of them are this bucket. The corpus is then its own evidence that the
missing course is real, and no row depends on what anybody remembers about UIUC.

**Scoring labels nothing.** `eval/GUIDELINES.md` forbids running `retrieve()` to decide a label —
what the system does with a question is the thing under test and cannot also be the thing that
labels it — so every shape above is decided by reading the export. The retrieval pass runs
afterwards, changes no row, and answers a different question: which rows the evidence gate would
refuse on the score alone.

```
retrieval            bm25 top-1, gate 15.0 (evidence.thresholds)
top_score            min 7.4375  max 39.9679  above the gate 17 of 25

Above the gate -- the gate would not refuse these, so a Day 4 refusal here is the model's and not
the evidence check's:
  v1-2005  invented      19.4353  2026-sp:PHIL-100   (ZZZ 100 alone 6.5751)
  v1-2007  out_of_term   29.1631  2026-sp:CS-527     (CS 428 alone 27.1554)
  v1-2010  out_of_term   31.0221  2026-sp:BADM-205   (GGIS 205 alone 31.0221)
  v1-2014  field_absent  25.4301  2026-sp:CS-225     (CS 225 alone 25.4301)
  …
```

**17 of 25 clear the gate, and `code_scores` says why.** Each declared code is scored *alone*
beside the question it sits in. Every bare invented code is refused (6.32–14.49, all under 15) and
three of the five invented rows that name one clear the gate anyway once the code is inside an
ordinary student sentence (15.03–20.76): the rest of the sentence matches real documents. So a `bm25: 15.0` gate separates
an invented course from a real one only when the query is the bare code, which no student types.
Two shapes cannot be gated at all, by construction — a `field_absent` row names a course that **is**
in the corpus, and a real out-of-term course is one the corpus talks about (`CS 428` alone scores
27.16 on CS 527, whose prerequisite sentence names it). For those the refusal has to come from the
answer node's grounding, which is the thing Day 5's abstention machinery has to calibrate.

The sharpest row is `v1-2010`, *"what is GGIS 205 about?"*: BADM 205's whole description is
*"Same as GGIS 205. See GGIS 205."*, so the retriever returns a document that names the course, is
about the course, and carries none of its content — the one cross-reference of 497 that does not
resolve inside the export.

A constructed row records a **different** provenance from a drafted one — a shape, a witness and a
gate reading where #56's records a batch, a model and a leakage overlap. `parse_drafts` takes the
key set to hold a file to, so both files go through one reader; #59 sees two shapes, which is the
honest thing for it to see.

```json
{
  "id": "v1-2007", "bucket": "unanswerable",
  "question": "how many credits is CS 428?",
  "gold_answer": "This catalog covers 2026 spring only, and CS 428 is not offered in it.",
  "gold_doc_ids": [], "should_abstain": true,
  "provenance": {
    "shape": "out_of_term", "source": "constructed", "codes": ["CS 428"],
    "named_by": "CS 527", "field": null, "note": "GUIDELINES U1, and PR #51's example. …",
    "config_hash": "c73b197d", "retrieval_mode": "bm25",
    "top_score": 29.1631, "top_doc_id": "2026-sp:CS-527",
    "code_scores": {"CS 428": 27.1554}, "gate": 15.0, "above_gate": true
  }
}
```

Three rows are not new: *"how many credits is CS 428?"* is `eval/GUIDELINES.md`'s boundary case
**U1**, *"what should I take next semester?"* is its **A2**, and *"is there a course about quantum
origami?"* is the negative query PR #51 measured at 11.06. A boundary case that is argued in the
guidelines and absent from the data is a rule nothing tests.

## Multi-hop bucket

This bucket comes off the prereq graph rather than out of a drafting prompt
([#58](../../issues/58)): a model shown ten course records writes one-hop questions and labels
them `multihop`, and a bucket full of those measures nothing. So the 25 rows are a table in
`eval/scripts/build_multihop.py`, built against `data/prereqs.json`, and the script checks it
against the committed graph and the catalog's own sentences. [#56](../../issues/56) still
drafts `multihop` candidates; this writes `multihop_graph_{hash}.jsonl` so the two never share a file.

```bash
uv run python -m eval.scripts.build_multihop      # no index, no model, no clock
```

**Five question shapes, so the bucket is not 25 rewordings of "what do I need before X".** Each
walks the same graph differently, and `unlocks` walks it backwards — the shape a system that only
implements `get_prereq_chain` has nothing to answer with.

| shape | rows | what a row declares |
|---|---|---|
| `chain` | 6 | the bounded chain: what X needs, and what those need |
| `reachable` | 6 | a `target`: is Y required for X, directly or indirectly |
| `eligibility` | 5 | `held`: the courses the question grants the student |
| `unlocks` | 4 | the dependents, which must list the root as a prerequisite |
| `distance` | 4 | how many courses stand between the student and X |

**Depth is two measures and they disagree.** `get_prereq_chain` layers by *shortest* distance,
which is what #38's answer node needs; #58's census — 738 courses at depth ≥ 2, deepest chain 13 —
is *longest* chain. Both are computed and recorded per row, because by levels the same graph has
**718** at depth ≥ 2 and its deepest is **9**. Every root clears 2 on **both**, so no row's
multi-hop-ness depends on which measure you picked; the stratification uses the longest-chain
depth, which is the one #58 sampled from.

```
depth                2:4  3:5  4:1  6:1  7:1  8:2  9:3  10:1  11:5  12:1  13:1   (longest chain; 15 at 5+)
cycles               6 rows name a course inside one of the 7
```

**Four records is the ceiling, and the question is what bounds the chain.** #58 wants
`gold_doc_ids` to list every course in the chain; `eval/GUIDELINES.md` caps a join at four records
and deletes an unbounded one outright. Those meet where the question does the bounding — not
*"everything before SE 495"* (27 courses, and a cycle) but *"is MATH 257 required for SE 495?"*,
whose answer rests on three records while its root still sits 13 levels deep. Six rows name a
course inside one of the seven cycles, which is where a walk without a visited set stops
returning.

**The tool proposes, the raw sentence disposes.** Writing these gold answers out of
`get_prereq_chain()` would make the set a regression test for the walker: if the walk is wrong the
gold is wrong the same way and Day 4 scores 100%. So every answer is written from the `prereqRaw`
sentences of the courses it cites, and the script re-derives where the exported `prereqs` list and
that sentence disagree. It is a **log, not a gate** — a row that has one must carry a `note`
saying which kind it is, and the run still writes.

```
tool vs raw          15 of 25 rows agree

Where the exported prerequisites and the course's own sentence differ -- the gold answers below
were written from the sentence:
  v1-1003  ATMS 202   drops MATH 221 (not in this term)
  v1-1009  MATH 412   drops MATH 347 (not in this term)
  v1-1010  AE 312     drops ME 200 (offered this term)
  v1-1018  ASTR 405   drops PHYS 213, PHYS 214 (offered this term)
  v1-1023  JOUR 340   drops JOUR 317, JOUR 217 (not in this term)
  …
```

**Ten of 25 rows sit on a sentence the export does not reproduce, and the four kinds are not the
same finding.** Across the whole graph, 46 courses drop a code their own sentence names while the
catalog still offers it, and the bucket carries one row of each kind deliberately:

- **a requirement dropped** — `AE 312` reads *"AE 202 and MATH 285. Credit or concurrent
  registration in ME 200."* and exports two codes. ME 200 **is** offered this term, so this is not
  the term filter: a continuation sentence naming a requirement was lost. Filed against
  `ingest/export_almabot.py`.
- **a recommendation dropped, correctly** — `ASTR 405` ends *"Recommended: PHYS 213 and PHYS 214"*
  and exports neither. The scanner cannot tell that from the case above; the row's `note` is where
  the reviewer says so, which is why the log is not a bug list.
- **a negation read as a mention** — `JOUR 340` names `JOUR 217` only to say it may **not** be
  taken concurrently.
- **the term filter turning a disjunction into a requirement** — `ATMS 202` requires *"MATH 220 or
  MATH 221."* and exports `["MATH 220"]`, because MATH 221 is not in 2026 spring. This is #58's
  own `ABE 228` case, and it is the commonest of the four.

One more that no scan finds and a reviewer does: `MATH 220`'s sentence asks for *an adequate ALEKS
placement score … demonstrating knowledge of topics of MATH 115*, and the export made that an edge
to MATH 115. The code is named, so the two "agree" — and the requirement is a placement score, not
a course. `v1-1003` is written against the sentence and says so.

A graph-built row records a **third** provenance shape: where in the graph the root sits, what the
tool would have said, and whether it agreed with the text. `parse_drafts` takes the key set to hold
a file to, so #56's, #57's and this one all go through one reader.

```json
{
  "id": "v1-1010", "bucket": "multihop",
  "question": "is MATH 285 required for AE 434, directly or indirectly?",
  "gold_answer": "Yes. AE 434 requires AE 312, and AE 312 requires AE 202 and MATH 285, plus credit or concurrent registration in ME 200.",
  "gold_doc_ids": ["2026-sp:AE-434", "2026-sp:AE-312", "2026-sp:MATH-285"],
  "should_abstain": false,
  "provenance": {
    "shape": "reachable", "source": "prereq-graph", "root": "AE 434", "target": "MATH 285",
    "held": [], "depth": 11, "levels": 7, "closure": 14, "cycle_members": [],
    "target_in_closure": true, "raw_agrees": false,
    "disagreements": [{"course": "AE 312", "dropped_offered": ["ME 200"],
                       "dropped_absent": [], "added": []}],
    "note": "the yes case, and the row carrying the clearest parser bug in the graph. …",
    "config_hash": "c73b197d"
  }
}
```

`target_in_closure` is the tool's proposal, recorded and not acted on: `MATH 213` is reachable
from `CS 374` and the answer is still **no**, because it is one of two alternatives. The same gap
runs through the other shapes — `MATH 241` is in `MATH 412`'s closure only because something else
in there requires it, so *"I have taken MATH 241, can I take MATH 412?"* is no as well. Membership
is not requirement, and that is most of what this bucket tests.

## Verification pass

The 3–4 hours that make this a **hand-verified** eval set rather than a pile of model output
([#59](../../issues/59)). Everything a machine can decide has already run — #54's schema and
validator, #56's id and overlap checks, #57's and #58's corpus checks — so what is left is the
three questions no check can answer, and `eval/scripts/review.py` exists to spend the reviewer's
minute on those and nothing else.

```bash
uv run python -m eval.scripts.review                 # the pass, resumable, one row at a time
uv run python -m eval.scripts.review --limit 20      # a sitting
uv run python -m eval.scripts.review --stats         # per-bucket counts and the top edit reasons
uv run python -m eval.scripts.review --promote       # accepted + edited -> eval/datasets/v1.jsonl
```

**The chunk text is on the screen, or check 2 does not happen.** Every cited chunk is printed in
full beside the question and the drafted answer, out of `data/chunks.jsonl` (or re-rendered from
the committed `data/courses.json` on a machine that has not built it). A reviewer who has to open
another window to read a citation approves citations by vibe at item 40, and a row whose answer is
right and whose citation is wrong is worse than a wrong row: it rewards a system that retrieved
the wrong document.

```
------------------------------------------------------------------------------------------------
v1-1001  multihop -- two or more records must be joined
eval/datasets/drafts/multihop_graph_c73b197d.jsonl:1   row 1 of 50

Q  what do I need before FR 104, and what do those courses require in turn?
A  FR 104 requires FR 103, FR 103 requires FR 102, and FR 102 requires FR 101 -- and each of the
   three sentences also accepts a placement score or the matching year of high school French …
   should_abstain: false   gold_doc_ids: 4

   1. is the gold_answer right?  against the catalog, not against plausibility
   2. do the gold_doc_ids support it?  the cited chunk must contain the fact asserted
   3. is the bucket right?  eval/GUIDELINES.md, and its eight boundary cases

cited chunks
  [1] 2026-sp:FR-104
      FR 104 — Intermediate French II
      Credits: 4
      …
      Prerequisites: FR 103 or equivalent, or a placement score showing high school achievement
      equivalent to FR 103.

   the answer names FR 104, FR 103, FR 102, FR 101

verdict?  [a]ccept  [e]dit  [d]elete  [q]uit >
```

**The flags are prompts, never verdicts.** A row that trips something in `eval/GUIDELINES.md` is
marked with the rule and nothing else — a flag that resolved itself would be the machine judging
the row, which is the thing this pass exists to not do.

| flag | the rule behind it |
|---|---|
| a gold id the validator rejects | #54 — shown here while the reviewer can still fix it, rather than at promotion |
| more than four cited records | "Which bucket": past four it is a list question, so narrow it or drop it |
| a credit question | rule 2 — 1,415 courses have a credit *range* and a single number in the chunk. This stack's export cannot say which, so the flag names the course and prints the one-liner over `origin/main`; once [#17](../../issues/17) lands it names the course outright |
| #56's top-decile overlap | the question may be retrievable by its own wording, which inflates Day 4's recall |
| a code in the answer that no cited chunk carries | check 2, mechanically: a citation about the right topic is still the wrong id |

**`v1.jsonl` is the drafts replayed through the log, not a second copy of them.** Every judgement
appends to `eval/datasets/review_log.jsonl` — `{id, verdict, reason, field_changed, before, after,
ts}`, one line per changed field, flushed before the next row is shown — and `--promote` rebuilds
the dataset from the draft files and that log. Editing rows into `v1.jsonl` directly would give
two files that can disagree, and the one that would be wrong is the one documenting the work. An
edit records the draft value it replaced, so a draft that moved after it was reviewed stops the
promotion instead of quietly promoting text nobody read.

**Nothing reaches the dataset unreviewed.** `--promote` refuses while any drafted row has no
verdict, refuses on a verdict naming a row no draft holds, and re-validates the promoted rows
through `eval.validate` — the composition of a draft file and a log is what Day 4 reads, and a row
is only as checked as the last thing that touched it. A short bucket is a **warning**: deleting a
bad item is what verification is for, and a check that fails on 49 facts argues for keeping a bad
row.

**Resumable by id, and interleaved.** A row with a verdict is not shown again unless `--redo` asks
for it, and then it is shown with its edits applied; the queue deals one row per bucket in turn so
that fifty `fact` rows in a row cannot be what stops the reviewer reading the chunk text. The log
is the one artifact here that carries a clock — every other one moves only when the run moved,
because it is regenerable, and a record of when a person judged what is not. `--stats` prints the
accept rate over the first half of the judgements against the second, which is what a
late-session cluster of accepts looks like.

**Status: the tool, not the pass.** The pass itself is a person's to run, and nothing in this
repository fabricates one — there is no `review_log.jsonl` and no `v1.jsonl` until somebody has
read the rows. All four buckets are now drafted — 152 candidate rows — so the pass is runnable
end to end; what it needs is a person, and nothing here writes a verdict on their behalf.
`tests/test_review.py` holds both artifacts the moment they land, including the invariant that
`v1.jsonl` is exactly the drafts replayed through the log.

## Model-assisted triage

`eval/scripts/triage.py` shortens the verification pass without replacing it. One model call per
drafted row answers the same questions the reviewer answers — against `eval/GUIDELINES.md` and
the cited chunk text — and `review --triage` shows the result under each row while a person still
types the verdict.

```bash
uv run --env-file .env python -m eval.scripts.triage      # read every unread row
uv run python -m eval.scripts.review --triage             # the pass, with the suggestions shown
```

**It is not the review log, and no reader can confuse the two.** The suggestions go to
`eval/datasets/triage.jsonl`, whose rows carry neither a `verdict` nor a `ts`; `read_triage`
refuses a file with either, `review.read_entries` refuses a triage row by name on the first line,
and `--promote` cannot see this file at all. A machine's opinion about a row is not evidence that
the row was checked, and the only file that is has to stay the one a person wrote into.

**The suggestion is derived, not asked for.** The model answers four questions — is the answer
right, do the citations support it, which bucket is this, and does the row belong in the set at
all — and `suggest()` maps those to accept / edit / delete. The first three are #59's checks and
#60's fields in their own words, so a triage answer and an annotator's are comparable values
rather than two vocabularies; the fourth exists because *"this row should be deleted"* is a
verdict the other three cannot express (a row can be correct in every way and still be one
`GUIDELINES.md` rule 2 deletes). Asking the guidelines' own questions and computing the verdict
means the mapping is a function a test pins with no model in the loop, and a reviewer who
disagrees can see **which** answer they disagree with.

**A third model.** The rows were drafted by `gpt-4o-mini` and `gpt-4.1-mini`, so the default here
is `gpt-4.1` — neither of them. A drafting model asked to check its own question is being asked
whether it made a mistake. `--model` overrides it, every row records who read it, and none of
this touches `config.yaml`.

**It anchors the reviewer, and that is the price.** A person shown `accept` accepts more often
than one shown nothing. Three things follow, all deliberate: the suggestion is rendered **last**,
after the three checks and after every cited record, so the reading happens before the opinion
arrives; `--triage` is opt-in, and the pass without it is the pass #59 shipped; and
[#62](../../issues/62)'s report **discloses** the assistance whenever the file exists — which
model, against which guidelines digest — and measures it, by printing how often the reviewer
agreed with the suggestion as a 3×3 table. That rate is an upper bound on the influence rather
than a measurement of it: what an unassisted reader would have decided is not recoverable
afterwards, and the off-diagonal cells are the only direct evidence in the report that the
reading happened.

## Sampled re-read

`eval/scripts/recheck.py` exists because the review log cannot tell a row that was **read** from
a row that was **confirmed** — an accept is an accept whether it took two seconds or two minutes,
and that distinction is the whole difference between a human-verified set and a machine-verified
one. So a stratified sample of already-judged rows is read again properly, and the second verdict
is compared against the first.

```bash
uv run python -m eval.scripts.recheck            # draw the sample, print the re-read command
uv run python -m eval.scripts.recheck --result   # what the re-read overturned
```

**It needs no new file.** #59's log is append-only and replays latest-timestamp-wins, so a row
judged twice carries both verdicts and `overturns()` reads the pair straight off it. That is the
property which makes this measurable at all, and the reason the review log carries a clock when
nothing else in the repo does.

**Stratified over bucket *and verdict*, with a floor of one row per cell.** The verdict matters
more than the bucket: an accept that should have been a delete leaves a wrong row for Day 4 to
score against, while a delete that should have been an accept throws a row away and **nothing
downstream will ever say so**. Deletes are the smaller group and the less recoverable error, so
every delete a *model* recommended is taken whole, and the rule 2 deletes are sampled like
anything else since those came from a lookup rather than a judgement. The floor of one is what
keeps a one-row cell — `(multihop, delete)` holds exactly one — from being rounded out of the
sample by a proportional share.

**What comes out is an interval, not a verdict on the pass.** Nothing overturned at n=30 bounds
the pass's error rate at **11.4%**; that is worth stating and is not the same claim as *the pass
was right*. #62 puts the rate and its Wilson interval in the Verification section, in the
limitations, and in the **headline** — and where the pass was model-assisted the headline stops
saying "reviewed by hand" and says "judged", with what the reviewer actually did in the sentence
after it. A headline claiming "by hand" over a log of confirmations would be the one sentence in
that report a reader could catch it out on.

`review --redo` takes ids for this, so a thirty-row re-read does not walk the whole queue again;
the earlier verdict is printed above each row, and a different one replaces it.

## Second annotator

Fifteen rows, one other person, the same `eval/GUIDELINES.md` and no conversation
([#60](../../issues/60)). This is what separates "I wrote an eval set" from "I built a
human-annotated eval set": a single-annotator dataset has no measurable reliability, and the first
reviewer who asks *"how do you know your labels are any good?"* has no answer without it.

```bash
uv run python -m eval.scripts.agreement           # draw the sample, write the packet
uv run python -m eval.scripts.agreement --score   # the returned sheet -> report/day3_agreement_{hash}.md
```

The export writes a handoff packet beside the dataset: `agreement_sample.jsonl` (the fifteen rows
with the first annotator's labels — the answer key), `agreement_sample.meta.json` (the seed, the
allocation and what was drawn), `agreement_sheet.tsv` (blind), `agreement_citations.md` (pass two),
`agreement_instructions.md`, and `agreement_questions.md`.

**Stratified 6 / 3 / 3 / 3, seeded per bucket.** Fifteen rows drawn uniformly from 120 miss
`ambiguous` entirely 5.4% of the time and bring back fewer than three of it 52.6% of the time, and
`ambiguous` is the bucket whose boundary is a judgement call (**A2** against **U1**) rather than a
lookup. Each bucket draws from its own stream over its sorted ids, so deleting a `fact` row during
verification does not re-draw the other three buckets; the fifteen are then shuffled once, because
a sheet whose first six rows are all `fact` hands over the stratification.

**Blind, and two labels leak anyway.** `bucket` and `should_abstain` are stripped — showing them
turns the exercise into a review of the first annotator's work and pushes agreement toward 100% —
but the sheet has to show the candidate answer, and two things follow that no column-stripping
fixes. A refusal is visibly a refusal, so the abstain/non-abstain split is given away and what the
bucket field measures is the two *within-pair* calls, `fact` against `multihop` and `unanswerable`
against `ambiguous`, which are the two the guidelines spend their boundary cases on. And the
citation count nearly determines the bucket — `unanswerable` carries exactly zero gold ids,
`multihop` at least two — so the cited records are held back for **pass two**, opened only after
`bucket` and `answer_correct` are filled in and saved. The report says both, next to the numbers.

**Percent agreement *and* Cohen's κ, per field.** With 50 of 120 rows in `fact`, two annotators who
both guessed `fact` every time would agree about 42% of the time having done nothing, so raw
percent agreement on four skewed buckets is inflated by chance and the gap between the two numbers
is the informative part. Reported honestly at n=15: the interval is a Wilson score interval from
the counts, not a flat ±25 points — 14 of 15 is 93.3% with an interval of **[70%, 99%]**, 23 points
of room below the estimate. κ gets no interval, because its asymptotic standard error is not
trustworthy at fifteen items.

**Two of the three fields break κ, in different ways, and the report names which.** `v1.jsonl`
holds only rows the first annotator accepted or edited, so the fields being scored are the fields
the set was filtered on. Their `answer_correct` is a constant `yes`, which forces `p_e = p_o` and
so κ **exactly** 0 however well the two agree. Their `citations_support` is not constant but it is
mechanical — `n/a` exactly where the row cites nothing, which the second annotator reads off the
same empty list — so those rows agree by construction and carry the field-level number up with
them; the table therefore also prints `citations_support` over the cited rows only, and on that
subset annotator 1 is a constant `yes` again. What percent agreement estimates on both fields is
**residual label error in the frozen set** — how often a second reader would reject a row Day 4 is
about to score — not inter-annotator reliability. A reliability κ there needs the sample drawn from
`datasets/drafts/` before promotion, where the first annotator's verdicts vary.

**Disagreements are the useful output.** Each one is itemised with both labels, the guideline rule
the second annotator says they applied (the sheet has a `rule` column for exactly this) and their
note; a disagreement with no rule attributed is marked as outstanding rather than passed over,
since one that cannot be traced to a rule cannot be fixed by editing the guidelines. Questions
asked during annotation go in `agreement_questions.md` — written once and never overwritten — and
are answered by amending `GUIDELINES.md` *afterwards*, with [#62](../../issues/62) recording that
the amendment came after the measurement.

**Status: the tool, not the measurement.** The sample cannot be drawn yet — `eval/datasets/v1.jsonl`
is #59's `--promote` output and does not exist, because no row has been verified yet — so this branch commits no sample, no sheet and no report, and the
export says which work has to happen first rather than writing an empty packet. Nothing here
fabricates a second annotator's judgements or a number computed from them.
`tests/test_agreement.py` drives both commands end to end over synthetic datasets in `tmp_path`,
asserts the arithmetic (Wilson's bounds, κ's two degeneracies, the confusion matrix) against
hand-checkable counts, and holds the real packet the moment the dataset lands.

## Frozen split

`eval/scripts/split.py` ([#61](../../issues/61)) cuts the eval set in half — a calibration half
and a test half — and freezes the result into the dataset. Day 5's conformal prediction
calibrates on `calib`; Day 4 reports every final number on `test`. A threshold tuned on the rows
it is later scored against produces a coverage figure that describes the tuning, so the two
halves have to be fixed before anything is measured.

```bash
uv run python -m eval.scripts.split                # freeze eval/datasets/v1.jsonl
uv run python -m eval.scripts.split --dry-run      # the table, written nowhere
uv run python -m eval.scripts.split --verify       # re-derive and compare; exits 1 on drift
```

```
eval/datasets/v1.jsonl  120 rows  seed 61  config c73b197d

bucket            n   calib   test
fact             50      25     25
multihop         25      13     12   (odd row -> calib)
unanswerable     25      12     13   (odd row -> test)
ambiguous        20      10     10
total           120      60     60

calibration n=60: coverage 1-alpha is attainable only for alpha >= 1/61 = 0.0164,
so 98.4% is the ceiling and 99% is out of reach at any threshold.
```

**Stratified, not uniform.** 50/50 *within each bucket*. A uniform draw of 60 from 120 puts the
twenty `ambiguous` rows somewhere other than 10/10 about 81% of the time and splits them 6/14 or
worse 8.5% of the time; at twenty rows that decides how much of Day 5's calibration is spent on
the bucket where abstention is hardest. Odd buckets alternate which half takes the extra row, in
`BUCKETS` order, so neither half is systematically larger.

**Reproducible and frozen are different properties, and the tool provides both.** The seed is
`eval.split_seed` in `config.yaml` and the assignment is derived from each bucket's **sorted**
ids — sorted, because a split derived from file order re-splits the dataset the first time
someone sorts the file. The frozen artifact is the `split` field committed on every row: that is
the contract, and the seed is its provenance. `--verify` re-derives from the seed and holds the
committed file to it, `tests/test_split.py` runs it over the committed dataset, and
`eval/datasets/v1.meta.json`
records the file's `sha256` beside the counts, so an edit to the eval set is a reviewable event
rather than an invisible one. `load_split()` refuses a file whose digest has moved.

**The rule, by name — the test half is read once per milestone, not once per experiment.** Day 5
touches `calib` only; every final metric is reported on `test`. The loader makes the wrong thing
hard: `load_split("calib")` / `load_split("test")` takes a required argument, and there is no
`load_all()` for tuning code to reach for.

```python
from eval.scripts.split import load_split

rows = load_split("calib")      # Day 5 calibrates here
rows = load_split("test")       # Day 4 reports here, once it is reporting
```

**What sixty calibration rows cost.** Split conformal takes the `ceil((n+1)(1-α))`-th smallest
calibration score, and that index has to exist — so `α ≥ 1/(n+1)`. At n = 60 that is
α ≥ 1/61 ≈ 0.0164: **98.4% coverage is the ceiling and 99% is unreachable at any threshold**. The
lever is the split ratio and the time to move it is before anything is measured, which is why the
number is here rather than in a Day 5 post-mortem.

**Adding `eval:` to `config.yaml` moved the full hash** from `7980f15f` to `c73b197d`. It reaches
no index key — `chunk_key` `05d00726`, `dense_key` `094143d8` and `bm25_key` `877a6610` are all
where they were — so nothing built was orphaned. The two Day 3 draft files were rebuilt under the
new hash (byte-identical apart from the `config_hash` they stamp); `report/day2_smoke_7980f15f.md`
keeps its name, because it is named after the config that produced it.

**Status: the tool, not the split.** `eval/datasets/v1.jsonl` is #59's `--promote` output and does
not exist — the 152 drafted rows have not been verified — so nothing is
frozen here and no dataset or sidecar is committed. The checks over the committed file are written
and skip until it lands; the table above is a run over a synthetic 120-row set with the target
counts, which is also what `tests/test_split.py` drives.

## Day 3 report

`eval/scripts/day3_report.py` ([#62](../../issues/62)) is Day 3's acceptance in one file:
`report/day3_evalset_{hash}.md`, six sections, every number in them read out of a committed
artifact.

```bash
uv run python -m eval.scripts.day3_report --check     # what exists, and what makes the rest
uv run python -m eval.scripts.day3_report             # write report/day3_evalset_{hash}.md
uv run python -m eval.scripts.day3_report --dry-run   # print it, write nothing
```

**Composition** (the counts as they are, not as they were planned, plus the question-length and
depth distributions, `fact`'s subject coverage and #57's four failure shapes), **provenance**
(which model drafted which rows, under which prompt version, from which seed and how many
batches), **verification** (#59's per-bucket accept / edit / delete counts and the reasons
given), **agreement** (#60's numbers), **split** (#61's seed, digest and conformal ceiling), and
**limitations**.

**The headline sentence is the count, not the claim.** *"N rows drafted, M reviewed by hand, K
edited, J deleted"* is rendered by one function and printed twice — at the top and over the
verification table — so the sentence a reader quotes and the table under it cannot disagree.

**The agreement section is #60's own rendering, one heading level down.** #62 asks for exactly
what `report/day3_agreement_{hash}.md` already prints — the per-field table with its Wilson
interval, the bucket confusion matrix, the itemised disagreements — and a second implementation
of those tables is two reports that can disagree about a number a reader will compare across
them. `demote()` moves the headings and nothing else; a test asserts the demoted string appears
in the report byte for byte.

**Limitations is a section, and it is computed.** The five things wrong with this dataset are
known in advance but their sizes are not, so each is rendered from the run's own figures: what
share of rows a second person actually read, the residual question/chunk overlap distribution
after #56 rewrote its worst decile, and the widest 95% interval a per-bucket accuracy on the test
half can have. A limitation written as fixed prose goes stale the moment the numbers move, which
is worse than none because it reads as though somebody checked.

**Three properties the acceptance names, and how each is held.** There is no clock anywhere in
the body — #59's log carries the timestamps and this prints none of them — so a re-run over
unchanged inputs rewrites the file byte for byte and a diff under `report/` means an artifact
moved. Every path goes through `rel_to_repo()`, so no home directory reaches a tracked file. And
no metric here is computed on the test half: the counting tables read the whole file through
`eval.validate.parse_items`, `load_split` is not imported, and `tests/test_day3_report.py`
asserts over the module source that it stays that way.

**It refuses to write rather than print a section that says "not measured yet".** A file called
`day3_evalset_{hash}.md` is the claim that Day 3 happened, and one with an empty verification
section is that claim made about a dataset nobody read. `--check` is what runs today:

```
report report/day3_evalset_c73b197d.md  config c73b197d

  ok       drafts                 eval/datasets/drafts
  MISSING  review log             eval/datasets/review_log.jsonl
                                  uv run python -m eval.scripts.review
  MISSING  dataset                eval/datasets/v1.jsonl
                                  uv run python -m eval.scripts.review --promote
  MISSING  split sidecar          eval/datasets/v1.meta.json
                                  uv run python -m eval.scripts.split
  MISSING  agreement sample       eval/datasets/agreement_sample.jsonl
                                  uv run python -m eval.scripts.agreement
  ...

2 draft files in eval/datasets/drafts, 50 of 120 rows drafted

6 inputs missing, so the report is not written.
```

**Status: the tool, not the report.** No `report/day3_evalset_*.md` is committed, because the
dataset it reports on does not exist — the 152 drafted rows have not been verified, there is
no second annotator and nothing has been split. Writing the file anyway would
mean inventing every number in it, which is the one thing a report about how an eval set was
built cannot do. `tests/test_day3_report.py` builds the whole of Day 3 in `tmp_path` — four draft
files, a review log with an accept, an edit and a delete in it, the promoted dataset, the freeze
and a filled sheet — and checks the report against it; the last test asserts that the inputs are
still missing here, and says to delete itself when they land.

## Retrieval ablation

`eval/scripts/ablation.py` ([#75](../../issues/75)) runs every row of one half that carries
`gold_doc_ids` (`fact` and `multihop`) through `retrieve(question, mode, k=5)` for `dense`, `bm25`
and `hybrid`, and writes one line per (row, mode) with Recall@5 and MRR@5 from `eval/metrics.py`.
`unanswerable` and `ambiguous` rows are skipped, and the skips are counted by bucket and printed.

```bash
uv run --env-file .env python -m eval.scripts.ablation --split calib   # choose and debug here
uv run --env-file .env python -m eval.scripts.ablation --split test    # the reported number
```

The output goes to `eval/runs/ablation_{split}_{hash}.jsonl` plus a `.meta.json` sidecar that holds
the config hash, `dense_key`, `bm25_key`, the dataset sha256 from the freeze, and the row ids.
**The run file is the artifact.** The bootstrap (#76) and the table (#77) read it and never
re-query an index. `--split` is required. Each half gets its own file because the strongest
single mode is picked on `calib` and tested on `test`. The runner is deterministic, but the
live dense side is not: embedding the same query twice can give slightly different vectors,
and two calib runs differed in 1 of 41 dense lines.

## Paired bootstrap

`eval/scripts/bootstrap.py` ([#76](../../issues/76)) reads the two committed run files and puts a
95% percentile interval on every mean, plus one on hybrid − the strongest single mode. It never
calls `retrieve()`.

```bash
uv run python -m eval.scripts.bootstrap    # -> eval/runs/bootstrap_{full_hash}.json
```

- **Paired.** One draw of B = 10,000 index vectors over the test questions is shared by every
  mode, and the difference is taken per resample from those same indices.
  `paired_bootstrap(scores, B, seed)` is the pure function; the rest is I/O. It runs in well
  under a second.
- **Strongest single is chosen on `calib`, per metric, and tested on `test`.** Choosing it on the
  test half would compare hybrid with the larger of two noisy means. On both metrics it is bm25.
- **A degenerate interval is labelled as one.** A mode that scores 1 on every question has [1, 1],
  which means no variance was seen, not that the mode is certain. With n = 41, Recall@5 on the 29
  `fact` rows moves in steps of 1/29.

Test half, n = 41 (29 fact + 12 multihop), seed 76:

| metric | dense | bm25 | hybrid | hybrid − bm25 |
|---|---|---|---|---|
| Recall@5 | 0.705 [0.571, 0.833] | 0.870 [0.778, 0.951] | 0.866 [0.783, 0.935] | −0.004 [−0.085, 0.073], not significant |
| MRR@5 | 0.637 [0.502, 0.764] | 0.923 [0.850, 0.984] | 0.802 [0.707, 0.891] | −0.120 [−0.225, −0.023], significant |

**The bootstrap settings do not move the config hash.** `eval.bootstrap` is read only after every
other artifact exists, so `config.AFTER_THE_RUNS` leaves it out of `cfg.hash`, which stays
`c73b197d`. Day 3's drafts and reports and the #75 run files keep their names. `cfg.full_hash`
(`8a1980f9`) covers every value, and it names only the bootstrap result.

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

## Day 2 goal

One message in, one of three exits out:

```python
from agent.graph import run
run("What do I need before CS 374?")["exit"]   # 'answer' | 'refuse_handoff' | 'out_of_scope'
```

- Five intents, a dispatch table over three tools and `retrieve()`, a threshold on the evidence,
  and an answer whose citations are verified before anyone reads it
- The same three tools served over MCP — the external surface, not the graph's transport
- Acceptance: ten handwritten queries from the command line, all three exits, one report

Progress is tracked in GitHub Issues (milestone: Day 2).
