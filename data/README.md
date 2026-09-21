# data/

Retrieval corpus, re-exported from the catalog CSV by `ingest/export_almabot.py`. Rebuild with:

```bash
uv run python -m ingest.export_almabot     # reads data.source from config.yaml, prints stats
```

The input is pinned by `data.source_sha256` in `config.yaml` and checked on every run, so these
files and the indexes built from them always belong to the same CSV. After replacing the CSV:
`uv run python config.py --source-digest`, update the pin, re-export.

| File | Source | Status |
|---|---|---|
| `courses.json` | `~/Downloads/course-catalog.csv` (12,823 section rows → 4,502 courses) via `ingest/export_almabot.py` | done — 2.9 MB, term `2026-sp` |
| `prereqs.json` | Same source; `prereqCodes` / `coreqCodes` filtered to courses in the catalog | done — 2,086 prerequisite + 34 corequisite edges |
| `requirements.json` | Gen Ed rules from `almabot-web/src/lib/gened.ts`; major requirements are not in the CSV, source TBD | done — 13 Gen Ed categories, `majors: []` |
| `chunks.jsonl` | `ingest/chunk.py`, one chunk per course, with `doc_id` | done — 4,502 chunks, 3.1 MB, **not committed** (rebuildable, see below) |

Note: `almabot-web/public/courses.json` truncates `desc` to 240 chars and `attrs` to 160, and has no
term field. **Do not copy it over.** Re-export the full text from the CSV.

CSV columns: Year, Term, YearTerm, Subject, Number, Name, Description, Credit Hours, Section Info,
Degree Attributes, Schedule Information, CRN, Section, Status Code, Part of Term, Section Title,
Section Credit Hours, Section Status, Enrollment Status, Type, Type Code, Start Time, End Time,
Days of Week, Room, Building, Instructors. Only one term is present: 2026 Spring.

## courses.json

One record per course, sorted by (subject, number). Course-level CSV columns are identical across
a course's section rows in this export (verified: 0 of 4,502 courses disagree), so the first row of
each code wins.

| Field | Notes |
|---|---|
| `code`, `subject`, `number` | `"CS 374"`, `"CS"`, `"374"` |
| `name` | HTML entities decoded |
| `term` | From `YearTerm`, falling back to `Year` + `Term`; checked against `data.term` |
| `credits` | First number in `Credit Hours`, or the maximum when the first is 0 (`"0 TO 4 hours."` → 4.0) |
| `desc` | **Full** description, untruncated (p50 366 chars, max 2,429) |
| `attrs` | Full `Degree Attributes` |
| `prereqRaw` | The **whole** requirements block: from the `Prerequisite:` label to the end of the description |
| `prereqCodes` | Codes required *before* the course, from the prerequisite statement — **not** filtered to the catalog |
| `coreqCodes` | Codes to be taken *alongside*, from an explicit corequisite sentence |
| `sectionCount`, `partOfTerm`, `types` | Section metadata, aggregated over the course's rows |

## prereqs.json

```json
[{"course": "AE 461", "prereqs": ["AE 321", "AE 352"], "coreqs": ["AE 323", "AE 353"],
  "raw": "AE 321 and AE 352. Credit or concurrent registration in AE 323 and AE 353."}]
```

`prereqs` and `coreqs` are `prereqCodes` / `coreqCodes` filtered to courses that exist in this
catalog — the same filter `dag.ts::prereqs()` applies at read time. Courses with no surviving
reference are omitted entirely.

**Build the prerequisite DAG from `prereqs` only.** Corequisites are mutual often enough
(DANC 498 ↔ DANC 499, MACS 480 ↔ MACS 481) that folding them in would turn correct data into
cycles.

**That is necessary and not sufficient: `prereqs` alone is not acyclic, so every walk needs a
visited set.** Seven strongly-connected components over 19 courses, no self-loops:

| cycle | |
|---|---|
| `THEA 375 → 376 → 377 → 378 →` | 4-cycle |
| `THEA 475 → 476 → 477 → 478 →` | 4-cycle |
| `CS 225 → CS 374 → MATH 412 → CS 225` | 3-cycle |
| `ACCY 301 ↔ ACCY 302`, `CHEM 204 ↔ CHEM 205`, `MCB 581 ↔ MCB 582`, `SE 311 ↔ SE 312` | mutual pairs |

16 of those 19 are corequisites after all, stated *inside* the prerequisite sentence ("SE 310;
concurrent registration in SE 312."), where `coreq_codes` cannot reach them — it only reads a
continuation sentence. The other three are disjunction members: `MATH 412` is one alternative in
`CS 225`'s "One of CS 173, CS 413, MATH 213, MATH 314, MATH 347, MATH 412", and `MATH 412` lists
`CS 374`, which requires `CS 225`.

Walking the graph from each of the 1,135 records: the deepest chain is 9 levels by shortest
distance (`CEE 560`, `ECE 454`, `ECE 575`, `ECE 585`, `ME 547`) and 13 courses along the longest
cycle-free path (`SE 495`); the largest transitive closure is 47 courses (`IS 557`), the median 4;
165 of the 1,135 closures contain one of the cycles above. `agent/tools/prereqs.py` is the walk.

## requirements.json

The 13 campus Gen Ed categories as declarative rules: a course matches when its `attrs` contains
**any** of `match.any` and **none** of `match.none`. `courseCount` is how many courses in this
export match, derived in the same run.

`majors` is an empty placeholder: major and degree requirements are not in `course-catalog.csv`,
which lists sections only. Filling it in needs a separate source (catalog.illinois.edu program
pages, or a department plan). `agent/tools/requirements.py` refuses every program except Gen Ed on
those grounds, and raises if this list ever stops being empty without that tool learning to read it.

572 courses carry a `Degree Attributes` string but only **570 satisfy a category**: `ACE 436` and
`CMN 361` are labelled `Global Classroom course.`, which is a designation and not a Gen Ed. 299 of
the 570 satisfy more than one category (244 once #14 lands), because a course can carry a Gen Ed
area and a Cultural Studies designation at the same time.

## chunks.jsonl

What the indexes embed: one JSON object per line, one line per course.

```bash
uv run python -m ingest.chunk                # -> data/chunks.jsonl + data/chunks.meta.json
uv run python -m ingest.chunk --show CS 374  # print one rendered chunk and exit
```

```json
{"doc_id": "2026-sp:CS-374", "text": "CS 374 — Introduction to Algorithms...",
 "metadata": {"code": "CS 374", "subject": "CS", "number": 374, "level": 300, "credits": 4.0,
              "term": "2026-sp", "prereq_codes": ["CS 173", "CS 225", "MATH 213"],
              "gened_ids": []}}
```

`doc_id` is `{term}:{SUBJ}-{NUM}` — readable in a ranked list, and term-scoped so a second term can
join the same collection without colliding. `text` is rendered by `render_chunk()` alone, under the
template named by `chunk.template_version`; see [the chunking section of the top-level
README](../README.md#chunking). `prereq_codes` is the **unfiltered** list, so a reference to a
course not offered this term stays visible in metadata even though `prereqs.json` drops it.

### Metadata: what each field is good for

Every field below is *displayable* — it comes back on a `Hit` from either retriever. The
**Filter with** column is what Chroma's `where` can do with it once the dense index holds it;
`chroma_metadata()` in `ingest/chunk.py` owns the encoding and `decode_chroma_metadata()` reverses
it, so a BM25 hit (which reads this file) and a dense hit (which reads Chroma) agree.

| Field | In Chroma | Filter with | Notes |
| --- | --- | --- | --- |
| `code` | `str` | `$eq`, `$in`, `$ne` | The display form, and the only one that keeps the catalog's zero padding (`MUS 090`) |
| `subject` | `str` | `$eq`, `$in`, `$ne` | |
| `number` | `int` | `$eq`, `$in`, `$gte`/`$gt`/`$lte`/`$lt` | 90–799 here. An int because Chroma compares numbers, not numeric strings |
| `level` | `int` | `$eq`, `$in`, ranges | `number` rounded down to the hundred, so "400-level" is one clause. `MUS 090` is level `0` |
| `credits` | `float` | `$eq`, `$in`, ranges | |
| `term` | `str` | `$eq`, `$in`, `$ne` | |
| `prereq_codes` | `list[str]`, **absent when empty** | `$contains`, `$not_contains` | The unfiltered list; the prerequisite *graph* is `prereqs.json` |
| `gened_ids` | `list[str]`, **absent when empty** | `$contains`, `$not_contains` | Ids from `requirements.json`, not the raw `Degree Attributes` string |

```python
{"gened_ids": {"$contains": "cultus"}}                       # 101 courses
{"$and": [{"subject": "CS"}, {"level": 400}]}                #  47
{"$and": [{"subject": "CS"}, {"number": {"$gte": 400}}]}     #  70
{"prereq_codes": {"$contains": "CS 225"}}                    #  31
```

Three things about that encoding are worth knowing before writing a filter, all pinned by
`tests/test_chroma_filters.py`:

- **`$contains` on metadata is list membership, not a substring test.** It matches nothing against
  a string value (`{"subject": {"$contains": "CS"}}` → no rows). The substring operator of the same
  name applies to the chunk *text*, and lives under `where_document`.
- **An empty list is not storable**, so the key is left out of the record entirely rather than
  written as `[]` or as an empty string. A course with no Gen Ed therefore never matches
  `$contains` and always matches `$not_contains`, which is what those questions mean anyway.
  `decode_chroma_metadata()` puts the `[]` back, so a `Hit` always carries both list fields.
- **The metadata encoding is not part of any index key.** `chunk_key` covers the chunk *text*
  (term, strategy, template, field list), so re-encoding metadata leaves the name of an existing
  collection unchanged: a collection built before this encoding keeps the old one until it is
  rebuilt with `--rebuild`.

Neither file is committed, unlike the three above: they are rebuilt from those by one command with
no private input and no API key, and a template change rewrites all 3 MB. `chunks.meta.json` records
the term, strategy, template, field list and token stats of the run that wrote `chunks.jsonl`, so an
index build can tell whether the chunks on disk were built under its own config. Compare its
`chunk_key` — a hash of exactly the four values that decide chunk content — and not `config_hash`,
which also moves when `seed`, `bm25` or `fusion` is tuned and the chunks on disk stay valid.
`tokens` is `null` when the run could not fetch the tokenizer vocabulary, which also means those
chunks were never checked against the 8191-token embedding limit.

## How the requirements block is read

The catalog puts the whole enrollment-requirements block after the `Prerequisite:` label, running
to the end of the description. `prereqRaw` keeps all of it; 2,728 courses have text there that the
frontend regex dropped at the first sentence break — including 23 whose block was cut inside an
abbreviation (`"Prerequisite: Ph.D. standing in BADM"` became `"Ph.D"`).

Codes are then read out of that block by sentence, because the sentences say different things:

| Sentence | Becomes |
|---|---|
| The prerequisite statement (first sentence) | `prereqCodes` |
| `Credit or concurrent registration in AE 323 and AE 353.` | `coreqCodes` |
| `Concurrent enrollment in CS 225 is preferred.` | neither — hedged |
| `May not be taken concurrently with ANTH 391.` | neither — negated |
| `Recommended: PHYS 213 and PHYS 214.` | neither — optional |
| `Credit is not given for both AE 461 and ME 461.` | neither — an exclusion |

Reading codes out of the whole block indiscriminately would add 70 edges, of which ~15 are
outright wrong (`CS 277 cannot be taken concurrently with CS 225`, `Credit not given for both
HORT 199 and HORT 101`). The rules above are deliberately conservative: every code the frontend
extracted is still extracted, and nothing is added without an explicit statement.

## Known data quality issues

Counts from the 2026-sp export; `ingest/export_almabot.py` prints them on every run.

- **497 descriptions are cross-references only** (`"Same as HIST 200. See HIST 200."`). They carry
  no topical text, so both retrievers will rank them poorly. Template `v1` renders them as they
  are; following the reference is #15.
- **1 course has an empty description** (`CWL 593`).
- **779 of 2,865 prerequisite references dangle** — a course mentioned in prerequisite text that is
  not offered in 2026-sp. They stay in `courses.json:prereqCodes` and are dropped from
  `prereqs.json`.
- **Only 572 of 4,502 courses carry any `Degree Attributes`**, so Gen Ed coverage is sparse by
  nature, not by parsing loss.
- **`socbeh` over-matches.** The catalog writes both social-science categories under one prefix,
  `Social & Beh Sci - Soc Sci` / `Social & Beh Sci - Beh Sci`, so the `Beh Sci` substring rule
  (inherited verbatim from `gened.ts`) matches all 118 Soc Sci courses on top of the 24 real
  Behavioral Sciences ones — 142 instead of 24. Affects `requirements.json:courseCount` and every
  chunk's `gened_ids`: #14.
- **Prerequisite text is prose, not a grammar.** `prereqCodes` is a bag of codes: "One of CS 173,
  MATH 213; CS 225" loses the "one of" structure, so a planner cannot tell alternatives from
  conjunctions. **744 of the 1,135 requirement statements contain a disjunction** (`or`, `one of`),
  and for **397** of them the code list has more than one entry — `ABE 228` reads "One of MATH 220,
  MATH 221, MATH 234." and exports `["MATH 220", "MATH 234"]`, where the student needs one. **299
  statements** also offer a "consent of …" escape hatch no code list can express. So a consumer
  must carry `raw` and quote it rather than read the list as a conjunction; `agent/tools/prereqs.py`
  puts it on every node of a chain. Parsing that structure into an AND/OR tree is a separate job.
