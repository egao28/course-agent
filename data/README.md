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
| `courses.json` | `~/Downloads/course-catalog.csv` (12,823 section rows → 4,502 courses) via `ingest/export_almabot.py` | done — 3.2 MB, term `2026-sp` |
| `prereqs.json` | Same source; `prereqCodes` / `coreqCodes` filtered to courses in the catalog | done — 2,086 prerequisite + 34 corequisite edges |
| `requirements.json` | Gen Ed rules from `almabot-web/src/lib/gened.ts`; major requirements are not in the CSV, source TBD | done — 13 Gen Ed categories, `majors: []` |
| `chunks.jsonl` | `ingest/chunk.py`, one chunk per course, with `doc_id` | done — 4,502 chunks, 3.2 MB, **not committed** (rebuildable, see below) |

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
| `creditsRaw` | The `Credit Hours` column **verbatim** (`"3 OR 4 hours."`) — the only lossless form |
| `creditsMin`, `creditsMax` | Inclusive bounds parsed from it (3.0 / 4.0), equal for a fixed-credit course |
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

## requirements.json

The 13 campus Gen Ed categories as declarative rules: a course matches when its `attrs` contains
**any** of `match.any` and **none** of `match.none`. `courseCount` is how many courses in this
export match, derived in the same run.

**Sub-category rules match on the `- ` separator**, not on the bare label: the catalog writes a
sub-category as `"<group> - <name>"` (`"Social & Beh Sci - Soc Sci course."`), and the bare labels
`gened.ts` tests are not safe substrings of that shape — `"Beh Sci"` also occurs inside the *group*
name `Social & Beh Sci`, so `socbeh` used to claim all 118 Soc Sci courses on top of its own 24
(#14). Anchoring is what keeps the 13 rules disjoint; it left the other 12 categories' membership
unchanged, and it made the second spelling `gened.ts` ORs into `cultwest` (`"Western course"`)
redundant, since that string only ever occurs as the tail of `"Cultural Studies - Western course"`.
`almabot-web/src/lib/gened.ts` still carries the unanchored rules.

`majors` is an empty placeholder: major and degree requirements are not in `course-catalog.csv`,
which lists sections only. Filling it in needs a separate source (catalog.illinois.edu program
pages, or a department plan).

## chunks.jsonl

What the indexes embed: one JSON object per line, one line per course.

```bash
uv run python -m ingest.chunk                # -> data/chunks.jsonl + data/chunks.meta.json
uv run python -m ingest.chunk --show CS 374  # print one rendered chunk and exit
```

```json
{"doc_id": "2026-sp:CS-374", "text": "CS 374 — Introduction to Algorithms...",
 "metadata": {"code": "CS 374", "subject": "CS", "number": "374", "credits_min": 4.0,
              "credits_max": 4.0, "term": "2026-sp",
              "prereq_codes": ["CS 173", "CS 225", "MATH 213"], "gened_ids": []}}
```

`doc_id` is `{term}:{SUBJ}-{NUM}` — readable in a ranked list, and term-scoped so a second term can
join the same collection without colliding. `text` is rendered by `render_chunk()` alone, under the
template named by `chunk.template_version` (`v1` renders the catalog fields as they stand, `v2`
de-duplicates the requirements block and follows cross-references); see [the chunking section of
the top-level README](../README.md#templates). `prereq_codes` is the **unfiltered** list, so a reference to a
course not offered this term stays visible in metadata even though `prereqs.json` drops it.

Neither file is committed, unlike the three above: they are rebuilt from those by one command with
no private input and no API key, and a template change rewrites all 3 MB. `chunks.meta.json` records
the term, strategy, template, field list and token stats of the run that wrote `chunks.jsonl`, so an
index build can tell whether the chunks on disk were built under its own config. Compare its
`chunk_key` — a hash of exactly the four values that decide chunk content — and not `config_hash`,
which also moves when `seed`, `bm25` or `fusion` is tuned and the chunks on disk stay valid.
`tokens` is `null` when the run could not fetch the tokenizer vocabulary, which also means those
chunks were never checked against the 8191-token embedding limit.

## How credit hours are read

**1,415 of 4,502 courses (31%) are variable-credit** — 834 written `X TO Y`, 581 `X OR Y` — so no
single number describes the catalog. `creditsRaw` keeps the column verbatim and the chunk renders
the whole range:

| `Credit Hours` | `creditsRaw` | min / max | rendered |
|---|---|---|---|
| `3 hours.` | `"3 hours."` | 3.0 / 3.0 | `Credits: 3` |
| `3 OR 4 hours.` | `"3 OR 4 hours."` | 3.0 / 4.0 | `Credits: 3 or 4` |
| `1 TO 5 hours.` | `"1 TO 5 hours."` | 1.0 / 5.0 | `Credits: 1 to 5` |
| `0 hours.` | `"0 hours."` | 0.0 / 0.0 | `Credits: 0` |

`credits_min <= n <= credits_max` is the filter for "can this be taken for n credits": it answers
for fixed- and variable-credit courses alike, and finds 2,184 courses that can be taken for 4
against the 1,009 a single number found. **The connector is kept because the two are different
claims** — a `3 OR 4` course is worth one of the two, a `1 TO 5` course anything between — which
also means the min/max *interval* over-reports an `OR` course: AAS 465 is 3 or 4, never 3.5. Read
`creditsRaw` (or the rendered line) when the answer has to be exact.

The export refuses to write a course whose `Credit Hours` it cannot read, rather than leaving
`null` bounds that every credit filter skips silently; all 67 distinct strings in the pinned CSV
parse. `Section Credit Hours` is a different column that can differ per section (3,257 of 12,823
rows disagree with the course-level one, e.g. AAS 465 has both a 3-hour and a 4-hour section) — the
course-level column is the one the catalog page shows, and the one exported here.

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
  are; `v2` follows the `See` clause and borrows the target's description, which resolves 496 of
  the 497 — the one that does not is `BADM 205`, pointing at `GGIS 205`, not offered this term.
  Neither template can fix the underlying export: `prereqCodes`, `credits` and `attrs` of a
  cross-listed course are still whatever the catalog filed under *its* code, not the target's.
- **1 course has an empty description** (`CWL 593`).
- **779 of 2,865 prerequisite references dangle** — a course mentioned in prerequisite text that is
  not offered in 2026-sp. They stay in `courses.json:prereqCodes` and are dropped from
  `prereqs.json`.
- **Only 572 of 4,502 courses carry any `Degree Attributes`**, so Gen Ed coverage is sparse by
  nature, not by parsing loss.
- **Gen Ed categories are small, and two of them share a group name.** `socsoc` is 118 courses and
  `socbeh` 24; both are written under the `Social & Beh Sci` prefix, which is why the rules match
  on the `- ` separator (#14). A rule added later has to do the same.
- **Prerequisite text is prose, not a grammar.** `prereqCodes` is a bag of codes: "One of CS 173,
  MATH 213; CS 225" loses the "one of" structure, so a planner cannot tell alternatives from
  conjunctions. Parsing that structure is a separate job.
