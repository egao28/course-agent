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
| `chunks.jsonl` | `ingest/chunk.py`, one chunk per course, with `doc_id` | pending |

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

## requirements.json

The 13 campus Gen Ed categories as declarative rules: a course matches when its `attrs` contains
**any** of `match.any` and **none** of `match.none`. `courseCount` is how many courses in this
export match, derived in the same run.

`majors` is an empty placeholder: major and degree requirements are not in `course-catalog.csv`,
which lists sections only. Filling it in needs a separate source (catalog.illinois.edu program
pages, or a department plan).

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
  no topical text, so both retrievers will rank them poorly — a candidate for following the
  reference at chunk time (#4).
- **1 course has an empty description** (`CWL 593`).
- **779 of 2,865 prerequisite references dangle** — a course mentioned in prerequisite text that is
  not offered in 2026-sp. They stay in `courses.json:prereqCodes` and are dropped from
  `prereqs.json`.
- **Only 572 of 4,502 courses carry any `Degree Attributes`**, so Gen Ed coverage is sparse by
  nature, not by parsing loss.
- **Prerequisite text is prose, not a grammar.** `prereqCodes` is a bag of codes: "One of CS 173,
  MATH 213; CS 225" loses the "one of" structure, so a planner cannot tell alternatives from
  conjunctions. Parsing that structure is a separate job.
