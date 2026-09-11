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
| `prereqs.json` | Same source; `prereqCodes` filtered to courses in the catalog | done — 2,080 edges over 1,123 courses |
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
| `prereqRaw` | The prerequisite sentence, verbatim |
| `prereqCodes` | Every course code in `prereqRaw`, minus self — **not** filtered to the catalog |
| `sectionCount`, `partOfTerm`, `types` | Section metadata, aggregated over the course's rows |

## prereqs.json

`[{"course": "CS 374", "prereqs": ["CS 173", "CS 225"], "raw": "One of CS 173, MATH 213; CS 225"}]`

`prereqs` is `prereqCodes` filtered to courses that exist in this catalog — the same filter
`dag.ts::prereqs()` applies at read time. Courses with no surviving reference are omitted entirely.

## requirements.json

The 13 campus Gen Ed categories as declarative rules: a course matches when its `attrs` contains
**any** of `match.any` and **none** of `match.none`. `courseCount` is how many courses in this
export match, derived in the same run.

`majors` is an empty placeholder: major and degree requirements are not in `course-catalog.csv`,
which lists sections only. Filling it in needs a separate source (catalog.illinois.edu program
pages, or a department plan).

## Known data quality issues

Counts from the 2026-sp export; `ingest/export_almabot.py` prints them on every run.

- **497 descriptions are cross-references only** (`"Same as HIST 200. See HIST 200."`). They carry
  no topical text, so both retrievers will rank them poorly — a candidate for following the
  reference at chunk time (#4).
- **1 course has an empty description** (`CWL 593`).
- **778 of 2,858 prerequisite references dangle** — a course mentioned in prerequisite text that is
  not offered in 2026-sp. They stay in `courses.json:prereqCodes` and are dropped from
  `prereqs.json`.
- **Only 572 of 4,502 courses carry any `Degree Attributes`**, so Gen Ed coverage is sparse by
  nature, not by parsing loss.
- `prereqRaw` stops at the first sentence break (inherited from `build_data.py`), so a
  multi-sentence prerequisite is captured only up to that point.
