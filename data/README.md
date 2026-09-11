# data/

| File | Source | Status |
|---|---|---|
| `courses.json` | `~/Downloads/course-catalog.csv` (12,823 section rows → 4,502 courses) via `ingest/export_almabot.py` | pending (see issue) |
| `prereqs.json` | Same source; `prereqCodes` edges (~2,800), keeping only courses that exist in the catalog | pending |
| `requirements.json` | Gen Ed rules from `almabot-web/src/lib/gened.ts`; major requirements are not in the CSV, source TBD | pending |
| `chunks.jsonl` | `ingest/chunk.py`, one chunk per course, with `doc_id` | pending |

Note: `almabot-web/public/courses.json` truncates `desc` to 240 chars and `attrs` to 160, and has no
term field. **Do not copy it over.** Re-export the full text from the CSV.

CSV columns: Year, Term, YearTerm, Subject, Number, Name, Description, Credit Hours, Section Info,
Degree Attributes, Schedule Information, CRN, Section, Status Code, Part of Term, Section Title,
Section Credit Hours, Section Status, Enrollment Status, Type, Type Code, Start Time, End Time,
Days of Week, Room, Building, Instructors. Only one term is present: 2026 Spring.
