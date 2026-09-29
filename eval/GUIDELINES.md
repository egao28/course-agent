# Annotation guidelines — eval set v1

The spec comes first. The drafting prompt ([#56](../../../issues/56)) is derived from it, the
verification pass ([#59](../../../issues/59)) judges against it, and the second annotator
([#60](../../../issues/60)) is handed this file and nothing else — an agreement number computed
against a spec written afterwards measures nothing. One page: a criterion that does not fit here
is not decidable.

The vocabulary is `eval/schema.py`'s: four buckets, six fields, gold ids in `2026-sp:CS-374` form.

## What the knowledge base is

Three things, and nothing else:

- **`data/chunks.jsonl`** — one chunk per course, 4,502 of them, **term `2026-sp` only**. Each
  prints the title, credits, term, the full description, the prerequisite sentence verbatim, and
  the degree attributes. This is what "the catalog says" means.
- **The prerequisite graph** — `data/prereqs.json`, 1,135 courses with edges, read through
  `get_prereq_chain`.
- **The Gen Ed set and the two hand-encoded majors** — `data/requirements.json`, 13 Gen Ed
  categories plus `cs-bs` and `statistics-bslas`, read through `check_requirement`.

Outside it by construction: instructors, difficulty or quality, enrolment and seats, meeting times
and rooms, tuition, textbooks, any other term, and the student's own record. A question about one
of those is `unanswerable` however well known the answer is elsewhere.

Three exported fields reach no chunk text: `sectionCount`, `partOfTerm`, `types`. `get_course`
returns `section_count`, so the agent can answer such a question — but nothing supports it, so no
gold id can, and it is not a v1 row (boundary case **U2**).

## When a `gold_answer` is correct

1. **Semantic equivalence for the prose, a code for every course.** `4`, `4 credits`, `4 credit
   hours` and `four credit hours` all pass. *"The data structures course is 4 credit hours"* does
   not: Day 4 cannot score a claim whose subject is ambiguous, so write *"CS 225 is 4 credit
   hours."* Codes keep the catalog's space (`CS 225`); only `gold_doc_ids` use the dash.
2. **Credit counts: fixed-value courses only.** The chunk prints `Credits:` as a single number
   even where the catalog gives a range — AAS 465 is "3 OR 4 hours." and its chunk says
   `Credits: 3`. 1,415 of 4,502 courses are affected ([#17](../../../issues/17) is fixed on `main`,
   not in this stack), so for those any credit answer is wrong about either the catalog or the
   corpus and the row is deleted, not repaired. The list, when in doubt:
   ```bash
   git show origin/main:data/courses.json | jq -r '.[]|select(.creditsMin!=.creditsMax)|.code'
   ```
3. **Prerequisite disjunctions follow the sentence, never the code list.** `prereq_codes` has the
   booleans flattened out of it — 744 of the 1,135 raw statements say "or" or "one of" — so
   CS 225's ten codes are two choices, not ten requirements. Turning *"One of CS 173, MATH 213;
   CS 225"* into *"CS 173, MATH 213 and CS 225"* is **wrong**, not loose, and quoting the
   catalog's own sentence is the one repair that introduces no new claim.
4. **A gold id supports the claim, not the topic.** Every fact the answer asserts must be readable
   in the text of a cited chunk. An id that is merely *about* the right course is a wrong id: it
   rewards a system that retrieved the wrong document, which is what this set exists to catch.

## When a refusal is correct

A refusal is correct when **the knowledge base does not contain the answer**. It is not correct
when the answer is in there and retrieval missed it — that gap is what Day 4 measures and Day 5
improves, and relabelling a hard-to-retrieve question `unanswerable` raises the score by deleting
the measurement.

So label from the source: open the chunk, or call `get_course` / `get_prereq_chain` /
`check_requirement`. Never run the agent or `retrieve()` to decide a label — what the system does
with a question is the thing under test, and it cannot also be the thing that labels it.

**Term scope is absence.** CS 421, CS 423, CS 425 and CS 427 are in the export and CS 428 is not;
it reads like a catalog gap and it is a scope boundary, and either way the answer is not here.

**The two abstaining buckets are not interchangeable.** An `unanswerable` row's `gold_answer` says
what the knowledge base does not have and carries no gold ids. An `ambiguous` row's `gold_answer`
**is the clarifying question** that should have been asked.

## Which bucket

**`fact` vs `multihop`: count records, not reasoning steps.** One chunk that carries every asserted
fact is a `fact` row, however many courses its sentences name. A `multihop` row needs its second
record to be load-bearing: remove it and the answer changes or disappears.

**Cap the join at four records.** More than that is a list question — narrow it (*"does RHET 105
satisfy Composition I?"*) or drop it (*"which courses satisfy Advanced Composition?"* — 83).

**Bounded chains only.** The prereq graph holds 7 cycles over 19 courses, so "everything before X"
need not terminate. The question states its depth, or the row is deleted.

**`ambiguous` vs `unanswerable`: one clarifying question is the whole test.** If naming the missing
thing makes the knowledge base answer it, the row is `ambiguous`; if no clarification helps, it is
`unanswerable`. A description with exactly one plausible referent is neither — it is a `fact`.

## Eight boundary cases

Not illustrations — the eight calls that were actually hard, two per bucket, with the reason.

### fact

- **F1.** *"What are the prerequisites for CS 374?"* → **`fact`**, `["2026-sp:CS-374"]`.
  Hops are records, not codes: CS 374's own chunk prints *"One of CS 173, MATH 213; CS 225."*, so
  one document answers it and the three codes buy no second id.
- **F2.** *"How many credits is the data structures course?"* → **`fact`**, `["2026-sp:CS-225"]`.
  Three titles contain the phrase — CS 225 *Data Structures*, CS 277, CI 487 — but only one **is**
  it, so the referent is unique; the answer still has to say *"CS 225 is 4 credit hours."*

### multihop

- **M1.** *"What are the prerequisites for ECE 374?"* → **`multihop`**,
  `["2026-sp:ECE-374", "2026-sp:CS-374"]`.
  ECE 374's whole description is *"Same as CS 374. See CS 374."* and its chunk reads
  `Prerequisites: None listed`, so a one-id version of this row certifies *"none"* — wrong, here
  and for the 497 courses whose description is nothing but a cross-reference.
- **M2.** *"List every course you have to take before MATH 412."* → **`multihop` only with a depth
  in the question**, otherwise deleted.
  MATH 412 → CS 374 → CS 225 → MATH 412 is one of the graph's 7 cycles, so the transitive answer
  does not terminate: *"…and the prerequisites of those"* is a row, "everything" is not.

### unanswerable

- **U1.** *"How many credits is CS 428?"* → **`unanswerable`**, no gold ids.
  A real course the 2026-sp export does not contain; term scope is part of the knowledge-base
  definition, so this is a boundary and not a gap to be filled from memory.
- **U2.** *"How many sections of CS 374 are running?"* → **deleted**, not `unanswerable`.
  `get_course` returns `section_count`, so the agent can answer it and no chunk supports the
  number: `unanswerable` would score a right answer as a missing refusal, and `fact` would score
  it as a citation failure.

### ambiguous

- **A1.** *"What are the prerequisites for 374?"* → **`ambiguous`**, `gold_answer` = *"Which 374 —
  BADM, CS, ECE or HIST?"*
  Four courses carry the number and one clarifying question picks between them, which is the test;
  the row abstains even though CS 374 is the likeliest reading.
- **A2.** *"What should I take next semester?"* → **`unanswerable`**, not `ambiguous`.
  No clarification closes this: what is missing is a transcript, a standing and a future term's
  schedule — *"which major?"* would only yield a plan the catalog cannot certify.

## Intent

Added for Day 6 ([#102](../../../issues/102)). The above labels a row's *bucket*. This labels
what the question *asks for*, in the router's five labels (`INTENTS` in `agent/state.py`), so
intent accuracy has something to be scored against. It goes in the sidecar
`eval/datasets/v1_intents.jsonl`, one row per **test** id: `{"id", "gold_intents", "note"}`.
`v1.jsonl` is frozen and is not touched.

The labels mean what the router is told they mean (`ROUTER_GLOSSES`, `agent/prompts.py`):
`course_fact` is one course and one attribute, `prereq_chain` is what has to come before a course,
`degree_requirement` is progress against a program, `compare_recommend` is a choice among courses,
and `out_of_scope` is not about catalog courses at all.

1. **Label the question, not the answer.** Whether the catalog can answer it, and which bucket
   the row is in, do not decide the intent. A refusal is made later, by whichever step looked and
   came back empty.
2. **A set, not one label.** `gold_intents` holds every label whose gloss the question, read
   literally, falls under. The router is right if its label is in the set. Most rows get one
   label. Two needs a `note` saying why, and three or more means the row is unclear, not generous.
3. **A course's prerequisites are both.** *"What are the prerequisites for X?"* →
   `["course_fact", "prereq_chain"]`, because X's own chunk prints the sentence (**F1**). Asking
   for a chain, or whether one course leads to another, is `["prereq_chain"]` only.
4. **One course against one requirement is both.** *"Does X satisfy Composition I?"* →
   `["course_fact", "degree_requirement"]`. *"What is left for the major?"* is
   `["degree_requirement"]` only.
5. **`out_of_scope` is never the default for a refusal.** A question about a course the catalog
   lacks, or a field it does not carry, is still about courses.

Label from the question text and this file only. Do not open `eval/runs/signals_test_*.jsonl`
(the router's own predictions), and do not run the router or the agent.

### ambiguous

- **IA1.** *"what are the prerequisites for 440?"* → `["course_fact", "prereq_chain"]`.
  The missing referent does not change what is asked. A question that names no course is still a
  prerequisite question (router rule 3), so it gets the set rule 3 above gives a named course.
- **IA2.** *"which course is intended for science and science-oriented students?"* →
  `["course_fact", "compare_recommend"]`.
  It reads as a lookup of one course by an attribute, or as a choice among the courses that could
  be it. Neither reading is wrong, and `out_of_scope` is.

### unanswerable

- **IU1.** *"who teaches CS 225?"* → `["course_fact"]`.
  Instructors are outside the knowledge base, but the question asks for one attribute of one
  course. The refusal belongs to the evidence gate, not the router (rule 5 above).
- **IU2.** *"what time does the dining hall close on Sundays?"* → `["out_of_scope"]`.
  This is not about courses at all. Neither is tuition: a question about the university that a
  course catalog does not answer is `out_of_scope` (router rule 4).

All four are calib questions. The labelled half is test, and no test question is quoted here.

## Amending this file

A rule agreed verbally is a rule that does not exist, because this file is the whole instruction to
the second annotator. Amend it, and record in the Day 3 report ([#62](../../../issues/62)) what
changed and which rows were re-judged under it. An amendment to the Intent section goes in the
Day 6 report ([#106](../../../issues/106)) instead.
