# Day 2 smoke: ten queries, one of three exits each

One message in, one of three exits out, once per query, from the command line.
Produced by:

```bash
uv run --env-file .env python -m eval.scripts.day2_smoke
```

|  |  |
| --- | --- |
| config hash | `7980f15f` |
| catalog term | `2026-sp` |
| retrieval mode | `bm25` (`agent.retrieval_mode`) |
| index | `indexes/bm25_877a6610.pkl` |
| evidence gate | `top_score`, bm25 threshold `15.0` |
| prompts | router `v2`, answer `v1` |
| answers written by | `gpt-4o-mini` |
| queries | `eval/datasets/day2_queries.jsonl`, 10 rows |

## What is and is not measured here

**This run called `gpt-4o-mini`**: once per query to classify, and once more for every query that got past the gate. The intent columns below are the router's own work, and every answer is the model's, kept only because its citations checked out.

- **A live run is not reproducible, and `temperature: 0` does not make it one.** Nothing else in this file can move a line -- there is no clock in it and the renderer is a function of the run -- so a diff under `report/` after a live re-run is the model having answered differently, which it does. Read a single row as one sample and not as the behaviour of the system.
- **Only the lexical half of the pipeline ran.** `agent.retrieval_mode` is `bm25`, so every retrieval row here was answered by BM25 alone and the hybrid path is plumbing-verified, not result-verified by this report. Which mode runs is a config value, so switching it is a new hash and a new report rather than a footnote to this one.

What the run does establish: the graph, the dispatch table, the three tools, the evidence gate, the citation check and the two fixed refusals hold together over ten real questions, and all three exits are reachable from the command line.

## Exits

| exit | expected | actual |
| --- | --- | --- |
| `answer` | 5 | 4 |
| `refuse_handoff` | 4 | 5 |
| `out_of_scope` | 1 | 1 |

**All three exits fired.** That is Day 2's acceptance, and this is the count.

## Expected against actual

| # | query | intent (expected → actual) | exit (expected → actual) |  |
| --- | --- | --- | --- | --- |
| 1 | How many credits is MATH 241? | `course_fact` | `answer` |  |
| 2 | What do I need before CS 374? | `prereq_chain` | `answer` |  |
| 3 | Which Gen Eds do I still need if I have taken RHET 105 and MATH 241? | `degree_requirement` | `answer` |  |
| 4 | Which intro machine learning course should I take? | `compare_recommend` | `answer` |  |
| 5 | What does the algorithms course cover? | `course_fact` | `answer` → `refuse_handoff` | ✗ |
| 6 | What is left for the Physics major if I have taken PHYS 211? | `degree_requirement` | `refuse_handoff` |  |
| 7 | What are the prerequisites? | `prereq_chain` | `refuse_handoff` |  |
| 8 | How many credits is CS 428? | `course_fact` | `refuse_handoff` |  |
| 9 | Is there a course about quantum origami? | `compare_recommend` | `refuse_handoff` |  |
| 10 | Who is the head football coach at Illinois? | `out_of_scope` | `out_of_scope` |  |

## Disagreements

1 of 10 rows did something other than what the dataset expects. They are listed here rather than re-labelled in `day2_queries.jsonl`: the expectation is what the row should do, and the gap is the finding.

**5. What does the algorithms course cover?**

- expected `answer` by way of `course_fact`, got `refuse_handoff` by way of `course_fact`
- evidence: retrieve(bm25), 5 hits; gate: refused -- 11.7914 < 15.0 (bm25)
- why the row is in the set: The other half of the course_fact row: same intent, no code in the question, so the fact has to come out of the retriever. expected_exit is what this should do once the dense index exists; on bm25 alone it does not, which is a finding and not a mislabelled row.

## The 10 runs

One section per query, each reproducible on its own with the command under it.

### 1. How many credits is MATH 241?

*A course fact that names its course: the dispatch table's lookup row, answered from the one catalog record rather than from five ranked chunks. The shortest path through the graph that still ends in a cited answer.*

```bash
uv run --env-file .env python -m agent "How many credits is MATH 241?"
```

|  | expected | actual |  |
| --- | --- | --- | --- |
| intent | `course_fact` | `course_fact` | classified |
| exit | `answer` | `answer` |  |

**Evidence** — get_course, 1 block. Gate: passed -- an exact catalog lookup carries no score, so no threshold was consulted.

| # | score | doc_id | first line of the block |
| --- | --- | --- | --- |
| 1 | — | `MATH 241` | MATH 241 — Calculus III |

**Answer** — written by the model, citing `MATH 241` — each checked against the evidence above.

> MATH 241 is 4 hours [MATH 241].

### 2. What do I need before CS 374?

*The prerequisite closure, walked to exhaustion and cycle-safe. CS 374's own statement is a disjunction ("One of CS 173, MATH 213; CS 225."), so the evidence block has to carry the catalog's sentence and not only the flattened layers.*

```bash
uv run --env-file .env python -m agent "What do I need before CS 374?"
```

|  | expected | actual |  |
| --- | --- | --- | --- |
| intent | `prereq_chain` | `prereq_chain` | classified |
| exit | `answer` | `answer` |  |

**Evidence** — get_prereq_chain, 1 block. Gate: passed -- an exact catalog lookup carries no score, so no threshold was consulted.

| # | score | doc_id | first line of the block |
| --- | --- | --- | --- |
| 1 | — | `CS 374` | CS 374 — what the catalog requires before it |

**Answer** — written by the model, citing `CS 374` — each checked against the evidence above.

> Before CS 374, you need "one of CS 173, MATH 213; CS 225" [CS 374].

### 3. Which Gen Eds do I still need if I have taken RHET 105 and MATH 241?

*The campus Gen Ed audit -- the one requirement set this export can decide in full, since a Gen Ed is a rule about a course's Degree Attributes. Both codes are in the 2026-sp catalog, so neither lands in unknown_courses.*

```bash
uv run --env-file .env python -m agent "Which Gen Eds do I still need if I have taken RHET 105 and MATH 241?"
```

|  | expected | actual |  |
| --- | --- | --- | --- |
| intent | `degree_requirement` | `degree_requirement` | classified |
| exit | `answer` | `answer` |  |

**Evidence** — check_requirement, 1 block. Gate: passed -- an exact catalog lookup carries no score, so no threshold was consulted.

| # | score | doc_id | first line of the block |
| --- | --- | --- | --- |
| 1 | — | `General Education` | General Education — requirement audit |

**Answer** — written by the model, citing `General Education` — each checked against the evidence above.

> You still need 11 Gen Eds: Advanced Composition (1 more), Quantitative Reasoning I (1 more), Physical Sciences (1 more), Life Sciences (1 more), Historical & Philosophical (1 more), Literature & the Arts (1 more), Social Sciences (1 more), Behavioral Sciences (1 more), Western/Comparative Cultures (1 more), Non-Western Cultures (1 more), and US Minority Cultures (1 more) [General Education].

### 4. Which intro machine learning course should I take?

*Open-ended: several courses could be it and a choice is asked for, so the row retrieves and the answer ranks. The only answering row whose evidence has to clear the gate's threshold rather than bypass it as an exact lookup.*

```bash
uv run --env-file .env python -m agent "Which intro machine learning course should I take?"
```

|  | expected | actual |  |
| --- | --- | --- | --- |
| intent | `compare_recommend` | `compare_recommend` | classified |
| exit | `answer` | `answer` |  |

**Evidence** — retrieve(bm25), 5 hits. Gate: passed -- 18.2746 >= 15.0 (bm25).

| # | score | doc_id | first line of the block |
| --- | --- | --- | --- |
| 1 | 18.2746 | `2026-sp:IS-557` | IS 557 — Applied Machine Learning: Team Projects |
| 2 | 15.2363 | `2026-sp:IS-327` | IS 327 — Concepts of Machine Learning |
| 3 | 11.7456 | `2026-sp:CS-562` | CS 562 — Advanced Topics in Security, Privacy, and Machine Lear… |
| 4 | 11.7339 | `2026-sp:LER-526` | LER 526 — Machine Learning Applications in HR |
| 5 | 11.3325 | `2026-sp:ACCY-577` | ACCY 577 — Machine Learning for Accounting |

**Answer** — written by the model, citing `2026-sp:IS-557` — each checked against the evidence above.

> The catalog does not carry a recommendation for which intro machine learning course to take [2026-sp:IS-557].

### 5. What does the algorithms course cover?

*The other half of the course_fact row: same intent, no code in the question, so the fact has to come out of the retriever. expected_exit is what this should do once the dense index exists; on bm25 alone it does not, which is a finding and not a mislabelled row.*

```bash
uv run --env-file .env python -m agent "What does the algorithms course cover?"
```

|  | expected | actual |  |
| --- | --- | --- | --- |
| intent | `course_fact` | `course_fact` | classified |
| exit | `answer` | `refuse_handoff` | ✗ |

**Evidence** — retrieve(bm25), 5 hits. Gate: refused -- 11.7914 < 15.0 (bm25).

| # | score | doc_id | first line of the block |
| --- | --- | --- | --- |
| 1 | 11.7914 | `2026-sp:ECON-300` | ECON 300 — Everyday Economics for Non-Majors |
| 2 | 11.6450 | `2026-sp:ERAM-557` | ERAM 557 — Meaning Patterns: Semiotics and the Interpretation o… |
| 3 | 10.6480 | `2026-sp:HDFS-322` | HDFS 322 — US Latina and Latino Families |
| 4 | 9.7806 | `2026-sp:UP-230` | UP 230 — Transportation Planning: Fundamentals and Innovations |
| 5 | 8.8244 | `2026-sp:IE-531` | IE 531 — Algorithms for Data Analytics |

**Answer** — fixed text: neither refusal exit calls a model.

> I could not answer this from the 2026-sp course catalog, so I am handing it over rather than guessing.
>
> What you asked: What does the algorithms course cover?
> Where I stopped: nothing matched closely enough -- the best bm25 match scored 11.7914, against the bm25 threshold of 15.0
> What would get it answered: try the words the catalog itself uses (a subject code, a course title), or ask for one course at a time
>
> An academic advisor can also see what this catalog export cannot -- your transcript, program requirements outside the Gen Eds, and sections as they actually fill.

### 6. What is left for the Physics major if I have taken PHYS 211?

*A degree this export cannot audit. #39 wrote this row as majors: [] and #31's unsupported refusal; #32 has since transcribed two programs, so the refusal now comes one step earlier -- the dispatch node reads no program it can audit out of the question (no_program) and calls no tool at all.*

```bash
uv run --env-file .env python -m agent "What is left for the Physics major if I have taken PHYS 211?"
```

|  | expected | actual |  |
| --- | --- | --- | --- |
| intent | `degree_requirement` | `degree_requirement` | classified |
| exit | `refuse_handoff` | `refuse_handoff` |  |

**Evidence** — none -- the tool it needed reported no_program. Gate: refused -- no_program: there is nothing to be confident about.

**Answer** — fixed text: neither refusal exit calls a model.

> I could not answer this from the 2026-sp course catalog, so I am handing it over rather than guessing.
>
> What you asked: What is left for the Physics major if I have taken PHYS 211?
> Where I stopped: I could not tell which program to audit -- this export carries the campus General Education rules and a small number of hand-transcribed degrees, and auditing you against a program you are not in is wrong in the direction that costs a semester
> What would get it answered: name it: 'gened' for the campus General Education requirements, or one of the degrees this export carries, the way the catalog spells it -- for anything else, the program's catalog page and your department advisor are the source
>
> An academic advisor can also see what this catalog export cannot -- your transcript, program requirements outside the Gen Eds, and sections as they actually fill.

### 7. What are the prerequisites?

*A prerequisite question that names no course. Which one was meant is not something to guess at: answering confidently about the wrong course is worse than not answering, so the hand-off asks for a code.*

```bash
uv run --env-file .env python -m agent "What are the prerequisites?"
```

|  | expected | actual |  |
| --- | --- | --- | --- |
| intent | `prereq_chain` | `prereq_chain` | classified |
| exit | `refuse_handoff` | `refuse_handoff` |  |

**Evidence** — none -- the tool it needed reported no_course_code. Gate: refused -- no_course_code: there is nothing to be confident about.

**Answer** — fixed text: neither refusal exit calls a model.

> I could not answer this from the 2026-sp course catalog, so I am handing it over rather than guessing.
>
> What you asked: What are the prerequisites?
> Where I stopped: the question does not name a course, and which one was meant is not something to guess at -- answering confidently about the wrong course is worse than not answering
> What would get it answered: name it by code: 'CS 374', 'cs 374' and 'CS374' all reach the same catalog record
>
> An academic advisor can also see what this catalog export cannot -- your transcript, program requirements outside the Gen Eds, and sections as they actually fill.

### 8. How many credits is CS 428?

*A real UIUC course that is not in the 2026-sp export. From inside the graph "there is no such course" and "it is not offered in the one term this export covers" are the same observation, and the hand-off says both rather than picking one.*

```bash
uv run --env-file .env python -m agent "How many credits is CS 428?"
```

|  | expected | actual |  |
| --- | --- | --- | --- |
| intent | `course_fact` | `course_fact` | classified |
| exit | `refuse_handoff` | `refuse_handoff` |  |

**Evidence** — none -- the tool it needed reported course_not_found. Gate: refused -- course_not_found: there is nothing to be confident about.

**Answer** — fixed text: neither refusal exit calls a model.

> I could not answer this from the 2026-sp course catalog, so I am handing it over rather than guessing.
>
> What you asked: How many credits is CS 428?
> Where I stopped: I looked the course up and this catalog does not have it -- either there is no such course, or it is not offered in the one term this export covers
> What would get it answered: check the subject and the number, or ask for the code it is cross-listed under -- the catalog lists 'CS 374' and 'ECE 374' as separate records
>
> An academic advisor can also see what this catalog export cannot -- your transcript, program requirements outside the Gen Eds, and sections as they actually fill.

### 9. Is there a course about quantum origami?

*A topical query with nothing behind it. BM25 always returns something -- the top hit is a real course about something else -- so this is the row the evidence gate exists for. The label is arguable (a code-free course_fact retrieves too, and lands in the same place); worth watching when the router is live.*

```bash
uv run --env-file .env python -m agent "Is there a course about quantum origami?"
```

|  | expected | actual |  |
| --- | --- | --- | --- |
| intent | `compare_recommend` | `compare_recommend` | classified |
| exit | `refuse_handoff` | `refuse_handoff` |  |

**Evidence** — retrieve(bm25), 5 hits. Gate: refused -- 11.0561 < 15.0 (bm25).

| # | score | doc_id | first line of the block |
| --- | --- | --- | --- |
| 1 | 11.0561 | `2026-sp:ECE-305` | ECE 305 — Quantum Systems I |
| 2 | 10.7406 | `2026-sp:PHYS-370` | PHYS 370 — Introduction to Quantum Information and Computing |
| 3 | 10.4175 | `2026-sp:ECE-406` | ECE 406 — Quantum Optics and Devices |
| 4 | 8.9002 | `2026-sp:CHEM-542` | CHEM 542 — Quantum Mech and Spectroscopy |
| 5 | 8.7075 | `2026-sp:ECE-532` | ECE 532 — Compnd Semicond & Diode Lasers |

**Answer** — fixed text: neither refusal exit calls a model.

> I could not answer this from the 2026-sp course catalog, so I am handing it over rather than guessing.
>
> What you asked: Is there a course about quantum origami?
> Where I stopped: nothing matched closely enough -- the best bm25 match scored 11.0561, against the bm25 threshold of 15.0
> What would get it answered: try the words the catalog itself uses (a subject code, a course title), or ask for one course at a time
>
> An academic advisor can also see what this catalog export cannot -- your transcript, program requirements outside the Gen Eds, and sections as they actually fill.

### 10. Who is the head football coach at Illinois?

*About the university and not about its courses. The one exit the router can take on its own, before any evidence is gathered -- and the one refusal that is not about a course this catalog happens to be missing.*

```bash
uv run --env-file .env python -m agent "Who is the head football coach at Illinois?"
```

|  | expected | actual |  |
| --- | --- | --- | --- |
| intent | `out_of_scope` | `out_of_scope` | classified |
| exit | `out_of_scope` | `out_of_scope` |  |

**Evidence** — not gathered -- the router's own exit ended the run. Gate: not reached -- no evidence was gathered for this run.

**Answer** — fixed text: neither refusal exit calls a model.

> That is outside what I can answer. I work from one source -- the University of Illinois course catalog for 2026-sp -- which lets me answer questions about course descriptions and credits, prerequisite chains, and progress against the Gen Ed requirements. I have not tried to answer this one from general knowledge.
