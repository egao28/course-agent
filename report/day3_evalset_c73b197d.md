# Day 3 — the evaluation set: what it is, who made it, who checked it

**167 rows drafted, 167 judged, 0 edited, 40 deleted** — leaving 127 in `eval/datasets/v1.jsonl`, of which 15 were read again by a second annotator. That pass was model-assisted: it agreed with the machine on 141 of 167 rows; 30 of them were read a second time and 0 verdicts were overturned (0%, 95% CI [0%, 11%]).

|  |  |
| --- | --- |
| config hash | `c73b197d` |
| catalog term | `2026-sp` |
| dataset | `eval/datasets/v1.jsonl`, 127 rows, sha256 `2a93182221bd8546…` |
| split | seed `61`, 64 calib, 63 test |
| annotation spec | `eval/GUIDELINES.md`, sha256 `75ec993375ada0e1…` |
| review log | `eval/datasets/review_log.jsonl`, 223 lines |
| second annotator | `eval/datasets/agreement_sheet_filled.tsv`, 15 rows |
| drafts | `eval/datasets/drafts/ambiguous_c73b197d.jsonl`, `eval/datasets/drafts/fact_c73b197d.jsonl`, `eval/datasets/drafts/multihop_graph_c73b197d.jsonl`, `eval/datasets/drafts/unanswerable_c73b197d.jsonl` |

Produced by:

```bash
uv run python -m eval.scripts.day3_report
```

## Composition

Counts as they are, not as they were planned. `target` is `eval.schema.RULES`; a bucket below it is verification having deleted rows, which is what verification is for.

| bucket | target | final | calib | test | abstain | what it tests |
| --- | --- | --- | --- | --- | --- | --- |
| `fact` | 50 | **58** | 29 | 29 | false | one catalog record answers it |
| `multihop` | 25 | **24** | 12 | 12 | false | two or more records must be joined |
| `unanswerable` | 25 | 25 | 13 | 12 | true | this catalog cannot answer it, so there is nothing to cite |
| `ambiguous` | 20 | 20 | 10 | 10 | true | under-specified; gold_answer holds the clarifying question |
| **total** | 120 | **127** | 64 | 63 |  |  |

**Question length**, in words.

| bucket | words |
| --- | --- |
| `fact` | n 58 min 6 p50 7 p90 9 max 14 |
| `multihop` | n 24 min 9 p50 12 p90 15 max 15 |
| `unanswerable` | n 25 min 4 p50 7 p90 9 max 10 |
| `ambiguous` | n 20 min 4 p50 8 p90 10 max 11 |
| **all** | n 127 min 4 p50 7 p90 12 max 15 |

**Multi-hop depth** — the longest prerequisite chain each row's walk covers, cycle-safe (`provenance.depth`, #58).

| depth | rows |
| --- | --- |
| 2 | 3 |
| 3 | 5 |
| 4 | 1 |
| 6 | 1 |
| 7 | 1 |
| 8 | 2 |
| 9 | 3 |
| 10 | 1 |
| 11 | 5 |
| 12 | 1 |
| 13 | 1 |

n 24 min 2 p50 8 p90 11 max 13. Each row also records `levels` and `closure`; the three disagree on a branching graph, which is #58's finding and not an error here.

**`fact` subject coverage** — 58 rows cite 54 subjects of the 187 in the catalog, 50 of them once. Most cited: `ENGL` 2, `TE` 2, `ECON` 2, `HT` 2, `DTX` 1, `RHET` 1, `MCB` 1, `PLPA` 1, `HORT` 1, `BCOG` 1.

**`unanswerable` failure shapes** — four different claims, and only one of them is about the retriever (#57).

| shape | table | in `v1` | what makes it unanswerable |
| --- | --- | --- | --- |
| `invented` | 6 | 6 | the course does not exist (an invented code, or an invented subject and no code) |
| `out_of_term` | 7 | 7 | real course, outside the export's one term |
| `field_absent` | 6 | 6 | real course, fact the knowledge base does not carry |
| `not_course` | 6 | 6 | not about courses at all |

17 of 25 score above the evidence gate on their own question (`provenance.above_gate`): the gate is not what will refuse them, so Day 4's abstention on this bucket tests the answer node and not the threshold.

## Provenance

Every row carries the record of how it was made, and `kept` is how many of a file's rows survived the verification pass below.

| file | bucket | source | model | prompt | seed | batches | kept |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `eval/datasets/drafts/ambiguous_c73b197d.jsonl` | `ambiguous` | drafted by model | `gpt-4.1-mini`, `gpt-4o-mini` | `v1` | `42` | 40 | 20/48 |
| `eval/datasets/drafts/fact_c73b197d.jsonl` | `fact` | drafted by model | `gpt-4.1-mini`, `gpt-4o-mini` | `v1` | `42` | 23 | 58/69 |
| `eval/datasets/drafts/multihop_graph_c73b197d.jsonl` | `multihop` | prereq-graph | — | — | — | — | 24/25 |
| `eval/datasets/drafts/unanswerable_c73b197d.jsonl` | `unanswerable` | constructed | — | — | — | — | 25/25 |

Rows in `eval/datasets/v1.jsonl` by drafting model: `gpt-4.1-mini` 49, `gpt-4o-mini` 29. A row with no model was constructed or walked out of the prerequisite graph rather than drafted — see `source`.

## Verification

**167 rows drafted, 167 judged, 0 edited, 40 deleted** — leaving 127 in `eval/datasets/v1.jsonl`, of which 15 were read again by a second annotator. That pass was model-assisted: it agreed with the machine on 141 of 167 rows; 30 of them were read a second time and 0 verdicts were overturned (0%, 95% CI [0%, 11%]).

| bucket | drafted | accept | edit | delete | unreviewed |
| --- | --- | --- | --- | --- | --- |
| `fact` | 69 | 58 | 0 | 11 | 0 |
| `multihop` | 25 | 24 | 0 | 1 | 0 |
| `unanswerable` | 25 | 25 | 0 | 0 | 0 |
| `ambiguous` | 48 | 20 | 0 | 28 | 0 |
| **total** | **167** | **127** | **0** | **40** | **0** |

From `eval/datasets/review_log.jsonl`, 223 lines — one per changed field, so an edit touching three fields is three lines sharing a timestamp. A bucket's edit rate is how much Day 4 should trust the pipeline that drafted it, which is why it is here rather than summarised.

**Deletes**

| reason | rows |
| --- | --- |
| `surplus: the bucket has its 20 and this shape is covered` | 26 |
| `variable credits` | 10 |
| `no citation to support` | 1 |
| `ambiguity` | 1 |
| `could not verified` | 1 |
| `d` | 1 |

**Did judgement drift?** accept rate: 94% over the first 111 judgements, 66% over the last 112. This is an ordering, not a duration: the log carries the clock and this report carries none of it, so a re-run rewrites this file byte for byte.

**The pass was model-assisted.** `gpt-4.1` read 167 rows against `eval/GUIDELINES.md` first (`eval/datasets/triage.jsonl`), and the reviewer saw its suggestion under each row's chunk text before typing a verdict. They agreed on 141 of 167 rows (84%).

| suggested ↓ / judged → | accept | edit | delete |
| --- | --- | --- | --- |
| **accept** | 127 | 0 | 26 |
| **edit** | 0 | 0 | 0 |
| **delete** | 0 | 0 | 14 |

Read the agreement rate as an **upper bound** on how much the suggestions moved the pass, not as a measurement of it: what an unassisted reader would have decided about these rows is not recoverable afterwards. The off-diagonal cells are the rows where a person overrode a machine.

**30 of the 167 judged rows were judged twice**, drawn stratified over bucket and verdict at seed `42` (`eval/datasets/recheck.meta.json`), including every delete a model asked for. 0 verdicts were overturned: **0%, 95% CI [0%, 11%]**.

| bucket | verdict | judged | re-read |
| --- | --- | --- | --- |
| `ambiguous` | `accept` | 46 | 8 |
| `ambiguous` | `delete` | 2 | 2 |
| `fact` | `accept` | 45 | 8 |
| `fact` | `delete` | 9 | 2 |
| `multihop` | `accept` | 24 | 4 |
| `multihop` | `delete` | 1 | 1 |
| `unanswerable` | `accept` | 25 | 5 |

Nothing overturned. That bounds the pass's error rate at **11%** and is not the same claim as *the pass was right*; it is what 30 rows support.

## Agreement

15 rows of 127 (11.8%) were judged a second time, blind, against `eval/GUIDELINES.md` and nothing else. The tables below are #60's own, as they appear in `report/day3_agreement_c73b197d.md` — one rendering, so the two reports cannot disagree about a number.

### How this sheet came back

> `agreement_sheet_filled.tsv` was **not filled in by the second annotator.** They read the fifteen rows and reported the outcome verbally to the first annotator, who relayed it; the file was then written from that account. The two `answer_correct` disagreements (`v1-0043`, `v1-3014`) are theirs. Everything else in the file is *"the rest agreed"*, transcribed rather than typed by them.

Everything below is computed from the sheet as it stands. The arithmetic cannot tell a
transcribed sheet from a typed one, which is why this section is here and above the numbers
rather than among the caveats.

### Agreement

| field | agreed | percent | 95% CI (Wilson) | Cohen's κ | p_e |
| --- | --- | --- | --- | --- | --- |
| `bucket` | 15/15 | 100.0% | [80%, 100%] | +1.000 | 0.280 |
| `answer_correct` | 13/15 | 86.7% | [62%, 96%] | +0.000 | 0.867 |
| `citations_support` | 15/15 | 100.0% | [80%, 100%] | +1.000 | 0.520 |
| `should_abstain` (derived) | 15/15 | 100.0% | [80%, 100%] | +1.000 | 0.520 |
| `citations_support` (cited rows only) | 9/9 | 100.0% | [70%, 100%] | — | 1.000 |

`p_e` is the chance agreement κ divides out. The interval is on the percent agreement only: κ's
asymptotic standard error is not trustworthy at n=15, and an interval that is wrong is worse than
an absent one.

**Where κ does not mean what it usually means:**

- `answer_correct` — κ 0 by construction: annotator 1 has no variance on this field in a promoted dataset, which forces p_e = p_o. Read the percent agreement, not this.
- `citations_support` (cited rows only) — κ undefined: both annotators used a single category, so chance agreement is 1 and κ is 0/0.

### Bucket confusion

Rows = annotator 1 (`v1.jsonl`), columns = annotator 2.

| 1 ↓ / 2 → | fact | multihop | unanswerable | ambiguous |
| --- | --- | --- | --- | --- |
| **fact** | 6 | 0 | 0 | 0 |
| **multihop** | 0 | 3 | 0 | 0 |
| **unanswerable** | 0 | 0 | 3 | 0 |
| **ambiguous** | 0 | 0 | 0 | 3 |

**Read this as two 2×2 blocks, not one 4×4.** The sheet strips `bucket` and `should_abstain`, but
it has to show the candidate answer — field 2 is *is this answer right* — and a refusal is visibly
a refusal. So the abstain/non-abstain split was largely given away, and the two off-diagonal
blocks (`fact`/`multihop` against `unanswerable`/`ambiguous`) should be near-empty for that reason
rather than because the guidelines are clear. The measurement lives in the two on-diagonal blocks:
`fact` against `multihop`, and `unanswerable` against `ambiguous` — which are also the two calls
`GUIDELINES.md` spends its boundary cases on (**F1**/**M1**, **A2**/**U1**).

### Disagreements

2 disagreements over 2 rows. This list is the revision plan for `GUIDELINES.md` v2. Where one shows the
**first** annotator was wrong, the fix goes into `v1.jsonl` through #59's log — and the number
above stays as it was measured, before the fix.

| id | field | annotator 1 | annotator 2 | rule cited | note |
| --- | --- | --- | --- | --- | --- |
| `v1-0043` | `answer_correct` | `yes` | `no` | — | — |
| `v1-3014` | `answer_correct` | `yes` | `no` | — | — |

#### `v1-0043` — `answer_correct`

**Q.** what are the prerequisites for the museum internship course?

Annotator 1 says `yes`, annotator 2 says `no`.

No rule cited — **attribute one by hand before #62**: a disagreement whose guideline rule is unidentified cannot be fixed by editing the guidelines.

#### `v1-3014` — `answer_correct`

**Q.** what are the prerequisites for 597?

Annotator 1 says `yes`, annotator 2 says `no`.

No rule cited — **attribute one by hand before #62**: a disagreement whose guideline rule is unidentified cannot be fixed by editing the guidelines.

Questions the annotator asked while working are in `eval/datasets/agreement_questions.md`; they are evidence that `eval/GUIDELINES.md` is incomplete, and any amendment made in answer to one came **after** this measurement — the guidelines digest in the header is the version the sheet was filled in against.

## Split

Stratified 50/50 within each bucket, derived from the seed over each bucket's sorted ids and frozen into the `split` field of every row (#61). The per-bucket halves are in the Composition table above; this is the provenance of them.

|  |  |
| --- | --- |
| seed | `61` (`eval.split_seed`) |
| sha256 | `2a93182221bd8546d5bed8e121d0b4bde9a8b624a2197c67f5e87120d36cfce6` |
| sidecar | `eval/datasets/v1.meta.json` |
| check | `python -m eval.scripts.split --verify` re-derives from the seed, and the digest matches |

**What 64 calibration rows cost Day 5.** Split conformal prediction takes the `ceil((n+1)(1-α))`-th smallest calibration score, and that index has to exist. That bounds α below at 1/65 = 0.0154, so **98.5% coverage is the ceiling** and 99% is out of reach at any threshold. The lever is the split ratio and the time to move it is before anything is measured, which is why the number is here.

**No number in this report is computed on either half.** Everything above is a count over the whole file; Day 4 reports on `test`, once, and Day 5 calibrates on `calib`.

## Limitations

Written before anyone asks, and sized from the artifacts above rather than from the plan.

- **One annotator on 112 of 127 rows.** 11.8% were double-annotated. The κ on that 11.8% is the only evidence there is about the other 88.2%, and at n=15 it is an estimate with a wide interval — the Wilson bounds in the Agreement section, not the point estimate, are what that sample supports.

- **Drafted and evaluated by the same model family.** `gpt-4o-mini` both wrote questions here and is the model under test (`llm.model`), so a blind spot it has is a blind spot in the questions and in the answers, and the two cannot be told apart. Human verification does not remove the correlation: checking that an answer is *right* does not detect that the question was *easy for this model*.

- **Questions derived from chunks share vocabulary with them, which flatters BM25.** #56 measures the Jaccard overlap between each drafted question and the record it came from, and rewrites the worst decile. The residual over the rows in this set: n 58 min 0.025 p50 0.073 p90 0.118 max 0.191. That is the distribution to read, not a claim that the leakage was solved; #57's and #58's rows are not drafted from a record and carry no overlap at all.

- **63 test items.** A per-bucket accuracy on this set is a wide interval whatever it comes out at — `fact` 29 rows ±17 points, `multihop` 12 rows ±25 points, `unanswerable` 12 rows ±25 points, `ambiguous` 10 rows ±26 points, as 95% Wilson half-widths at the worst case p=0.5 (a result near 0 or 1 is tighter). Day 4 must report intervals and not bare percentages, and this report says so first.

- **One term, one institution.** `2026-sp`, 4,502 courses. Nothing measured on this set generalises past that, and the `unanswerable` bucket depends on the scope boundary being exactly this one: `out_of_term` rows are real courses that this export does not carry, and they stop being unanswerable the moment a second term is indexed.

- **`should_abstain` is one bit where two buckets want different behaviour.** 25 `unanswerable`, 20 `ambiguous` rows carry it. An `ambiguous` row should come back with the clarifying question in its `gold_answer`; an `unanswerable` row should come back with a refusal and no question at all. The bit cannot tell those apart, so Day 4's abstention metrics must condition on `bucket` (#54).

- **The verification pass was model-assisted, and a suggestion anchors a reviewer.** `gpt-4.1` read 167 rows first and the reviewer saw its accept / edit / delete and its one-sentence reason under each row before judging it, against guidelines `75ec993375ada0e1…`. A person shown a suggestion agrees with it more often than one shown nothing, so the accept rate in the Verification section is not the accept rate of an unassisted pass, and the agreement rate reported there bounds the effect rather than measuring it. 30 of the rows were judged a second time and 0 verdicts were overturned, which puts the pass's error rate at 0% with a 95% interval of [0%, 11%] — the other 137 rows are covered by that interval and by nothing else. No suggestion reached the review log: they are a separate append-only file (`eval/datasets/triage.jsonl`) whose rows carry neither a `verdict` nor a `ts`, which is what makes the two unmistakable for each other.

- **The second annotator's sheet is not their own typing.** How it came back is in `eval/datasets/agreement_sheet_filled.provenance.md` and is quoted above the numbers in the Agreement section. The arithmetic cannot tell a transcribed sheet from a typed one, so the distinction survives only because it is written down. What it costs concretely: 15 of the 15 judged rows cite no guideline rule, so a disagreement cannot be traced to the line of `eval/GUIDELINES.md` it came from — which is the main thing #60 draws this sample for — and the two-pass order that keeps `bucket` from being judged after the citations are opened is unverifiable here. The labels are a real second reader's; the record of them is not first-hand.

## Reproduction

```bash
uv run python -m eval.scripts.day3_report
```

This file is a function of the dataset, its sidecar, the draft files and theirs, the review log
and the returned sheet — and of `config.yaml`, which names it. There is no timestamp and no duration
anywhere in it, so a re-run over unchanged inputs rewrites it byte for byte and a diff under
`report/` is one of those artifacts having changed. The digests in the header are what pins that:
the dataset's is #61's sidecar, and the guidelines' is the version the second annotator was
handed.
