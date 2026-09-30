# Day 3 — inter-annotator agreement

Config hash `c73b197d`. 15 rows judged by a second annotator against `eval/GUIDELINES.md` and nothing else.

|  |  |
| --- | --- |
| sampled from | `eval/datasets/v1.jsonl` (127 rows) |
| dataset sha256 | `2a93182221bd8546…` |
| seed | `42` |
| allocation | fact 6, multihop 3, unanswerable 3, ambiguous 3 |
| drawn | 6 fact, 3 multihop, 3 unanswerable, 3 ambiguous |
| sheet | `eval/datasets/agreement_sheet_filled.tsv` |
| guidelines sha256 | `75ec993375ada0e1…` |

## How this sheet came back

> `agreement_sheet_filled.tsv` was **not filled in by the second annotator.** They read the fifteen rows and reported the outcome verbally to the first annotator, who relayed it; the file was then written from that account. The two `answer_correct` disagreements (`v1-0043`, `v1-3014`) are theirs. Everything else in the file is *"the rest agreed"*, transcribed rather than typed by them.

Everything below is computed from the sheet as it stands. The arithmetic cannot tell a
transcribed sheet from a typed one, which is why this section is here and above the numbers
rather than among the caveats.

## Agreement

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

## Bucket confusion

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

## Disagreements

2 disagreements over 2 rows. This list is the revision plan for `GUIDELINES.md` v2. Where one shows the
**first** annotator was wrong, the fix goes into `v1.jsonl` through #59's log — and the number
above stays as it was measured, before the fix.

| id | field | annotator 1 | annotator 2 | rule cited | note |
| --- | --- | --- | --- | --- | --- |
| `v1-0043` | `answer_correct` | `yes` | `no` | — | — |
| `v1-3014` | `answer_correct` | `yes` | `no` | — | — |

### `v1-0043` — `answer_correct`

**Q.** what are the prerequisites for the museum internship course?

Annotator 1 says `yes`, annotator 2 says `no`.

No rule cited — **attribute one by hand before #62**: a disagreement whose guideline rule is unidentified cannot be fixed by editing the guidelines.

### `v1-3014` — `answer_correct`

**Q.** what are the prerequisites for 597?

Annotator 1 says `yes`, annotator 2 says `no`.

No rule cited — **attribute one by hand before #62**: a disagreement whose guideline rule is unidentified cannot be fixed by editing the guidelines.

## What this does and does not measure

**n=15.** A fifteen-item agreement estimate is wide, and wider than the usual rule of thumb
suggests. 14 of 15 is 93.3% with a Wilson interval of **[70%, 99%]** — 23 points of room below the
estimate, not the ±13 a normal approximation gives at that count (and the ±25 quoted for n=15 is
the p̂ = 0.5 worst case). 13 of 15 reaches down to 62%. So this measurement catches a guidelines
document that does not work *at all*; it does not distinguish 85% from 95%, and no point estimate
from the table above should be quoted without its interval beside it.

**κ on `answer_correct` and `citations_support` is not a reliability number, for two different
reasons.** `v1.jsonl` holds only rows the first annotator accepted or edited — that is what
promotion means — so the fields being scored are the fields the set was filtered on.

On `answer_correct` their label is constant `yes`, `p_e` collapses onto `p_o`, and κ is exactly 0
or undefined however well the two annotators agree. On `citations_support` their label is not
constant but it is mechanical: `n/a` exactly where the row carries no gold ids, which the second
annotator reads off the same empty list. Those rows agree by construction and carry the
field-level number up with them, which is why the table also gives `citations_support` over the
cited rows only — and on that subset annotator 1 is a constant `yes` again.

What the percent agreement on both fields estimates is **residual label error in the frozen set**:
how often a second reader would reject a row Day 4 is about to score. That is worth knowing, and
it is a different quantity from inter-annotator reliability. A reliability κ on these fields needs
the sample drawn from `eval/datasets/drafts/` before promotion, where annotator 1's verdicts vary.

**The bucket number is narrower than four-way.** See the note under the confusion matrix: the
candidate answer's wording carries `should_abstain`, so the blind part of the bucket call is the
two within-pair distinctions.

**The two-pass order is an honour system.** Nothing here can prove `agreement_citations.md` was
opened after the sheet's first two columns were filled and saved. The `note` column is where a
changed pass-1 answer is recorded, and a sheet that records none may simply mean there were none.

**One annotator, one sitting.** 15 rows read by one other person is a reliability estimate for
*this pair of readers*, not for the guidelines in general. Three annotators and Fleiss' κ would be
the honest version of this measurement; that is not what #60 asks for, and this number should not
be quoted as if it were.

**The guidelines did not move during annotation.** Questions in `agreement_questions.md` are
answered by amending `eval/GUIDELINES.md` afterwards. The page this was measured against is
sha256 `75ec993375ada0e1…`; if #62 reports a different one, the amendment happened after the
measurement, which is the intended order.

## Reproducing this

```bash
uv run python -m eval.scripts.agreement --seed 42 --force   # redraws the sample
uv run python -m eval.scripts.agreement --score
```

The sample is a function of the seed, the id set and the allocation alone — per-bucket streams over
sorted ids, so adding or deleting a row in one bucket does not move another bucket's draw. The
report carries no timestamp: nothing in it changes unless the sheet or the dataset did.
