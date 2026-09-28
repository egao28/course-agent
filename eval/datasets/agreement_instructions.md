# Second annotator — 15 items

15 rows from `eval/datasets/v1.jsonl`, drawn with seed `42`. You are the second annotator: another person
has already labelled these rows, and you are not being shown their labels. Please do not ask for
them, and please do not open `agreement_sample.jsonl` — it is the answer key.

## Read first, and read only this

`eval/GUIDELINES.md`. One page. It is the whole instruction, and that is the point of the
exercise: what is being measured is whether that page is enough for someone who did not write it,
so anything you have to ask is a finding and not a nuisance.

Log every question in `agreement_questions.md`, including the ones you answer yourself. They are the evidence
for what the guidelines leave open, and they are answered by amending the page *after* the
agreement is measured — so a question you asked and worked around is worth more here than a
question you asked and had answered.

## Two passes, in this order

The order is a control, not bureaucracy. The cited documents nearly give the bucket away — a row
with no citations is almost always a refusal, a row with three is almost always a join — so a
bucket judged after reading them measures very little.

**Pass 1 — `agreement_sheet.tsv`, without opening `agreement_citations.md`.** Fill in `bucket` and `answer_correct` for
all 15 rows, and save the file.

- `bucket` — one of `fact`, `multihop`, `unanswerable`, `ambiguous`. `GUIDELINES.md` → *Which bucket*, and the eight boundary cases.
- `answer_correct` — `yes` or `no`. Is the candidate answer right? Where the candidate answer is a
  refusal or a clarifying question, the question is *is this the right thing to refuse (or to
  ask)?* — `GUIDELINES.md` → *When a refusal is correct*.

To check an answer, go to the source: read the chunk, or call `get_course`, `get_prereq_chain`,
`check_requirement`. Do **not** run the agent or `retrieve()` to decide a label — what the system
does with a question is the thing under test, and it cannot also be the thing that labels it.

**Pass 2 — `agreement_citations.md`, then the last judgement.** It prints, per row, the records the first
annotator cited, in full. Fill in `citations_support`:

- `yes` — every fact the candidate answer asserts is readable in the text of a cited record.
- `no` — it is not. An id that is merely *about* the right course is a `no`: `GUIDELINES.md`
  rule 4.
- `n/a` — the row cites nothing.

Changing a pass-1 answer after reading the citations is allowed. Say so in `note` if you do: that
is exactly the leak the two passes exist to measure, and it is more useful recorded than avoided.

## The last two columns

- `rule` — which part of `GUIDELINES.md` you applied: a boundary case (`F1`, `M2`, `U2`, `A1` …),
  a numbered rule (`rule 3`), or a section name. On a row that was obvious, leave it empty. On a
  row that was not, this is the most useful cell in the sheet — where we disagree it says which
  rule was ambiguous, instead of leaving us to guess which one you were reading.
- `note` — anything else, in your words.

## When you are done

Save the sheet as `agreement_sheet_filled.tsv` and hand it back. It is committed exactly as you leave it; nobody
tidies it up afterwards.

Do not edit the `id`, `question` or `candidate_answer` columns, and do not add or remove rows: the
scorer matches on `id`, and it refuses a sheet that has lost one rather than quietly scoring
fourteen.
