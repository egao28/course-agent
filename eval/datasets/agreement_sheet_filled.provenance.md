# How this sheet came back

`agreement_sheet_filled.tsv` was **not filled in by the second annotator.** They read the fifteen
rows and reported the outcome verbally to the first annotator, who relayed it; the file was then
written from that account. The two `answer_correct` disagreements (`v1-0043`, `v1-3014`) are
theirs. Everything else in the file is *"the rest agreed"*, transcribed rather than typed by them.

## What that costs, item by item

- **No rule attributions.** The `rule` column is empty on both disagreements. A disagreement that
  cannot be traced to a line of `eval/GUIDELINES.md` cannot be used to amend it, which is the
  main thing #60 draws this sample for. The report marks both as outstanding.
- **No notes.** Nothing in their words survives, so the report has the labels and not the
  reasoning. On `v1-0043` the first annotator's own reading of the row — the answer never names
  `MUSE 390`, against rule 1 — is a guess at their reasoning and is *not* recorded as theirs.
- **The two-pass order is unverified.** The blind sheet exists so that `bucket` is judged before
  the cited records are opened (a row with no citations is almost always a refusal, so the
  citations nearly give the bucket away). Whether that order was kept is not observable here.
- **Row-level agreement is an assertion, not a record.** "The rest agreed" is thirteen rows and
  thirty-nine cells that nobody typed. If they in fact skimmed some of those rows, this file
  cannot tell anyone.

## What it does not cost

The two disagreements are a real second reader's, on rows the first pass accepted, and both are
defensible on the guidelines' own terms. `answer_correct` therefore has variance in it for the
first time — against a promoted dataset it is otherwise a constant `yes`, which forces Cohen's κ
to exactly 0. That much is a genuine finding and is what this sample was for.

## What would replace it

The same fifteen rows, typed by them, with the `rule` column filled in. `--force` will not
overwrite a filled sheet, so move this file aside deliberately and re-export; the numbers move
and the report says which version it was computed on.
