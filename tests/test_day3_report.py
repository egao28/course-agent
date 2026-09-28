"""Tests for eval/scripts/day3_report.py: the check, the six sections, and what may not be in them.

The thing under test is a report generator, so the fixture is the whole of Day 3 in ``tmp_path``:
four draft files with the provenance their three builders record, a review log with an accept, an
edit and a delete in it, the promoted dataset, #61's freeze and sidecar, and #60's packet with a
filled sheet that disagrees about one row. Nineteen rows rather than 120 -- the arithmetic has to
be checkable by reading the fixture, and a synthetic 120-row set would only make the expected
numbers longer.

Five groups.

The first is ``--check``, which is the only part of this tool that does anything today: it names
every artifact the report is built from and the command that produces each.

The second is the report over the fixture: every section #62 asks for, populated from the files
and not from the plan, and the headline sentence that has to agree with the verification table
under it.

The third is the Agreement section, which is #60's own rendering one heading level down. What is
asserted is that it *is* #60's -- the demoted string, byte for byte -- because the point of
reusing it is that the two reports cannot disagree about a number a reader will compare.

The fourth is the three properties the acceptance names and a reader cannot check by eye: no
clock anywhere in the body, a byte-identical re-run, and no local path. The test half is in there
too, by the only means available -- the module does not import ``load_split``.

The fifth is one test about the repository as it stands: none of the inputs exist, so no report
is committed, and that is the correct state rather than an omission.

**Nothing here writes into ``eval/datasets/`` or ``report/``.** The sources are a
:class:`~eval.scripts.day3_report.Sources` built in ``tmp_path``, and the one test that drives
``main`` monkeypatches both it and the report path -- a suite that wrote a review log or an
agreement report would be forging one, which is the rule ``tests/test_review.py`` and
``tests/test_agreement.py`` keep.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

from config import load_config, rel_to_repo
from eval.schema import BUCKETS, RULES, EvalItem, item_id, row_json
from eval.scripts import agreement, build_multihop, build_unanswerable, review, split, triage
from eval.scripts import day3_report as report
from eval.scripts.draft_items import ID_BLOCKS
from eval.scripts.draft_items import PROVENANCE_KEYS as DRAFTED_KEYS
from eval.validate import corpus_doc_ids

# --- a whole Day 3, in tmp_path -------------------------------------------------------------------

#: How many rows each bucket is drafted with. ``fact`` gets seven because one of them is deleted
#: at review and #60's allocation wants six to draw from.
DRAFTED = {"fact": 7, "multihop": 4, "unanswerable": 4, "ambiguous": 4}

#: Which draft file each bucket is written to. The names matter: #59 picks the provenance set a
#: file is held to off the prefix, so ``multihop_graph_`` is #58's and ``unanswerable_`` is #57's.
FILES = {
    "fact": "fact_abc12345.jsonl",
    "ambiguous": "ambiguous_abc12345.jsonl",
    "multihop": "multihop_graph_abc12345.jsonl",
    "unanswerable": "unanswerable_abc12345.jsonl",
}

#: Prose per bucket, formatted with the row's number so no two questions are the same -- two
#: identical questions are a validation error, and the promote step is what would report it.
PROSE = {
    "fact": ("how many credits is course {n}?", "Course {n} is 3 credit hours."),
    "multihop": ("what do I need before course {n}?", "Course {n} needs course {n} minus one."),
    "unanswerable": ("who teaches course {n}?", "This catalog does not carry instructors."),
    "ambiguous": ("what about {n}?", "Which course {n} do you mean -- the lab or the lecture?"),
}

#: Gold ids per bucket, inside what :data:`eval.schema.RULES` allows.
IDS = {"fact": 1, "multihop": 2, "unanswerable": 0, "ambiguous": 0}

#: The one edit reason and the one delete reason the log records, asserted by name below.
EDIT_REASON = "wording"
DELETE_REASON = "the question is answered by the record it cites"


@pytest.fixture(scope="module")
def known() -> frozenset[str]:
    """The real corpus. The rows are synthetic; their citations are not, because everything that
    reads a dataset here validates the gold ids against it."""
    return corpus_doc_ids()


@pytest.fixture(scope="module")
def doc_ids(known: frozenset[str]) -> list[str]:
    """Real ids, spread across the catalog's subjects so the coverage line has something to
    count, and sorted first so the fixture is the same set on every machine."""
    return sorted(known)[::71][:64]


@pytest.fixture
def cfg():
    return load_config()


def drafted_provenance(n: int, cfg, **overrides: Any) -> dict[str, Any]:
    """A #56 row's provenance, every key present, so #59's reader accepts the file."""
    provenance: dict[str, Any] = dict.fromkeys(DRAFTED_KEYS, None)
    provenance.update(
        {
            "batch": 1 + n % 2,
            "batch_doc_ids": [],
            "cited_doc_ids": [],
            "subjects": [],
            "model": "gpt-4o-mini",
            "prompt_version": "v1",
            "config_hash": cfg.hash,
            "seed": cfg["seed"],
            "overlap": round(0.10 + 0.02 * n, 3),
            "flagged": False,
            "credit_question": False,
        }
    )
    provenance.update(overrides)
    return provenance


def graph_provenance(n: int, cfg) -> dict[str, Any]:
    """A #58 row's provenance: a walk over the prerequisite graph, with its depth."""
    provenance: dict[str, Any] = dict.fromkeys(build_multihop.PROVENANCE_KEYS, None)
    provenance.update(
        {
            "shape": "chain",
            "source": build_multihop.SOURCE,
            "root": f"TEST {n}",
            "held": [],
            "depth": 2 + n % 3,
            "levels": 2 + n % 3,
            "closure": 2 + n % 3,
            "cycle_members": [],
            "raw_agrees": True,
            "disagreements": [],
            "note": "a chain",
            "config_hash": cfg.hash,
        }
    )
    return provenance


def constructed_provenance(n: int, cfg) -> dict[str, Any]:
    """A #57 row's provenance: one of the four failure shapes, and its gate reading."""
    shapes = list(build_unanswerable.SHAPES)
    provenance: dict[str, Any] = dict.fromkeys(build_unanswerable.PROVENANCE_KEYS, None)
    provenance.update(
        {
            "shape": shapes[n % len(shapes)],
            "source": build_unanswerable.SOURCE,
            "codes": [],
            "note": "constructed",
            "config_hash": cfg.hash,
            "retrieval_mode": "bm25",
            "top_score": 9.5 + n,
            "top_doc_id": None,
            "code_scores": {},
            "gate": 15.0,
            "above_gate": n == 0,
        }
    )
    return provenance


def make_row(bucket: str, n: int, cfg, doc_ids: Sequence[str]) -> dict[str, Any]:
    """One draft row: the six dataset fields, then the provenance its builder records."""
    question, answer = PROSE[bucket]
    number = ID_BLOCKS[bucket] + n
    item = EvalItem(
        id=item_id(number),
        bucket=bucket,
        question=question.format(n=number),
        gold_answer=answer.format(n=number),
        gold_doc_ids=list(doc_ids[n : n + IDS[bucket]]),
        should_abstain=RULES[bucket].should_abstain,
    )
    provenance = {
        "multihop": graph_provenance,
        "unanswerable": constructed_provenance,
    }.get(bucket, lambda n, cfg: drafted_provenance(n, cfg))(n, cfg)
    return {**row_json(item), "provenance": provenance}


def write_drafts(directory: Path, cfg, doc_ids: Sequence[str]) -> Path:
    """The four draft files and the two sidecars #56 writes beside its own."""
    directory.mkdir(parents=True, exist_ok=True)
    for bucket, name in FILES.items():
        rows = [make_row(bucket, n, cfg, doc_ids) for n in range(DRAFTED[bucket])]
        (directory / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
        meta: dict[str, Any] = {"bucket": bucket, "config_hash": cfg.hash, "rows": len(rows)}
        if bucket in {"fact", "ambiguous"}:
            meta.update(
                {
                    "seed": cfg["seed"],
                    "batch_size": 10,
                    "corpus": {"chunks": 4502, "subjects": 187},
                    "batches": [{"batch": 1, "model": "gpt-4o-mini"}, {"batch": 2}],
                }
            )
        (directory / name.replace(".jsonl", ".meta.json")).write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return directory


def write_log(path: Path, candidates: Sequence[review.Candidate]) -> Path:
    """A review log over every drafted row: two edits, one deleted ``fact`` row, the rest accepted.

    Through :func:`eval.scripts.review.append_verdict`, so the file has the shape #59 writes
    rather than the shape this test imagines. The timestamps are fixed, since a report that
    printed one would make this suite race a clock.
    """
    by_bucket = {
        bucket: [c for c in candidates if c.item.bucket == bucket] for bucket in BUCKETS
    }
    edited = {by_bucket["fact"][1].id, by_bucket["unanswerable"][0].id}
    deleted = {by_bucket["fact"][0].id}
    for n, candidate in enumerate(candidates):
        ts = f"2026-09-21T12:00:{n:02d}.000000Z"
        if candidate.id in deleted:
            verdict = review.Verdict(candidate.id, "delete", DELETE_REASON, ts)
        elif candidate.id in edited:
            answer = candidate.item.gold_answer
            verdict = review.Verdict(
                candidate.id,
                "edit",
                EDIT_REASON,
                ts,
                (review.Edit("gold_answer", answer, answer + " (checked)"),),
            )
        else:
            verdict = review.Verdict(candidate.id, "accept", None, ts)
        review.append_verdict(path, verdict)
    return path


def fill_sheet(pack: agreement.Packet, sample: dict[str, EvalItem]) -> Path:
    """The returned sheet: the first annotator's labels, with one bucket call flipped.

    By rewriting the exported file rather than writing one, which is what
    ``tests/test_agreement.py`` does and for the same reason: what comes back from a person is
    the file they were sent, with five columns typed into it.
    """
    lines = pack.sheet.read_text(encoding="utf-8").splitlines()
    out = [lines[0]]
    disagree = min(item.id for item in sample.values() if item.bucket == "fact")
    for line in lines[1:]:
        cells = line.split("\t")
        labels = agreement.first_annotator(sample[cells[0]])
        bucket = "multihop" if cells[0] == disagree else labels["bucket"]
        rule = "F1" if cells[0] == disagree else ""
        note = "reads as a join to me" if cells[0] == disagree else ""
        out.append(
            "\t".join(
                [
                    *cells[:3],
                    bucket,
                    labels["answer_correct"],
                    labels["citations_support"],
                    rule,
                    note,
                ]
            )
        )
    pack.filled.write_text("\n".join(out) + "\n", encoding="utf-8")
    return pack.filled


@pytest.fixture
def src(tmp_path: Path, cfg, known: frozenset[str], doc_ids: list[str]) -> report.Sources:
    """Day 3, complete: drafts, a log, the promoted dataset, the freeze, and a filled sheet."""
    datasets = tmp_path / "datasets"
    drafts = write_drafts(datasets / "drafts", cfg, doc_ids)
    candidates, found = review.load_candidates(review.find_drafts(drafts))
    assert not found, [f.violation.message for f in found]

    log = write_log(datasets / "review_log.jsonl", candidates)
    promotion = review.promote(candidates, review.read_verdicts(log), known)
    assert promotion.ok, review.promotion_report(promotion, datasets / "v1.jsonl")
    dataset = review.write_dataset(datasets / "v1.jsonl", promotion.rows)
    assert split.freeze(cfg, dataset, cfg["eval"]["split_seed"], StringIO(), known=known) == 0

    pack = agreement.packet(datasets)
    texts = {one: f"{one} — a course record." for row in promotion.rows for one in row.gold_doc_ids}
    assert agreement.export(cfg, dataset, 42, pack, texts=texts, out=StringIO()) == 0
    fill_sheet(pack, {item.id: item for item in agreement.load_dataset(pack.sample)})
    return report.Sources(
        dataset=dataset,
        drafts=drafts,
        log=log,
        packet=pack,
        # Named, and pointed at a file that does not exist: the default is the committed
        # eval/datasets/triage.jsonl, and a fixture that fell through to it would report on a
        # real model-assisted pass and fail the moment one happened.
        triage=datasets / "triage.jsonl",
    )


@pytest.fixture
def facts(cfg, src: report.Sources) -> report.Facts:
    return report.read_facts(cfg, src)


@pytest.fixture
def body(facts: report.Facts) -> str:
    return report.render_report(facts)


# --- the check ------------------------------------------------------------------------------------


def test_check_lists_every_input_and_what_produces_it(cfg, tmp_path: Path):
    """Nothing exists: eight lines, a command under each, and a non-zero exit."""
    empty = report.sources(
        dataset=tmp_path / "v1.jsonl",
        drafts=tmp_path / "drafts",
        log=tmp_path / "review_log.jsonl",
        packet=agreement.packet(tmp_path),
        advisory=tmp_path / "triage.jsonl",
    )
    out = StringIO()
    assert report.render_check(cfg, empty, out) == 1
    printed = out.getvalue()
    for one in report.inputs(empty):
        assert one.name in printed
        assert one.how in printed
    assert "7 inputs missing" in printed


def test_check_says_how_much_of_the_plan_is_drafted(cfg, src: report.Sources):
    """The gate on everything else, so it is a line of its own rather than a row in the table."""
    out = StringIO()
    assert report.render_check(cfg, src, out) == 0
    drafted = sum(DRAFTED.values())
    assert f"4 draft files in {rel_to_repo(src.drafts)}, {drafted} of 120 rows drafted" in (
        out.getvalue()
    )
    assert "Every input is present" in out.getvalue()


def test_read_facts_refuses_and_names_the_files_it_wanted(cfg, src: report.Sources):
    """A report is not written from six of seven artifacts, and the message says which is gone."""
    src.dataset.unlink()
    with pytest.raises(ValueError, match=r"v1\.jsonl"):
        report.read_facts(cfg, src)


def test_a_sheet_that_cannot_be_scored_stops_the_report(cfg, src: report.Sources):
    """#60 reports every fault in a sheet at once; this hands the reader to it, not past it."""
    lines = src.packet.filled.read_text(encoding="utf-8").splitlines()
    cells = lines[1].split("\t")
    cells[4] = "probably"
    lines[1] = "\t".join(cells)
    src.packet.filled.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="agreement --score"):
        report.read_facts(cfg, src)


# --- the six sections -----------------------------------------------------------------------------


def test_every_section_the_issue_asks_for_is_present(body: str):
    """Composition, provenance, verification, agreement, split, limitations -- and reproduction."""
    for heading in (
        "## Composition",
        "## Provenance",
        "## Verification",
        "## Agreement",
        "## Split",
        "## Limitations",
        "## Reproduction",
    ):
        assert heading in body


def test_the_headline_sentence_is_the_one_the_issue_names(facts: report.Facts, body: str):
    """N drafted, M reviewed, K edited, J deleted -- and it is one function, so it appears twice."""
    drafted = sum(DRAFTED.values())
    sentence = f"**{drafted} rows drafted, {drafted} reviewed by hand, 2 edited, 1 deleted**"
    assert body.count(sentence) == 2
    assert f"leaving {drafted - 1} in" in body


def test_composition_counts_the_dataset_and_not_the_plan(facts: report.Facts, body: str):
    """``target`` is #54's table, ``final`` is what verification left, bolded when they differ."""
    assert f"| `fact` | {RULES['fact'].target_n} | **6** |" in body  # 7 drafted, 1 deleted
    assert f"| `ambiguous` | {RULES['ambiguous'].target_n} | **4** |" in body
    assert f"| **total** | 120 | **{sum(DRAFTED.values()) - 1}** |" in body
    for bucket in BUCKETS:
        assert RULES[bucket].gloss in body


def test_composition_reads_the_two_distributions_off_the_rows(body: str):
    """Question length per bucket, and #58's depth -- which only the draft files know."""
    assert "**Question length**, in words." in body
    assert "**Multi-hop depth**" in body
    assert "| depth | rows |" in body


def test_the_unanswerable_shapes_come_from_provenance(facts: report.Facts, body: str):
    """#57's four claims, against the table sizes that file is built to."""
    assert "**`unanswerable` failure shapes**" in body
    for name, shape in build_unanswerable.SHAPES.items():
        assert f"| `{name}` | {shape.n} |" in body
        assert shape.gloss in body
    assert "1 of 4 score above the evidence gate" in body


def test_provenance_names_the_model_the_prompt_and_the_seed(facts: report.Facts, body: str):
    """Which model drafted which rows -- and that two of the four files had no model at all."""
    assert "`gpt-4o-mini`" in body
    assert f"`{build_multihop.SOURCE}`" in body or build_multihop.SOURCE in body
    assert build_unanswerable.SOURCE in body
    assert "| `v1` |" in body
    assert "6/7" in body  # fact: one row deleted at review


def test_verification_reports_per_bucket_verdicts_and_the_reasons(body: str):
    """The table #59's ``--stats`` prints, plus the reasons, which are the useful half."""
    assert "| bucket | drafted | accept | edit | delete | unreviewed |" in body
    assert f"| `{EDIT_REASON}` | 2 |" in body
    assert DELETE_REASON in body
    assert "**Did judgement drift?**" in body


def test_split_records_the_seed_the_digest_and_the_ceiling(facts: report.Facts, body: str):
    """#61's sidecar, printed where #62 was always going to read it from."""
    assert f"`{facts.sidecar['seed']}` (`eval.split_seed`)" in body
    assert f"`{facts.sidecar['sha256']}`" in body
    n_calib = facts.sidecar["conformal"]["n_calib"]
    assert f"bounds α below at 1/{n_calib + 1}" in body
    assert "coverage is the ceiling" in body
    assert "re-derives from the seed, and the digest matches" in body


def test_limitations_are_computed_and_not_recited(facts: report.Facts, body: str):
    """Each of the six is sized from this run: the share double-annotated, the interval, the
    overlap."""
    section = body.split("## Limitations", 1)[1]
    assert section.count("\n- **") == 6
    assert "**One annotator on 3 of 18 rows.**" in section
    assert "83.3% were double-annotated" in section
    assert "`gpt-4o-mini`" in section and "is the model under test" in section
    assert "n 10 min 0.100" in section  # the residual question/chunk overlap, over #56's rows
    assert "`fact` 3 rows ±" in section
    assert "2026-sp" in section and "4,502 courses" in section
    assert "must condition on `bucket`" in section


def test_an_unassisted_pass_gets_no_assistance_section_or_bullet(body: str):
    """The caveat has to be absent exactly when it does not apply."""
    assert "model-assisted" not in body
    assert body.split("## Limitations", 1)[1].count("\n- **") == 6


@pytest.fixture
def assisted(src: report.Sources, cfg) -> report.Facts:
    """The same pass, with a machine's suggestion recorded for every row it judged.

    Two of them disagree with the verdict the fixture's log holds, so the agreement rate is a
    number and not 100%.
    """
    verdicts = review.read_verdicts(src.log)
    for n, (row_id, _) in enumerate(sorted(verdicts.items())):
        triage.append_triage(
            src.triage,
            triage.Triage(
                id=row_id,
                suggested="accept" if n % 7 else "delete",
                reason="the chunk settles it.",
                checks={
                    "answer_correct": "yes",
                    "citations_support": "yes",
                    "bucket": "fact",
                    "belongs": "yes",
                },
                model="gpt-4.1",
                guidelines_sha256="a" * 64,
                config_hash=cfg.hash,
            ),
        )
    return report.read_facts(cfg, src)


def test_an_assisted_pass_is_disclosed_and_measured(assisted: report.Facts):
    """#62 may not report an accept rate from an assisted pass without saying it was one."""
    body = report.render_report(assisted)
    assert "**The pass was model-assisted.**" in body
    assert "`gpt-4.1`" in body
    assert "| suggested ↓ / judged → | accept | edit | delete |" in body
    assert "upper bound" in body
    section = body.split("## Limitations", 1)[1]
    assert section.count("\n- **") == 7
    assert "anchors a" in section
    assert "no suggestion reached the review log" in section


def test_the_agreement_rate_counts_the_rows_both_judged(assisted: report.Facts):
    """Every judged row has a suggestion in this fixture, and two of them disagree."""
    pairs = [
        (assisted.advisory[one].suggested, verdict.verdict)
        for one, verdict in assisted.verdicts.items()
        if one in assisted.advisory
    ]
    agreed = sum(1 for a, b in pairs if a == b)
    body = report.render_report(assisted)
    assert f"They agreed on {agreed} of {len(pairs)} rows" in body
    assert agreed < len(pairs), "the fixture is meant to disagree somewhere"


def test_the_advisory_file_is_an_optional_input(cfg, src: report.Sources):
    """--check lists it, and its absence does not stop the report."""
    out = StringIO()
    assert report.render_check(cfg, src, out) == 0
    assert "machine triage" in out.getvalue()
    assert not any(one.required for one in report.inputs(src) if one.name == "machine triage")


# --- the agreement section is #60's own -----------------------------------------------------------


def test_the_agreement_tables_are_the_ones_agreement_py_renders(facts: report.Facts, body: str):
    """Byte for byte, one level down. Two renderings of one number is two reports that disagree."""
    pairs = agreement.pairs_for("bucket", facts.judged, facts.sample)
    assert report.demote(agreement.render_confusion(pairs)) in body
    found = agreement.disagreements(facts.judged, facts.sample)
    assert found, "the fixture flips one bucket call, so there is one disagreement to itemise"
    assert report.demote(agreement.render_disagreements(found, len(facts.judged))) in body


def test_the_agreement_section_sits_under_one_heading(body: str):
    """Demoted, so #60's ``## Agreement`` does not become a second top-level section here."""
    assert body.count("\n## Agreement") == 1
    assert "\n### Agreement" in body
    assert "\n### Bucket confusion" in body
    assert "\n### Disagreements" in body


def test_demote_moves_headings_and_leaves_sentences_alone():
    """``#59's log`` at the start of a line is a sentence; ``## Agreement`` is a heading."""
    given = "## Agreement\n\n#59's log is the evidence.\n\n### One row\n#\n"
    want = "### Agreement\n\n#59's log is the evidence.\n\n#### One row\n#\n"
    assert report.demote(given) == want


def test_the_share_double_annotated_is_stated_where_the_numbers_are(facts: report.Facts, body: str):
    """The interval, not the point estimate, is what fifteen of a hundred and twenty supports."""
    section = body.split("## Agreement", 1)[1].split("## Split", 1)[0]
    assert f"{len(facts.judged)} rows of {len(facts.rows)}" in section
    assert "95% CI (Wilson)" in section


# --- what may not be in it ------------------------------------------------------------------------


def test_there_is_no_clock_anywhere_in_the_body(facts: report.Facts, body: str):
    """#59's log carries the timestamps and this report prints none of them."""
    assert not re.search(r"\d{4}-\d\d-\d\dT\d\d:\d\d", body)
    for entry in facts.entries:
        assert entry["ts"] not in body


def test_the_report_regenerates_byte_identically(facts: report.Facts, body: str):
    """The acceptance line: unchanged inputs, unchanged file, so a diff means an artifact moved."""
    assert report.render_report(facts) == body


def test_no_local_path_reaches_the_file(src: report.Sources, body: str):
    """Every path goes through ``rel_to_repo``, so a path inside the repository is written
    relative to it and no home directory reaches a tracked file."""
    assert str(Path.home()) not in body
    assert "`eval/GUIDELINES.md`" in body


def test_the_test_half_is_never_loaded():
    """The one property no fixture can show: ``load_split`` is not reachable from this module.

    Asserted over the source because that is what the acceptance is about -- no metric in this
    report is computed on the test half, and the way that stays true is that the module counts
    the file and never asks for a half of it.
    """
    source = Path(report.__file__).read_text(encoding="utf-8")
    assert "load_split" not in source.replace("``load_split`` is deliberately not imported", "")


def test_main_writes_the_report_and_prints_the_summary(
    cfg, src: report.Sources, tmp_path: Path, monkeypatch, capsys
):
    """End to end, with the sources and the report path pointed away from the repository."""
    written = tmp_path / "day3_evalset_test.md"
    monkeypatch.setattr(report, "sources", lambda **kwargs: src)
    monkeypatch.setattr(report, "report_path", lambda stem, cfg: written)
    assert report.main([]) == 0
    assert written.read_text(encoding="utf-8").startswith("# Day 3 —")
    printed = capsys.readouterr().out
    assert f"drafted                {sum(DRAFTED.values())}" in printed
    assert "wrote" in printed


def test_dry_run_writes_nothing(cfg, src: report.Sources, tmp_path: Path, monkeypatch, capsys):
    """The report on stdout, and no file -- #61's and #60's flag, with the same meaning."""
    written = tmp_path / "day3_evalset_test.md"
    monkeypatch.setattr(report, "sources", lambda **kwargs: src)
    monkeypatch.setattr(report, "report_path", lambda stem, cfg: written)
    assert report.main(["--dry-run"]) == 0
    assert not written.exists()
    assert "## Limitations" in capsys.readouterr().out


# --- the repository as it stands ------------------------------------------------------------------


def test_no_day3_report_is_committed_because_the_pass_has_not_run(cfg):
    """The status this branch ships in, asserted rather than described.

    ``v1.jsonl`` is #59's ``--promote`` output; 70 of the 120 rows are undrafted and no row has
    been verified. Until they are, this report cannot be populated from committed artifacts, and
    a file under ``report/`` would be the claim that it was.
    """
    gaps = report.missing(report.sources())
    assert gaps, "inputs have landed -- write the report and drop this test"
    assert not report.report_path(report.STEM, cfg).exists()
    assert report.main(["--check"]) == 1

