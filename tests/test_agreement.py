"""Tests for eval/scripts/agreement.py: the draw, the blind sheet, the two numbers, the packet.

Six groups, and the order is the order the tool runs in.

The first three are over a **synthetic 120-row dataset** built in ``tmp_path`` -- the bucket counts
`eval/schema.py` targets, and nothing else real about it. What these check is arithmetic and
blinding, and a fixture keeps the expected numbers readable: an agreement of 12 of 15 has to be
12 of 15 because the test said so, not because a real dataset happened to hold that.

The fourth group is the one that is about this dataset rather than about the tool. ``v1.jsonl``
holds only rows the first annotator accepted, so their label on two of the three fields is a
constant, and Cohen's κ on those fields is 0 or undefined however well the annotators agree. That
is a property of the sampling frame and not of anyone's judgement, and it is asserted here so that
a later change which made κ look healthy on those fields would fail rather than be believed.

The fifth drives the two commands end to end over ``tmp_path`` packets. The sixth is one test
about the repository as it stands: ``eval/datasets/v1.jsonl`` does not exist yet, so the export
cannot have run, and that is why this branch commits no sample, no sheet and no report.

**Nothing here writes into ``eval/datasets/`` or ``report/``.** Every test passes its own
:class:`~eval.scripts.agreement.Packet` and its own report path. A filled sheet is a person's work
and an agreement report is a measurement; a test suite that wrote either would be forging them,
which is the same rule ``tests/test_review.py`` keeps about the review log.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from io import StringIO
from itertools import groupby
from math import comb
from pathlib import Path

import pytest

from config import load_config, rel_to_repo
from eval.schema import BUCKETS, RULES, EvalItem, item_id, row_json
from eval.scripts import agreement, review

# --- a dataset ------------------------------------------------------------------------------------

#: The shape of ``v1.jsonl`` as #54 targets it, so the draw's 6/3/3/3 has something to draw from.
COUNTS = {bucket: RULES[bucket].target_n for bucket in BUCKETS}

#: How many gold ids a row of each bucket carries, inside what :data:`eval.schema.RULES` allows.
IDS = {"fact": 1, "multihop": 2, "unanswerable": 0, "ambiguous": 0}

#: Prose per bucket. No bucket name and no ``true``/``false`` appears in any of it -- the blinding
#: test asserts the sheet contains no label, and a question that said "fact" would pass it falsely.
PROSE = {
    "fact": ("how many credits is TEST {n}?", "TEST {n} is 3 credit hours."),
    "multihop": ("what do I need before TEST {n}?", "TEST {n} needs TEST 100, which needs 99."),
    "unanswerable": ("who teaches TEST {n}?", "This catalog does not carry instructors."),
    "ambiguous": ("what about {n}?", "Which TEST {n} do you mean -- the lab or the lecture?"),
}


def make_item(n: int, bucket: str) -> EvalItem:
    """One synthetic row, valid against the schema and inside its bucket's rule."""
    question, answer = PROSE[bucket]
    return EvalItem(
        id=item_id(n),
        bucket=bucket,
        question=question.format(n=n),
        gold_answer=answer.format(n=n),
        gold_doc_ids=[f"2026-sp:TEST-{n:03d}-{k}" for k in range(IDS[bucket])],
        should_abstain=RULES[bucket].should_abstain,
    )


def make_rows(counts: dict[str, int] | None = None) -> list[EvalItem]:
    """120 rows in bucket order: ids are minted in pipeline order, as #56 mints them."""
    counts = COUNTS if counts is None else counts
    rows: list[EvalItem] = []
    for bucket in BUCKETS:
        for _ in range(counts[bucket]):
            rows.append(make_item(len(rows) + 1, bucket))
    return rows


def write_dataset(path: Path, rows: Sequence[EvalItem]) -> Path:
    """The rows as a JSONL file -- what :func:`agreement.load_dataset` reads."""
    path.write_text(
        "".join(json.dumps(row_json(row), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    return write_dataset(tmp_path / "v1.jsonl", make_rows())


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def pack(tmp_path: Path) -> agreement.Packet:
    directory = tmp_path / "datasets"
    directory.mkdir()
    return agreement.packet(directory)


def buckets_of(sample: agreement.Sample) -> list[str]:
    return [row.bucket for row in sample.rows]


# --- the draw -------------------------------------------------------------------------------------


def test_the_allocation_is_the_one_in_the_issue():
    """6 / 3 / 3 / 3, fifteen in total. A count that drifted would move every interval below."""
    assert agreement.ALLOCATION == {
        "fact": 6,
        "multihop": 3,
        "unanswerable": 3,
        "ambiguous": 3,
    }
    assert agreement.SAMPLE_N == 15


def test_the_draw_is_stratified(dataset: Path):
    """Every bucket contributes its allocation -- which is the point of stratifying at all."""
    sample = agreement.stratified_sample(agreement.load_dataset(dataset), seed=42)
    drawn = {draw.bucket: draw.got for draw in sample.draws}
    assert drawn == dict(agreement.ALLOCATION)
    assert len(sample.rows) == 15
    assert not sample.short


def test_uniform_sampling_is_what_the_allocation_avoids(dataset: Path):
    """The premise, not the tool: 15 of 120 drawn uniformly does not reliably see ``ambiguous``.

    Exactly rather than empirically -- the hypergeometric probabilities over a 120-row set with
    20 ``ambiguous`` rows -- because the interesting number is close enough to a half that a
    200-seed estimate of it lands on either side of any threshold worth asserting.

    A uniform draw misses the bucket outright 5.4% of the time and brings back fewer than three
    of it 52.6% of the time. The stratified draw takes three, at every seed.
    """
    rows = agreement.load_dataset(dataset)
    n_ambiguous = sum(1 for row in rows if row.bucket == "ambiguous")
    assert (n_ambiguous, len(rows)) == (20, 120)

    def hypergeometric(k: int) -> float:
        return comb(n_ambiguous, k) * comb(len(rows) - n_ambiguous, 15 - k) / comb(len(rows), 15)

    assert hypergeometric(0) == pytest.approx(0.054, abs=0.001)
    assert sum(hypergeometric(k) for k in range(3)) == pytest.approx(0.526, abs=0.001)
    for seed in range(50):
        sample = agreement.stratified_sample(rows, seed=seed)
        assert buckets_of(sample).count("ambiguous") == 3


def test_the_draw_is_reproducible_from_the_seed(dataset: Path):
    rows = agreement.load_dataset(dataset)
    first = agreement.stratified_sample(rows, seed=42)
    again = agreement.stratified_sample(rows, seed=42)
    assert [row.id for row in first.rows] == [row.id for row in again.rows]
    assert first.seed == 42


def test_the_draw_is_reproducible_from_the_ids_and_not_the_line_order(dataset: Path):
    """Shuffling the file changes nothing: the pools are sorted before they are sampled."""
    rows = agreement.load_dataset(dataset)
    shuffled = list(reversed(rows))
    assert sorted(row.id for row in agreement.stratified_sample(rows, 42).rows) == sorted(
        row.id for row in agreement.stratified_sample(shuffled, 42).rows
    )


def test_another_seed_draws_another_sample(dataset: Path):
    rows = agreement.load_dataset(dataset)
    assert {row.id for row in agreement.stratified_sample(rows, 42).rows} != {
        row.id for row in agreement.stratified_sample(rows, 43).rows
    }


def test_one_buckets_pool_does_not_move_another_buckets_draw(dataset: Path):
    """The reason each bucket gets its own stream, asserted.

    Delete ten ``fact`` rows and the ``fact`` draw changes, as it must. Nothing else does. One
    shared stream would move all four, so a row deleted during verification would silently
    re-draw the other three buckets and "reproducible from the recorded seed" would hold only
    while the dataset never changed -- which is exactly when it is not needed.
    """
    rows = agreement.load_dataset(dataset)
    fewer = [row for row in rows if row.bucket != "fact"] + [
        row for row in rows if row.bucket == "fact"
    ][:40]
    before = {d.bucket: d.ids for d in agreement.stratified_sample(rows, 42).draws}
    after = {d.bucket: d.ids for d in agreement.stratified_sample(fewer, 42).draws}
    assert before["fact"] != after["fact"]
    for bucket in ("multihop", "unanswerable", "ambiguous"):
        assert before[bucket] == after[bucket]


def test_the_sheet_order_is_not_grouped_by_bucket(dataset: Path):
    """A sheet whose first six rows are all ``fact`` hands over the stratification.

    Asserted as runs rather than as an exact order: what matters is that the buckets interleave,
    not which permutation a given seed produces.
    """
    rows = agreement.load_dataset(dataset)
    for seed in range(20):
        sample = agreement.stratified_sample(rows, seed=seed)
        runs = len(list(groupby(buckets_of(sample))))
        assert runs > len(BUCKETS), f"seed {seed} drew the sample in blocks"


def test_a_short_bucket_is_recorded_and_not_raised(tmp_path: Path):
    """Verification deletes rows. A bucket that cannot fill its share says so and carries on."""
    counts = {**COUNTS, "ambiguous": 1}
    path = write_dataset(tmp_path / "short.jsonl", make_rows(counts))
    sample = agreement.stratified_sample(agreement.load_dataset(path), seed=42)
    assert len(sample.rows) == 13
    short = {draw.bucket: (draw.wanted, draw.got) for draw in sample.short}
    assert short == {"ambiguous": (3, 1)}


def test_a_missing_dataset_names_the_work_that_has_to_happen_first():
    with pytest.raises(ValueError) as exc:
        agreement.load_dataset(Path("eval/datasets/nope.jsonl"))
    message = str(exc.value)
    assert "eval.scripts.review" in message
    assert "--promote" in message


def test_a_dataset_that_is_not_one_is_refused(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "v1-0001", "bucket": "banana"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="bad row"):
        agreement.load_dataset(path)


# --- the sheet ------------------------------------------------------------------------------------


@pytest.fixture
def sample(dataset: Path) -> agreement.Sample:
    return agreement.stratified_sample(agreement.load_dataset(dataset), seed=42)


def flat(text: str) -> str:
    """Whitespace collapsed -- for asserting on report prose without minding where it wrapped."""
    return " ".join(text.split())


def sheet_rows(path: Path) -> list[list[str]]:
    """The returned sheet as cells, header dropped."""
    return [line.split("\t") for line in path.read_text(encoding="utf-8").splitlines()[1:]]


def fill(
    pack: agreement.Packet,
    judge: Callable[[str], Sequence[str]],
    sheet: Path | None = None,
) -> Path:
    """Fill the exported sheet with ``judge(id) -> (bucket, answer_correct, support, rule, note)``.

    Deliberately by rewriting the exported file rather than by writing a sheet from scratch: what
    comes back from a person is the file they were sent with five columns typed into it, and a
    test that built its own would not notice an export that wrote the wrong header.
    """
    sheet = pack.filled if sheet is None else sheet
    lines = pack.sheet.read_text(encoding="utf-8").splitlines()
    out = [lines[0]]
    for line in lines[1:]:
        cells = line.split("\t")
        out.append("\t".join([*cells[:3], *judge(cells[0])]))
    sheet.write_text("\n".join(out) + "\n", encoding="utf-8")
    return sheet


def agreeing(sample: agreement.Sample) -> Callable[[str], Sequence[str]]:
    """A second annotator who reproduces the first annotator's labels exactly."""
    labels = {row.id: agreement.first_annotator(row) for row in sample.rows}
    return lambda item_id: (*(labels[item_id][f] for f in agreement.FIELDS), "", "")


def exported(
    cfg, dataset: Path, pack: agreement.Packet, capsys=None, **kwargs
) -> agreement.Sample:
    """Run the export over a synthetic corpus and hand back the sample it drew."""
    rows = agreement.load_dataset(dataset)
    seed = kwargs.pop("seed", 42)
    sample = agreement.stratified_sample(rows, seed=seed)
    texts = {
        doc_id: f"{doc_id}\nCredits: 3\nPrerequisites: None listed."
        for row in rows
        for doc_id in row.gold_doc_ids
    }
    code = agreement.export(cfg, dataset, seed, pack, texts=texts, **kwargs)
    assert code == 0
    return sample


def test_the_sheet_carries_no_label(cfg, dataset: Path, pack: agreement.Packet):
    """``bucket``, ``should_abstain`` and the gold ids are all absent from the exported sheet.

    The three of them, not just the two #60 names: the id *count* per bucket is fixed tightly
    enough by :data:`eval.schema.RULES` -- exactly zero for ``unanswerable``, two or more for
    ``multihop`` -- that a sheet listing them labels most of the rows by arithmetic.
    """
    sample = exported(cfg, dataset, pack)
    body = pack.sheet.read_text(encoding="utf-8")
    header, rest = body.split("\n", 1)
    assert header.split("\t") == list(agreement.SHEET_COLUMNS)
    for bucket in BUCKETS:
        assert bucket not in rest
    for row in sample.rows:
        for doc_id in row.gold_doc_ids:
            assert doc_id not in rest
    assert "true" not in rest.casefold()


def test_the_sheet_has_one_row_per_sampled_row_and_five_empty_columns(
    cfg, dataset: Path, pack: agreement.Packet
):
    sample = exported(cfg, dataset, pack)
    rows = sheet_rows(pack.sheet)
    assert [cells[0] for cells in rows] == [row.id for row in sample.rows]
    for cells in rows:
        assert cells[1] and cells[2]
        assert cells[3:] == ["", "", "", "", ""]


def test_cell_collapses_whitespace():
    """The one thing TSV cannot survive is a tab inside a cell, so nothing written can hold one."""
    assert agreement.cell("a\tb\n c  d ") == "a b c d"


def test_a_filled_sheet_reads_back_as_it_was_filled(cfg, dataset: Path, pack: agreement.Packet):
    sample = exported(cfg, dataset, pack)
    fill(pack, agreeing(sample))
    judged, problems = agreement.read_sheet(pack.filled, tuple(row.id for row in sample.rows))
    assert not problems
    assert len(judged) == 15
    for row in judged:
        assert row.labels == agreement.first_annotator(sample.by_id[row.id])


def test_the_reader_accepts_the_spellings_a_person_types(
    cfg, dataset: Path, pack: agreement.Packet
):
    """``Yes``, ``N/A`` and ``y`` are not errors. A fifteen-row re-read over case is not a check."""
    sample = exported(cfg, dataset, pack)
    fill(pack, lambda _: ("FACT", "Yes", "N/A", "", ""))
    judged, problems = agreement.read_sheet(pack.filled, tuple(row.id for row in sample.rows))
    assert not problems
    assert {row.labels["bucket"] for row in judged} == {"fact"}
    assert {row.labels["answer_correct"] for row in judged} == {"yes"}
    assert {row.labels["citations_support"] for row in judged} == {"n/a"}


def test_the_reader_reports_every_fault_at_once(cfg, dataset: Path, pack: agreement.Packet):
    """Four faults, four problems, and the denominator is never quietly reduced.

    A reader that stopped at the first one would send a person back to their spreadsheet once per
    blank cell; a reader that dropped the bad rows instead would compute a percent agreement over
    twelve rows and call it fifteen.
    """
    sample = exported(cfg, dataset, pack)
    ids = [row.id for row in sample.rows]
    lines = pack.sheet.read_text(encoding="utf-8").splitlines()
    filled = [lines[0]]
    for n, line in enumerate(lines[1:]):
        cells = line.split("\t")
        if n == 0:
            cells[3] = "banana"
        elif n == 1:
            cells[3] = "fact"
        else:
            cells[3:] = ["fact", "yes", "n/a", "", ""]
        filled.append("\t".join(cells))
    del filled[3]
    filled.append("v1-9999\tq\ta\tfact\tyes\tn/a\t\t")
    pack.filled.write_text("\n".join(filled) + "\n", encoding="utf-8")
    judged, problems = agreement.read_sheet(pack.filled, tuple(ids))
    assert len(judged) == 12
    messages = {(p.id, p.column): p.message for p in problems}
    assert "one of fact, multihop" in messages[(ids[0], "bucket")]
    assert "is empty" in messages[(ids[1], "answer_correct")]
    assert messages[(ids[2], None)] == "is in the sample and not in the sheet"
    assert messages[("v1-9999", "id")] == "is not in the sample"


def test_an_edited_header_is_refused_before_any_row_is_read(
    cfg, dataset: Path, pack: agreement.Packet
):
    sample = exported(cfg, dataset, pack)
    fill(pack, agreeing(sample))
    body = pack.filled.read_text(encoding="utf-8").splitlines()
    body[0] = body[0].replace("citations_support", "citations")
    pack.filled.write_text("\n".join(body) + "\n", encoding="utf-8")
    judged, problems = agreement.read_sheet(pack.filled, tuple(row.id for row in sample.rows))
    assert not judged
    assert len(problems) == 1
    assert "it must be exactly" in problems[0].message


def test_a_row_that_lost_a_tab_is_reported_as_one(cfg, dataset: Path, pack: agreement.Packet):
    sample = exported(cfg, dataset, pack)
    fill(pack, agreeing(sample))
    body = pack.filled.read_text(encoding="utf-8").splitlines()
    body[1] = body[1].replace("\t", " ", 1)
    pack.filled.write_text("\n".join(body) + "\n", encoding="utf-8")
    _, problems = agreement.read_sheet(pack.filled, tuple(row.id for row in sample.rows))
    assert any("columns" in p.message for p in problems)


def test_a_row_judged_twice_is_reported(cfg, dataset: Path, pack: agreement.Packet):
    sample = exported(cfg, dataset, pack)
    fill(pack, agreeing(sample))
    body = pack.filled.read_text(encoding="utf-8").splitlines()
    body.append(body[1])
    pack.filled.write_text("\n".join(body) + "\n", encoding="utf-8")
    _, problems = agreement.read_sheet(pack.filled, tuple(row.id for row in sample.rows))
    assert any("already judged" in p.message for p in problems)


# --- the two numbers ------------------------------------------------------------------------------


def test_wilson_cannot_leave_the_unit_interval():
    """The reason it is Wilson and not p ± 1.96·se: 15 of 15 must not have an upper bound of 1.2.

    The upper bound at full agreement is exactly 1 -- ``centre + half`` collapses to
    ``(1 + z²/n)/(1 + z²/n)`` -- and the lower bound does not collapse with it, which is the other
    half of the point: 15 of 15 is [79.6%, 100%] and not a certainty.
    """
    lo, hi = agreement.wilson(15, 15)
    assert hi == pytest.approx(1.0)
    assert lo == pytest.approx(0.796, abs=0.001)


def test_wilson_is_asymmetric_at_the_top_of_the_range():
    """14 of 15 is 93.3%, and the interval reaches 23 points below it and 5 above.

    The number the report quotes, asserted here so the caveat cannot drift from the arithmetic.
    """
    lo, hi = agreement.wilson(14, 15)
    assert (round(100 * lo, 1), round(100 * hi, 1)) == (70.2, 98.8)


def test_wilson_widens_toward_a_half():
    widths = [agreement.wilson(k, 15)[1] - agreement.wilson(k, 15)[0] for k in (15, 14, 13, 8)]
    assert widths == sorted(widths)


def test_kappa_is_one_when_two_categories_are_used_and_agreed():
    pairs = [("yes", "yes")] * 8 + [("no", "no")] * 8
    kappa = agreement.cohen_kappa(pairs, ("yes", "no"))
    assert kappa.value == pytest.approx(1.0)
    assert kappa.p_e == pytest.approx(0.5)
    assert kappa.note is None


def test_kappa_is_zero_at_chance():
    """Half right, with balanced marginals: p_o = p_e = 0.5, and κ says the agreement is free."""
    pairs = [("yes", "yes"), ("yes", "no"), ("no", "yes"), ("no", "no")]
    kappa = agreement.cohen_kappa(pairs, ("yes", "no"))
    assert kappa.value == pytest.approx(0.0)
    assert kappa.p_e == pytest.approx(0.5)


def test_kappa_goes_negative_below_chance():
    pairs = [("yes", "no"), ("no", "yes")]
    assert agreement.cohen_kappa(pairs, ("yes", "no")).value == pytest.approx(-1.0)


def test_kappa_is_undefined_when_both_annotators_are_unanimous():
    """p_e = 1 makes κ a literal 0/0, and 0.0 would read as "chance-level" for total agreement."""
    kappa = agreement.cohen_kappa([("yes", "yes")] * 15, ("yes", "no"))
    assert kappa.value is None
    assert kappa.p_e == pytest.approx(1.0)
    assert "undefined" in kappa.note


def test_kappa_is_forced_to_zero_when_one_annotator_has_no_variance():
    """The degeneracy this dataset actually reaches, at three different rates of disagreement.

    A constant first marginal makes p_e = p_o, so κ is exactly 0 whether the second annotator
    disagreed once or seven times. The number carries no information at all, and the note is how
    the report says so instead of printing ``+0.000`` next to 93.3%.
    """
    for wrong in (1, 4, 7):
        pairs = [("yes", "no")] * wrong + [("yes", "yes")] * (15 - wrong)
        kappa = agreement.cohen_kappa(pairs, ("yes", "no"))
        assert kappa.value == pytest.approx(0.0)
        assert "by construction" in kappa.note


def test_percent_agreement_is_the_count_over_the_rows():
    pairs = [("fact", "fact")] * 12 + [("fact", "multihop")] * 3
    score = agreement.score_field(pairs, "bucket")
    assert (score.agreed, score.n) == (12, 15)
    assert score.pct == "80.0%"


def test_the_confusion_matrix_keeps_a_row_for_a_bucket_nobody_used():
    """Four by four, always. An empty row is the statement that the sample held none of it."""
    pairs = [("fact", "fact")] * 6 + [("multihop", "fact")] * 3
    matrix = agreement.confusion(pairs, BUCKETS)
    assert len(matrix) == len(BUCKETS)
    assert all(len(row) == len(BUCKETS) for row in matrix)
    assert sum(sum(row) for row in matrix) == 9
    assert matrix[BUCKETS.index("unanswerable")] == [0, 0, 0, 0]
    assert matrix[BUCKETS.index("multihop")][BUCKETS.index("fact")] == 3


# --- what a promoted dataset forces ---------------------------------------------------------------


def test_the_first_annotator_is_constant_on_two_of_the_three_fields(dataset: Path):
    """``v1.jsonl`` holds only accepted rows, so two of the three labels carry no information.

    Not a quirk of the synthetic fixture: it follows from what promotion means. Asserted so that
    the caveat in the report is checkable rather than argued.
    """
    rows = agreement.load_dataset(dataset)
    labels = [agreement.first_annotator(row) for row in rows]
    assert {label["answer_correct"] for label in labels} == {"yes"}
    for row, label in zip(rows, labels, strict=True):
        assert label["citations_support"] == ("yes" if row.gold_doc_ids else "n/a")
        assert label["bucket"] == row.bucket


def test_kappa_on_answer_correct_cannot_be_a_reliability_number(
    cfg, dataset: Path, pack: agreement.Packet
):
    """The finding, end to end: a real disagreement rate, and a κ that stays at zero through it.

    Three of fifteen answers rejected is 80% agreement and an estimate of residual label error
    worth having. κ reports +0.000 for it, because annotator 1 said ``yes`` fifteen times. The
    report has to print both, and say which one means something.
    """
    sample = exported(cfg, dataset, pack)
    rejected = [row.id for row in sample.rows][:3]
    fill(
        pack,
        lambda item_id: (
            sample.by_id[item_id].bucket,
            "no" if item_id in rejected else "yes",
            agreement.first_annotator(sample.by_id[item_id])["citations_support"],
            "",
            "",
        ),
    )
    judged, problems = agreement.read_sheet(pack.filled, tuple(row.id for row in sample.rows))
    assert not problems
    score = agreement.score_field(
        agreement.pairs_for("answer_correct", judged, sample.by_id), "answer_correct"
    )
    assert (score.agreed, score.n) == (12, 15)
    assert score.pct == "80.0%"
    assert score.kappa.value == pytest.approx(0.0)
    assert "by construction" in score.kappa.note


# --- the two commands -----------------------------------------------------------------------------


def test_the_export_writes_the_whole_packet(cfg, dataset: Path, pack: agreement.Packet):
    exported(cfg, dataset, pack)
    for path in (*pack.written, pack.questions):
        assert path.exists(), path.name
    assert not pack.filled.exists()


def test_the_sidecar_records_everything_the_draw_was_a_function_of(
    cfg, dataset: Path, pack: agreement.Packet
):
    """"Sampling reproducible from the recorded seed" is an acceptance line for this file."""
    sample = exported(cfg, dataset, pack)
    meta = json.loads(pack.meta.read_text(encoding="utf-8"))
    assert meta["seed"] == 42
    assert meta["allocation"] == dict(agreement.ALLOCATION)
    assert meta["sample_n"] == 15
    assert meta["target_n"] == 15
    assert meta["dataset_rows"] == 120
    assert len(meta["dataset_sha256"]) == 64
    assert len(meta["guidelines_sha256"]) == 64
    drawn = {draw["bucket"]: draw["ids"] for draw in meta["draws"]}
    for draw in sample.draws:
        assert drawn[draw.bucket] == list(draw.ids)
    assert sorted(i for ids in drawn.values() for i in ids) == sorted(
        row.id for row in sample.rows
    )


def test_the_export_refuses_to_overwrite_a_packet(cfg, dataset: Path, pack: agreement.Packet):
    exported(cfg, dataset, pack)
    before = pack.sheet.read_text(encoding="utf-8")
    out = StringIO()
    assert agreement.export(cfg, dataset, 43, pack, texts={}, out=out) == 2
    assert "--force" in out.getvalue()
    assert pack.sheet.read_text(encoding="utf-8") == before


def test_force_redraws_the_packet(cfg, dataset: Path, pack: agreement.Packet):
    exported(cfg, dataset, pack)
    before = pack.sheet.read_text(encoding="utf-8")
    exported(cfg, dataset, pack, force=True, seed=43)
    assert pack.sheet.read_text(encoding="utf-8") != before


def test_the_export_will_not_redraw_under_a_filled_sheet(
    cfg, dataset: Path, pack: agreement.Packet
):
    """Even with ``--force``. The failure this prevents is silent, which is why it is not a warning.

    Re-drawing leaves a file of judgements about rows that are no longer the sample. If the ids
    happen to overlap it scores; the number it produces is a mixture of two samples and nothing
    in the report would say so.
    """
    sample = exported(cfg, dataset, pack)
    fill(pack, agreeing(sample))
    before = pack.sheet.read_text(encoding="utf-8")
    out = StringIO()
    assert agreement.export(cfg, dataset, 43, pack, force=True, texts={}, out=out) == 2
    assert "already been filled in" in out.getvalue()
    assert pack.sheet.read_text(encoding="utf-8") == before


def test_a_short_draw_is_a_decision(cfg, tmp_path: Path, pack: agreement.Packet):
    """Nothing is written, and the message says what the choice is. The sample size *is* the CI."""
    path = write_dataset(tmp_path / "short.jsonl", make_rows({**COUNTS, "ambiguous": 1}))
    out = StringIO()
    assert agreement.export(cfg, path, 42, pack, texts={}, out=out) == 1
    assert "--allow-short" in out.getvalue()
    assert "ambiguous wanted 3 rows and the dataset has 1" in out.getvalue()
    assert not pack.sheet.exists()
    out = StringIO()
    assert agreement.export(cfg, path, 42, pack, allow_short=True, texts={}, out=out) == 0
    assert len(sheet_rows(pack.sheet)) == 13


def test_the_questions_log_is_written_once(cfg, dataset: Path, pack: agreement.Packet):
    """A re-export must not wipe what the annotator has already written in it."""
    exported(cfg, dataset, pack)
    pack.questions.write_text("## v1-0001 — is a lab a course?\n", encoding="utf-8")
    exported(cfg, dataset, pack, force=True)
    assert pack.questions.read_text(encoding="utf-8") == "## v1-0001 — is a lab a course?\n"


def test_the_citations_packet_prints_every_cited_record_and_says_when_there_are_none(
    cfg, dataset: Path, pack: agreement.Packet
):
    sample = exported(cfg, dataset, pack)
    body = pack.citations.read_text(encoding="utf-8")
    for row in sample.rows:
        assert f"## `{row.id}`" in body
        for doc_id in row.gold_doc_ids:
            assert f"### `{doc_id}`" in body
            assert f"{doc_id}\nCredits: 3" in body
    abstaining = [row for row in sample.rows if not row.gold_doc_ids]
    assert abstaining
    assert body.count("`citations_support` is `n/a`") == len(abstaining)


def test_the_instructions_put_the_passes_in_order(cfg, dataset: Path, pack: agreement.Packet):
    """The two-pass order is the blinding control, so the instructions have to state it."""
    exported(cfg, dataset, pack)
    body = pack.instructions.read_text(encoding="utf-8")
    assert body.index("Pass 1") < body.index("Pass 2")
    assert f"without opening `{pack.citations.name}`" in body
    assert "eval/GUIDELINES.md" in body
    assert pack.questions.name in body
    assert pack.filled.name in body
    assert pack.sample.name in body


def test_the_report_carries_the_numbers_and_the_caveats(
    cfg, dataset: Path, pack: agreement.Packet, tmp_path: Path
):
    sample = exported(cfg, dataset, pack)
    fill(pack, agreeing(sample))
    report = tmp_path / "day3_agreement_test.md"
    out = StringIO()
    assert agreement.score(cfg, pack, report=report, out=out) == 0
    body = report.read_text(encoding="utf-8")
    for heading in ("# Day 3", "## Agreement", "## Bucket confusion", "## Disagreements"):
        assert heading in body
    assert "## What this does and does not measure" in body
    assert "## Reproducing this" in body
    for field in (*agreement.FIELDS, agreement.DERIVED):
        assert f"`{field}`" in body
    assert "15/15" in body
    assert "100.0%" in body
    assert "[80%, 100%]" in body
    assert "residual label error in the frozen set" in flat(body)
    assert rel_to_repo(report) in out.getvalue()


def test_the_report_says_where_kappa_does_not_mean_what_it_usually_means(
    cfg, dataset: Path, pack: agreement.Packet, tmp_path: Path
):
    """Perfect agreement: κ is undefined on every field, and the report must not print 0.000."""
    sample = exported(cfg, dataset, pack)
    fill(pack, agreeing(sample))
    report = tmp_path / "r.md"
    agreement.score(cfg, pack, report=report, out=StringIO())
    body = report.read_text(encoding="utf-8")
    assert "Where κ does not mean what it usually means" in flat(body)
    assert "+0.000" not in body
    assert "is not a reliability number" in flat(body)
    # Two lines are unanimous on both sides, and they are exactly the two the caveats name:
    # `answer_correct`, where annotator 1 is constant `yes` by promotion, and the restricted
    # citations line, where annotator 1 is constant `yes` again once the mechanical `n/a` rows
    # are taken out. The field-level `citations_support` spans two categories and scores +1.000,
    # which is the inflation the restricted line exists to expose.
    assert flat(body).count("undefined: both annotators used a single category") == 2
    assert "`answer_correct` — κ undefined" in flat(body)
    assert "`citations_support` (cited rows only) — κ undefined" in flat(body)
    assert "+1.000" in body
    # The line the field-level citations number cannot be read without.
    assert "`citations_support` (cited rows only)" in body


def test_the_report_is_a_function_of_the_sheet(
    cfg, dataset: Path, pack: agreement.Packet, tmp_path: Path
):
    """No clock in it, so a diff under ``report/`` means the sheet or the dataset moved."""
    sample = exported(cfg, dataset, pack)
    fill(pack, agreeing(sample))
    first, second = tmp_path / "a.md", tmp_path / "b.md"
    agreement.score(cfg, pack, report=first, out=StringIO())
    agreement.score(cfg, pack, report=second, out=StringIO())
    assert first.read_bytes() == second.read_bytes()


def test_score_refuses_a_faulty_sheet_and_writes_nothing(
    cfg, dataset: Path, pack: agreement.Packet, tmp_path: Path
):
    sample = exported(cfg, dataset, pack)
    fill(pack, lambda item_id: (sample.by_id[item_id].bucket, "", "yes", "", ""))
    report = tmp_path / "r.md"
    out = StringIO()
    assert agreement.score(cfg, pack, report=report, out=out) == 1
    assert not report.exists()
    assert "denominator" in out.getvalue()
    assert "answer_correct" in out.getvalue()


def test_dry_run_prints_the_report_and_writes_no_file(
    cfg, dataset: Path, pack: agreement.Packet, tmp_path: Path
):
    sample = exported(cfg, dataset, pack)
    fill(pack, agreeing(sample))
    report = tmp_path / "r.md"
    out = StringIO()
    assert agreement.score(cfg, pack, dry_run=True, report=report, out=out) == 0
    assert not report.exists()
    assert "# Day 3 — inter-annotator agreement" in out.getvalue()


def test_every_disagreement_is_itemised_with_the_rule_the_annotator_cited(
    cfg, dataset: Path, pack: agreement.Packet, tmp_path: Path
):
    """The useful output. Two disagreements, one with a rule cited and one without.

    The one without is marked as outstanding rather than passed over: a disagreement whose
    guideline rule is unidentified cannot be turned into an edit to the guidelines, which is the
    whole reason the list exists.
    """
    sample = exported(cfg, dataset, pack)
    facts = [row.id for row in sample.rows if row.bucket == "fact"]
    labelled, bare = facts[0], facts[1]

    def judge(item_id: str) -> Sequence[str]:
        if item_id == labelled:
            return ("multihop", "yes", "yes", "M1", "the second record looked load-bearing")
        if item_id == bare:
            return ("multihop", "yes", "yes", "", "")
        return (*(agreement.first_annotator(sample.by_id[item_id])[f] for f in agreement.FIELDS),
                "", "")

    fill(pack, judge)
    report = tmp_path / "r.md"
    assert agreement.score(cfg, pack, report=report, out=StringIO()) == 0
    body = report.read_text(encoding="utf-8")
    assert "2 disagreements over 2 rows" in body
    assert f"### `{labelled}` — `bucket`" in body
    assert "Rule cited: `M1`." in body
    assert "the second record looked load-bearing" in body
    assert f"### `{bare}` — `bucket`" in body
    assert "attribute one by hand before #62" in body
    assert "13/15" in body


def test_the_derived_abstain_line_separates_the_leak_from_the_measurement(
    cfg, dataset: Path, pack: agreement.Packet, tmp_path: Path
):
    """A ``fact`` row called ``multihop`` is a bucket disagreement and not an abstain one.

    Which is the point of the derived line: the abstain half of the bucket call is given away by
    the candidate answer's wording, so it should be reported separately from the within-pair
    calls that are actually blind.
    """
    sample = exported(cfg, dataset, pack)
    swapped = [row.id for row in sample.rows if row.bucket == "fact"][:2]

    def judge(item_id: str) -> Sequence[str]:
        theirs = agreement.first_annotator(sample.by_id[item_id])
        bucket = "multihop" if item_id in swapped else theirs["bucket"]
        return (bucket, theirs["answer_correct"], theirs["citations_support"], "", "")

    fill(pack, judge)
    judged, _ = agreement.read_sheet(pack.filled, tuple(row.id for row in sample.rows))
    bucket = agreement.score_field(
        agreement.pairs_for("bucket", judged, sample.by_id), "bucket"
    )
    derived = agreement.score_field(
        agreement.derived_pairs(judged, sample.by_id), agreement.DERIVED
    )
    assert bucket.agreed == 13
    assert derived.agreed == 15


# --- the repository as it stands ------------------------------------------------------------------


def test_the_promoted_dataset_is_either_absent_or_samplable(cfg):
    """One test, correct in both states of this stack.

    ``eval/datasets/v1.jsonl`` is written by #59's ``--promote``, and on this branch it does not
    exist: the drafting of the ``fact`` and ``ambiguous`` buckets is not finished, so no row has
    been verified and there is nothing to sample. That is why this branch commits no sample, no
    sheet and no report -- and the assertion is that the tool *says* so, rather than producing an
    empty packet.

    Once the dataset does land, the same test draws from it and checks the allocation over the
    real rows. A test that simply asserted the file was missing would start failing on the day
    the work it is waiting for was finished.
    """
    if not review.DATASET_PATH.exists():
        with pytest.raises(ValueError) as exc:
            agreement.load_dataset(review.DATASET_PATH)
        assert "eval.scripts.review" in str(exc.value)
        return
    rows = agreement.load_dataset(review.DATASET_PATH)
    sample = agreement.stratified_sample(rows, seed=cfg["seed"])
    assert len(sample.rows) == agreement.SAMPLE_N
    assert {draw.bucket: draw.got for draw in sample.draws} == dict(agreement.ALLOCATION)


def test_the_packet_over_real_rows_and_the_real_corpus(
    cfg, tmp_path: Path, pack: agreement.Packet
):
    """One run over the committed drafts, whose prose and chunk text are real.

    #57's and #58's 50 rows, with their ``provenance`` stripped so they parse as dataset rows.
    Two buckets only, so the draw is short and the export has to be told so -- and the citations
    come out of ``data/courses.json`` through the same fallback :mod:`eval.scripts.review` uses,
    which is the path a machine that has not built the corpus takes.
    """
    rows = []
    for path in sorted((review.DRAFTS_DIR).glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            raw = json.loads(line)
            raw.pop("provenance", None)
            rows.append(EvalItem.model_validate(raw))
    assert len(rows) == 50
    dataset = write_dataset(tmp_path / "drafts.jsonl", rows)
    out = StringIO()
    assert agreement.export(cfg, dataset, 42, pack, allow_short=True, out=out) == 0
    assert "fact wanted 6 rows and the dataset has 0" in out.getvalue()
    assert len(sheet_rows(pack.sheet)) == 6
    body = pack.citations.read_text(encoding="utf-8")
    cited = [doc_id for row in rows if row.gold_doc_ids for doc_id in row.gold_doc_ids]
    assert any(f"### `{doc_id}`" in body for doc_id in cited)
    assert "Credits:" in body
