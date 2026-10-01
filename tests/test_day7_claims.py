"""Tests for the claims page -- ``report/day7_claims.md`` (#120), held to the files it cites.

The page is a table, which makes it a different thing to test from the two prose pages
``tests/test_final_report.py`` covers. A claim is one row: the claim in the register of the
measurement, the file and key path every number in it comes from, the rows it rests on, and what
it does not say. So the rules this file holds are the ones a table can break:

* **No number in a claim is typed in.** :data:`final_report.CLAIMS` is hand-written wording with
  the numbers interpolated from the files each row names, and
  :func:`test_no_number_is_typed_into_a_claim` refuses a digit in the wording. A claim whose
  numbers were typed would be a claim that stops being true silently, which is the one failure
  mode a file of claims cannot have.
* **Every number in a row is read back through that row's own links** (#120's acceptance, which
  is #107's rule applied to a table). ``final_report.py --check`` does it, so the expensive test
  here is to run it on the committed page and then to break the page on purpose, one way at a
  time: a digit changed in a cell, a row's source link deleted, a key path misspelt.
* **The bullets are held to the same rule as the prose in the report.** The claims in resume
  wording are not generated -- they are the author's, and the page ships with the section empty --
  so what is tested is the mechanism that will hold them: a bullet quoting the table with the
  row's ``[src]`` link passes, and the same bullet without the link does not.
* **Every claim says what it does not say.** A row with an empty fourth column is a number
  without its caveat, and #120's acceptance is that the caveat is in the column rather than
  omitted.

What no test here can do is judge whether a claim is a fair reading of the file it cites. That is
the last column's job, and the report's limitations section is the longer form of it.
"""

from __future__ import annotations

import re

import pytest

from eval.scripts import final_report as fr

pytest.importorskip("scipy")

PAGE = fr.CLAIMS_PAGE

#: The four columns #120 commissioned, in order.
COLUMNS = ("the claim", "where every number in it comes from", "what it rests on",
           "what it does not say")

#: The sections #120 commissioned: the bullets sit above the table, and the refusals below it.
SECTIONS = ("## The bullets", "## The claims", "## What may not go on the line",
            "## Checking this file")

#: ``[text](path)`` or ``[text](path "title")`` for a local file, as the page links to one.
LINK_RE = re.compile(r"\[[^\]]+\]\((?P<path>(?!https?:)[^)\s#]+)(?: \"[^\"]*\")?\)")

#: Where the page leaves room for the bullets it does not write.
EMPTY = "_Not written yet._"

#: The one allowlist entry that is dangerous inside a generated row: `` of 127 `` has the shape
#: of a course code (``CS 374``), so a measurement that happens to follow a two-to-four-letter
#: word would be masked out of the check instead of read back. No row here quotes a course code,
#: so inside a row the pattern firing at all means a number has gone unchecked.
COURSE_CODE = next(pattern for pattern, why in fr.ALLOWED if "course code" in why)


@pytest.fixture(scope="module")
def cfg():
    from config import load_config

    return load_config()


@pytest.fixture(scope="module")
def page() -> fr.Page:
    return next(one for one in fr.pages(fr.Sources()) if one.path == PAGE)


@pytest.fixture(scope="module")
def text() -> str:
    return PAGE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def facts(cfg):
    return fr.read_facts(cfg, fr.Sources())


@pytest.fixture(scope="module")
def bodies(facts):
    return fr.render_blocks(facts)


@pytest.fixture(scope="module")
def rows(page, text: str) -> list[str]:
    return [row for _, row in fr.claim_rows(page, text)]


def problems(page, text: str, facts, bodies) -> list[str]:
    return [problem.message for problem in fr.check(page, text, facts, bodies)]


# --- the wording is written, the numbers are not ----------------------------------------------


def test_no_number_is_typed_into_a_claim() -> None:
    """Every number in the table is interpolated from a file; the wording carries none."""
    for claim in fr.CLAIMS:
        for field in (claim.line, claim.rests_on, claim.unsaid):
            found = fr.NUMBER_RE.findall(fr.mask(field))
            assert not found, (
                f"{claim.id} has {found} written into its wording; a number in a claim comes "
                "from claim_values() and the file the row names, never from this source file")


def test_no_number_is_typed_into_a_refusal() -> None:
    for one in fr.REFUSED:
        for field in (one.line, one.instead):
            assert not fr.NUMBER_RE.findall(fr.mask(field)), one.line


def test_every_claim_has_a_unique_id() -> None:
    ids = [claim.id for claim in fr.CLAIMS]
    assert len(set(ids)) == len(ids), "an id names one row, and a bullet names one id"


def test_every_claim_says_what_it_does_not_say() -> None:
    """#120's fourth column is not optional: a claim with no caveat is a claim with none found."""
    for claim in fr.CLAIMS:
        assert claim.unsaid.strip(), f"{claim.id} has an empty last column"


def test_every_claim_names_at_least_one_source() -> None:
    for claim in fr.CLAIMS:
        assert claim.sources, f"{claim.id} cites nothing"
        for one in claim.sources:
            assert one.file in fr.CLAIM_FILES, f"{claim.id} names the unknown file {one.file}"
            assert one.keys, f"{claim.id} cites {one.file} with no key path"


def test_every_cited_key_path_resolves(facts) -> None:
    """Every key path a row names is one its file actually has a value at."""
    for claim in (*fr.CLAIMS, *fr.REFUSED):
        for one in claim.sources:
            path = fr.CLAIM_FILES[one.file](facts)
            for key in one.keys:
                assert fr.candidates(path, key) is not None, (
                    f"{path.name} has no key path {key!r}")


# --- the committed page ------------------------------------------------------------------------


def test_check_passes_on_the_committed_page(page, text: str, facts, bodies) -> None:
    assert problems(page, text, facts, bodies) == []


def test_the_generator_is_idempotent(page, text: str, bodies) -> None:
    assert fr.splice(page, text, bodies) == text, (
        "the committed page is not what the generator writes today; run "
        "`python -m eval.scripts.final_report`")


def test_every_claim_is_a_row_on_the_page(rows: list[str]) -> None:
    for claim in fr.CLAIMS:
        assert any(f"**{claim.id}**" in row for row in rows), f"{claim.id} is not on the page"


def test_every_row_carries_a_source_or_no_number(rows: list[str]) -> None:
    """The rule in miniature: a row with a number in it names the file the number came from."""
    for row in rows:
        if fr.NUMBER_RE.findall(fr.mask(row)):
            assert fr.SRC_RE.search(row), row


def test_no_row_hides_a_number_behind_the_course_code_rule(rows: list[str]) -> None:
    """A claim worded so that its number looks like a course code is a number nothing checks."""
    for row in rows:
        bare = fr.CODE_RE.sub(" ", fr.TARGET_RE.sub(" ", row))
        found = COURSE_CODE.findall(bare)
        assert not found, (
            f"{found} in this row has the shape of a course code, so `mask` removes it before "
            f"the check reads it; word the claim so the number does not follow a short word: "
            f"{row}")


def test_the_four_columns_are_the_ones_commissioned(page, text: str) -> None:
    header = text[next(one.start for one in fr.blocks_in(page, text) if one.name == "claims"):]
    cells = tuple(cell.strip() for cell in header.splitlines()[0].strip("|").split("|"))
    assert cells == COLUMNS


def test_the_sections_are_in_the_commissioned_order(text: str) -> None:
    found = [line for line in text.splitlines() if line.startswith("## ")]
    assert tuple(found) == SECTIONS


def test_the_bullets_sit_above_the_table(text: str) -> None:
    """#120 puts the bullets above the table they quote, so a reader meets the claim first."""
    assert text.index("## The bullets") < text.index("<!-- BEGIN gen:claims -->")


def test_the_bullets_are_not_written_by_the_generator(text: str) -> None:
    """The claims in resume wording are the author's. The page ships with the room for them."""
    assert EMPTY in text
    assert text.index(EMPTY) < text.index("<!-- BEGIN gen:claims -->")


def test_every_local_link_resolves(text: str) -> None:
    missing = [path for path in LINK_RE.findall(text)
               if not (PAGE.parent / path).resolve().exists()]
    assert missing == []


def test_the_page_points_at_the_limitations_section(text: str) -> None:
    """#120's second acceptance: no claim contradicts them, so the page says where they are."""
    assert "README.md#7-limitations" in text


def test_the_refused_lines_are_the_ones_the_issue_names() -> None:
    """The four lines #120 forbids by name, each one refused by a number rather than by taste."""
    lines = " ".join(one.line for one in fr.REFUSED).lower()
    for forbidden in ("improved retrieval", "deployed", "latency", "reduced hallucination",
                      "interval dropped"):
        assert forbidden in lines, f"nothing in REFUSED covers {forbidden!r}"


def test_the_corpus_counts_are_read_from_the_export(facts) -> None:
    """C9's numbers are counted from the committed corpus, and checked against the README."""
    assert facts.corpus == fr._corpus(fr.Sources())
    assert facts.corpus["courses"] > 0 and facts.corpus["prereq_edges"] > 0


# --- breaking it on purpose --------------------------------------------------------------------


def test_a_digit_changed_in_a_claim_fails(page, text: str, facts, bodies) -> None:
    rows = str(len(facts.dataset))
    broken = text.replace(f"evaluation set, {rows} questions",
                          f"evaluation set, {int(rows) + 1} questions")
    assert broken != text
    assert any("is not a value in" in one for one in problems(page, broken, facts, bodies))


def test_a_row_with_its_source_link_deleted_fails(page, text: str, facts, bodies) -> None:
    broken = fr.SRC_RE.sub("", text, count=1)
    assert broken != text
    assert any("has no [src] link in its row" in one
               for one in problems(page, broken, facts, bodies))


def test_a_mistyped_key_path_in_a_row_fails(page, text: str, facts, bodies) -> None:
    broken = text.replace('v1.jsonl "split")', 'v1.jsonl "splits")', 1)
    assert broken != text
    assert any("has no key path 'splits'" in one
               for one in problems(page, broken, facts, bodies))


def test_a_claim_table_outside_its_markers_is_refused(page, text: str, facts, bodies) -> None:
    """A second table nothing generates is a row of numbers with no owner."""
    broken = text.replace(EMPTY, "| a claim | from nowhere |")
    assert any("a table outside a gen: block" in one
               for one in problems(page, broken, facts, bodies))


# --- the bullets, when they are written ---------------------------------------------------------


def test_a_bullet_that_quotes_the_table_passes(page, text: str, facts, bodies) -> None:
    rows = len(facts.dataset)
    bullet = (f"- Built and hand-verified a {rows}-row evaluation set "
              '[src](../eval/datasets/v1.jsonl "split").')
    assert problems(page, text.replace(EMPTY, bullet), facts, bodies) == []


def test_the_same_bullet_without_its_link_fails(page, text: str, facts, bodies) -> None:
    rows = len(facts.dataset)
    bullet = f"- Built and hand-verified a {rows}-row evaluation set."
    assert any("has no [src] link in its paragraph" in one
               for one in problems(page, text.replace(EMPTY, bullet), facts, bodies))


def test_a_bullet_the_file_does_not_support_fails(page, text: str, facts, bodies) -> None:
    bullet = ('- Built and hand-verified a 500-row evaluation set '
              '[src](../eval/datasets/v1.jsonl "split").')
    assert any("is not a value in" in one
               for one in problems(page, text.replace(EMPTY, bullet), facts, bodies))


# --- the page's scope ---------------------------------------------------------------------------


def test_only_the_claims_page_carries_the_claims_blocks() -> None:
    for one in fr.pages(fr.Sources()):
        carried = set(one.blocks) & set(fr.CLAIMS_BLOCKS)
        assert carried == (set(fr.CLAIMS_BLOCKS) if one.path == PAGE else set()), one.path


def test_the_report_still_carries_every_other_block() -> None:
    assert set(fr.REPORT_BLOCKS) | set(fr.CLAIMS_BLOCKS) == set(fr.BLOCKS)


def test_the_claims_page_is_an_input_of_the_generator(cfg) -> None:
    paths = [one.path for one in fr.inputs(cfg, fr.Sources())]
    assert PAGE in paths, "a page the generator writes is a page it refuses to write without"
