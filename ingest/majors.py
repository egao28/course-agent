"""The hand-encoded degree requirements, and the evaluator for the rules they are written in.

``course-catalog.csv`` is a list of *sections*: 27 columns of when a class meets and who teaches
it, and not one of them is a degree requirement. So ``requirements.json`` carried ``majors: []``
from #3 onward, and ``check_requirement()`` (#31) could answer "what Gen Eds do I still need?"
while "am I on track for my CS degree?" -- the question a course-planning agent exists for -- was
unanswerable. This module is #32's answer: two programs transcribed by hand from the catalog's own
program pages into ``ingest/majors.json``, in the declarative shape ``GENED`` already uses.

**Two programs, not all of them.** Computer Science, BS and Statistics, BSLAS -- one from each of
the two colleges the demo queries touch, out of roughly 250 the campus offers. Scraping the whole
catalog was the other option (#32's option 2) and is a different piece of work: 250 prose pages,
a parser to test, and a new source to pin. Two hand-written programs are auditable by a human in
five minutes, which is the property that matters for a tool a student might act on.

**Every rule is written from a course list, and rules that are not course lists say so.** A degree
is more than its courses: 128 hours, a 2.0 technical GPA, 40 hours of upper-division work, a
language placement, an approved science-elective list that is not in this export. None of that can
be decided from a transcript of course codes, so none of it is *guessed* -- those rows carry
``{"kind": "advising"}`` and the audit reports them as unchecked, with the catalog's own text and
a link. An audit that quietly omits them would read as "you are done" to a student who is not.

**The rules evaluate against codes, not against catalog records.** ``data/courses.json`` is one
term: ``MATH 221`` is required by both programs and is not offered in 2026-sp, and neither is
``ENG 300``. Checking requirements against the catalog index would report a student who has taken
MATH 221 as still needing it -- wrong in the one direction they cannot detect. :func:`credit_rule`
therefore matches normalized code strings, and the catalog is used only to *describe* coverage:
:func:`build_majors` records which named codes this term does not offer, so the audit can say so.

**What the catalog is still used for** is refusing an encoding that has rotted. A group whose
courses have all vanished from the catalog is the shape a renumbered program takes (CS 125 became
CS 124 and CS 128 in 2021, and nothing announced it in the data), so :func:`build_majors` raises
on it rather than exporting a requirement nobody can satisfy. Individual absences are the one-term
corpus and are recorded, not fatal -- see ``data/README.md``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from config import REPO_ROOT, rel_to_repo

#: The hand-written source, next to the code that reads it -- not in ``data/``, which holds only
#: what the exporter generates. ``data/requirements.json`` carries the exported copy.
MAJORS_PATH = REPO_ROOT / "ingest" / "majors.json"

#: A course code in the one spelling this file may use: the catalog's own, "CS 374". Codes here
#: are written by a human against a program page, so an unrecognizable one is a typo and not an
#: input to be normalized -- ``agent.tools.courses.normalize_code`` is for what a *student* types.
CODE_RE = re.compile(r"[A-Z]{2,4} \d{3}")

#: ``all_of`` every course; ``n_of`` any ``n`` of a named list; ``n_of_matching`` any ``n`` courses
#: matching a code pattern (the technical electives: "six CS 400-level courses"); ``advising`` for
#: a requirement a course list cannot decide, which is reported and never counted.
RULE_KINDS = frozenset({"all_of", "n_of", "n_of_matching", "advising"})

#: Keys every major and every group must carry. ``source`` and ``catalogYear`` are filled in per
#: group by :func:`build_majors` when the group does not override them, so a requirement in the
#: exported file always names the page it was read off -- an audit that cannot be cited is an
#: audit a student has to take on faith.
MAJOR_KEYS = ("id", "label", "college", "source", "aliases", "groups")
GROUP_KEYS = ("id", "label", "group", "rule")


class Credit(NamedTuple):
    """What a transcript does to one requirement group.

    ``credited`` lists every course counted, even past ``n``: a student who took both CS 210 and
    CS 211 took both, and an audit that hid one would look like it had lost a course.
    ``still_needed`` is how many more are required, and ``options`` names the courses that would
    supply them -- empty for a pattern rule, where the answer is a range and not a list.
    """

    credited: list[str]
    still_needed: int
    options: list[str]


# --- normalizing ----------------------------------------------------------------------------------


def normalize_program(major: str) -> str:
    """``"  Computer Science, BS  "`` -> ``"computer science bs"``: the form aliases are matched in.

    Lower-cased with every run of non-alphanumerics collapsed to one space, so "Gen-Ed", "CS, BS"
    and "cs bs" all land on one string. Matching is exact against that form and never partial:
    "statistics and computer science" is a *different* program from "statistics", and a partial
    match would audit a student against a degree they are not taking.
    """
    return re.sub(r"[^a-z0-9]+", " ", major.lower()).strip()


def matches_pattern(match: Mapping[str, Any], code: str) -> bool:
    """Whether a course code is one a ``n_of_matching`` rule accepts.

    ``include`` is an escape hatch for a code outside the number range that the catalog names
    anyway (CS 397 among the 400-level technical electives); ``exclude`` removes codes inside it,
    either because the catalog excludes them or because another group already requires them.
    """
    if code in match.get("exclude", ()):
        return False
    if code in match.get("include", ()):
        return True
    subject, _, number = code.partition(" ")
    if subject not in match["subjects"] or not number.isdigit():
        return False
    return match["numberMin"] <= int(number) <= match["numberMax"]


def rule_codes(rule: Mapping[str, Any]) -> list[str]:
    """Every course code a rule writes down, exclusions included -- what spelling is checked over.

    A pattern rule writes down only its literal ``include`` / ``exclude`` codes; what it accepts
    beyond those is a range.
    """
    if rule["kind"] in ("all_of", "n_of"):
        return list(rule["courses"])
    if rule["kind"] == "n_of_matching":
        return [*rule["match"].get("include", ()), *rule["match"].get("exclude", ())]
    return []


def named_codes(rule: Mapping[str, Any]) -> list[str]:
    """The codes a student could *take* to satisfy a rule, which is what coverage is measured over.

    Everything :func:`rule_codes` returns except a pattern's exclusions: CS 400 is not a course
    this degree is missing, it is one the technical electives refuse, and listing it among the
    courses the term does not offer would read as a gap in the requirement.
    """
    if rule["kind"] == "n_of_matching":
        return list(rule["match"].get("include", ()))
    return rule_codes(rule)


# --- the audit ------------------------------------------------------------------------------------


def required_count(rule: Mapping[str, Any]) -> int:
    """How many courses a rule takes. ``all_of`` is ``n_of`` over its whole list."""
    return len(rule["courses"]) if rule["kind"] == "all_of" else int(rule["n"])


def credit_rule(rule: Mapping[str, Any], taken: Sequence[str]) -> Credit:
    """Apply one rule to a transcript of normalized course codes.

    ``taken`` is code strings and not catalog records, deliberately: a required course this term
    does not offer (MATH 221, ENG 300) must still be creditable to the student who took it.

    An ``advising`` rule is not applied at all -- it comes back as zero of everything, and the
    caller reports it as unchecked rather than as satisfied or unsatisfied.
    """
    kind = rule["kind"]
    if kind == "advising":
        return Credit(credited=[], still_needed=0, options=[])
    if kind == "n_of_matching":
        credited = [code for code in taken if matches_pattern(rule["match"], code)]
        return Credit(credited, max(0, required_count(rule) - len(credited)), [])
    held = set(taken)
    # In the rule's own order, which is the catalog's: a student reads "CS 124, CS 128, ..." in the
    # order the program page lists them, not in the order they happened to take them.
    credited = [code for code in rule["courses"] if code in held]
    options = [code for code in rule["courses"] if code not in held]
    still_needed = max(0, required_count(rule) - len(credited))
    return Credit(credited, still_needed, options if still_needed else [])


# --- reading and checking the source ----------------------------------------------------------


def load_majors(path: Path | None = None) -> dict[str, Any]:
    """The whole hand-written document, checked. Raises rather than export a rule nobody wrote."""
    path = MAJORS_PATH if path is None else Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{rel_to_repo(path)} is missing; it is the source of data/majors")
    with path.open(encoding="utf-8") as f:
        doc = json.load(f)
    for key in ("catalogYear", "source", "majors"):
        if not doc.get(key):
            raise ValueError(f"{rel_to_repo(path)} has no {key!r}")
    check_majors(doc["majors"], rel_to_repo(path))
    return doc


def check_majors(majors: Sequence[Mapping[str, Any]], where: str) -> None:
    """Refuse a set of programs that would audit a student wrongly.

    Read this as a list of the wrong answers it prevents. A duplicate program id or alias makes
    which degree was audited depend on list order. A rule kind nobody implements would be skipped
    silently, reporting a requirement as met. ``n`` above the length of its own list is a
    requirement no transcript can satisfy, and ``n`` of zero is one every transcript satisfies. A
    code written outside the catalog's spelling ("CS374", "cs 374") matches nothing a student ever
    took, so the audit would report a course they hold as still needed -- the failure that makes a
    wrong audit worse than no audit.

    Runs over the hand-written source *and* over the exported copy in ``data/requirements.json``,
    which is the same shape with the derived fields added: the tool that reads that file has no
    other way to know a hand edit broke it.
    """
    seen_ids: set[str] = set()
    aliases: dict[str, str] = {}
    for major in majors:
        missing = [key for key in MAJOR_KEYS if not major.get(key)]
        if missing:
            raise ValueError(f"{where}: major {major.get('id') or major} is missing {missing}")
        if major["id"] in seen_ids:
            raise ValueError(f"{where}: two programs share the id {major['id']!r}")
        seen_ids.add(major["id"])
        for alias in [*major["aliases"], major["label"], major["id"]]:
            normalized = normalize_program(alias)
            if not normalized:
                raise ValueError(f"{where}: {major['id']} has a blank alias")
            owner = aliases.setdefault(normalized, major["id"])
            if owner != major["id"]:
                raise ValueError(
                    f"{where}: {normalized!r} names both {owner!r} and {major['id']!r}, so which "
                    "degree a student is audited against would depend on the order of this file"
                )
        check_groups(major, where)


def check_groups(major: Mapping[str, Any], where: str) -> None:
    """The per-group half of :func:`check_majors`, for one program."""
    seen: set[str] = set()
    for group in major["groups"]:
        label = f"{where}: {major['id']}/{group.get('id') or group}"
        missing = [key for key in GROUP_KEYS if not group.get(key)]
        if missing:
            raise ValueError(f"{label} is missing {missing}")
        if group["id"] in seen:
            raise ValueError(f"{label} appears twice, so it would be reported twice")
        seen.add(group["id"])
        rule = group["rule"]
        if rule.get("kind") not in RULE_KINDS:
            raise ValueError(
                f"{label} has rule kind {rule.get('kind')!r}, which no evaluator implements; "
                f"known kinds are {sorted(RULE_KINDS)}"
            )
        if rule["kind"] == "advising" and not group.get("note"):
            raise ValueError(
                f"{label} is an advising requirement with no note, so the audit would report it as "
                "unchecked without saying what it is"
            )
        for code in rule_codes(rule):
            if not CODE_RE.fullmatch(code):
                raise ValueError(
                    f"{label} names {code!r}, which is not a course code as the catalog spells "
                    "one ('CS 374'); it would match nothing a student has taken"
                )
        check_rule_size(rule, label)


def check_rule_size(rule: Mapping[str, Any], label: str) -> None:
    """``n`` has to be satisfiable and non-trivial, and a list has to hold distinct courses."""
    if rule["kind"] == "advising":
        return
    codes = rule["courses"] if rule["kind"] != "n_of_matching" else rule_codes(rule)
    duplicates = sorted({code for code in codes if codes.count(code) > 1})
    if duplicates:
        raise ValueError(f"{label} names {duplicates} twice, which would credit them twice")
    needed = required_count(rule)
    if needed < 1:
        raise ValueError(f"{label} requires {needed} courses, so every transcript satisfies it")
    if rule["kind"] == "n_of_matching":
        match = rule["match"]
        if not match.get("subjects"):
            raise ValueError(f"{label} matches no subject, so it accepts nothing")
        if match["numberMin"] > match["numberMax"]:
            raise ValueError(
                f"{label} matches numbers {match['numberMin']}-{match['numberMax']}, an empty range"
            )
        open_pattern = {**match, "exclude": []}
        stranded = [c for c in match.get("exclude", ()) if not matches_pattern(open_pattern, c)]
        if stranded:
            raise ValueError(
                f"{label} excludes {stranded}, which its own pattern never matched: a typo, not a "
                "rule, and the course it meant to exclude is still counting"
            )
    elif needed > len(rule["courses"]):
        raise ValueError(
            f"{label} requires {needed} of {len(rule['courses'])} courses, which no transcript "
            "can satisfy"
        )


# --- the exported copy ----------------------------------------------------------------------------


def build_majors(
    courses: Sequence[Mapping[str, Any]], doc: Mapping[str, Any] | None = None
) -> list[dict[str, Any]]:
    """The programs as ``data/requirements.json`` carries them: checked, cited, and measured.

    Each group gains the page and catalog year it came from (its own, or the program's), how many
    courses in *this* export could count toward it, and which of the codes it names this term does
    not offer. The count is the same sanity check ``courseCount`` is for a Gen Ed category, with
    one difference that matters: a Gen Ed category with no courses is a broken rule, while a
    requirement naming a course that is not offered this spring is a normal Tuesday -- MATH 221 is
    required by both programs and runs in the fall. :func:`check_coverage` draws the line at zero,
    and is run by the exporter against the pinned catalog rather than here, so a fixture of
    seventeen courses can still build a document to test the audit with.
    """
    doc = load_majors() if doc is None else doc
    known = {course["code"] for course in courses}
    built = []
    for major in doc["majors"]:
        groups = []
        for group in major["groups"]:
            rule = group["rule"]
            named = named_codes(rule)
            not_offered = [code for code in named if code not in known]
            if rule["kind"] == "n_of_matching":
                count = sum(1 for code in known if matches_pattern(rule["match"], code))
            else:
                count = len(named) - len(not_offered)
            groups.append(
                {
                    **group,
                    "source": group.get("source", major["source"]),
                    "catalogYear": group.get("catalogYear", doc["catalogYear"]),
                    "courseCount": count,
                    "notOffered": not_offered,
                }
            )
        built.append(
            {
                **major,
                "catalogYear": major.get("catalogYear", doc["catalogYear"]),
                "groups": groups,
            }
        )
    return built


def check_coverage(majors: Sequence[Mapping[str, Any]], where: str) -> None:
    """Refuse an encoding this catalog has stopped recognizing.

    Runs on the built copy, against the pinned CSV, and asks one question per group: does this
    catalog offer *any* of the courses that could satisfy it? One missing course is the one-term
    corpus -- MATH 221 is required by both programs and runs in the fall. A group with nothing
    left is the shape a renumbered program takes (CS 125 became CS 124 and CS 128 in 2021, and no
    field in the data announced it), and exporting it would tell a student to take a course that
    does not exist.
    """
    for major in majors:
        for group in major["groups"]:
            if group["rule"]["kind"] == "advising" or group["courseCount"]:
                continue
            raise ValueError(
                f"{where}: {major['id']}/{group['id']} requires "
                f"{named_codes(group['rule']) or 'courses'} and "
                "this catalog offers none of them, which is what a renumbered program looks like; "
                f"re-read {group['source']} before exporting it"
            )
