"""Gold intents for the test half: the sidecar's shape, its validator, and the meta stamp (#102).

    uv run python -m eval.intents                         # check the sidecar and its meta
    uv run python -m eval.intents --stamp --labeller NAME # check it, then write the meta

``v1.jsonl`` carries no intent, and it is frozen (sha ``2a931822...``), so the labels live beside
it in ``eval/datasets/v1_intents.jsonl``: one row per **test** id, ``{"id", "gold_intents",
"note"}``. The rules are the Intent section of ``eval/GUIDELINES.md``.

**A person writes the labels; this only checks them.** Nothing here proposes, pre-fills or
suggests an intent. Doing that would mean reading the router's predictions or running it, and
the labels exist to score the router.

**A set, so the checks are about sets.** Every label must be one of
:data:`agent.state.INTENTS`, with no repeats and at least one. A set of two or more needs a
``note`` saying why (GUIDELINES rule 2). Three or more is a warning, not an error: the
guidelines call such a row unclear, and deciding that is the labeller's call.

**All 63 or it fails.** A missing test id is an error, not a short count. Intent accuracy over
whichever rows happened to get labelled is a number about the labeller's order.

**The meta is what makes the labels citable.** ``v1_intents.meta.json`` records who labelled,
the digests of the frozen dataset and of the sidecar, and a digest of the Intent section they
were labelled under. The last is there because an amended rule after labelling changes what
the labels mean, and #106 has to say which version they follow. :func:`stamp` writes the meta
only over a sidecar with no errors. The check fails when any recorded digest no longer matches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.state import INTENTS, Intent
from config import REPO_ROOT, file_sha256, rel_to_repo
from eval.schema import EvalItem
from eval.scripts.review import DATASET_PATH
from eval.scripts.split import load_split
from eval.validate import Violation, plural, pydantic_violations, render, summary

SIDECAR_PATH = REPO_ROOT / "eval" / "datasets" / "v1_intents.jsonl"
META_PATH = SIDECAR_PATH.with_suffix(".meta.json")
GUIDELINES_PATH = REPO_ROOT / "eval" / "GUIDELINES.md"

#: The router's predictions for the test half. They are never read here. The meta lists which
#: of them existed when the labels were stamped, because a blind labelling can't be proven
#: while they sit in the tree (#102 "Discipline", reported in #106).
PREDICTIONS_GLOB = "eval/runs/signals_test_*.jsonl"

#: More labels than this is a warning: GUIDELINES rule 2, "three or more means the row is unclear".
MAX_QUIET_LABELS = 2


class IntentRow(BaseModel):
    """One sidecar row. Shape only; the cross-row checks are :func:`check_rows`'s."""

    model_config = ConfigDict(extra="forbid")

    id: str
    gold_intents: list[Intent] = Field(min_length=1)
    note: str


class Labelled(NamedTuple):
    """A row that parsed, and the line it came from."""

    line: int
    row: IntentRow


# --- reading --------------------------------------------------------------------------------------


def parse_sidecar(path: Path) -> tuple[list[Labelled], list[Violation]]:
    """Every row that parses, and one violation per fault in the rows that do not."""
    rows: list[Labelled] = []
    violations: list[Violation] = []
    with Path(path).open(encoding="utf-8") as f:
        for line, text in enumerate(f, start=1):
            if not text.strip():
                continue
            try:
                raw = json.loads(text)
            except json.JSONDecodeError as exc:
                violations.append(Violation(line, None, None, f"not JSON: {exc.msg}"))
                continue
            if not isinstance(raw, dict):
                kind = type(raw).__name__
                violations.append(Violation(line, None, None, f"not an object, but a {kind}"))
                continue
            found = raw.get("id") if isinstance(raw.get("id"), str) else None
            try:
                rows.append(Labelled(line, IntentRow.model_validate(raw)))
            except ValidationError as exc:
                violations.extend(pydantic_violations(line, found, exc))
    return rows, violations


def labelled_half(dataset: Path = DATASET_PATH) -> tuple[str, ...]:
    """The test half's ids, in file order. :func:`load_split` refuses a moved dataset digest."""
    return tuple(item.id for item in load_split("test", dataset))


def all_ids(dataset: Path = DATASET_PATH) -> dict[str, str | None]:
    """Every v1 id and its half, so a calib id can be named as calib and not as unknown."""
    with Path(dataset).open(encoding="utf-8") as f:
        items = [EvalItem.model_validate_json(line) for line in f if line.strip()]
    return {item.id: item.split for item in items}


def intent_section(path: Path = GUIDELINES_PATH) -> str:
    """``## Intent`` up to the next ``## `` heading: the rules the labels were made under."""
    text = Path(path).read_text(encoding="utf-8")
    _, found, tail = text.partition("\n## Intent\n")
    if not found:
        raise ValueError(f"{rel_to_repo(path)} has no '## Intent' section to label under")
    section, _, _ = tail.partition("\n## ")
    return section


def section_sha256(path: Path = GUIDELINES_PATH) -> str:
    """The digest of :func:`intent_section` -- what the meta stamps, so edited rules show up."""
    return hashlib.sha256(intent_section(path).encode("utf-8")).hexdigest()


# --- the checks -----------------------------------------------------------------------------------


def check_rows(
    rows: Sequence[Labelled],
    wanted: Sequence[str],
    halves: dict[str, str | None],
    unparsed: frozenset[str] = frozenset(),
) -> Iterator[Violation]:
    """Per row: a known test id, no repeated label, a note where the set has two. Then coverage.

    ``unparsed`` holds the ids of rows that did not parse. They already have a message each, so
    they are not reported a second time as missing.
    """
    seen: dict[str, int] = {}
    for line, row in rows:
        first = seen.get(row.id)
        if first is not None:
            yield Violation(line, row.id, "id", f"already labelled on line {first}")
        else:
            seen[row.id] = line
        if row.id not in halves:
            yield Violation(line, row.id, "id", "is not a v1 id")
        elif halves[row.id] != "test":
            yield Violation(
                line, row.id, "id", f"is a {halves[row.id]} row; only the test half is labelled"
            )

        labels = row.gold_intents
        repeated = sorted({label for label in labels if labels.count(label) > 1})
        if repeated:
            yield Violation(
                line, row.id, "gold_intents", f"repeats {', '.join(repeated)}; it is a set"
            )
        distinct = len(set(labels))
        if distinct > 1 and not row.note.strip():
            yield Violation(
                line,
                row.id,
                "note",
                f"is empty, but the set has {distinct} labels; say why both readings stand",
            )
        if distinct > MAX_QUIET_LABELS:
            yield Violation(
                line,
                row.id,
                "gold_intents",
                f"{distinct} labels; the guidelines call a row like this unclear, not generous",
                "warning",
            )

    missing = [i for i in wanted if i not in seen and i not in unparsed]
    if missing:
        shown = ", ".join(missing[:5])
        more = "" if len(missing) <= 5 else f", and {len(missing) - 5} more"
        yield Violation(
            None,
            None,
            None,
            f"{plural(len(missing), 'test id')} not labelled ({shown}{more}); "
            f"all {len(wanted)} are, or intent accuracy is a number about the labelling order",
        )


def check_meta(
    meta: dict[str, Any] | None, sidecar: Path, dataset: Path, guidelines: Path
) -> Iterator[Violation]:
    """The stamp exists, names a labeller, and every digest it records is still true."""
    if meta is None:
        yield Violation(
            None,
            None,
            None,
            f"no meta at {rel_to_repo(sidecar.with_suffix('.meta.json'))}; "
            "run `python -m eval.intents --stamp --labeller NAME`",
        )
        return
    if not str(meta.get("labeller") or "").strip():
        yield Violation(None, None, "labeller", "is empty; the meta names the person who labelled")
    expected = {
        "dataset_sha256": (file_sha256(dataset), "the frozen dataset changed"),
        "sha256": (file_sha256(sidecar), "the labels were edited after stamping; re-stamp"),
        "guidelines_intent_sha256": (
            section_sha256(guidelines),
            (
                "the Intent section changed after labelling; record the amendment in #106, "
                "re-judge the rows it touches and re-stamp"
            ),
        ),
    }
    for key, (actual, why) in expected.items():
        if meta.get(key) != actual:
            yield Violation(
                None, None, key, f"records {meta.get(key)}, the file is {actual}: {why}"
            )


def validate(
    sidecar: Path = SIDECAR_PATH,
    dataset: Path = DATASET_PATH,
    guidelines: Path = GUIDELINES_PATH,
    *,
    with_meta: bool = True,
) -> list[Violation]:
    """Every check over the sidecar, in file order, then the file-wide findings."""
    sidecar = Path(sidecar)
    if not sidecar.exists():
        raise FileNotFoundError(f"{rel_to_repo(sidecar)} does not exist")
    rows, violations = parse_sidecar(sidecar)
    if not rows and not violations:
        return [Violation(None, None, None, "holds no rows")]
    unparsed = frozenset(v.item_id for v in violations if v.item_id is not None)
    violations.extend(check_rows(rows, labelled_half(dataset), all_ids(dataset), unparsed))
    violations.sort(key=lambda v: (v.line is None, v.line or 0, v.field or ""))
    if with_meta:
        violations.extend(check_meta(read_meta(sidecar), sidecar, dataset, guidelines))
    return violations


# --- the stamp ------------------------------------------------------------------------------------


def read_meta(sidecar: Path) -> dict[str, Any] | None:
    """The sidecar's ``.meta.json`` as a dict, or ``None`` when it has never been stamped."""
    path = Path(sidecar).with_suffix(".meta.json")
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def meta(sidecar: Path, dataset: Path, guidelines: Path, labeller: str) -> dict[str, Any]:
    """What :func:`stamp` writes: provenance, digests, and the counts #104 and #106 cite."""
    rows, _ = parse_sidecar(sidecar)
    sets = [set(r.row.gold_intents) for r in rows]
    return {
        "labeller": labeller,
        "sidecar": rel_to_repo(sidecar),
        "sha256": file_sha256(sidecar),
        "dataset": rel_to_repo(dataset),
        "dataset_sha256": file_sha256(dataset),
        "guidelines_intent_sha256": section_sha256(guidelines),
        "rows": len(rows),
        "rows_with_label": {label: sum(label in s for s in sets) for label in INTENTS},
        "rows_with_more_than_one_label": sum(len(s) > 1 for s in sets),
        "predictions_in_tree": sorted(p.name for p in REPO_ROOT.glob(PREDICTIONS_GLOB)),
    }


def stamp(
    labeller: str,
    sidecar: Path = SIDECAR_PATH,
    dataset: Path = DATASET_PATH,
    guidelines: Path = GUIDELINES_PATH,
) -> tuple[Path | None, list[Violation]]:
    """Write the meta if the sidecar has no errors. Return its path (or ``None``) and findings."""
    if not labeller.strip():
        raise ValueError("--labeller is the person who labelled the rows, and it is empty")
    violations = validate(sidecar, dataset, guidelines, with_meta=False)
    if any(v.level == "error" for v in violations):
        return None, violations
    path = Path(sidecar).with_suffix(".meta.json")
    content = meta(Path(sidecar), Path(dataset), Path(guidelines), labeller.strip())
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path, violations


# --- the command ----------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Check the gold-intent sidecar, and with ``--stamp`` write its meta if it has no errors."""
    parser = argparse.ArgumentParser(
        prog="python -m eval.intents",
        description="Check the test half's gold intents (#102); with --stamp, also write the meta.",
    )
    parser.add_argument("--sidecar", default=str(SIDECAR_PATH), help="the gold-intent JSONL")
    parser.add_argument("--stamp", action="store_true", help="write the meta if the file passes")
    parser.add_argument("--labeller", default="", help="who labelled the rows (with --stamp)")
    args = parser.parse_args(argv)

    sidecar = Path(args.sidecar)
    if args.stamp:
        written, violations = stamp(args.labeller, sidecar)
    else:
        written, violations = None, validate(sidecar)
    if violations:
        print(render(sidecar, violations))
    print(summary(sidecar, violations))
    if written is not None:
        print(f"wrote {rel_to_repo(written)}")
    return 1 if any(v.level == "error" for v in violations) else 0


if __name__ == "__main__":
    sys.exit(main())
