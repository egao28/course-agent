"""A machine's reading of every drafted row, so that a person's is faster -- and never instead.

    uv run --env-file .env python -m eval.scripts.triage              # every unread row
    uv run --env-file .env python -m eval.scripts.triage --limit 20   # a slice of them
    uv run python -m eval.scripts.review --triage                     # the pass, with them shown

#59's pass is 152 rows read by a person, and that is the deliverable. This shortens it without
replacing it: one model call per drafted row answers the same four questions the reviewer answers,
against ``eval/GUIDELINES.md`` and the cited chunk text, and the answers are written to
``eval/datasets/triage.jsonl``. ``review --triage`` then shows each row's suggestion **under** the
chunk text, and the reviewer still types the verdict.

**This file is not the review log, and cannot be mistaken for one.** It is a different path with
different keys: :data:`TRIAGE_KEYS` against
:data:`eval.scripts.review.LOG_KEYS`, ``suggested`` rather than ``verdict``, and no timestamp.
:func:`eval.scripts.review.read_entries` refuses it by name on the first line, ``--promote``
cannot see it, and nothing downstream of the log reads this file at all. A machine's opinion about
a row is not evidence that the row was checked, and the only file that is has to stay the one a
person wrote into.

**It anchors the reviewer, and that is the cost.** A person shown ``accept`` accepts more often
than one shown nothing -- the effect is well enough established that it has to be priced in rather
than hoped away. Three things follow, and all three are deliberate:

- The suggestion is rendered **after** the three checks and after every cited chunk, so the
  reading happens before the opinion arrives, and it is labelled as a machine's.
- ``--triage`` is opt-in. The pass with no flag is the pass #59 shipped, and it stays runnable.
- :mod:`eval.scripts.day3_report` reports that the pass was model-assisted, in the limitations
  section, whenever this file exists -- and it says which model and against which guidelines
  digest. A model-assisted pass that the report did not disclose would be the report's one
  disqualifying omission.

**The suggestion is derived, not asked for.** The model answers four questions -- is the answer
right, do the citations support it, which bucket is this, does the row belong in the set at all --
and :func:`suggest` maps those to ``accept`` / ``edit`` / ``delete``. Asking a model for a verdict
directly gets a verdict with reasons attached to it afterwards; asking it the questions the
guidelines actually pose, and computing the verdict from the answers, means the mapping is one
function a test pins without a model in the loop, and a reviewer who disagrees can see *which*
answer they disagree with.

**A third model, by default.** The rows were drafted by ``gpt-4o-mini`` and ``gpt-4.1-mini``, so
the triage default is neither: a drafting model asked to check its own question is being asked
whether it made a mistake. ``--model`` overrides it, every row records the model that read it, and
none of this touches ``config.yaml`` -- the same reasoning, and the same mechanism, as
:func:`eval.scripts.draft_items.make_drafter`.

**Resumable, and it never rewrites a row.** Rows already in the file are skipped; ``--redo``
starts over for the ids it is given. The file is append-only for the same reason the review log
is: a run that failed on row 90 has read 89 rows, and paying for them twice is how this gets
abandoned.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, NamedTuple, TextIO

from config import REPO_ROOT, Config, file_sha256, load_config, rel_to_repo
from eval.schema import BUCKETS, RULES, EvalItem
from eval.scripts.agreement import GUIDELINES_PATH
from eval.scripts.draft_items import credit_question
from eval.scripts.review import (
    REVIEW_LOG_PATH,
    Candidate,
    Verdict,
    answer_codes,
    current,
    find_drafts,
    interleave,
    load_candidates,
    read_verdicts,
)
from eval.scripts.review import corpus_text as review_corpus_text
from eval.validate import plural
from ingest.tokenize import get_tokenizer
from llm import Chat, OpenAIChat, check_llm_config

#: Where the advisory file lives. Beside the drafts it is about, and **not** beside the review log
#: it must never be confused with.
TRIAGE_PATH = REPO_ROOT / "eval" / "datasets" / "triage.jsonl"

#: Every key a triage row carries. Deliberately not :data:`eval.scripts.review.LOG_KEYS`: no
#: ``verdict``, no ``ts``, no ``before`` / ``after``, and two keys that log lines never have. A
#: reader that expects one file and is handed the other fails on the first line.
TRIAGE_KEYS: tuple[str, ...] = (
    "id",
    "suggested",
    "reason",
    "checks",
    "model",
    "prompt_version",
    "guidelines_sha256",
    "config_hash",
)

#: Bumped whenever the prompt below changes in a way a model could notice, exactly as #56 bumps
#: its own: a file triaged under two prompts is two readings, and the rows say which they had.
TRIAGE_PROMPT_VERSION = "v2"

#: Neither of the two models that drafted these rows. See the module docstring.
DEFAULT_MODEL = "gpt-4.1"

#: The four questions, and what each answer may be. The first three are #59's checks and #60's
#: fields, in their words, so that a triage answer and an annotator's are comparable values and
#: not two vocabularies. ``belongs`` is the fourth because "this row should not exist" is a
#: verdict the other three cannot express: a row can have a right answer, supporting citations
#: and the right bucket and still be one ``GUIDELINES.md`` rule 2 deletes.
CHECKS: Mapping[str, tuple[str, ...]] = {
    "answer_correct": ("yes", "no", "unsure"),
    "citations_support": ("yes", "no", "n/a"),
    "bucket": BUCKETS,
    "belongs": ("yes", "no"),
}


class Triage(NamedTuple):
    """One row's machine reading: the four answers, the sentence, and who read it."""

    id: str
    suggested: str
    reason: str
    checks: dict[str, str]
    model: str
    prompt_version: str = TRIAGE_PROMPT_VERSION
    guidelines_sha256: str = ""
    config_hash: str = ""

    def to_json(self) -> dict[str, Any]:
        """The reading as it is written to the file, provenance fields included."""
        return {
            "id": self.id,
            "suggested": self.suggested,
            "reason": self.reason,
            "checks": dict(self.checks),
            "model": self.model,
            "prompt_version": self.prompt_version,
            "guidelines_sha256": self.guidelines_sha256,
            "config_hash": self.config_hash,
        }

    def line(self, bucket: str) -> str:
        """One screen line per answer that is not the clean one, for ``review --triage``.

        Only the answers that would change something: a reviewer does not need to be told that a
        machine also thought the answer was right, and a triage block that restated all four
        would bury the one that mattered.
        """
        parts = []
        if self.checks.get("answer_correct") != "yes":
            parts.append(f"answer {self.checks.get('answer_correct')}")
        if self.checks.get("citations_support") == "no":
            parts.append("citations do not support it")
        if self.checks.get("bucket") != bucket:
            parts.append(f"reads as {self.checks.get('bucket')}")
        if self.checks.get("belongs") == "no":
            parts.append("should not be in the set")
        if self.suggested == "delete" and self.checks.get("belongs") != "no":
            # The mechanical half: rule 2 is not one of the four answers, so a delete the model
            # did not ask for has to name where it came from or the line reads "no objection".
            parts.append("deleted by the rule 2 lookup")
        found = "; ".join(parts) if parts else "no objection"
        return f"{self.suggested} -- {found}. {self.reason}"


_VARIABLE: dict[str, frozenset[str] | None] = {}

#: The last commit whose ``data/courses.json`` carries ``creditsRaw`` / ``creditsMin`` /
#: ``creditsMax`` -- the #25/#17 merge, which is ``main`` as it stood immediately before #126.
#:
#: Not ``origin/main``, which is where this used to point. #126 resolved ``data/courses.json`` to
#: the six-day stack's copy on purpose (the stack keeps its chunk template), and that copy carries
#: a single ``credits`` float. So the branch that had the answer no longer does, and no branch in
#: the repo does: #17 is merged in history and out of the tree. A pinned commit is what is left,
#: and it is the better shape anyway -- which courses the *catalog* gives a range for is a fact
#: about one export, not about whatever a moving branch happens to hold today.
#:
#: Reachable from ``origin/main``'s first-parent history, so a ``fetch-depth: 0`` checkout finds
#: it. A shallow clone does not, and reads as unavailable rather than as zero.
CREDIT_RANGE_REF = "c9ff4de76555f09f3080af575408af8e001aa3c6"


def variable_credit_codes(ref: str = CREDIT_RANGE_REF) -> frozenset[str] | None:
    """Course codes whose catalog credits are a range, read from ``ref``. ``None`` if unavailable.

    ``GUIDELINES.md`` rule 2 deletes a credit question about one of these, and the rule is a
    **lookup**, not a judgement: the chunk prints ``Credits: 3`` whether the catalog says "3
    hours." or "3 OR 4 hours.", so neither a model nor a person can settle it from the record in
    front of them. No export in the working tree carries ``creditsMin`` / ``creditsMax`` -- which
    is why :func:`eval.scripts.review.credit_ranges` returns ``None`` here -- so this reads them
    out of :data:`CREDIT_RANGE_REF` instead, and the 1,415 affected courses are decided by the
    lookup rather than by whoever remembered to check.

    ``None``, not an empty set, in both of the ways this can fail to answer:

    * git cannot produce the file at all -- no remote, a shallow clone, a rewritten ref;
    * git produces a file that has no ``creditsMin`` key in it, which is what ``origin/main``
      became in #126 and what made this return 0 for 1,415 courses (#128).

    The second is the one worth guarding, because it looks like a successful lookup. An empty set
    would mean "no course has a range", which silently switches the rule off, and
    :func:`eval.scripts.review.credit_ranges` has refused it on the same grounds since #59.
    """
    if ref not in _VARIABLE:
        try:
            body = subprocess.run(
                ["git", "show", f"{ref}:data/courses.json"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            courses = json.loads(body)
            if not courses or "creditsMin" not in courses[0]:
                _VARIABLE[ref] = None
            else:
                _VARIABLE[ref] = frozenset(
                    course["code"]
                    for course in courses
                    if course["creditsMin"] != course["creditsMax"]
                )
        except (OSError, subprocess.CalledProcessError, json.JSONDecodeError, KeyError):
            _VARIABLE[ref] = None
    return _VARIABLE[ref]


def rule_two(item: EvalItem, variable: frozenset[str] | None, tokenizer: Any) -> list[str]:
    """The variable-credit courses a credit *answer* names, which rule 2 deletes the row for.

    Three halves have to hold, and the third is the one that is easy to get wrong.

    The question has to be about credits -- one about prerequisites that happens to name AAS 465
    is not a rule 2 row. The course has to have a range -- a credit question about a fixed-credit
    course is a perfectly good ``fact`` row. And **the row has to assert a credit count at all**:
    rule 2 sits under *"When a ``gold_answer`` is correct"* and deletes the row because any
    number it gives is wrong about either the catalog or the corpus. An abstaining row gives no
    number. An ``unanswerable`` row's answer says the knowledge base does not have it, and an
    ``ambiguous`` row's answer *is the clarifying question* -- *"which 199 do you mean, UKR 199
    or SOC 590?"* is correct whether or not either course has a credit range, and deleting it
    would throw away a row that tests ambiguity on the grounds of a claim it never made.

    Empty when the lookup is unavailable, which leaves the row to the reviewer and to the
    on-screen flag #59 already prints.
    """
    if variable is None or item.should_abstain or not credit_question(item.question, tokenizer):
        return []
    named = answer_codes(f"{item.question} {item.gold_answer}")
    return [code for code in named if code in variable]


def suggest(checks: Mapping[str, str], bucket: str, rule2: Sequence[str] = ()) -> str:
    """``accept`` / ``edit`` / ``delete`` from the four answers. The mapping, in one place.

    ``rule2`` is the mechanical half -- the variable-credit codes :func:`rule_two` found, which
    the model was told not to judge -- and it outranks everything, because a row whose credit
    answer is wrong about either the catalog or the corpus cannot be repaired into a right one.

    ``delete`` only where the row should not be in the set: a wrong answer is an *edit*, because
    the question is usually fine and the answer is what was got wrong, and a triage that
    recommended deleting every row with a wrong answer would recommend throwing away the rows
    verification exists to repair. ``unsure`` about the answer is an edit too -- it is a row that
    needs a person's attention, which is what the suggestion is for.
    """
    if rule2 or checks.get("belongs") == "no":
        return "delete"
    clean = (
        checks.get("answer_correct") == "yes"
        and checks.get("citations_support") in {"yes", "n/a"}
        and checks.get("bucket") == bucket
    )
    return "accept" if clean else "edit"


# --- the prompt -----------------------------------------------------------------------------------

#: What the model is and what it is doing. No persona, and one instruction that matters more than
#: the rest: the guidelines are the authority, not what the model would have written.
TRIAGE_SYSTEM = (
    "You are checking one row of a hand-built evaluation set against the annotation guidelines "
    "you are given and against the catalog records quoted for you. Judge the row as written. Do "
    "not rewrite it, do not prefer a question you would have asked instead, and do not use "
    "anything you know about this university that is not in the records quoted below. Reply with "
    "one JSON object and nothing else."
)

#: The body. The guidelines go in whole -- they are one page, they are the file the reviewer and
#: the second annotator are both held to, and a paraphrase of them here would be a third
#: specification nobody is checking against.
TRIAGE_TEMPLATE = """\
# The annotation guidelines

{guidelines}

# The row

bucket as drafted: {bucket} -- {gloss}
question: {question}
gold_answer: {gold_answer}
gold_doc_ids: {gold_doc_ids}
should_abstain: {should_abstain}

# The cited catalog records, in full

{records}

# What to answer

Reply with one JSON object with exactly these five keys:

- "answer_correct": "yes" | "no" | "unsure" -- is gold_answer correct for this question, judged
  against the records above and the guidelines' rules on correctness? "unsure" if the records
  quoted do not settle it.
- "citations_support": "yes" | "no" | "n/a" -- is every fact the answer asserts readable in the
  text of a cited record? "n/a" when the row cites none, which is correct for a refusal row.
- "bucket": "fact" | "multihop" | "unanswerable" | "ambiguous" -- which bucket the row belongs
  in by the guidelines' rules, which may be the one it was drafted as.
- "belongs": "yes" | "no" -- "no" only if the row should be deleted from the set outright rather
  than corrected: it is answerable from the records above and labelled a refusal, or it asks
  something the records cannot settle and is labelled answerable. Do **not** consider rule 2
  (credit counts of variable-credit courses) here: the records quoted above print one number
  whether or not the catalog gives a range, so that rule cannot be judged from them and is
  decided outside this reply.
- "reason": one short sentence, naming the guideline rule or the record line that decides it.
"""

#: How a row with no cited records reads in the prompt. The absence is the thing being checked on
#: a refusal row, so it is stated rather than left as an empty section.
NO_RECORDS = (
    "None. This row cites no catalog record, which the guidelines require of an `unanswerable` "
    "row and allow for an `ambiguous` one. What to check is the claim of absence."
)


def render_records(item: EvalItem, texts: Mapping[str, str]) -> str:
    """Every cited chunk, in full, headed by its id -- what the reviewer sees on screen."""
    if not item.gold_doc_ids:
        return NO_RECORDS
    blocks = []
    for n, gold in enumerate(item.gold_doc_ids, start=1):
        text = texts.get(gold, "-- not in the corpus --")
        blocks.append(f"[{n}] doc_id: {gold}\n{text}")
    return "\n\n".join(blocks)


def render_prompt(item: EvalItem, texts: Mapping[str, str], guidelines: str) -> str:
    """The whole user message for one row."""
    return TRIAGE_TEMPLATE.format(
        guidelines=guidelines.strip(),
        bucket=item.bucket,
        gloss=RULES[item.bucket].gloss,
        question=item.question,
        gold_answer=item.gold_answer,
        gold_doc_ids=", ".join(item.gold_doc_ids) or "(none)",
        should_abstain=str(item.should_abstain).lower(),
        records=render_records(item, texts),
    )


# --- reading the reply ----------------------------------------------------------------------------


def parse_reply(text: str) -> dict[str, str]:
    """The model's JSON as four answers and a sentence, or an exception naming the fault.

    A fence is unwrapped rather than refused, like #56's: a model that wraps its JSON has not
    made a judgement error, and throwing the row away would cost a call to punish formatting.
    """
    body = text.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        body = body.removeprefix("json").strip()
    try:
        raw = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError(f"reply is not JSON: {exc.msg}") from None
    if not isinstance(raw, dict):
        raise TypeError(f"reply is a {type(raw).__name__}, not an object")
    checks: dict[str, str] = {}
    for field, allowed in CHECKS.items():
        value = raw.get(field)
        if not isinstance(value, str) or value.strip().lower() not in allowed:
            raise ValueError(
                f"{field}={value!r} is not one of {', '.join(allowed)}"
            )
        checks[field] = value.strip().lower()
    reason = raw.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason is missing; it is the half a reviewer reads")
    checks["reason"] = " ".join(reason.split())
    return checks


# --- the file -------------------------------------------------------------------------------------


def read_triage(path: Path | None = None) -> dict[str, Triage]:
    """The advisory file as one reading per row, last line winning. A missing file is empty.

    Refuses a review log outright rather than reading six of its seven keys as a triage row: the
    two files are one directory apart and one of them is evidence.
    """
    path = TRIAGE_PATH if path is None else Path(path)
    if not path.exists():
        return {}
    out: dict[str, Triage] = {}
    with path.open(encoding="utf-8") as f:
        for line, text in enumerate(f, start=1):
            if not text.strip():
                continue
            raw = json.loads(text)
            if "verdict" in raw or "ts" in raw:
                raise ValueError(
                    f"{rel_to_repo(path)}:{line} carries 'verdict' or 'ts', which makes it a "
                    "review log entry and not a triage row. A machine's suggestion is not a "
                    "verdict; keep the two files apart."
                )
            missing = [key for key in TRIAGE_KEYS if key not in raw]
            if missing:
                raise ValueError(
                    f"{rel_to_repo(path)}:{line} is missing {', '.join(missing)}; a triage row "
                    f"carries all {len(TRIAGE_KEYS)} keys"
                )
            out[raw["id"]] = Triage(
                id=raw["id"],
                suggested=raw["suggested"],
                reason=raw["reason"],
                checks=dict(raw["checks"]),
                model=raw["model"],
                prompt_version=raw["prompt_version"],
                guidelines_sha256=raw["guidelines_sha256"],
                config_hash=raw["config_hash"],
            )
    return out


def append_triage(path: Path, row: Triage) -> None:
    """Add one reading and fsync, before the next call is made. Append-only, like the log."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row.to_json(), ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# --- the run --------------------------------------------------------------------------------------


def make_reader(
    cfg: Mapping[str, Any], model: str | None = None, client: Any | None = None
) -> Chat:
    """The client this run reads with: a third model by default. See the module docstring."""
    section = check_llm_config(cfg)
    return OpenAIChat(
        model or DEFAULT_MODEL,
        temperature=float(section["temperature"]),
        max_tokens=int(section["max_tokens"]),
        client=client,
    )


class Run(NamedTuple):
    """What one triage run did: the rows it read, the ones it could not, and the ones it skipped."""

    read: list[Triage]
    failed: list[tuple[str, str]]
    skipped: int


def triage(
    cfg: Config,
    candidates: Sequence[Candidate],
    chat: Chat,
    texts: Mapping[str, str],
    path: Path,
    done: Mapping[str, Triage],
    out: TextIO,
    limit: int | None = None,
    guidelines: str | None = None,
    verdicts: Mapping[str, Verdict] | None = None,
    variable: frozenset[str] | None | Literal[False] = False,
) -> Run:
    """One model call per unread row, written before the next one is made.

    The row read is the row **as the review pass would show it** -- #59's ``current()``, so a row
    already edited by hand is triaged as edited and not as drafted. A triage of the draft behind
    an edit would tell the reviewer about a row they have already fixed.
    """
    text = GUIDELINES_PATH.read_text(encoding="utf-8") if guidelines is None else guidelines
    digest = file_sha256(GUIDELINES_PATH) if guidelines is None else ""
    judged = read_verdicts() if verdicts is None else verdicts
    # False means "look it up"; None means "the lookup is unavailable", which is a different
    # state and one a test has to be able to ask for.
    ranges = variable_credit_codes() if variable is False else variable
    tokenizer = get_tokenizer(cfg)
    run = Run([], [], 0)
    for candidate in candidates:
        if candidate.id in done:
            run = run._replace(skipped=run.skipped + 1)
            continue
        if limit is not None and len(run.read) + len(run.failed) >= limit:
            break
        item = current(candidate, judged.get(candidate.id))
        try:
            reply = chat.complete(
                render_prompt(item, texts, text), system=TRIAGE_SYSTEM
            )
            checks = parse_reply(reply)
        except (TypeError, ValueError, RuntimeError) as exc:
            run.failed.append((candidate.id, str(exc)[:160]))
            out.write(f"  {candidate.id}  could not be read: {str(exc)[:120]}\n")
            continue
        reason = checks.pop("reason")
        rule2 = rule_two(item, ranges, tokenizer)
        if rule2:
            reason = (
                f"GUIDELINES rule 2: {', '.join(rule2)} has a credit range in the catalog and a "
                f"single number in the chunk, so this row is deleted and not repaired. {reason}"
            )
        row = Triage(
            id=candidate.id,
            suggested=suggest(checks, item.bucket, rule2),
            reason=reason,
            checks=checks,
            model=getattr(chat, "model", "unknown"),
            guidelines_sha256=digest,
            config_hash=cfg.hash,
        )
        append_triage(path, row)
        run.read.append(row)
        out.write(f"  {row.id}  {row.suggested:<7} {row.reason[:88]}\n")
    return run


def summary(run: Run, path: Path, total: int) -> str:
    """What the terminal gets: the three counts, the failures, and where the file is."""
    counts = {name: sum(1 for row in run.read if row.suggested == name) for name in
              ("accept", "edit", "delete")}
    lines = [
        "",
        (
            f"read                   {len(run.read)} of {total} drafted rows "
            f"({run.skipped} already in the file)"
        ),
        "suggested              "
        + ", ".join(f"{name} {n}" for name, n in counts.items()),
    ]
    if run.failed:
        lines.append(f"unread                 {plural(len(run.failed), 'row')}")
    ranges = variable_credit_codes()
    lines.append(
        "rule 2 lookup          "
        + (
            f"{len(ranges)} variable-credit courses from c9ff4de"
            if ranges is not None
            else "UNAVAILABLE -- git show c9ff4de:data/courses.json gave no creditsMin, so no "
            "row was deleted for rule 2; the reviewer's on-screen flag is the only check"
        )
    )
    lines += [
        f"wrote                  {rel_to_repo(path)}",
        "",
        "These are suggestions, not verdicts. Run the pass with them on screen:",
        "  uv run python -m eval.scripts.review --triage",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Read every drafted row with a model and write an advisory suggestion per row.

    A draft that does not parse stops the run before anything is written: a triage of a row that
    does not parse is nothing. Exits 1 only when every attempted row failed.
    """
    parser = argparse.ArgumentParser(
        prog="python -m eval.scripts.triage",
        description="Read every drafted row against eval/GUIDELINES.md with a model, and write "
        "an advisory suggestion per row to eval/datasets/triage.jsonl. Not a verdict, and not "
        "the review log: `review --triage` shows these and a person still decides.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"read with this model instead of {DEFAULT_MODEL}, which is deliberately neither "
        "of the two that drafted these rows",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="stop after this many rows (a sitting)"
    )
    parser.add_argument(
        "--redo",
        nargs="+",
        default=None,
        metavar="ID",
        help="re-read these ids even though the file already holds them; the later line wins",
    )
    parser.add_argument("--triage-file", type=Path, default=None, help="where the file goes")
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help=f"the review log to read edits from (default: {rel_to_repo(REVIEW_LOG_PATH)}); a "
        "row already edited by hand is triaged as edited, not as drafted",
    )
    parser.add_argument("--drafts-dir", type=Path, default=None, help="where the drafts are")
    parser.add_argument("--config", default=None, help="config file path (default: config.yaml)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config) if args.config else load_config()
    path = TRIAGE_PATH if args.triage_file is None else args.triage_file
    candidates, found = load_candidates(find_drafts(args.drafts_dir))
    if found:
        for one in found[:10]:
            print(f"{rel_to_repo(one.path)}: {one.violation.message}", file=sys.stderr)
        print("\nFix the drafts first; a triage of a row that does not parse is nothing.",
              file=sys.stderr)
        return 1
    done = read_triage(path)
    if args.redo:
        done = {key: value for key, value in done.items() if key not in set(args.redo)}
    queue = interleave(candidates)
    chat = make_reader(cfg, args.model)
    run = triage(
        cfg,
        queue,
        chat,
        review_corpus_text(),
        path,
        done,
        sys.stdout,
        args.limit,
        verdicts=read_verdicts(args.log),
    )
    print(summary(run, path, len(candidates)))
    return 1 if run.failed and not run.read else 0


if __name__ == "__main__":
    sys.exit(main())
