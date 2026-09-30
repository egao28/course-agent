"""The LLM judge's first call on every row of a signals run: ``judge_correct`` and one line of why.

    uv run --env-file .env python -m eval.scripts.judge --split calib
    uv run --env-file .env python -m eval.scripts.judge --split test

One line per row of #87's ``signals_{split}_{signals_key}.jsonl`` goes to
``eval/runs/judge_{split}_{correctness_key}.jsonl``, with a sidecar. **This is not the label.** The
label is ``correct``, and only a person gives it (``eval/scripts/review_signals.py``). The judge's
verdict is what the reviewer is shown and what the reviewer overturns or accepts, and the count of
overturns is the one piece of evidence on whether the judge can be trusted without a person on
Day 6.

**One prompt per bucket**, from ``correctness.judge_prompts``, because ``gold_answer`` means
something different in each (``eval/schema.py``). For ``fact`` and ``multihop`` it is the answer.
For ``unanswerable`` it is one acceptable refusal, so the judge is told the answer need not match
it. For ``ambiguous`` it is the clarifying question that should have been asked.

**What is judged is the row's ``answer``**, the answer node's temperature-0 draft, which is what
all three signals were taken from. That includes a draft the citation check discarded for the
hand-off exit: the signals describe that draft, so the label has to as well. ``exit`` is carried
into this file and put on the reviewer's screen, so a person deciding whether a discarded draft
"counts" sees that it was discarded.

**Parsing does not guess.** The reply's first line must open with ``correct`` or ``incorrect``
(any case), alone or followed by a period or colon and the reason -- gpt-4o-mini put both on one
line in 10 of 46 calib replies despite the prompt asking for two. Anything else, a hedge like
"correct, mostly" included, makes ``judge_correct`` null, with the reply kept in ``judge_raw`` and
counted. A null is not a verdict. The reviewer still labels the row, and the
agreement tables count it as its own column.

The gold answer comes from the frozen dataset, joined on ``id``. The signals file does not carry it,
and a row whose ``id`` the dataset does not have is refused by name.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple, TextIO

import openai

from config import (
    Config,
    correctness_key,
    file_sha256,
    load_config,
    rel_to_repo,
    signals_key,
)
from eval.schema import BUCKETS, SPLITS, EvalItem, Split
from eval.scripts.ablation import write_lines
from eval.scripts.draft_items import write_json
from eval.scripts.review import DATASET_PATH
from eval.scripts.signals import RUNS_DIR, fields_of, run_path, sha256_text
from eval.scripts.split import META_SUFFIX, load_split, meta_path, read_sidecar
from llm import Chat, check_llm_config, make_chat

#: The placeholders every bucket's prompt must carry, exactly.
PLACEHOLDERS = frozenset({"question", "gold", "answer"})

#: A first line that counts: the verdict word alone, or followed by ``.``/``:`` and the reason.
VERDICT_RE = re.compile(r"(correct|incorrect)(?:[.:](?:\s+(.*))?)?", re.IGNORECASE)

VERDICTS = {"correct": True, "incorrect": False}

#: What a run file line carries, in order. The review tool reads the first five.
KEYS = ("id", "bucket", "split", "judge_correct", "rationale", "judge_raw", "answer_sha256", "exit")


def judge_path(split: Split, cfg: Config, runs_dir: Path = RUNS_DIR) -> Path:
    """``eval/runs/judge_{split}_{correctness_key}.jsonl``: one per signals run and prompt set."""
    return Path(runs_dir) / f"judge_{split}_{correctness_key(cfg)}.jsonl"


# --- config ---------------------------------------------------------------------------------------


def prompts(cfg: Mapping[str, Any]) -> dict[str, str]:
    """``correctness.judge_prompts``, one per bucket, refused by name if one cannot be filled."""
    section = cfg.get("correctness")
    if not section:
        raise ValueError(
            "config has no `correctness` section; see config.yaml for the one #88 added"
        )
    table = section.get("judge_prompts")
    if not isinstance(table, Mapping):
        # ValueError, not TypeError: a bad config is reported by main() like every other one.
        raise ValueError("correctness.judge_prompts must map each bucket to a prompt")  # noqa: TRY004
    missing = [bucket for bucket in BUCKETS if bucket not in table]
    extra = sorted(set(table) - set(BUCKETS))
    if missing or extra:
        raise ValueError(
            "correctness.judge_prompts must have exactly one prompt per bucket "
            f"({', '.join(BUCKETS)}); missing {', '.join(missing) or 'none'}, "
            f"unknown {', '.join(extra) or 'none'}"
        )
    for bucket in BUCKETS:
        template = table[bucket]
        if not isinstance(template, str) or not template.strip():
            raise ValueError(f"correctness.judge_prompts.{bucket} must be a non-empty string")
        found = fields_of(template)
        if found != PLACEHOLDERS:
            raise ValueError(
                f"correctness.judge_prompts.{bucket} must use exactly the placeholders "
                f"{', '.join(sorted(PLACEHOLDERS))}; it uses {', '.join(sorted(found)) or 'none'}"
            )
    return {bucket: table[bucket] for bucket in BUCKETS}


# --- parsing ------------------------------------------------------------------------------------


class Parsed(NamedTuple):
    """A judge reply read apart: the verdict, and the reason it gave for it."""

    #: ``None`` when the reply did not parse -- an unparsed judgement, not a "no".
    verdict: bool | None
    rationale: str | None


def parse_reply(reply: str) -> Parsed:
    """The verdict from the first line, and the reason after it. Anything else is ``None``."""
    lines = [line.strip() for line in reply.strip().splitlines() if line.strip()]
    if not lines:
        return Parsed(None, None)
    match = VERDICT_RE.fullmatch(lines[0])
    if match is None:
        return Parsed(None, " ".join(lines[1:]) or None)
    rest = [match.group(2)] if match.group(2) else []
    return Parsed(VERDICTS[match.group(1).lower()], " ".join(rest + lines[1:]) or None)


# --- the run ------------------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """One object per non-blank line."""
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(text) for text in f if text.strip()]


def read_run(path: Path) -> list[dict[str, Any]]:
    """A signals run file, refused by name if it is not there -- a half nobody has collected."""
    path = Path(path)
    if not path.exists():
        raise ValueError(
            f"{rel_to_repo(path)} does not exist: run `python -m eval.scripts.signals --split "
            "<half>` first. There is nothing to judge until that half has answers."
        )
    return read_jsonl(path)


def gold_by_id(rows: Sequence[EvalItem]) -> dict[str, EvalItem]:
    """The dataset keyed by row id, for joining a run line back to its gold answer."""
    return {row.id: row for row in rows}


def judge_line(
    line: Mapping[str, Any], item: EvalItem, templates: Mapping[str, str], chat: Chat
) -> dict[str, Any]:
    """One row's verdict, or its failure recorded as a null -- never a lost row."""
    prompt = templates[line["bucket"]].format(
        question=line["question"], gold=item.gold_answer, answer=line["answer"]
    )
    try:
        raw: str | None = chat.complete(prompt)
    except (RuntimeError, openai.APIError) as error:
        raw = None
        parsed = Parsed(None, f"error: {error}")
    else:
        parsed = parse_reply(raw)
    return {
        "id": line["id"],
        "bucket": line["bucket"],
        "split": line["split"],
        "judge_correct": parsed.verdict,
        "rationale": parsed.rationale,
        "judge_raw": raw,
        "answer_sha256": sha256_text(line["answer"]),
        "exit": line["exit"],
    }


def tally(lines: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Per bucket: how many the judge called correct, incorrect, or could not read."""
    counts: dict[str, Counter[str]] = {bucket: Counter() for bucket in BUCKETS}
    for line in lines:
        verdict = line["judge_correct"]
        counts[line["bucket"]][
            "unparsed" if verdict is None else "correct" if verdict else "incorrect"
        ] += 1
    return {bucket: dict(counts[bucket]) for bucket in BUCKETS if counts[bucket]}


def judge_split(
    cfg: Config,
    split: Split,
    out: TextIO,
    dataset: Path = DATASET_PATH,
    runs_dir: Path = RUNS_DIR,
    known: frozenset[str] | None = None,
    chat: Chat | None = None,
) -> Path:
    """Judge every row of one half's signals run, and write the judge file and its sidecar."""
    templates = prompts(cfg)
    source = run_path(split, cfg, runs_dir)
    lines = read_run(source)
    items = gold_by_id(load_split(split, dataset, known))
    unknown = [line["id"] for line in lines if line["id"] not in items]
    if unknown:
        raise ValueError(
            f"{rel_to_repo(source)} has {len(unknown)} id(s) the {split} half of "
            f"{rel_to_repo(Path(dataset))} does not: {', '.join(unknown[:5])}"
        )
    if chat is None:
        check_llm_config(cfg)
        chat = make_chat(cfg)
    print(f"{split}: judging {len(lines)} rows of {rel_to_repo(source)}", file=out)
    judged: list[dict[str, Any]] = []
    for n, line in enumerate(lines, start=1):
        result = judge_line(line, items[line["id"]], templates, chat)
        judged.append(result)
        verdict = result["judge_correct"]
        word = "unparsed" if verdict is None else "correct" if verdict else "incorrect"
        print(f"  [{n:>3}/{len(lines)}] {line['id']} {line['bucket']:<12} {word}", file=out)

    path = write_lines(judge_path(split, cfg, runs_dir), judged)
    dataset_meta = read_sidecar(meta_path(Path(dataset))) or {}
    write_json(
        path.with_suffix(META_SUFFIX),
        {
            "run": rel_to_repo(path),
            "sha256": file_sha256(path),
            "config_hash": cfg.hash,
            "signals_key": signals_key(cfg),
            "correctness_key": correctness_key(cfg),
            "signals_run": rel_to_repo(source),
            "signals_run_sha256": file_sha256(source),
            "dataset": rel_to_repo(Path(dataset)),
            "dataset_sha256": dataset_meta.get("sha256"),
            "split": split,
            "model": chat.model,
            "prompts_sha256": {bucket: sha256_text(templates[bucket]) for bucket in BUCKETS},
            "note": (
                "judge_correct is the LLM judge's first call, not the label; `correct` comes "
                "only from a person's verdict in the review log"
            ),
            "n_lines": len(judged),
            "by_bucket": tally(judged),
        },
    )
    counts = tally(judged)
    for bucket, by in counts.items():
        print(f"  {bucket:<12} " + ", ".join(f"{n} {k}" for k, n in sorted(by.items())), file=out)
    print(
        f"wrote {rel_to_repo(path)} ({len(judged)} lines, correctness key "
        f"{correctness_key(cfg)}). Not labels: run eval.scripts.review_signals next.",
        file=out,
    )
    return path


def main(argv: Sequence[str] | None = None) -> int:
    """Judge one half's answers with the model and write the verdicts; a refusal exits 1."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", required=True, choices=SPLITS, help="which half to judge")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH, help="a frozen dataset")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR, help="where the runs are")
    args = parser.parse_args(argv)
    try:
        judge_split(load_config(), args.split, sys.stdout, args.dataset, args.runs_dir)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
