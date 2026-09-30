"""The two confidence signals that need a written answer: verbalized, and self-consistency.

#87 measured them on every eval row, and #92's post-answer gate measures them on every live answer.
They are computed here, once, so the number the gate compares is produced by the same prompts, the
same calls and the same parser as the number the threshold was fitted on. A second copy in the
gate would be a gate calibrated on a different signal from the one it reads.

* **verbalized** -- one follow-up call (``signals.verbalized_prompt``) that checks the answer
  against the evidence and ends on ``Confidence: <integer>``. The integer, in [0, 100], is the
  signal. Any other last line is ``None``, never 50 or 0.
* **self_consistency** -- ``signals.samples`` answers drawn with the answer node's own prompt at
  ``signals.sample_temperature``, then a judge call (``signals.judge_prompt``) on every pair. The
  signal is agreeing pairs / pairs: with 3 samples, one of {0, 1/3, 2/3, 1}. One unreadable
  verdict makes the whole score ``None``, not a smaller ratio.

The settings live in the ``signals`` section of ``config.yaml``, which :func:`settings` validates.
``llm`` is never imported here -- a client is passed in -- so :mod:`agent.graph` still imports with
no ``openai`` behind it.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Mapping, Sequence
from string import Formatter
from typing import TYPE_CHECKING, Any, NamedTuple

from agent.prompts import render_evidence
from agent.state import Evidence

if TYPE_CHECKING:  # `llm` pulls in openai
    from llm import Chat

#: The placeholders each prompt must carry, exactly -- a prompt missing ``{answer}`` would ask for
#: confidence in nothing, and one with a stray ``{field}`` would raise one row into the run.
PLACEHOLDERS: dict[str, frozenset[str]] = {
    "verbalized_prompt": frozenset({"question", "evidence", "answer"}),
    "judge_prompt": frozenset({"question", "a", "b"}),
}

#: The last line of a verbalized reply that counts: ``Confidence: <integer>``, and nothing else on
#: it. Case and a trailing ``%`` or period are forgiven; range is checked separately.
CONFIDENCE_RE = re.compile(r"confidence:\s*([0-9]{1,3})\s*%?\.?", re.IGNORECASE)

#: A judge reply that counts, after lowercasing and dropping one trailing period.
VERDICTS = {"yes": True, "no": False}


# --- config ---------------------------------------------------------------------------------------


class Settings(NamedTuple):
    """The ``signals`` section, validated."""

    samples: int
    sample_temperature: float
    verbalized_prompt: str
    judge_prompt: str


def fields_of(template: str) -> set[str]:
    """The named ``{}`` placeholders in a format template, so a prompt can be checked before use."""
    return {name for _, name, _, _ in Formatter().parse(template) if name is not None}


def settings(cfg: Mapping[str, Any]) -> Settings:
    """The ``signals`` section, refused by name if a value would make a run that cannot happen."""
    section = cfg.get("signals")
    if not section:
        raise ValueError("config has no `signals` section; see config.yaml for the one #87 added")
    missing = [key for key in Settings._fields if key not in section]
    if missing:
        raise ValueError(f"config signals section is missing {', '.join(missing)}")
    samples = section["samples"]
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 2:
        # One sample has no pair to judge, so its score would be 0/0.
        raise ValueError(f"signals.samples must be an integer >= 2, got {samples!r}")
    temperature = section["sample_temperature"]
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not 0 < temperature <= 2
    ):
        # 0 would draw the temperature-0 answer n more times, and agreement would measure nothing.
        raise ValueError(
            f"signals.sample_temperature must be a number in (0, 2], got {temperature!r}"
        )
    for key, wanted in PLACEHOLDERS.items():
        template = section[key]
        if not isinstance(template, str) or not template.strip():
            raise ValueError(f"signals.{key} must be a non-empty string")
        found = fields_of(template)
        if found != wanted:
            raise ValueError(
                f"signals.{key} must use exactly the placeholders "
                f"{', '.join(sorted(wanted))}; it uses {', '.join(sorted(found)) or 'none'}"
            )
    return Settings(
        samples, float(temperature), section["verbalized_prompt"], section["judge_prompt"]
    )


# --- parsing the two replies ----------------------------------------------------------------------


def parse_verbalized(reply: str) -> int | None:
    """The integer on the reply's ``Confidence:`` last line, in [0, 100], or ``None``.

    Only the last non-empty line is read, so a number in the reasoning above it can never be
    taken for the answer. Never a default: a null is counted, a 50 is a lie.
    """
    lines = [line.strip() for line in reply.strip().splitlines() if line.strip()]
    match = CONFIDENCE_RE.fullmatch(lines[-1]) if lines else None
    if match is None:
        return None
    value = int(match.group(1))
    return value if 0 <= value <= 100 else None


def parse_verdict(reply: str) -> bool | None:
    """``True`` for yes, ``False`` for no, ``None`` for anything else."""
    text = reply.strip().lower()
    return VERDICTS.get(text.removesuffix("."))


def pairs(n: int) -> list[tuple[int, int]]:
    """Every unordered pair of ``n`` samples, in a fixed order: (0, 1), (0, 2), (1, 2) for 3."""
    return list(itertools.combinations(range(n), 2))


def agreement(verdicts: Sequence[bool | None]) -> float | None:
    """Agreeing pairs / pairs, or ``None`` if any verdict is unreadable -- not a smaller ratio."""
    if any(verdict is None for verdict in verdicts):
        return None
    return sum(1 for verdict in verdicts if verdict) / len(verdicts)


# --- the two measurements -------------------------------------------------------------------------


class Verbalized(NamedTuple):
    """The parsed confidence, and the reply it was read from."""

    value: int | None
    raw: str


class Consistency(NamedTuple):
    """The agreement ratio, and everything it was computed from."""

    value: float | None
    samples: list[str]
    pair_agree: list[bool | None]
    judge_raw: list[str]


def verbalized(
    chat: Chat, config: Settings, question: str, docs: Sequence[Evidence], answer: str
) -> Verbalized:
    """One follow-up call rating ``answer`` against ``docs``. Costs 1 chat call."""
    raw = chat.complete(
        config.verbalized_prompt.format(
            question=question, evidence=render_evidence(docs), answer=answer
        )
    )
    return Verbalized(parse_verbalized(raw), raw)


def self_consistency(
    chat: Chat, config: Settings, question: str, prompt: str, system: str | None
) -> Consistency:
    """``samples`` fresh answers to ``prompt``, and a judge on every pair.

    ``prompt`` and ``system`` must be exactly the answer node's, so the samples answer the question
    the scored answer did. Costs ``samples + samples * (samples - 1) / 2`` chat calls: 6 with the
    shipped 3 samples.
    """
    samples = [
        chat.complete(prompt, system=system, temperature=config.sample_temperature)
        for _ in range(config.samples)
    ]
    judged = [
        chat.complete(config.judge_prompt.format(question=question, a=samples[i], b=samples[j]))
        for i, j in pairs(config.samples)
    ]
    verdicts = [parse_verdict(reply) for reply in judged]
    return Consistency(agreement(verdicts), samples, verdicts, judged)


def calls_per_answer(signal: str, config: Settings) -> int:
    """How many chat calls the post-answer gate adds to one answered query, for ``signal``."""
    if signal == "verbalized":
        return 1
    if signal == "self_consistency":
        return config.samples + len(pairs(config.samples))
    return 0
