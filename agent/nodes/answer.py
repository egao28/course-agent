"""Node 4: turn the evidence into an answer, and nothing else into an answer.

    check_evidence ──► answer ──► END
                          └─────► refuse_handoff

The last node, and the only one that puts a model's words in front of a person. Three constraints
make it worth being a module rather than one call:

**Answer only from the evidence.** The model knows a great deal about this university's courses
from pretraining, all of it from some other year of the catalog and none of it checkable against
the corpus this repo exported. The prompt (:data:`agent.prompts.ANSWER_SYSTEM_PROMPT`) says so in
its first rule, the evidence is the only course data in the message, and nothing else is offered
to it -- no tools, no second turn, no history.

**If the evidence does not contain it, say so.** As a normal answer -- "the catalog entry for CS 374
doesn't list who teaches it" -- not an apology and not a value that is probably right. That is the
one behaviour a fake client cannot demonstrate, which is why the prompt spends an example on it and
why #39's live run is where it gets checked.

**Cite ``doc_id``, and verify the citation afterwards.** Every id the answer cites in square
brackets must be a ``doc_id`` this run actually put in ``docs``. One that is not goes into
``signals["citations_unknown"]`` and the whole answer is discarded for the hand-off exit: a model
that invented a citation invented the claim it was supporting, and an unchecked citation is worse
than none, because it looks like verification. The ids are course codes, chunk ids and program
labels this corpus already uses, so the check is a set lookup and not a judgement.

**An answer that cites nothing is discarded too**, and that is what rule 4 of the prompt is for: a
"the catalog does not say" answer cites the block it looked in, so *every* answer this node accepts
can carry a citation, and a draft with none is not a modest answer -- it is an untraceable one.
``signals["citations_verified"]`` is the list that makes the two cases distinguishable afterwards:
``[]`` is "it cited nothing", a non-empty list beside ``citations_unknown`` is "it cited something
real and something invented", and ``None`` means this node never ran.

**Matching is by identifier, not by resemblance.** A citation matches a block when the two strings
are equal after collapsing whitespace and case -- ``[cs 374]`` cites ``CS 374``, and nothing looser
than that. In particular a citation is never resolved by *containment*: an answer about CS 225 that
cites ``[CS 374]`` while the CS 374 chunk is in ``docs`` would pass any "the id appears in some
block" rule, and would be exactly the wrong answer with exactly the reassuring bracket after it.
The near misses this strictness costs -- a chunk id cited as its course code, ``[1]`` for the first
block -- are recorded in ``citations_unknown`` as themselves, so Day 5 can count them and decide
whether the prompt or this parser is what should change. That is #35's argument for an exact label
match, one node further on.

**The two refusal exits stay fixed text.** This node writes ``exit`` and, when it keeps the draft,
``answer``; a discarded draft writes ``exit`` alone and
:func:`agent.nodes.exits.refuse_handoff_node` mints the sentence, so the hand-off message a person
reads is written in one place however the run got there. Nothing here generates a refusal -- a
model asked to write one negotiates with itself and produces a partial answer with a hedge in front
of it.

**A cut-off completion raises rather than being handed on.** :meth:`llm.OpenAIChat.complete` treats
``finish_reason == "length"`` as an error, and the answer node is why: a truncated answer keeps its
citation and loses the claim the citation was supporting, which is the one failure this node's
check cannot see. Raise ``llm.max_tokens`` rather than ship half an answer. An API failure -- a
429, a timeout -- propagates for #35's reason: a billing problem reported as a hand-off is a
refusal in #39's report with nothing behind it. So does a client that returns something other than
a ``str``, which is a broken fake rather than a model failure, and whose ``repr`` would arrive here
carrying square brackets of its own.

The client is injectable and is built on the first query, exactly as in
:mod:`agent.nodes.route_intent` -- ``llm`` is imported inside the function because it reaches
``openai`` through ``embedding.py``, the ``llm`` config section is validated when the node is built
(it reads no key and costs nothing), and an injected ``chat=`` skips that check. Everything above
except constraint 1 is therefore testable with no credits: prompt assembly, the citation parse, the
downgrade, and the exit each node claims.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

from agent.nodes import Node
from agent.prompts import ANSWER_SYSTEM_PROMPT, answer_user_prompt
from agent.state import AgentState, Evidence
from config import Config

if TYPE_CHECKING:  # `llm` pulls in openai; see the module docstring
    from llm import Chat

#: One bracketed citation. Non-greedy by construction -- ``[^\[\]]`` cannot cross a bracket -- so
#: ``"... [CS 374]. ... [CS 225]."`` yields two spans rather than one span holding the sentence
#: between them. Markdown emphasis and code fences do not use square brackets, and the prompt asks
#: for no link syntax, so a bracket in a draft is a citation attempt.
CITATION_RE = re.compile(r"\[([^\[\]]*)\]")

#: What separates two ids inside one pair of brackets, when a bracket is not an id itself. Applied
#: only after the whole bracket has failed to match a ``doc_id``, because a program's doc_id *is*
#: comma-shaped -- ``Computer Science, BS`` -- and splitting first would turn the one citation this
#: repo's requirement rows can carry into two fabrications.
CITATION_SPLIT_RE = re.compile(r"[;,]")


class Citations(NamedTuple):
    """What the answer cited, split by whether this run can show the block it names.

    Both lists are in order of first appearance and hold no repeats: they go into ``signals``,
    where Day 5 counts them, and "cited CS 999 four times" is one fabrication.
    """

    #: The ids the answer cited that are in ``docs``, spelled the way ``docs`` spells them.
    verified: list[str]
    #: The ids it cited that are not -- each kept as the answer wrote it, for the count and for
    #: the hand-off sentence, which quotes it back.
    unknown: list[str]


def normalize_id(text: str) -> str:
    """A ``doc_id`` as it is compared: whitespace collapsed, case folded.

    The two differences that are not fabrications. ``[cs 374]`` and ``[CS  374]`` name the block
    headed ``CS 374`` and there is no ambiguity about which one; a difference in the *characters*
    of the id -- ``CS374``, ``CS-374``, ``2026-sp:CS 374`` -- is left as a miss on purpose. This
    module is not the place where a course code is respelled (:func:`agent.tools.courses.
    normalize_code` is, and it has no opinion about chunk ids or program labels), and one rule that
    holds for all three id shapes is worth more here than three that each fit one.
    """
    return " ".join(text.split()).casefold()


def citation_spans(answer: str) -> list[str]:
    """The contents of every pair of brackets in ``answer``, in order, stripped.

    Empty brackets are dropped rather than recorded: ``[]`` names no id, so there is nothing to
    look up and nothing for a report to count. An answer whose only bracket is empty has cited
    nothing, which is the case :func:`verify_citations` already refuses.
    """
    spans = (match.group(1).strip() for match in CITATION_RE.finditer(answer))
    return [span for span in spans if span]


def split_citation(span: str) -> list[str]:
    """The ids in one bracket that was not a ``doc_id`` itself -- ``"CS 374, CS 225"`` -> two.

    The prompt asks for one id per pair of brackets; this is what happens when it did not get it,
    and it is a parse rather than a repair: each part is then looked up on its own terms, so a
    bracket holding one real id and one invented one contributes exactly one fabrication.
    """
    parts = (part.strip() for part in CITATION_SPLIT_RE.split(span))
    return [part for part in parts if part]


def verify_citations(answer: str, docs: Sequence[Evidence]) -> Citations:
    """Every id ``answer`` cites, checked against the ids this run actually retrieved.

    A bracket is matched whole first and only then split on commas and semicolons, so
    ``[Computer Science, BS]`` -- a real ``doc_id`` from the requirement row -- stays one citation,
    while ``[CS 374, CS 225]`` becomes the two the writer meant.
    """
    known = {normalize_id(block.doc_id): block.doc_id for block in docs}
    verified: list[str] = []
    unknown: list[str] = []

    def record(target: list[str], value: str) -> None:
        if value not in target:
            target.append(value)

    for span in citation_spans(answer):
        whole = known.get(normalize_id(span))
        if whole is not None:
            record(verified, whole)
            continue
        for part in split_citation(span):
            found = known.get(normalize_id(part))
            if found is not None:
                record(verified, found)
            else:
                record(unknown, part)
    return Citations(verified, unknown)


def answer_node(cfg: Config, chat: Chat | None = None) -> Node:
    """Build the answer node: the evidence in, an answer with checkable citations out.

    ``chat`` replaces the client for the lifetime of the node -- anything with a ``complete``
    method is a :class:`llm.Chat` -- which is how the prompt assembly, the citation check and the
    downgrade are tested with no network and no credits.

    Nothing else is injectable. There is no "skip the check" switch and no threshold on how many
    citations are enough: the check is the node's reason to exist, and a run whose citations were
    not verified must not be reachable from a config or a keyword, because #39's report would have
    no column to say so in.
    """
    if chat is None:
        from llm import check_llm_config

        check_llm_config(cfg)
    built: list[Chat] = []

    def chat_client() -> Chat:
        """The client, made once per node and only if a query actually reaches it."""
        if chat is not None:
            return chat
        if not built:
            from llm import make_chat

            built.append(make_chat(cfg))
        return built[0]

    def node(state: AgentState) -> dict[str, Any]:
        docs = state.get("docs") or []
        if not docs:
            raise ValueError(
                "the answer node was reached with no evidence; the gate (#37) returns "
                "evidence_ok=False for an empty `docs` and the run hands over, so this is a "
                "wiring bug in agent/graph.py rather than a query"
            )
        draft = chat_client().complete(
            answer_user_prompt(state["query"], docs), system=ANSWER_SYSTEM_PROMPT
        )
        if not isinstance(draft, str):
            # A client that broke the `complete() -> str` contract (llm.py), which means a fake:
            # raised rather than repr()'d, because a repr carries square brackets of its own and
            # would be reported as a fabricated citation -- a client bug filed as a model failure.
            raise TypeError(
                f"the chat client returned {type(draft).__name__}; llm.Chat.complete returns the "
                "completion as a str (see llm.py) -- there is nothing to verify citations in"
            )
        citations = verify_citations(draft, docs)
        if citations.unknown:
            # The draft is dropped, not shown with a warning: a claim whose citation cannot be
            # found is a claim with nothing behind it, and the rest of the answer was written by
            # the same pass. `citations_unknown` is what the hand-off message quotes.
            return {
                "exit": "refuse_handoff",
                "signals": {
                    "citations_verified": citations.verified,
                    "citations_unknown": citations.unknown,
                },
            }
        if not citations.verified:
            # Nothing invented, nothing traceable either. Rule 4 of the prompt means even "the
            # catalog does not say" carries a citation, so an uncited draft is not a modest answer.
            return {
                "exit": "refuse_handoff",
                "signals": {"citations_verified": [], "citations_unknown": None},
            }
        return {
            "answer": draft,
            "exit": "answer",
            "signals": {"citations_verified": citations.verified, "citations_unknown": None},
        }

    node.__name__ = "answer"
    return node
