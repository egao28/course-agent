"""Prompt to text: the one chat client both of the agent's model calls go through.

``agent.nodes.route_intent`` classifies a query into one of five intents and ``agent.nodes.answer``
writes the grounded answer from the evidence. Those are the only two places this repo asks for a
chat completion, and neither names a model of its own -- the arrangement ``embedding.py`` already
has for the one encoder the build side and the query side share. What model that is comes from
``llm`` in ``config.yaml``::

    llm:
      provider: openai
      model: gpt-4o-mini
      temperature: 0        # a classification and a grounded answer are both deterministic tasks
      max_tokens: 800

That section is part of the config hash, so a report is named after the prompt settings that
produced it. It reaches **neither index key** -- ``dense_key`` hashes the chunk text and the
encoder, ``bm25_key`` the chunk text and the ``bm25`` section -- which is the point: adding a chat
model to this repo must not rename, and so orphan, an index that cost money to build.

Every call is injectable, and that is not a nicety. The account behind this repo has no credits
(an API call returns 429 ``insufficient_quota``), so until it has, the agent's nodes run against a
fake: anything with a ``complete`` method is a :class:`Chat`, and ``make_chat(cfg, client=...)``
replaces the SDK object underneath :class:`OpenAIChat` with a stub. Prompt assembly, label
validation and the citation check are therefore all testable with no network and no
``OPENAI_API_KEY``; the live run is the last step of Day 2, not a prerequisite for any of it.

:class:`StandInChat`, at the end of this file, is the one canned client that ships: it answers
with the first line of the evidence it was handed and cites that block's id, so that Day 2's smoke
report can be produced end to end before there are credits. It is not a model, it is reachable
only through an explicit ``--fake-model``, and it calls itself ``stand-in`` wherever it is printed
-- a report that read as though a model had been in the loop would be worse than no report.

:meth:`OpenAIChat.complete` returns a plain ``str`` rather than a response object precisely so that
such a fake is three lines long. Nothing here counts tokens: an embedding run bills per corpus and
is worth a cost line, while a chat run bills per query and the usage page is the honest source.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Protocol

# One key, one message for a missing one: embedding.py made the first API call in this repo and
# owns that check, so this module reuses it instead of writing a second sentence about .env.
from embedding import openai_client

#: The ``llm.provider`` values this module implements. ``embedding.provider`` also honours
#: ``local``, deliberately not mirrored here: the models that fit on this laptop route badly and
#: answer worse, so a config asking for a local chat route would name a run that cannot happen.
PROVIDERS = ("openai",)


class Chat(Protocol):
    """What the router and the answer node need: a model name, one prompt in, one string out."""

    model: str

    def complete(
        self, prompt: str, *, system: str | None = None, temperature: float | None = None
    ) -> str:
        """One prompt in, the completion text out -- no message list, no streaming, no tool calls.

        ``system`` and ``temperature`` override the implementation's defaults when given.
        """
        ...


# --- config ---------------------------------------------------------------------------------------


def check_llm_config(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate the ``llm`` section against what this module implements; return it.

    Reported as a sentence naming the offending key and the file to fix it in, never as a
    ``KeyError`` from whichever line happened to read it first -- the rule
    :func:`embedding.check_embedding_config` follows, for the same reason: these values are hashed
    into every report name, so a config describing a run that cannot happen is worth refusing
    before the run rather than after it.
    """
    if "llm" not in cfg:
        raise ValueError(
            "config has no `llm` section; it must set provider, model, temperature and max_tokens "
            "(see config.yaml) -- the intent router and the answer node both read it"
        )
    section = cfg["llm"]
    missing = [
        key for key in ("provider", "model", "temperature", "max_tokens") if key not in section
    ]
    if missing:
        raise ValueError(
            f"config llm section is missing {', '.join(missing)}; it must set provider, model, "
            "temperature and max_tokens (see config.yaml)"
        )
    provider = section["provider"]
    if provider not in PROVIDERS:
        raise ValueError(
            f"unknown llm.provider {provider!r}; known providers: {', '.join(PROVIDERS)}"
        )
    model = section["model"]
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"llm.model must be a non-empty string, got {model!r}")
    temperature = section["temperature"]
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not 0 <= temperature <= 2
    ):
        raise ValueError(f"llm.temperature must be a number in [0, 2], got {temperature!r}")
    max_tokens = section["max_tokens"]
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        # True is an int in Python and would cap every completion at a single token.
        raise ValueError(f"llm.max_tokens must be a positive integer, got {max_tokens!r}")
    return section


# --- providers ------------------------------------------------------------------------------------


class OpenAIChat:
    """One chat completion per :meth:`complete`, and no conversation state between them.

    Both callers are single-turn -- a classification, and an answer written from the evidence it is
    given -- so there is no history worth keeping, and keeping none means a query cannot inherit
    anything from the one before it.

    ``max_retries`` is above the SDK default for the reason ``OpenAIEmbedder`` raises it too: a
    smoke run is ten queries and two calls each, and a rate-limited request that gives up costs
    the report a row rather than a retry.

    ``timeout`` is per request, and far below the SDK's 600 s default: #87's signal run makes ~600
    calls a half, and one request that hung without an error stalled a live run for over ten
    minutes on 2026-09-28. A timed-out request is retried like any other.
    """

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 800,
        client: Any | None = None,
        max_retries: int = 5,
        timeout: float = 60.0,
    ) -> None:
        """Build a client for ``model``, or take ``client`` as given so a test can inject a fake.

        ``temperature`` and ``max_tokens`` are the per-call defaults; :meth:`complete` overrides
        the first of them per call.
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = (
            openai_client(max_retries=max_retries, timeout=timeout) if client is None else client
        )

    def complete(
        self, prompt: str, *, system: str | None = None, temperature: float | None = None
    ) -> str:
        """The model's reply to ``prompt``, stripped; ``system`` leads the message list if given.

        ``temperature`` overrides ``llm.temperature`` for this one call and no other. The router
        and the answer node never pass it -- both are deterministic tasks -- and Day 5's
        self-consistency samples (#87) do, at ``signals.sample_temperature``: a per-call value
        rather than a second client, so the samples go through the same model, the same
        ``max_tokens`` and the same three refusals below as the answer they are compared with.

        The reply comes back stripped because one caller compares it against five exact labels and
        the other renders it as markdown, so neither has a use for leading whitespace.

        Three ways a completion can arrive without an answer in it, all raised rather than returned
        as a string the caller would have to re-inspect: no choices at all, a cut-off completion,
        and an empty (or refused) message. The cut-off case matters most for the answer node: a
        truncated answer keeps its citation and loses the claim the citation was supporting, which
        is exactly the failure the citation check cannot see.
        """
        messages: list[dict[str, str]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens,
        )
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise RuntimeError(f"chat completion for model {self.model!r} returned no choices")
        choice = choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            raise RuntimeError(
                f"chat completion was cut off at the max_tokens ceiling ({self.max_tokens}); "
                "raise llm.max_tokens in config.yaml rather than use a half-written answer"
            )
        message = choice.message
        content = getattr(message, "content", None)
        if content is None or not content.strip():
            refusal = getattr(message, "refusal", None)
            raise RuntimeError(
                f"chat completion for model {self.model!r} came back empty"
                + (f" (refusal: {refusal})" if refusal else "")
            )
        return content.strip()


def make_chat(cfg: Mapping[str, Any], client: Any | None = None) -> Chat:
    """The chat client ``config.yaml`` asks for, validated first.

    ``client`` replaces the SDK object inside :class:`OpenAIChat`, so a caller gets the real prompt
    assembly and the real error handling over a stub transport -- and skips the
    ``OPENAI_API_KEY`` check, because nothing will be sent. That is how every node is tested, and
    how the smoke run works at all while the account has no credits.
    """
    section = check_llm_config(cfg)
    if section["provider"] == "openai":
        return OpenAIChat(
            section["model"],
            temperature=float(section["temperature"]),
            max_tokens=int(section["max_tokens"]),
            client=client,
        )
    # Unreachable through check_llm_config: here so that adding a PROVIDERS entry without adding a
    # client for it fails by name instead of quietly returning an OpenAI one.
    raise ValueError(f"llm.provider {section['provider']!r} has no client in llm.py")


# --- the stand-in ---------------------------------------------------------------------------------


#: What :class:`StandInChat` reads out of an answer prompt: the first evidence block's ``doc_id``
#: and the first line of its text. The shape is :data:`agent.prompts.EVIDENCE_TEMPLATE`'s, matched
#: here rather than imported -- ``agent`` imports this module, and the dependency runs one way
#: only -- and ``tests/test_llm.py`` renders a real prompt through it, so the two cannot drift
#: without failing.
STANDIN_BLOCK_RE = re.compile(r"^doc_id: (?P<doc_id>.+)\n-{3}\n(?P<line>.+)$", re.MULTILINE)


class StandInChat:
    """Not a model: the canned completion that lets a run finish while the account has no credits.

    It answers with the first line of the first evidence block and cites that block's ``doc_id``,
    so everything around the words is real -- the prompt #38 assembled, the citation parse, the
    verification against ``docs``, the exit the node claims -- and only the words are canned. That
    is the most a run can honestly demonstrate with nothing behind it, and it is what
    ``--fake-model`` puts behind the answer node in ``python -m agent`` and in Day 2's smoke run.

    **It cannot classify, and refuses rather than guessing.** A prompt with no evidence block in it
    is the router's (#35), and a stand-in that returned a label there would be a routing decision
    nobody made -- reported in #39's table beside labels a model did produce. Force the label with
    ``--intent`` instead; #35 then records ``intent_raw = None``, which is how a forced row stays
    visible as one.

    **Nothing it produces may be reported as a model's work.** ``model`` says ``stand-in`` rather
    than naming the configured model, the trace prints it on every run, and the smoke report says
    it in its header and beside every answer -- a report that reads as though a model was in the
    loop when it was not is worse than no report.
    """

    #: What the trace and the report print as the source of the answer. Deliberately not the
    #: configured model name: a stand-in run must never be mistaken for a run of ``gpt-4o-mini``.
    model = "stand-in"

    def complete(
        self, prompt: str, *, system: str | None = None, temperature: float | None = None
    ) -> str:
        """The first evidence block's opening line, with that block's id cited after it.

        ``temperature`` is accepted and ignored: a canned reply has nothing to sample.
        """
        match = STANDIN_BLOCK_RE.search(prompt)
        if match is None:
            raise ValueError(
                "a stand-in has nothing to answer with: it quotes the evidence in the prompt, and "
                "this prompt carries none -- a classification (#35) is not something it can fake, "
                "so force the label with --intent or put a real client behind the router"
            )
        return f"{match['line'].strip()} [{match['doc_id'].strip()}]"
