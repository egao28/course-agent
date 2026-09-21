"""Tests for llm.py: the config guards, what a completion returns, and the injected client.

No network and no API key anywhere here, which is the whole point of the module: the account has
no credits, so the router and the answer node have to be buildable and testable against a fake.
The OpenAI route is exercised through a stub that mimics the shape of a real response, and the
:class:`llm.Chat` protocol through a canned client -- a node depends on ``complete``, not on the
SDK, and both stand-ins prove it separately.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import llm
from config import Config, bm25_key, dense_key, load_config
from ingest.chunk import chunk_key

BASE: dict = {
    "llm": {"provider": "openai", "model": "gpt-4o-mini", "temperature": 0, "max_tokens": 800}
}


@pytest.fixture
def cfg() -> Config:
    return Config(BASE)


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch) -> None:
    """Nothing in this file may need a key; a test that does has to ask for one explicitly."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


class FakeCompletions:
    """The ``client.chat.completions`` namespace, recording what it was asked for."""

    def __init__(
        self,
        content: str | None = "course_fact",
        finish_reason: str = "stop",
        choices: int = 1,
        refusal: str | None = None,
    ) -> None:
        self.content, self.finish_reason = content, finish_reason
        self.choices, self.refusal = choices, refusal
        self.calls: list[dict] = []

    def create(self, model: str, messages: list[dict], temperature: float, max_tokens: int):
        self.calls.append(
            {
                "model": model,
                "messages": list(messages),
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        message = SimpleNamespace(content=self.content, refusal=self.refusal)
        items = [
            SimpleNamespace(message=message, finish_reason=self.finish_reason)
            for _ in range(self.choices)
        ]
        return SimpleNamespace(choices=items)


def fake_client(**kwargs) -> SimpleNamespace:
    return SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(**kwargs)))


class CannedChat:
    """A whole :class:`llm.Chat` in three lines -- what every Day 2 node is tested against."""

    model = "canned"

    def __init__(self, reply: str) -> None:
        self.reply, self.prompts = reply, []

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        self.prompts.append((system, prompt))
        return self.reply


def classify(chat: llm.Chat, query: str) -> str:
    """A stand-in for #35's node: the only thing a caller needs from a chat client."""
    return chat.complete(query, system="Reply with one label.")


# --- config guards --------------------------------------------------------------------------------


def test_check_llm_config_returns_the_section(cfg):
    assert llm.check_llm_config(cfg)["model"] == "gpt-4o-mini"


def test_the_shipped_config_describes_a_run_that_can_happen():
    # The live config.yaml, not a fixture: its llm section is hashed into every report name, so it
    # has to name a provider this module actually has a client for.
    section = llm.check_llm_config(load_config())
    assert section["provider"] in llm.PROVIDERS


def test_check_llm_config_reports_a_missing_section_as_a_sentence():
    with pytest.raises(ValueError, match="no `llm` section"):
        llm.check_llm_config(Config({"seed": 42}))


def test_check_llm_config_names_the_keys_it_needs():
    with pytest.raises(ValueError, match="missing model, temperature, max_tokens"):
        llm.check_llm_config(Config({"llm": {"provider": "openai"}}))


@pytest.mark.parametrize("provider", ["local", "cohere"])
def test_check_llm_config_rejects_a_provider_with_no_client(cfg, provider):
    # `local` included on purpose: embedding.provider honours it and llm.provider does not, so a
    # config that copied the wrong axis must be told, not quietly given an OpenAI client.
    other = Config({"llm": {**cfg["llm"], "provider": provider}})
    with pytest.raises(ValueError, match="unknown llm.provider"):
        llm.check_llm_config(other)


@pytest.mark.parametrize("model", ["", "   ", 3])
def test_check_llm_config_rejects_an_empty_model(cfg, model):
    with pytest.raises(ValueError, match="llm.model"):
        llm.check_llm_config(Config({"llm": {**cfg["llm"], "model": model}}))


@pytest.mark.parametrize("temperature", [None, "0", True, -0.5, 2.5])
def test_check_llm_config_rejects_a_temperature_outside_the_api_range(cfg, temperature):
    # True is an int in Python, and a temperature the API refuses would fail one call into a run.
    with pytest.raises(ValueError, match="llm.temperature"):
        llm.check_llm_config(Config({"llm": {**cfg["llm"], "temperature": temperature}}))


@pytest.mark.parametrize("max_tokens", [0, -1, 1.5, "800", True])
def test_check_llm_config_rejects_a_bad_max_tokens(cfg, max_tokens):
    with pytest.raises(ValueError, match="llm.max_tokens"):
        llm.check_llm_config(Config({"llm": {**cfg["llm"], "max_tokens": max_tokens}}))


@pytest.mark.parametrize("temperature", [0, 0.0, 1, 2])
def test_check_llm_config_accepts_ints_and_floats_in_range(cfg, temperature):
    # YAML reads `temperature: 0` as an int; the section must not have to be written as `0.0`.
    assert llm.check_llm_config(Config({"llm": {**cfg["llm"], "temperature": temperature}}))


# --- the OpenAI route -----------------------------------------------------------------------------


def test_openai_chat_sends_the_system_prompt_first_and_the_settings_from_the_config():
    chat = llm.OpenAIChat("gpt-4o-mini", temperature=0.0, max_tokens=800, client=fake_client())
    assert chat.complete("How many credits is CS 374?", system="Reply with one label.") == (
        "course_fact"
    )

    call = chat._client.chat.completions.calls[0]
    assert call["model"] == "gpt-4o-mini"
    assert call["temperature"] == 0.0
    assert call["max_tokens"] == 800
    assert [m["role"] for m in call["messages"]] == ["system", "user"]
    assert call["messages"][0]["content"] == "Reply with one label."
    assert call["messages"][1]["content"] == "How many credits is CS 374?"


def test_openai_chat_sends_one_message_when_there_is_no_system_prompt():
    chat = llm.OpenAIChat("m", client=fake_client())
    chat.complete("hi")
    assert [m["role"] for m in chat._client.chat.completions.calls[0]["messages"]] == ["user"]


def test_openai_chat_strips_the_reply():
    # #35 compares the reply against five exact labels, so the stripping belongs here, once.
    chat = llm.OpenAIChat("m", client=fake_client(content="  prereq_chain\n"))
    assert chat.complete("what comes before SE 495?") == "prereq_chain"


def test_openai_chat_keeps_no_state_between_calls():
    chat = llm.OpenAIChat("m", client=fake_client())
    chat.complete("first")
    chat.complete("second")
    calls = chat._client.chat.completions.calls
    assert [c["messages"][-1]["content"] for c in calls] == ["first", "second"]
    assert all(len(c["messages"]) == 1 for c in calls)  # no history carried over


def test_openai_chat_refuses_a_response_with_no_choices():
    with pytest.raises(RuntimeError, match="returned no choices"):
        llm.OpenAIChat("m", client=fake_client(choices=0)).complete("hi")


def test_openai_chat_refuses_a_cut_off_completion_and_names_the_ceiling():
    # A truncated answer keeps its citation and loses the claim it supported -- the one failure
    # #38's citation check cannot see, so it is an error here rather than a returned string.
    chat = llm.OpenAIChat("m", max_tokens=8, client=fake_client(finish_reason="length"))
    with pytest.raises(RuntimeError, match=r"cut off at the max_tokens ceiling \(8\)"):
        chat.complete("summarize the catalog")


@pytest.mark.parametrize("content", [None, "   "])
def test_openai_chat_refuses_an_empty_completion(content):
    with pytest.raises(RuntimeError, match="came back empty"):
        llm.OpenAIChat("m", client=fake_client(content=content)).complete("hi")


def test_openai_chat_quotes_the_refusal_when_the_model_declined():
    client = fake_client(content=None, refusal="I cannot help with that")
    with pytest.raises(RuntimeError, match="refusal: I cannot help with that"):
        llm.OpenAIChat("m", client=client).complete("hi")


def test_openai_chat_needs_a_key_only_when_it_builds_its_own_client():
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        llm.OpenAIChat("m")


# --- dispatch -------------------------------------------------------------------------------------


def test_make_chat_builds_an_openai_client_from_the_config(cfg, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    chat = llm.make_chat(cfg)
    assert isinstance(chat, llm.OpenAIChat)
    assert chat.model == "gpt-4o-mini"
    assert chat.temperature == 0.0  # the config's int, as a float
    assert chat.max_tokens == 800


def test_make_chat_takes_an_injected_client_and_never_looks_for_a_key(cfg):
    # The credits blocker in one test: with no OPENAI_API_KEY in the environment, a node still
    # gets a working Chat, with this module's prompt assembly and error handling intact.
    chat = llm.make_chat(cfg, client=fake_client(content="degree_requirement"))
    assert classify(chat, "Which Gen Eds do I still need?") == "degree_requirement"


def test_make_chat_validates_before_it_builds_anything(cfg):
    with pytest.raises(ValueError, match="unknown llm.provider"):
        llm.make_chat(Config({"llm": {**cfg["llm"], "provider": "x"}}))


def test_make_chat_refuses_a_listed_provider_with_no_client(cfg, monkeypatch):
    # PROVIDERS and the dispatch in make_chat are two lists that have to agree; adding to one and
    # forgetting the other must fail by name instead of handing back an OpenAI client.
    monkeypatch.setattr(llm, "PROVIDERS", ("openai", "anthropic"))
    other = Config({"llm": {**cfg["llm"], "provider": "anthropic"}})
    with pytest.raises(ValueError, match="has no client in llm.py"):
        llm.make_chat(other)


# --- the protocol ---------------------------------------------------------------------------------


def test_a_canned_client_is_a_chat_and_records_what_it_was_asked():
    # What every Day 2 node's tests will pass in: no client, no config, no network.
    chat = CannedChat("out_of_scope")
    assert classify(chat, "What's the weather in Champaign?") == "out_of_scope"
    assert chat.prompts == [("Reply with one label.", "What's the weather in Champaign?")]


def test_the_llm_section_renames_no_index():
    # #28's acceptance, asserted rather than eyeballed: adding `llm:` to config.yaml moves the full
    # config hash (the reports get new names) and must move none of the three keys an artifact on
    # disk is named after. The live config is the one that matters here -- a fixture could agree
    # while the shipped file had put the section somewhere that does reach a key.
    live = load_config()
    without = Config({k: v for k, v in live.to_dict().items() if k != "llm"})

    assert "llm" in live
    assert live.hash != without.hash
    assert chunk_key(live) == chunk_key(without)
    assert dense_key(live, chunk_key(live)) == dense_key(without, chunk_key(without))
    assert bm25_key(live, chunk_key(live)) == bm25_key(without, chunk_key(without))


def test_both_clients_satisfy_the_same_caller():
    over_the_sdk = llm.OpenAIChat("m", client=fake_client(content="compare_recommend"))
    canned = CannedChat("compare_recommend")
    query = "Which intro ML course should I take?"
    assert classify(over_the_sdk, query) == classify(canned, query)
