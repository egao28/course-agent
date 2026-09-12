"""Tests for embedding.py: the config guards, provider dispatch, and what a batch returns.

No network and no API key anywhere here. The OpenAI route is exercised through a fake client that
mimics the shape of a real response, and the local route through a fake encoder -- importing the
real ``sentence_transformers`` would download a model, and the point of these tests is the wiring
around the provider, not the provider.

``no_sentence_transformers`` fakes the *absence* of the optional extra rather than relying on it
being missing from this venv, so the error-message test means the same thing after
``uv sync --extra local``.
"""

from __future__ import annotations

import builtins
from types import SimpleNamespace

import pytest

import embedding as em
from config import Config, load_config

BASE: dict = {
    "embedding": {"provider": "openai", "model": "text-embedding-3-small", "batch_size": 256}
}


@pytest.fixture
def cfg() -> Config:
    return Config(BASE)


class FakeEmbeddings:
    """The ``client.embeddings`` namespace, recording what it was asked for."""

    def __init__(self, dim: int = 4, tokens: int | None = 7, shuffle: bool = False) -> None:
        self.dim, self.tokens, self.shuffle = dim, tokens, shuffle
        self.calls: list[dict] = []

    def create(self, model: str, input: list[str]):  # the API's own parameter name
        self.calls.append({"model": model, "input": list(input)})
        items = [
            SimpleNamespace(index=i, embedding=[float(i)] * self.dim) for i in range(len(input))
        ]
        if self.shuffle:
            items.reverse()
        usage = None if self.tokens is None else SimpleNamespace(total_tokens=self.tokens)
        return SimpleNamespace(data=items, usage=usage)


def fake_client(**kwargs) -> SimpleNamespace:
    return SimpleNamespace(embeddings=FakeEmbeddings(**kwargs))


class FakeEncoder:
    """A sentence-transformers model stand-in: records kwargs, returns fixed vectors."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def encode(self, texts, **kwargs):
        self.calls.append({"texts": list(texts), **kwargs})
        return [[0.6, 0.8] for _ in texts]


@pytest.fixture
def no_sentence_transformers(monkeypatch) -> None:
    """A venv without the ``local`` extra, whether or not this one has it."""
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("No module named 'sentence_transformers'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


# --- config guards --------------------------------------------------------------------------------


def test_check_embedding_config_returns_the_section(cfg):
    assert em.check_embedding_config(cfg)["model"] == "text-embedding-3-small"


def test_the_shipped_config_names_a_provider_this_module_implements():
    # The live config.yaml, not a fixture: its embedding section is hashed into the collection
    # name, so it has to describe a build that can actually happen.
    assert em.check_embedding_config(load_config())["provider"] in em.PROVIDERS


def test_check_embedding_config_reports_a_missing_section_as_a_sentence():
    with pytest.raises(ValueError, match="no `embedding` section"):
        em.check_embedding_config(Config({"seed": 42}))


def test_check_embedding_config_names_the_keys_it_needs():
    cfg = Config({"embedding": {"provider": "openai"}})
    with pytest.raises(ValueError, match="missing model, batch_size"):
        em.check_embedding_config(cfg)


def test_check_embedding_config_rejects_an_unknown_provider(cfg):
    other = Config({"embedding": {**cfg["embedding"], "provider": "cohere"}})
    with pytest.raises(ValueError, match="unknown embedding.provider"):
        em.check_embedding_config(other)


@pytest.mark.parametrize("model", ["", "   ", 3])
def test_check_embedding_config_rejects_an_empty_model(cfg, model):
    other = Config({"embedding": {**cfg["embedding"], "model": model}})
    with pytest.raises(ValueError, match="embedding.model"):
        em.check_embedding_config(other)


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, "256", True])
def test_check_embedding_config_rejects_a_bad_batch_size(cfg, batch_size):
    # True is an int in Python and would batch one chunk per request: 4,502 API calls.
    other = Config({"embedding": {**cfg["embedding"], "batch_size": batch_size}})
    with pytest.raises(ValueError, match="embedding.batch_size"):
        em.check_embedding_config(other)


# --- cost -----------------------------------------------------------------------------------------


def test_estimate_cost_is_the_list_price_per_million_tokens():
    assert em.estimate_cost("text-embedding-3-small", 1_000_000) == pytest.approx(0.02)
    assert em.estimate_cost("text-embedding-3-small", 522_658) == pytest.approx(0.0104, abs=1e-4)


@pytest.mark.parametrize(("model", "tokens"), [("bge-m3", 1000), ("text-embedding-3-small", None)])
def test_estimate_cost_is_none_when_it_cannot_be_known(model, tokens):
    # A report that says nothing beats one that quotes a made-up price.
    assert em.estimate_cost(model, tokens) is None


# --- the OpenAI route -----------------------------------------------------------------------------


def test_openai_embedder_returns_one_vector_per_text_with_the_billed_tokens():
    embedder = em.OpenAIEmbedder("text-embedding-3-small", client=fake_client())
    result = embedder.embed(["a", "b", "c"])

    assert len(result.vectors) == 3
    assert result.tokens == 7
    assert embedder._client.embeddings.calls[0]["model"] == "text-embedding-3-small"


def test_openai_embedder_restores_input_order_from_the_item_index():
    # The vector for input i must be vectors[i] even if the API returns the items in another
    # order; a silently mislabelled vector is a wrong answer no test downstream could see.
    embedder = em.OpenAIEmbedder("m", client=fake_client(shuffle=True))
    result = embedder.embed(["first", "second", "third"])
    assert [v[0] for v in result.vectors] == [0.0, 1.0, 2.0]


def test_openai_embedder_refuses_a_short_response():
    client = fake_client()
    original = client.embeddings.create
    client.embeddings.create = lambda model, input: original(model=model, input=input[:-1])
    with pytest.raises(RuntimeError, match="returned 2 vectors for 3 inputs"):
        em.OpenAIEmbedder("m", client=client).embed(["a", "b", "c"])


def test_openai_embedder_skips_the_call_for_no_texts():
    embedder = em.OpenAIEmbedder("m", client=fake_client())
    assert embedder.embed([]) == em.Vectors([], 0)
    assert embedder._client.embeddings.calls == []


def test_openai_embedder_tolerates_a_response_without_usage():
    embedder = em.OpenAIEmbedder("m", client=fake_client(tokens=None))
    assert embedder.embed(["a"]).tokens is None


def test_openai_client_names_the_missing_key_and_where_to_put_it(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        em.openai_client()


def test_openai_client_is_built_when_the_key_is_there(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert em.openai_client().api_key == "sk-test"  # constructed, nothing sent


# --- the local route ------------------------------------------------------------------------------


def test_local_embedder_normalizes_and_reports_no_token_cost():
    encoder = FakeEncoder()
    result = em.LocalEmbedder("BAAI/bge-m3", encoder=encoder).embed(["a", "b"])

    assert result.vectors == [[0.6, 0.8], [0.6, 0.8]]
    assert result.tokens is None  # nothing was billed, so nothing is reported as billed
    assert encoder.calls[0]["normalize_embeddings"] is True


def test_local_embedder_skips_the_call_for_no_texts():
    encoder = FakeEncoder()
    assert em.LocalEmbedder("m", encoder=encoder).embed([]) == em.Vectors([], None)
    assert encoder.calls == []


def test_local_embedder_points_at_the_extra_when_it_is_not_installed(no_sentence_transformers):
    with pytest.raises(RuntimeError, match=r"uv sync --locked --extra local"):
        em.LocalEmbedder("BAAI/bge-m3")


# --- dispatch -------------------------------------------------------------------------------------


def test_make_embedder_routes_on_the_provider(cfg, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    embedder = em.make_embedder(cfg)
    assert isinstance(embedder, em.OpenAIEmbedder)
    assert embedder.model == "text-embedding-3-small"


def test_make_embedder_routes_local_without_touching_openai(cfg, no_sentence_transformers):
    local = Config({"embedding": {**cfg["embedding"], "provider": "local", "model": "BAAI/bge-m3"}})
    # Reaching the extra's error proves the local branch was taken -- no API key was consulted.
    with pytest.raises(RuntimeError, match="sentence-transformers is not installed"):
        em.make_embedder(local)


def test_make_embedder_validates_before_it_builds_anything(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="unknown embedding.provider"):
        em.make_embedder(Config({"embedding": {"provider": "x", "model": "m", "batch_size": 1}}))
