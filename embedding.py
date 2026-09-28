"""Text to vector: the one encoder both sides of the dense index go through.

``ingest.build_dense`` embeds 4,502 chunks once; ``retrieval.dense`` embeds one query per search.
Those two must produce vectors in the *same* space -- a query encoded by a different model than the
corpus returns confident nonsense, with no error anywhere -- so both call :func:`make_embedder`
and neither names a model of its own. What model that is comes from ``embedding`` in
``config.yaml``, which is hashed into the collection name, so a corpus and a query can only
disagree if the collection was built under a different config than the one being queried.

Two providers::

    embedding:
      provider: openai        # text-embedding-3-small over the API; needs OPENAI_API_KEY
      model: text-embedding-3-small
      batch_size: 256

    embedding:
      provider: local         # BGE-M3 through sentence-transformers; needs `uv sync --extra local`
      model: BAAI/bge-m3

Both return :class:`Vectors`; nothing downstream knows which one ran. Their embeddings live in
different spaces and have different dimensions, but ``provider`` and ``model`` are part of the
config hash, so switching providers names a new collection rather than polluting the old one.

``tokens`` is what the provider says it billed, and is ``None`` for a local model that bills
nothing -- so the cost line ``build_dense`` prints is a *reported* number, not an estimate of one.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, Protocol

import openai

#: The ``embedding.provider`` values this module implements.
PROVIDERS = ("openai", "local")

#: OpenAI list price in USD per 1M input tokens (as published 2026-09). Only ever used for the
#: cost line of a build report -- an out-of-date entry misreports a diagnostic, never a vector --
#: and an unknown model simply prints no price rather than a guessed one.
PRICE_PER_MTOK: dict[str, float] = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
}


class Vectors(NamedTuple):
    """One batch of embeddings, with the token count the provider reported for it."""

    vectors: list[list[float]]
    #: Tokens the provider billed, or ``None`` when it does not report any (local models).
    tokens: int | None


class Embedder(Protocol):
    """What ``build_dense`` and ``DenseRetriever`` need: a model name and a batch call."""

    model: str

    def embed(self, texts: Sequence[str]) -> Vectors: ...


# --- config ---------------------------------------------------------------------------------------


def check_embedding_config(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate the ``embedding`` section against what this module implements; return it.

    Every value here is hashed into the collection name, so a config naming a provider nothing
    honours would tag an index with a description of how it was *not* built. Reported as a
    sentence naming the offending key, not as a ``KeyError`` from the first line that reads it.
    """
    if "embedding" not in cfg:
        raise ValueError(
            "config has no `embedding` section; it must set provider, model and batch_size "
            "(see config.yaml) -- all three are hashed into the collection name"
        )
    section = cfg["embedding"]
    missing = [key for key in ("provider", "model", "batch_size") if key not in section]
    if missing:
        raise ValueError(
            f"config embedding section is missing {', '.join(missing)}; it must set provider, "
            "model and batch_size (see config.yaml) -- all three are hashed into the collection "
            "name"
        )
    provider = section["provider"]
    if provider not in PROVIDERS:
        raise ValueError(
            f"unknown embedding.provider {provider!r}; known providers: {', '.join(PROVIDERS)}"
        )
    model = section["model"]
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"embedding.model must be a non-empty string, got {model!r}")
    batch_size = section["batch_size"]
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"embedding.batch_size must be a positive integer, got {batch_size!r}")
    return section


def estimate_cost(model: str, tokens: int | None) -> float | None:
    """USD for ``tokens`` of ``model``, or ``None`` when either is unknown."""
    price = PRICE_PER_MTOK.get(model)
    if price is None or tokens is None:
        return None
    return price * tokens / 1_000_000


# --- providers ------------------------------------------------------------------------------------


def openai_client(**kwargs: Any) -> openai.OpenAI:
    """An OpenAI client, refusing early and by name when the key is missing.

    The SDK's own error for an unset key arrives without saying where this repo expects it, and
    would otherwise surface halfway into a build script. ``.env`` is read by ``uv run --env-file``
    here; wiring ``python-dotenv`` so a plain ``uv run`` picks it up is #11.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Either export it, or put it in .env "
            "(`cp -n .env.example .env`) and run with `uv run --env-file .env ...`"
        )
    return openai.OpenAI(**kwargs)


class OpenAIEmbedder:
    """``text-embedding-3-*`` over the API, one request per batch.

    The SDK retries a rate-limited or 5xx request on its own; ``max_retries`` is raised above its
    default because a build is ~18 sequential requests and losing one of them means re-running the
    whole thing (or paying for a ``--rebuild``).

    ``timeout`` is per request and well below the SDK's 600 s default. A query-side embedding that
    hung without an error stalled #87's live signal run for over 40 minutes on 2026-09-28. A
    256-text build batch takes seconds, so 120 s leaves it plenty of room.
    """

    def __init__(
        self,
        model: str,
        client: Any | None = None,
        max_retries: int = 5,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self._client = (
            openai_client(max_retries=max_retries, timeout=timeout) if client is None else client
        )

    def embed(self, texts: Sequence[str]) -> Vectors:
        texts = list(texts)
        if not texts:
            return Vectors([], 0)
        response = self._client.embeddings.create(model=self.model, input=texts)
        # The API documents the response as parallel to the input, but each item carries its own
        # index; sorting by it means a reordered response can never silently mislabel a vector.
        data = sorted(response.data, key=lambda item: item.index)
        if len(data) != len(texts):
            raise RuntimeError(
                f"embeddings API returned {len(data)} vectors for {len(texts)} inputs"
            )
        usage = getattr(response, "usage", None)
        vectors = [list(item.embedding) for item in data]
        return Vectors(vectors, getattr(usage, "total_tokens", None))


class LocalEmbedder:
    """BGE-M3 (or any sentence-transformers model) on this machine, no API and no key.

    ``sentence-transformers`` is an optional extra: it pulls in torch, which is most of a
    gigabyte and pointless for the OpenAI route. Imported here rather than at module scope so
    that cost is only paid by a config that asks for it.

    Vectors are L2-normalized. Cosine ranking does not care, but it makes the raw vectors
    comparable with a dot product, which is what a future re-ranking step would reach for.
    """

    def __init__(self, model: str, encoder: Any | None = None) -> None:
        self.model = model
        if encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    f"embedding.provider is 'local' but sentence-transformers is not installed; "
                    f"run `uv sync --locked --extra local` to get it (and {model} on first use)"
                ) from exc
            encoder = SentenceTransformer(model)
        self._encoder = encoder

    def embed(self, texts: Sequence[str]) -> Vectors:
        texts = list(texts)
        if not texts:
            return Vectors([], None)
        encoded = self._encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return Vectors([[float(x) for x in vector] for vector in encoded], None)


def make_embedder(cfg: Mapping[str, Any]) -> Embedder:
    """The embedder ``config.yaml`` asks for, validated first."""
    section = check_embedding_config(cfg)
    model = section["model"]
    if section["provider"] == "openai":
        return OpenAIEmbedder(model)
    return LocalEmbedder(model)
