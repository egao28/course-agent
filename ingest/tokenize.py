"""Text to terms: the tokenizer both sides of the BM25 index go through.

``ingest.build_bm25`` tokenizes 4,502 chunks once; ``retrieval.bm25`` tokenizes one query per
search. BM25 matches terms literally -- there is no embedding to smooth over a difference -- so a
document and a query that spell the same thing differently simply do not match, and nothing
anywhere reports it. Hence one module, one function, and ``bm25.tokenizer`` in ``config.yaml``
hashed into the index name, so a query can only be tokenized differently from the corpus if the
index was built under a different config than the one being queried.

Three things ``simple_v1`` does, in the order they matter:

**Course codes collapse to three terms.** ``CS 374`` yields ``cs``, ``374`` and ``cs374``, and so
do ``CS374`` and ``cs 374`` -- the code is matched as one unit before the text is split into
words, so all three spellings produce the *same* term stream and score identically. Without the
glued term, ``CS374`` (one word) and ``CS 374`` (two) would share no term at all.

**A course's own code is a different term from a code it merely cites.** This is what
``code_boost`` is for, and the corpus is why. ``ECE 374``'s entire description is "Same as CS 374.
See CS 374." -- one of 497 cross-reference-only descriptions (#15) -- so for the query ``CS 374``
it is a 20-term document matching ``cs374`` twice, while CS 374's own 110-term chunk matches it
once. BM25 is right on its own terms and ranks the stub first; it just has no way to know that one
of those documents *is* CS 374 and the other points at it. So every chunk carries its own code as
a reserved term (``code:cs374``, from ``metadata.code``), and a query naming a code asks for it.
Repeated ``bm25.code_boost`` times, because once is not enough against a citation: measured on the
live 2026-sp corpus, the query ``CS 374`` ranks CS 374 itself 11th at boost 0, 2nd at 1 (25.6
against the stub's 26.6) and 1st at 2 (29.0 against 26.5). The reserved prefix ends in ``:``, a
character ``\\w+`` cannot produce, so the namespace cannot collide with a term from the corpus.

**Stopwords are prose, and a code is not prose.** :data:`STOPWORDS` is deliberately small -- the
words that carry no signal in a catalog of 4,502 short course descriptions -- and it filters the
word branch only. That matters for one subject: ``IS`` is both a real UIUC subject and a stopword,
and because the code is matched as a unit first, ``IS 407`` still yields ``is``, ``407`` and
``is407``. Keeping the subject is the point of emitting it separately -- it is what makes ``CS 374``
rank other CS courses above unrelated ones -- so the one subject that collides with the stopword
list is not the one that quietly loses it.

The one deliberate imprecision: the code pattern is a shape, not a subject list, so "of 100
students" also yields ``of100``. It is harmless -- the term is only reachable by a query spelling
the same two tokens the same way, and the words are emitted regardless -- and it keeps the
tokenizer independent of which subjects the catalog happens to offer this term.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

#: A course code as the catalog spells it: 2-4 letters, an optional space, three digits. Defined
#: once and used twice -- alone to find the codes in a text, and as a branch of :data:`TOKEN_RE`
#: so that a code is consumed as a unit before it can be split into words.
CODE = r"\b(?P<subject>[A-Za-z]{2,4})[ \t]*(?P<number>\d{3})(?!\w)"
CODE_RE = re.compile(CODE)
TOKEN_RE = re.compile(rf"{CODE}|(?P<word>\w+)")

#: Prefix of the reserved term that says a document *is* this course. ``:`` cannot occur in a
#: ``\w+`` term, so nothing in the corpus can produce one of these by accident.
IDENTITY_PREFIX = "code:"

#: Dropped from both documents and queries. Small on purpose: in a corpus of course descriptions,
#: "credits" and "prerequisite" are not stopwords (they are the field labels the template renders),
#: and anything subject-specific would be a retrieval decision hiding in a word list.
# fmt: off
STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has", "have", "in",
    "is", "it", "its", "of", "on", "or", "that", "the", "their", "then", "there", "these", "they",
    "this", "to", "was", "were", "which", "will", "with",
})
# fmt: on


# --- tokenizers -----------------------------------------------------------------------------------


def _simple_v1(text: str) -> list[str]:
    """Lowercase ``\\w+`` terms without stopwords, plus ``subject``/``number``/``glued`` per code.

    One left-to-right scan, codes first: a match of the code branch consumes the whole code, so
    the digits can never come back as a bare word and the three spellings of one code produce
    identical output.
    """
    tokens: list[str] = []
    for match in TOKEN_RE.finditer(text):
        word = match.group("word")
        if word is None:
            subject, number = match.group("subject").lower(), match.group("number")
            tokens += [subject, number, subject + number]
        elif (word := word.lower()) not in STOPWORDS:
            tokens.append(word)
    return tokens


#: Every tokenizer, keyed by ``bm25.tokenizer``. A new one is a new entry and a new config value,
#: never an edit to an existing function: the name is hashed into the index name, so changing what
#: ``simple_v1`` means would leave an index claiming to hold terms it does not.
TOKENIZERS: dict[str, Callable[[str], list[str]]] = {"simple_v1": _simple_v1}


def tokenize(text: str, name: str = "simple_v1") -> list[str]:
    """Terms of ``text`` under the named tokenizer."""
    try:
        tokenizer = TOKENIZERS[name]
    except KeyError:
        known = ", ".join(sorted(TOKENIZERS))
        raise ValueError(f"unknown bm25.tokenizer {name!r}; known tokenizers: {known}") from None
    return tokenizer(text)


# --- config ---------------------------------------------------------------------------------------


def check_tokenizer(cfg: Mapping[str, Any]) -> str:
    """Validate ``bm25.tokenizer`` and return it, as a sentence naming the key when it is wrong."""
    section = cfg.get("bm25")
    if not section:
        raise ValueError(
            "config has no `bm25` section; it must set tokenizer, k1, b and code_boost "
            "(see config.yaml) -- all of them are hashed into the BM25 index name"
        )
    if "tokenizer" not in section:
        raise ValueError(
            "config bm25 section is missing tokenizer -- which function turns text into the terms "
            f"the index holds (see config.yaml). Known tokenizers: {', '.join(sorted(TOKENIZERS))}"
        )
    name = section["tokenizer"]
    if name not in TOKENIZERS:
        known = ", ".join(sorted(TOKENIZERS))
        raise ValueError(f"unknown bm25.tokenizer {name!r}; known tokenizers: {known}")
    return name


def get_tokenizer(cfg: Mapping[str, Any]) -> Callable[[str], list[str]]:
    """The tokenizer ``cfg`` names. Build side and query side both get it from here."""
    return TOKENIZERS[check_tokenizer(cfg)]


def code_boost(cfg: Mapping[str, Any]) -> int:
    """``bm25.code_boost`` -- how many times a document repeats its own code; ``0`` disables it.

    Read through one guard, like ``retrieval.types.top_k_each``: it changes the term stream, so
    the build and any tool that reproduces it must agree on the number, and a config that never
    set it gets a sentence naming the key rather than a ``KeyError`` from whichever line read it.
    """
    section = cfg.get("bm25")
    if not section or "code_boost" not in section:
        raise ValueError(
            "config bm25 section is missing code_boost -- how many times a chunk repeats its own "
            "course code, so a query naming a code ranks the course itself above one that cites "
            "it (see config.yaml). 0 disables it"
        )
    value = section["code_boost"]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"bm25.code_boost must be a non-negative integer, got {value!r}")
    return value


# --- term streams ---------------------------------------------------------------------------------


def identity_token(code: str) -> str:
    """The reserved term for a course code: ``"CS 374"`` -> ``"code:cs374"``."""
    match = CODE_RE.fullmatch(code.strip())
    if match is None:
        raise ValueError(
            f"{code!r} is not a course code (2-4 letters and three digits, e.g. 'CS 374'), so no "
            "chunk can claim it as its own; set bm25.code_boost to 0 to index without identity "
            "terms"
        )
    return f"{IDENTITY_PREFIX}{match.group('subject').lower()}{match.group('number')}"


def mentioned_codes(text: str) -> list[str]:
    """Reserved terms for every course code ``text`` mentions, in order of appearance."""
    return [
        f"{IDENTITY_PREFIX}{m.group('subject').lower()}{m.group('number')}"
        for m in CODE_RE.finditer(text)
    ]


def document_tokens(
    chunk: Mapping[str, Any], tokenizer: Callable[[str], list[str]], boost: int
) -> list[str]:
    """The terms one chunk is indexed under: its text, plus its own code ``boost`` times.

    The repeat is what a fielded index would express as a field weight; rank_bm25 has no fields,
    so the weight is carried by term frequency, which BM25 saturates -- the reason the boost is a
    small integer and not a multiplier.
    """
    tokens = tokenizer(chunk["text"])
    if boost:
        code = (chunk.get("metadata") or {}).get("code")
        if not code:
            raise ValueError(
                f"chunk {chunk.get('doc_id')!r} has no metadata.code, so it cannot be indexed "
                "under its own course code; re-run `uv run python -m ingest.chunk`, or set "
                "bm25.code_boost to 0"
            )
        tokens += [identity_token(code)] * boost
    return tokens


def query_tokens(query: str, tokenizer: Callable[[str], list[str]]) -> list[str]:
    """The terms one query scores against: its text, plus the code it names, once.

    Once, whatever ``code_boost`` is: the boost is a property of the *document* -- how strongly a
    chunk claims its own code -- and rank_bm25 sums over the query terms it is given, so repeating
    it here would multiply the same evidence a second time.
    """
    return tokenizer(query) + mentioned_codes(query)


def corpus_tokens(
    chunks: Sequence[Mapping[str, Any]], tokenizer: Callable[[str], list[str]], boost: int
) -> list[list[str]]:
    """:func:`document_tokens` over the whole corpus, in ``chunks`` order -- the BM25 input."""
    return [document_tokens(chunk, tokenizer, boost) for chunk in chunks]
