"""Tests for ingest/tokenize.py: the code rule, the stopwords, and the identity term.

The property worth defending here is that the build side and the query side cannot disagree, so
most of these assert an *equality between two term streams* rather than a literal list: BM25 has
no way to report a mismatch, it just returns a confident, wrong ranking.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from config import Config, load_config
from ingest import tokenize as tk
from ingest.tokenize import (
    IDENTITY_PREFIX,
    TOKENIZERS,
    check_tokenizer,
    code_boost,
    corpus_tokens,
    document_tokens,
    identity_token,
    query_tokens,
    tokenize,
)

BASE_CONFIG = Path(__file__).parent / "fixtures" / "config_fixture.yaml"

SIMPLE = TOKENIZERS["simple_v1"]


@pytest.fixture
def cfg() -> Config:
    raw = load_config(BASE_CONFIG).to_dict()
    raw["bm25"] = {"k1": 1.5, "b": 0.75, "tokenizer": "simple_v1", "code_boost": 2}
    return Config(raw)


def chunk(doc_id: str = "2026-sp:CS-374", text: str = "CS 374 — Algorithms", **metadata):
    return {"doc_id": doc_id, "text": text, "metadata": {"code": "CS 374", **metadata}}


# --- course codes ---------------------------------------------------------------------------------


def test_a_code_yields_the_subject_the_number_and_the_two_joined():
    assert tokenize("CS 374") == ["cs", "374", "cs374"]


@pytest.mark.parametrize("spelling", ["CS 374", "cs 374", "CS374", "cs374", "Cs  374"])
def test_every_spelling_of_a_code_produces_the_same_terms(spelling):
    # The whole point of the glued term: "CS374" and "CS 374" share no word, so without it the
    # two spellings would score against disjoint term sets.
    assert tokenize(spelling) == tokenize("CS 374")


def test_a_code_inside_a_sentence_is_found():
    assert tokenize("Same as ECE 374.") == ["same", "ece", "374", "ece374"]


def test_a_four_digit_number_is_not_a_course_number():
    assert tokenize("MATH 2134") == ["math", "2134"]


def test_a_long_word_before_a_number_is_not_a_subject():
    # Subjects are 2-4 letters; "PHYSICS 101" must not glue on its last four.
    assert tokenize("PHYSICS 101") == ["physics", "101"]


def test_a_code_is_consumed_whole_so_the_digits_are_not_also_a_word():
    # One left-to-right scan: the number appears once, from the code branch, not twice.
    assert tokenize("CS 374").count("374") == 1


def test_a_code_is_not_glued_across_a_line_break():
    assert tokenize("taught by ECE\n374 staff") == ["taught", "ece", "374", "staff"]


# --- stopwords ------------------------------------------------------------------------------------


def test_stopwords_are_dropped_from_prose():
    assert tokenize("the analysis of algorithms") == ["analysis", "algorithms"]


def test_a_subject_that_is_also_a_stopword_survives_as_part_of_its_code():
    # IS is a real UIUC subject. The code is matched before words are filtered, so the subject
    # term -- the one that ranks other IS courses -- is not lost to the stopword list.
    assert tokenize("IS 407") == ["is", "407", "is407"]
    assert tokenize("this is fine") == ["fine"]


def test_case_is_folded():
    assert tokenize("Analysis Of ALGORITHMS") == tokenize("analysis of algorithms")


def test_punctuation_and_empty_text_produce_no_terms():
    assert tokenize("—.,;:!?") == []
    assert tokenize("") == []


# --- the identity term ----------------------------------------------------------------------------


def test_identity_token_is_the_code_under_a_reserved_prefix():
    assert identity_token("CS 374") == "code:cs374"
    assert identity_token(" cs374 ") == "code:cs374"


def test_identity_token_refuses_something_that_is_not_a_code():
    with pytest.raises(ValueError, match="not a course code"):
        identity_token("Introduction to Algorithms")


def test_no_text_can_produce_an_identity_term():
    # The prefix ends in ":", which \w+ cannot emit -- that is what keeps the namespace reserved.
    noisy = "code:cs374 CODE:CS374 code cs374 code: cs 374"
    assert not [t for t in tokenize(noisy) if t.startswith(IDENTITY_PREFIX)]


def test_a_document_claims_its_own_code_code_boost_times():
    tokens = document_tokens(chunk(), SIMPLE, 2)
    assert tokens.count("code:cs374") == 2
    assert tokens[:-2] == tokenize("CS 374 — Algorithms")


def test_code_boost_zero_indexes_the_text_alone():
    assert document_tokens(chunk(), SIMPLE, 0) == tokenize("CS 374 — Algorithms")


def test_a_document_without_a_code_is_refused_rather_than_quietly_unclaimed():
    bare = {"doc_id": "2026-sp:CS-374", "text": "CS 374", "metadata": {}}
    with pytest.raises(ValueError, match="no metadata.code"):
        document_tokens(bare, SIMPLE, 2)
    # ... unless nothing was going to claim it anyway.
    assert document_tokens(bare, SIMPLE, 0) == tokenize("CS 374")


def test_a_query_asks_for_the_code_it_names_once():
    # Once whatever the boost is: rank_bm25 sums over the query terms it is given, so repeating it
    # here would count the same evidence twice.
    assert query_tokens("CS 374", SIMPLE) == ["cs", "374", "cs374", "code:cs374"]


def test_a_query_asks_for_every_code_it_names():
    assert [
        t for t in query_tokens("CS 374 or ECE 374", SIMPLE) if t.startswith(IDENTITY_PREFIX)
    ] == [
        "code:cs374",
        "code:ece374",
    ]


def test_a_query_naming_no_code_asks_for_no_identity_term():
    assert query_tokens("machine learning", SIMPLE) == ["machine", "learning"]


def test_the_two_sides_agree_on_the_terms_of_a_code():
    # A document indexed under "CS 374" and a query naming it must meet on the same four terms.
    doc = set(document_tokens(chunk(), SIMPLE, 1))
    assert set(query_tokens("CS 374", SIMPLE)) <= doc


def test_corpus_tokens_keeps_the_corpus_order():
    chunks = [chunk("2026-sp:CS-374"), chunk("2026-sp:IS-407", "IS 407 — Data", code="IS 407")]
    assert [tokens[-1] for tokens in corpus_tokens(chunks, SIMPLE, 1)] == [
        "code:cs374",
        "code:is407",
    ]


# --- config ---------------------------------------------------------------------------------------


def test_get_tokenizer_returns_the_one_the_config_names(cfg):
    assert tk.get_tokenizer(cfg) is SIMPLE


def test_an_unknown_tokenizer_is_refused_by_name(cfg):
    named = Config({**cfg.to_dict(), "bm25": {**cfg["bm25"], "tokenizer": "porter_v9"}})
    with pytest.raises(ValueError, match="unknown bm25.tokenizer 'porter_v9'"):
        check_tokenizer(named)
    with pytest.raises(ValueError, match="unknown bm25.tokenizer"):
        tokenize("CS 374", "porter_v9")


def test_a_config_without_a_tokenizer_says_which_key_is_missing(cfg):
    bare = {k: v for k, v in cfg["bm25"].items() if k != "tokenizer"}
    with pytest.raises(ValueError, match="bm25 section is missing tokenizer"):
        check_tokenizer(Config({**cfg.to_dict(), "bm25": bare}))


def test_a_config_without_a_bm25_section_says_so(cfg):
    without = {k: v for k, v in cfg.to_dict().items() if k != "bm25"}
    with pytest.raises(ValueError, match="no `bm25` section"):
        check_tokenizer(Config(without))
    with pytest.raises(ValueError, match="missing code_boost"):
        code_boost(Config(without))


def test_code_boost_is_read_through_one_guard(cfg):
    assert code_boost(cfg) == 2


def test_a_config_without_a_code_boost_says_which_key_is_missing(cfg):
    bare = {k: v for k, v in cfg["bm25"].items() if k != "code_boost"}
    with pytest.raises(ValueError, match="missing code_boost"):
        code_boost(Config({**cfg.to_dict(), "bm25": bare}))


@pytest.mark.parametrize("value", [-1, 1.5, "2", True, None])
def test_code_boost_must_be_a_count(cfg, value):
    with pytest.raises(ValueError, match="code_boost must be a non-negative integer"):
        code_boost(Config({**cfg.to_dict(), "bm25": {**cfg["bm25"], "code_boost": value}}))
