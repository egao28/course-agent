"""What the Chroma metadata can actually be filtered on (#18).

The store is a real ``PersistentClient`` under ``tmp_path``, filled through the same
``chroma_metadata()`` the dense build uses, because the whole question here is what *Chroma*
does with an encoding -- a fake store would answer whatever the encoding pretended. No embedder:
the vectors are hand-written, since none of these assertions is about similarity.

The two filters the issue named are the ones a course-planning query needs -- "courses that
satisfy a Gen Ed category" and "400-level courses in a subject" -- plus the prerequisite lookup
the joined-string encoding made impossible. The rest of the file pins the chromadb behaviour the
encoding is built on, so a version bump that changes it fails here and not in a Day 4 ablation:
a list value is storable and matched with ``$contains``, an empty list is not storable at all,
and ``$contains`` against a plain string matches nothing.
"""

from __future__ import annotations

from typing import Any

import chromadb
import pytest

from ingest.chunk import chroma_metadata, chunk_metadata, decode_chroma_metadata

#: Gen Ed rules in the shape ``requirements.json`` writes, narrowed to the two used here.
CATEGORIES = [
    {"id": "humlit", "match": {"field": "attrs", "any": ["Lit & Arts"], "none": []}},
    {"id": "cultus", "match": {"field": "attrs", "any": ["US Minority"], "none": []}},
]

#: Courses in the shape the export writes, chosen to span the filters: one course per level that
#: matters, one with a Gen Ed and one without, one whose prerequisites name CS 225, and MUS 090 --
#: the single zero-padded number in the 2026-sp corpus.
COURSES = [
    {"code": "CS 225", "subject": "CS", "number": "225", "prereqCodes": ["CS 124"], "attrs": ""},
    {
        "code": "CS 374",
        "subject": "CS",
        "number": "374",
        "prereqCodes": ["CS 173", "CS 225", "MATH 213"],
        "attrs": "",
    },
    {"code": "CS 446", "subject": "CS", "number": "446", "prereqCodes": ["CS 374"], "attrs": ""},
    {
        "code": "AAS 100",
        "subject": "AAS",
        "number": "100",
        "prereqCodes": [],
        "attrs": "Cultural Studies - US Minority course.",
    },
    {
        "code": "ENGL 104",
        "subject": "ENGL",
        "number": "104",
        "prereqCodes": [],
        "attrs": "Humanities - Lit & Arts course. Cultural Studies - US Minority course.",
    },
    {"code": "MUS 090", "subject": "MUS", "number": "090", "prereqCodes": [], "attrs": ""},
]


def course(record: dict[str, Any]) -> dict[str, Any]:
    """``record`` with the fields ``chunk_metadata`` reads but this file does not vary."""
    return {"credits": 3.0, "term": "2026-sp", **record}


def doc_id(record: dict[str, Any]) -> str:
    return f"2026-sp:{record['subject']}-{record['number']}"


@pytest.fixture
def collection(tmp_path):
    """A collection filled exactly the way ``ingest.build_dense`` fills one."""
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    collection = client.create_collection(
        "courses_test", configuration={"hnsw": {"space": "cosine"}}, embedding_function=None
    )
    collection.add(
        ids=[doc_id(r) for r in COURSES],
        embeddings=[[1.0, float(i)] for i, _ in enumerate(COURSES)],
        metadatas=[chroma_metadata(chunk_metadata(course(r), CATEGORIES)) for r in COURSES],
    )
    return collection


def codes(result) -> set[str]:
    return {m["code"] for m in result["metadatas"]}


# --- the filters the issue asked for -------------------------------------------------------------


def test_gen_ed_category_filters_with_contains(collection):
    found = collection.get(where={"gened_ids": {"$contains": "cultus"}}, include=["metadatas"])
    assert codes(found) == {"AAS 100", "ENGL 104"}


def test_a_course_satisfying_several_categories_matches_each_of_them(collection):
    # The joined string could only ever be matched whole, so a two-category course was reachable
    # by neither of its categories alone.
    for category in ("humlit", "cultus"):
        found = collection.get(where={"gened_ids": {"$contains": category}}, include=["metadatas"])
        assert "ENGL 104" in codes(found)


def test_400_level_courses_in_a_subject(collection):
    found = collection.get(
        where={"$and": [{"subject": "CS"}, {"level": 400}]}, include=["metadatas"]
    )
    assert codes(found) == {"CS 446"}


def test_the_same_range_expressed_on_the_number_itself(collection):
    # `level` is the convenience; `number` is an int so open-ended ranges work too.
    found = collection.get(
        where={"$and": [{"subject": "CS"}, {"number": {"$gte": 300}}]}, include=["metadatas"]
    )
    assert codes(found) == {"CS 374", "CS 446"}


def test_courses_whose_prerequisites_include_a_given_course(collection):
    found = collection.get(where={"prereq_codes": {"$contains": "CS 225"}}, include=["metadatas"])
    assert codes(found) == {"CS 374"}


def test_a_filter_also_narrows_a_vector_search(collection):
    # The path retrieval actually takes: `where` on query(), not on get().
    found = collection.query(
        query_embeddings=[[1.0, 0.0]],
        n_results=5,
        where={"gened_ids": {"$contains": "cultus"}},
        include=["metadatas"],
    )
    assert {m["code"] for m in found["metadatas"][0]} == {"AAS 100", "ENGL 104"}


# --- what a course with no Gen Ed does -----------------------------------------------------------


def test_a_course_with_no_gen_ed_is_stored_without_the_key(collection):
    stored = collection.get(ids=["2026-sp:CS-374"], include=["metadatas"])["metadatas"][0]
    assert "gened_ids" not in stored
    assert stored["prereq_codes"] == ["CS 173", "CS 225", "MATH 213"]


def test_a_missing_key_matches_not_contains_but_never_contains(collection):
    contains = collection.get(where={"gened_ids": {"$contains": "cultus"}}, include=["metadatas"])
    without = collection.get(
        where={"gened_ids": {"$not_contains": "cultus"}}, include=["metadatas"]
    )
    assert "CS 374" not in codes(contains)
    assert "CS 374" in codes(without)  # no key at all still answers "does not satisfy cultus"


def test_what_comes_back_out_decodes_to_what_went_in(collection):
    # The round trip the dense retriever depends on, through a real store rather than in memory.
    for record in COURSES:
        stored = collection.get(ids=[doc_id(record)], include=["metadatas"])["metadatas"][0]
        assert decode_chroma_metadata(stored) == chunk_metadata(course(record), CATEGORIES)


def test_the_zero_padded_number_keeps_its_catalog_spelling_in_the_code(collection):
    stored = collection.get(ids=["2026-sp:MUS-090"], include=["metadatas"])["metadatas"][0]
    assert (stored["number"], stored["level"]) == (90, 0)
    assert stored["code"] == "MUS 090"  # the display form, unlike the int the filter compares


# --- the chromadb behaviour the encoding rests on ------------------------------------------------


def test_chroma_refuses_an_empty_list_which_is_why_the_key_is_dropped(collection):
    with pytest.raises(ValueError, match="non-empty"):
        collection.add(
            ids=["2026-sp:CS-999"], embeddings=[[1.0, 9.0]], metadatas=[{"gened_ids": []}]
        )


def test_contains_against_a_string_value_matches_nothing(collection):
    # $contains is a list-membership operator on metadata, not a substring test: the joined-string
    # encoding this issue replaced was unfilterable for exactly this reason.
    found = collection.get(where={"subject": {"$contains": "CS"}}, include=["metadatas"])
    assert codes(found) == set()
