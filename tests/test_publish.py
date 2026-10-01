"""Tests for #119's publication checklist: the license, and the data decision behind it.

Every item here was checked once by hand while writing #119. A checklist checked once is a
checklist that is true on the day it was written, which is not the property wanted of a repo that
is about to be public -- so each of the four things that would actually hurt is a test:

- ``LICENSE`` exists, is the MIT text, and says in the file itself that it does not reach ``data/``;
- ``pyproject.toml`` declares the same license, so the metadata and the file cannot disagree;
- ``data/README.md`` carries the redistribution decision, *with* the two email addresses named in
  it, because the cost of that decision being undocumented is the whole reason #119 exists;
- no tracked text file carries an absolute home path or anything shaped like an API key.

The last one is a tree scan, not a history scan. #119 scanned all 175 commits by hand and found
nothing; a per-run scan of every blob in history is too slow to be worth it, and a *new* key can
only arrive through a new commit, which this catches in the tree before it is ever reachable.
"""

from __future__ import annotations

import re
import subprocess
import tomllib

import pytest

from config import REPO_ROOT

LICENSE_PATH = REPO_ROOT / "LICENSE"
DATA_README = REPO_ROOT / "data" / "README.md"

#: The two institutional addresses the catalog descriptions contain (#119). Hard-coded here rather
#: than re-derived from ``data/courses.json`` on purpose: this asserts that the documentation names
#: them, and a test that read the corpus for the list would pass while the prose named neither.
KNOWN_EMAILS = ("sustaindesign@illinois.edu", "jjkramer@illinois.edu")

#: Key-shaped strings. Deliberately narrow -- each matches a real provider's format rather than
#: "a long string", so the scan has no false positives to suppress and nobody learns to ignore it.
SECRET_RE = re.compile(
    r"""sk-[A-Za-z0-9_-]{16,}        # OpenAI
      | ghp_[A-Za-z0-9]{20,}         # GitHub personal access token
      | AKIA[0-9A-Z]{16}             # AWS access key id
      | -----BEGIN[A-Z ]+PRIVATE\ KEY-----
    """,
    re.VERBOSE,
)

#: An absolute path into somebody's home directory. ``rel_to_repo()`` exists so that no generated
#: report ever writes one; this is what notices when a hand-written file does.
HOME_PATH_RE = re.compile(r"/Users/[a-z]|/home/[a-z]")

#: Files that legitimately contain the patterns above, and why. ``uv.lock`` holds registry URLs and
#: hashes; these two tests own the patterns themselves.
SCAN_EXEMPT = {
    "uv.lock",
    "tests/test_publish.py",
    "tests/test_day4_report.py",  # asserts no "/Users/" appears in the report it generates
}


def tracked_text_files() -> list[str]:
    """Every tracked file, relative to the repo root.

    ``git ls-files`` rather than a walk, so an untracked local artifact -- an index, a ``.env``, a
    scratch file -- is never scanned and never reported. Binary files are filtered by the callers,
    which skip whatever does not decode as UTF-8. Where git cannot answer at all (an unpacked
    tarball, a machine without git) the two scans skip: they would otherwise report an empty tree
    as clean, which is the one answer worse than no answer.
    """
    try:
        done = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:  # no git, or not a checkout
        pytest.skip(f"cannot list tracked files: {exc}")
    return [one for one in done.stdout.split("\0") if one]


@pytest.fixture(scope="module")
def license_text() -> str:
    return LICENSE_PATH.read_text()


def test_the_license_file_is_there() -> None:
    assert LICENSE_PATH.is_file(), "no LICENSE: the repo cannot be published (#119)"


def test_the_license_is_mit(license_text: str) -> None:
    """The MIT text, not a paraphrase of it -- three clauses nobody may reword."""
    assert license_text.startswith("MIT License\n")
    for clause in (
            "Permission is hereby granted, free of charge",
            "The above copyright notice and this permission notice shall be included in all",
            'THE SOFTWARE IS PROVIDED "AS IS"'):
        assert clause in license_text, f"MIT clause missing or reworded: {clause!r}"


def test_the_license_carries_a_year_and_a_holder(license_text: str) -> None:
    assert re.search(r"^Copyright \(c\) 20\d\d \S", license_text, re.MULTILINE)


def test_the_license_says_it_does_not_cover_the_corpus(license_text: str) -> None:
    """The one non-standard thing in the file, and the reason it is not standard.

    A bare MIT file on this repo would read as a licence over the catalog text under ``data/``,
    which is not the copyright holder's to give.
    """
    tail = license_text.split("---", 1)
    assert len(tail) == 2, "LICENSE has no note after the MIT text about what it does not cover"
    assert "data/README.md" in tail[1]


def test_pyproject_declares_the_same_license() -> None:
    """Metadata and file agree. Two places naming a licence is two places to disagree."""
    meta = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]
    assert meta["license"] == "MIT"
    assert meta["license-files"] == ["LICENSE"]


def test_the_data_readme_records_the_decision() -> None:
    """#119's acceptance: the redistribution decision is written down, not assumed."""
    text = DATA_README.read_text()
    assert "## Redistribution" in text, "data/README.md does not record the data decision (#119)"
    assert "MIT" in text, "the decision does not say what the code licence is"


def test_the_data_readme_names_both_email_addresses() -> None:
    """Named in the decision rather than found after the flip."""
    text = DATA_README.read_text()
    missing = [one for one in KNOWN_EMAILS if one not in text]
    assert not missing, f"addresses in data/courses.json that the decision does not name: {missing}"


def test_those_are_still_the_only_addresses_in_the_corpus() -> None:
    """The count is part of the claim, so a re-export that brings a third one fails here.

    Skipped rather than failed where the corpus is absent: it is committed, but a sparse checkout
    is not a reason to call the decision wrong.
    """
    corpus = REPO_ROOT / "data" / "courses.json"
    if not corpus.is_file():
        pytest.skip("data/courses.json not present")
    found = set(re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.(?:edu|com|org|net)",
                           corpus.read_text()))
    assert found == set(KNOWN_EMAILS), (
        f"the addresses in the corpus are no longer the two the decision names: {sorted(found)}")


def test_no_tracked_file_carries_a_key() -> None:
    """A tree scan for key-shaped strings. #119 scanned the history by hand and found none."""
    hits = []
    for name in tracked_text_files():
        if name in SCAN_EXEMPT:
            continue
        try:
            text = (REPO_ROOT / name).read_text()
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        if SECRET_RE.search(text):
            hits.append(name)
    assert not hits, f"key-shaped string in tracked file(s): {hits}"


def test_no_tracked_file_carries_an_absolute_home_path() -> None:
    """``rel_to_repo()`` keeps generated files clean; this covers the hand-written ones."""
    hits = []
    for name in tracked_text_files():
        if name in SCAN_EXEMPT:
            continue
        try:
            text = (REPO_ROOT / name).read_text()
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        if HOME_PATH_RE.search(text):
            hits.append(name)
    assert not hits, f"absolute home path in tracked file(s): {hits}"
