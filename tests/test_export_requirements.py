"""Tests for scripts/export_requirements.py and for the three files it writes.

Two groups, and the split between them is the point.

The first group tests the **generator**: the header it writes, the argv it builds, and what
``--check`` does when a committed file no longer matches what the lock would produce. Every one
of these runs against a fake exporter and a ``tmp_path`` repo -- none of them shells out to uv,
because what is being tested is the drift report, not uv.

The second group tests the **committed files**, and it deliberately does not use uv either. It
re-reads ``uv.lock`` with ``tomllib`` and asserts that every pin in every requirements file is
the version the lock holds, that every line carries a hash, and that the three files divide the
dependencies the way their headers claim. That matters because ``--check`` needs the uv binary,
so it can only ever run where uv is installed -- which is precisely not the machine these files
exist for. A reader who installs from ``requirements.txt`` with pip and no uv can still run this
suite and find out that the file drifted.

The floors in ``pyproject.toml`` are checked here too, against the same lock: a floor above the
locked version would make ``uv sync --locked`` and ``pip install -r requirements.txt`` disagree
about what is installable, which is the one way a floor can do damage rather than nothing.
"""

from __future__ import annotations

import re
import tomllib
from io import StringIO
from pathlib import Path

import pytest

from scripts import export_requirements as ex

REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENT = re.compile(r"^(?P<name>[A-Za-z0-9._-]+)==(?P<version>[^ ;\\]+)")

#: The five floors #116 asked for, and the one (`pydantic`) it named a number for itself.
FLOORED = ("chromadb", "openai", "tiktoken", "langgraph", "fastmcp", "pydantic")


def locked_versions() -> dict[str, str]:
    """Every package in ``uv.lock``, by normalised name, with the version it pins."""
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    return {package["name"]: package["version"] for package in lock["package"]}


def pinned(export: ex.Export) -> dict[str, str]:
    """The ``name == version`` pins in a committed requirements file."""
    found = {}
    for line in export.path.read_text(encoding="utf-8").splitlines():
        match = REQUIREMENT.match(line)
        if match:
            found[match["name"]] = match["version"]
    return found


def release(text: str) -> tuple[int, ...]:
    """A dotted release as a tuple of ints, for comparing a floor with a locked version."""
    return tuple(int(part) for part in re.findall(r"\d+", text.split("+")[0]))


def fake_repo(monkeypatch, tmp_path: Path, body: str) -> Path:
    """Point the generator at ``tmp_path`` and give it a fixed export body instead of uv."""
    monkeypatch.setattr(ex, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ex, "run_export", lambda export: body)
    return tmp_path


BODY = "chromadb==1.5.9 \\\n    --hash=sha256:abc\n"


# --- the generator ------------------------------------------------------------------------

def test_the_header_forbids_the_edit_and_names_the_command_that_undoes_it():
    text = ex.header(ex.EXPORTS[0])
    assert "do not edit by hand" in text
    assert ex.GENERATOR in text
    assert f"{ex.GENERATOR} --check" in text
    assert " ".join(ex.EXPORTS[0].command) in text


def test_every_header_line_is_a_comment_and_fits_the_line_length():
    for export in ex.EXPORTS:
        for line in ex.header(export).splitlines():
            assert line.startswith("#")
            assert len(line) <= 100, line


def test_the_body_is_uvs_output_untouched():
    rendered = ex.render(ex.EXPORTS[0], BODY)
    assert rendered.endswith(BODY)
    assert rendered[: -len(BODY)] == ex.header(ex.EXPORTS[0]) + "\n"


def test_the_command_locks_the_lock_and_keeps_the_project_out():
    for export in ex.EXPORTS:
        assert export.command[:3] == ["uv", "export", "--locked"]
        assert "--no-emit-project" in export.command
        assert "--no-header" in export.command


def test_the_three_files_ask_uv_three_different_questions():
    selectors = [export.selectors for export in ex.EXPORTS]
    assert len(set(selectors)) == len(selectors)
    assert len({export.filename for export in ex.EXPORTS}) == len(ex.EXPORTS)


def test_write_creates_every_file_then_reports_them_as_current(monkeypatch, tmp_path):
    fake_repo(monkeypatch, tmp_path, BODY)
    out = StringIO()
    assert ex.write(ex.EXPORTS, out) == 0
    assert out.getvalue().count("wrote ") == len(ex.EXPORTS)
    for export in ex.EXPORTS:
        assert (tmp_path / export.filename).read_text(encoding="utf-8") == ex.generate(export)

    again = StringIO()
    assert ex.write(ex.EXPORTS, again) == 0
    assert again.getvalue().count("already current") == len(ex.EXPORTS)


def test_check_passes_on_files_the_generator_just_wrote(monkeypatch, tmp_path):
    fake_repo(monkeypatch, tmp_path, BODY)
    ex.write(ex.EXPORTS, StringIO())
    out = StringIO()
    assert ex.check(ex.EXPORTS, out) == 0
    assert "match uv.lock" in out.getvalue()


def test_check_fails_and_prints_the_line_when_a_version_is_edited(monkeypatch, tmp_path):
    fake_repo(monkeypatch, tmp_path, BODY)
    ex.write(ex.EXPORTS, StringIO())
    edited = ex.EXPORTS[0]
    (tmp_path / edited.filename).write_text(
        ex.generate(edited).replace("1.5.9", "1.5.8"), encoding="utf-8"
    )
    out = StringIO()
    assert ex.check(ex.EXPORTS, out) == 1
    report = out.getvalue()
    assert "-chromadb==1.5.8" in report and "+chromadb==1.5.9" in report
    assert f"1 of {len(ex.EXPORTS)} files do not match uv.lock" in report
    assert ex.GENERATOR in report


def test_check_fails_when_a_file_is_missing_entirely(monkeypatch, tmp_path):
    fake_repo(monkeypatch, tmp_path, BODY)
    out = StringIO()
    assert ex.check(ex.EXPORTS, out) == 1
    assert f"{len(ex.EXPORTS)} of {len(ex.EXPORTS)} files do not match" in out.getvalue()


def test_check_fails_when_only_the_header_was_edited(monkeypatch, tmp_path):
    fake_repo(monkeypatch, tmp_path, BODY)
    ex.write(ex.EXPORTS, StringIO())
    export = ex.EXPORTS[0]
    (tmp_path / export.filename).write_text(BODY, encoding="utf-8")
    out = StringIO()
    assert ex.check(ex.EXPORTS, out) == 1
    assert "do not edit by hand" in out.getvalue()


def test_a_failing_export_is_reported_rather_than_raised(monkeypatch, tmp_path):
    monkeypatch.setattr(ex, "REPO_ROOT", tmp_path)

    def explode(export):
        raise RuntimeError("uv export exited 2: the lockfile is not up-to-date")

    monkeypatch.setattr(ex, "run_export", explode)
    assert ex.main(["--check"]) == 1


def test_only_narrows_the_run_to_one_file(monkeypatch, tmp_path):
    fake_repo(monkeypatch, tmp_path, BODY)
    assert ex.main(["--only", "requirements.txt"]) == 0
    assert (tmp_path / "requirements.txt").exists()
    assert not (tmp_path / "requirements-dev.txt").exists()


def test_dry_run_prints_and_writes_nothing(monkeypatch, tmp_path, capsys):
    fake_repo(monkeypatch, tmp_path, BODY)
    assert ex.main(["--dry-run", "--only", "requirements.txt"]) == 0
    assert "chromadb==1.5.9" in capsys.readouterr().out
    assert not (tmp_path / "requirements.txt").exists()


# --- the committed files, read without uv -------------------------------------------------

@pytest.mark.parametrize("export", ex.EXPORTS, ids=lambda e: e.filename)
def test_the_committed_file_exists_and_opens_with_its_generated_header(export):
    text = export.path.read_text(encoding="utf-8")
    assert text.startswith(ex.header(export)), (
        f"{export.filename}'s header is not the one the generator writes; "
        f"run `{ex.GENERATOR}`"
    )


@pytest.mark.parametrize("export", ex.EXPORTS, ids=lambda e: e.filename)
def test_every_pin_is_the_version_uv_lock_holds(export):
    locked = locked_versions()
    drifted = {
        name: (version, locked.get(name))
        for name, version in pinned(export).items()
        if locked.get(name) != version
    }
    assert not drifted, f"{export.filename} has drifted from uv.lock: {drifted}"


@pytest.mark.parametrize("export", ex.EXPORTS, ids=lambda e: e.filename)
def test_every_requirement_carries_at_least_one_hash(export):
    lines = export.path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if REQUIREMENT.match(line):
            following = lines[index + 1] if index + 1 < len(lines) else ""
            assert line.endswith("\\") and "--hash=sha256:" in following, (
                f"{export.filename}: {line} is a version without a file behind it"
            )


@pytest.mark.parametrize("export", ex.EXPORTS, ids=lambda e: e.filename)
def test_the_project_is_not_a_requirement_of_itself(export):
    assert "almabot" not in pinned(export)


def test_the_default_install_is_the_agent_stack_without_either_extra():
    default = pinned(ex.EXPORTS[0])
    assert {"chromadb", "langgraph", "openai", "rank-bm25", "tiktoken"} <= set(default)
    for heavy in ("matplotlib", "scikit-learn", "sentence-transformers", "torch"):
        assert heavy not in default, f"{heavy} is in requirements.txt but belongs to an extra"
    for tool in ("pytest", "ruff"):
        assert tool not in default


def test_the_full_file_adds_both_extras_and_nothing_else_loses_anything():
    default, full = pinned(ex.EXPORTS[0]), pinned(ex.EXPORTS[1])
    assert set(default) < set(full)
    assert {"matplotlib", "scikit-learn", "sentence-transformers", "torch"} <= set(full)
    assert "pytest" not in full


def test_the_dev_file_is_the_environment_ci_tests_in():
    dev = pinned(ex.EXPORTS[2])
    assert {"pytest", "ruff"} <= set(dev)
    assert {"matplotlib", "scikit-learn"} <= set(dev), "the dev group asks for almabot[analysis]"
    assert {"chromadb", "langgraph"} <= set(dev)
    for local_only in ("sentence-transformers", "torch"):
        assert local_only not in dev, "the local extra is not in the dev group"


# --- the floors in pyproject.toml ---------------------------------------------------------

def test_every_floor_is_a_lower_bound_at_or_below_what_the_lock_holds():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    locked = locked_versions()
    floors = {}
    for requirement in pyproject["project"]["dependencies"]:
        name, _, specifier = requirement.partition(">=")
        if specifier:
            floors[name] = specifier
    assert set(floors) == set(FLOORED), "the floored dependencies moved; #116 named these"
    for name, floor in floors.items():
        assert "==" not in floor, f"{name} is pinned in pyproject; uv.lock is the pin"
        assert release(floor) <= release(locked[name]), (
            f"{name}>={floor} is above the locked {locked[name]}: pip and uv would disagree"
        )


def test_the_floored_names_are_all_real_dependencies():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = {
        requirement.partition(">=")[0] for requirement in pyproject["project"]["dependencies"]
    }
    assert set(FLOORED) <= declared
