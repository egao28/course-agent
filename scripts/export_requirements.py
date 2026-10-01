"""The three ``requirements*.txt`` files, generated from ``uv.lock`` and never written by hand.

    uv run python -m scripts.export_requirements            # rewrite all three
    uv run python -m scripts.export_requirements --check    # verify them; exit 1 on drift
    uv run python -m scripts.export_requirements --dry-run  # print one, write nothing

``uv.lock`` already pins every dependency to an exact version with hashes, and ``uv sync
--locked`` refuses a stale lock. A hand-written ``requirements.txt`` beside it would be a second
source of truth that drifts silently -- worse than having none, because it looks authoritative.
So these files are not maintained: they are *derived*, by ``uv export``, and CI runs ``--check``
so that a file which stopped matching the lock fails the build instead of being discovered by
whoever installed from it.

**Three files, because there are three questions a reader can be asking** (:data:`EXPORTS`):

* ``requirements.txt`` -- the agent and retrieval stack, nothing else. This is what runs
  ``python -m agent`` and ``python -m retrieval``.
* ``requirements-full.txt`` -- that plus the ``analysis`` extra (Day 5's figure and AUROC) and the
  ``local`` extra (``sentence-transformers``, Day 4's cross-encoder). It is what reproduces the
  ``hybrid_rerank`` row; ``sentence-transformers`` pulls in torch, which is why it is not the
  default install.
* ``requirements-dev.txt`` -- the ``dev`` group: pytest, ruff, and, because that group asks for
  ``almabot[analysis]``, the whole runtime stack and the analysis extra with it. It is therefore
  not a small file, and it is the one that reproduces the environment CI tests in
  (``uv sync --locked --extra analysis``). The ``local`` extra is the only thing it leaves out.

**The hashes stay in.** ``uv export`` emits ``--hash`` lines by default and nothing here strips
them: a version number with no hash pins against a name on an index, not against a file, and pip
enforces the hashes it is given.

**``--no-emit-project``** keeps ``almabot`` itself out of its own requirements. The project is not
installed from an index -- ``pyproject.toml`` has no wheel anywhere -- and ``pytest``'s
``pythonpath = ["."]`` is what makes the modules importable from a checkout.

**The header is ours, the body is uv's.** ``uv export``'s own header names the command but does
not forbid an edit, and the point of these files is that an edit to them is meaningless. So each
file is generated with ``--no-header`` and given :func:`header` instead, which says both. That is
also why ``--check`` compares the rendered text rather than shelling out to ``diff``: the header
is part of what can go stale.
"""

import argparse
import difflib
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: How a reader regenerates every file here, printed in each one's header.
GENERATOR = "uv run python -m scripts.export_requirements"


@dataclass(frozen=True)
class Export:
    """One generated file: what it is called, which ``uv export`` selectors build it, and why."""

    filename: str
    selectors: tuple[str, ...]
    purpose: str

    @property
    def path(self) -> Path:
        """The file's place in the repository."""
        return REPO_ROOT / self.filename

    @property
    def command(self) -> list[str]:
        """The exact ``uv export`` argv whose output is this file's body."""
        return ["uv", "export", "--locked", *self.selectors,
                "--no-emit-project", "--no-header"]


EXPORTS: tuple[Export, ...] = (
    Export(
        filename="requirements.txt",
        selectors=("--no-dev",),
        purpose="The agent + retrieval stack: what `python -m agent` and `python -m retrieval` "
                "need, and nothing more.",
    ),
    Export(
        filename="requirements-full.txt",
        selectors=("--no-dev", "--extra", "analysis", "--extra", "local"),
        purpose="The above plus both extras -- `analysis` (Day 5's figure and AUROC) and `local` "
                "(sentence-transformers, and so torch). This is the install that reproduces Day "
                "4's `hybrid_rerank` row.",
    ),
    Export(
        filename="requirements-dev.txt",
        selectors=("--only-dev",),
        purpose="The dev group: pytest and ruff, and -- because that group asks for "
                "`almabot[analysis]` -- the runtime stack and the analysis extra with them. Add "
                "nothing to run the suite; this is the environment CI tests in.",
    ),
)


def header(export: Export) -> str:
    """The generated-by block that opens ``export``'s file, ending in a blank comment line."""
    notice = (
        f"Generated from uv.lock by `{GENERATOR}` -- do not edit by hand. An edit here is undone "
        f"by the next run of the generator, and CI fails on any file that has stopped matching "
        f"the lock (`{GENERATOR} --check`). uv.lock is the pin, hashes and all; this is a "
        f"rendering of it for readers, and for machines, without uv."
    )
    lines = [f"# {line}" for line in _wrap(notice)]
    lines += ["#", "# Everything below this header is the verbatim output of:",
              f"#     {' '.join(export.command)}", "#"]
    lines += [f"# {line}" for line in _wrap(export.purpose)]
    return "\n".join(lines + [""])


def _wrap(text: str, width: int = 96) -> list[str]:
    """Wrap ``text`` to comment width, by hand, so the header needs no import to format."""
    lines: list[str] = []
    line = ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if len(candidate) > width and line:
            lines.append(line)
            line = word
        else:
            line = candidate
    return lines + [line] if line else lines


def run_export(export: Export) -> str:
    """Run ``uv export`` for ``export`` and return its stdout, raising on a non-zero exit."""
    result = subprocess.run(export.command, cwd=REPO_ROOT, capture_output=True, text=True,
                            check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"{' '.join(export.command)} exited {result.returncode}:\n{result.stderr.strip()}"
        )
    return result.stdout


def render(export: Export, body: str) -> str:
    """The file's full text: :func:`header`, then uv's output, with one blank line between."""
    return f"{header(export)}\n{body}"


def generate(export: Export) -> str:
    """The text ``export``'s file should have right now, read out of the lock through uv."""
    return render(export, run_export(export))


def diff(export: Export, expected: str) -> str:
    """A unified diff from the committed file to ``expected``; empty when they agree."""
    committed = export.path.read_text(encoding="utf-8") if export.path.exists() else ""
    if committed == expected:
        return ""
    return "".join(difflib.unified_diff(
        committed.splitlines(keepends=True), expected.splitlines(keepends=True),
        fromfile=f"{export.filename} (committed)", tofile=f"{export.filename} (from uv.lock)",
    ))


def check(exports: Sequence[Export], out) -> int:
    """Report every file that has drifted from the lock; 0 when all three are current."""
    stale = 0
    for export in exports:
        body = diff(export, generate(export))
        if body:
            stale += 1
            out.write(body if body.endswith("\n") else body + "\n")
    if stale:
        out.write(f"\n{stale} of {len(exports)} files do not match uv.lock. "
                  f"Regenerate with `{GENERATOR}` and commit the result; do not edit them.\n")
        return 1
    out.write(f"all {len(exports)} requirements files match uv.lock\n")
    return 0


def write(exports: Sequence[Export], out) -> int:
    """Rewrite every file, saying which ones moved."""
    for export in exports:
        text = generate(export)
        current = export.path.read_text(encoding="utf-8") if export.path.exists() else None
        if text == current:
            out.write(f"{export.filename} is already current\n")
            continue
        export.path.write_text(text, encoding="utf-8")
        out.write(f"wrote {export.filename} ({text.count(chr(10))} lines)\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Rewrite the requirements files, or with ``--check`` verify them and write nothing."""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.export_requirements",
        description="Generate requirements.txt, requirements-full.txt and requirements-dev.txt "
                    "from uv.lock. uv.lock is the pin; these are renderings of it.",
    )
    parser.add_argument("--check", action="store_true",
                        help="verify the committed files and write nothing; exit 1 on drift")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the generated files instead of writing them")
    parser.add_argument("--only", choices=[e.filename for e in EXPORTS], default=None,
                        help="act on one file instead of all three")
    args = parser.parse_args(argv)

    exports = [e for e in EXPORTS if args.only in (None, e.filename)]
    try:
        if args.check:
            return check(exports, sys.stdout)
        if args.dry_run:
            for export in exports:
                sys.stdout.write(generate(export))
            return 0
        return write(exports, sys.stdout)
    except (RuntimeError, FileNotFoundError) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
