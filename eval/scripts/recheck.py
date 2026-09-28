"""A sampled re-read of the verification pass, so that "confirmed" has a number attached to it.

    uv run python -m eval.scripts.recheck                 # draw the sample, print the command
    uv run python -m eval.scripts.recheck --result        # what the re-read overturned

#59's pass leaves one verdict per row and no way to tell a row that was *read* from a row that
was *confirmed*. That distinction is the whole difference between a human-verified set and a
machine-verified one, and it cannot be recovered from the log afterwards: an accept is an accept
whether it took two seconds or two minutes.

So this draws a stratified sample of rows that already have a verdict, a person reads those
properly, and the second verdict is compared against the first. What comes out is an **overturn
rate** with a Wilson interval -- the share of the pass that a careful reading would have decided
differently -- and that number is what #62 reports in place of a claim.

**It needs no new file to record the re-read.** #59's log is append-only and replays
latest-timestamp-wins, so a row judged twice carries both verdicts, and :func:`overturns` reads
the pair off the log. That is the property that makes this measurable at all, and it is the
reason the log carries a clock when nothing else in the repo does.

**What the sample is stratified on.** Two things, and the second is the one that matters:

- the bucket, so no bucket's verdicts go unchecked;
- the **verdict**, because a delete and an accept fail differently. An accept that should have
  been a delete leaves a wrong row in the set, where Day 4 will score against it. A delete that
  should have been an accept throws a row away, and nothing downstream will ever say so. Deletes
  are the smaller group and the less recoverable error, so every delete a *model* recommended is
  in the sample by construction -- there are three of them -- and the rule 2 deletes are sampled
  like anything else, since those came from a lookup rather than a judgement.

**A re-read that agrees with the first pass is not a wasted one.** The point is the interval. At
n=30 with nothing overturned the 95% upper bound is 11.4%, which is worth stating and is not the
same claim as "the pass was right" -- and it is the number that says what a bigger re-read would
buy, if the first thirty rows come back clean.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple, TextIO

from config import REPO_ROOT, Config, load_config, rel_to_repo
from eval.scripts.agreement import wilson
from eval.scripts.draft_items import write_json
from eval.scripts.review import (
    Candidate,
    Verdict,
    find_drafts,
    load_candidates,
    read_entries,
    read_verdicts,
)
from eval.validate import plural

#: Seed, strata and the ids drawn -- the sidecar convention #56 established, and the record that
#: makes "thirty rows, drawn this way" checkable rather than asserted.
RECHECK_PATH = REPO_ROOT / "eval" / "datasets" / "recheck.meta.json"

#: How many rows a re-read covers. Thirty rather than fifteen because this interval is the only
#: evidence about the rest of the pass, and rather than sixty because the whole point is that it
#: is affordable: thirty rows read properly is about twenty minutes.
SAMPLE_N = 30


class Stratum(NamedTuple):
    """One (bucket, verdict) cell: what it holds, and what was drawn from it."""

    bucket: str
    verdict: str
    held: int
    ids: tuple[str, ...]


class Sample(NamedTuple):
    """The draw: every id in it, the cells it came from, and the seed that produced it."""

    ids: tuple[str, ...]
    strata: tuple[Stratum, ...]
    seed: int


def cells(
    candidates: Sequence[Candidate], judged: Mapping[str, Verdict]
) -> dict[tuple[str, str], list[str]]:
    """Judged ids by (bucket, verdict), each list sorted. Unjudged rows are not in scope.

    Sorted for the reason #60's sampler gives: a draw over file order re-draws the sample the
    first time somebody sorts a file, and this sample has to be reproducible from its seed.
    """
    out: dict[tuple[str, str], list[str]] = {}
    for candidate in candidates:
        verdict = judged.get(candidate.id)
        if verdict is None:
            continue
        out.setdefault((candidate.item.bucket, verdict.verdict), []).append(candidate.id)
    return {key: sorted(ids) for key, ids in sorted(out.items())}


def allot(free: Mapping[tuple[str, str], Sequence[str]], left: int) -> dict[tuple[str, str], int]:
    """How many rows to draw from each cell: one each, then the rest in proportion to size.

    **One each first**, which is the part a proportional share alone gets wrong. The cells are
    not the same size -- at the real counts ``(multihop, delete)`` holds exactly one row against
    ``(ambiguous, accept)``'s forty-six -- and a cell rounded out of the sample is a kind of
    verdict nobody checked. A one-row cell is also where a single mistake is the whole cell.

    Only when there are more cells than slots does that floor give way, smallest cell first:
    thirty slots over eight cells is not that case, and a caller asking for five rows has said
    something about their time rather than about stratification.

    The remainder goes to whichever cell is furthest below its proportional target, which is the
    largest-remainder rule stated as a loop -- and it stops handing rows to a cell that has run
    out, so a small cell cannot absorb slots it has no rows for.
    """
    counts = {key: (1 if ids else 0) for key, ids in free.items()}
    for key in sorted(free, key=lambda k: (len(free[k]), k)):
        if sum(counts.values()) <= left:
            break
        counts[key] = 0
    total = sum(len(ids) for ids in free.values()) or 1
    target = {key: len(ids) * left / total for key, ids in free.items()}
    while sum(counts.values()) < left:
        room = [key for key in free if counts[key] < len(free[key])]
        if not room:
            break
        counts[max(room, key=lambda k: (target[k] - counts[k], k))] += 1
    return counts


def draw(
    candidates: Sequence[Candidate],
    judged: Mapping[str, Verdict],
    seed: int,
    n: int = SAMPLE_N,
    always: Sequence[str] = (),
) -> Sample:
    """``n`` ids, stratified over (bucket, verdict), proportional to what each cell holds.

    ``always`` is taken whole before anything is drawn -- the deletes a model recommended, which
    are three rows and the least recoverable error in the pass. Everything else goes through
    :func:`allot`, which gives every non-empty cell one row before it shares the rest out.
    """
    pools = cells(candidates, judged)
    taken = {item_id for item_id in always if any(item_id in ids for ids in pools.values())}
    left = max(n - len(taken), 0)
    free = {key: [i for i in ids if i not in taken] for key, ids in pools.items()}
    counts = allot(free, left)
    strata: list[Stratum] = []
    drawn: set[str] = set(taken)
    for key in sorted(pools):
        bucket, verdict = key
        rng = random.Random(f"{seed}:{bucket}:{verdict}")
        picked = sorted(rng.sample(free[key], k=counts.get(key, 0))) if counts.get(key) else []
        here = sorted({i for i in pools[key] if i in taken} | set(picked))
        drawn.update(here)
        strata.append(Stratum(bucket, verdict, len(pools[key]), tuple(here)))
    return Sample(tuple(sorted(drawn)), tuple(strata), seed)


def model_deletes(judged: Mapping[str, Verdict], advisory: Mapping[str, Any]) -> list[str]:
    """Deleted rows whose deletion a model, not the rule 2 lookup, asked for.

    Taken whole into the sample: nine of the twelve deletes are a table lookup that a person
    cannot second-guess from the screen, and the other three are a judgement that nothing
    downstream will ever revisit -- a deleted row leaves no trace in the dataset to be wrong in.
    """
    return sorted(
        item_id
        for item_id, verdict in judged.items()
        if verdict.verdict == "delete"
        and item_id in advisory
        and not advisory[item_id].reason.startswith("GUIDELINES rule 2")
    )


# --- what the re-read found -----------------------------------------------------------------------


class Overturn(NamedTuple):
    """One row read twice: the verdict it had, and the verdict it has now."""

    id: str
    first: str
    second: str

    @property
    def changed(self) -> bool:
        return self.first != self.second


def overturns(ids: Sequence[str], entries: Sequence[Mapping[str, Any]]) -> list[Overturn]:
    """Every sampled row that carries two verdicts, first against last. See the module docstring.

    Rows judged once are **not** here: the sample is what was asked for, and a row in it that
    still has one verdict has not been re-read yet. :func:`progress` is what reports that.
    """
    asked = set(ids)
    seen: dict[str, list[tuple[str, str]]] = {}
    for entry in entries:
        if entry["id"] not in asked:
            continue
        stamps = seen.setdefault(entry["id"], [])
        pair = (entry["ts"], entry["verdict"])
        if not stamps or stamps[-1][0] != entry["ts"]:
            stamps.append(pair)
    out = []
    for item_id, stamps in sorted(seen.items()):
        if len(stamps) < 2:
            continue
        ordered = sorted(stamps)
        out.append(Overturn(item_id, ordered[0][1], ordered[-1][1]))
    return out


class Result(NamedTuple):
    """The number #62 reports in place of a claim about the pass."""

    reread: int
    asked: int
    changed: tuple[Overturn, ...]

    @property
    def rate(self) -> float | None:
        return len(self.changed) / self.reread if self.reread else None

    @property
    def interval(self) -> tuple[float, float]:
        return wilson(len(self.changed), self.reread) if self.reread else (0.0, 1.0)

    @property
    def done(self) -> bool:
        return self.reread >= self.asked


def result(ids: Sequence[str], entries: Sequence[Mapping[str, Any]]) -> Result:
    """The overturn rate over the rows of ``ids`` that have been read a second time."""
    found = overturns(ids, entries)
    return Result(len(found), len(ids), tuple(one for one in found if one.changed))


def pace(entries: Sequence[Mapping[str, Any]], ids: Sequence[str] | None = None) -> float | None:
    """Median seconds between consecutive judgements, or ``None`` with fewer than two.

    The one number that says how a pass was made rather than what it decided, and the reason
    #59's log carries a clock. It is a **pace, not a reading speed** -- a reviewer who stops to
    check a course against the catalog and comes back leaves one long gap and a run of short
    ones, so the median is the typical row and the tail is where the work went. Both are worth
    printing: a median under a second over a hundred rows is a keystroke rate, whatever the
    mean says.

    Derived from the timestamps rather than being one, so it is stable under a re-run of the
    report: the log is fixed, so the number is.

    ``ids`` restricts it to a subset and takes each row's **last** judgement, which is how the
    re-read's own pace is measured against the first pass's.
    """
    if ids is None:
        stamps = sorted({entry["ts"] for entry in entries})
    else:
        asked, last = set(ids), {}
        for entry in entries:
            if entry["id"] in asked:
                last[entry["id"]] = max(last.get(entry["id"], ""), entry["ts"])
        stamps = sorted(set(last.values()))
    if len(stamps) < 2:
        return None
    moments = [datetime.fromisoformat(one) for one in stamps]
    gaps = sorted(
        (moments[i + 1] - moments[i]).total_seconds() for i in range(len(moments) - 1)
    )
    return gaps[len(gaps) // 2]


def read_sample(path: Path | None = None) -> Sample | None:
    """The drawn sample as recorded, or ``None`` when none has been drawn."""
    path = RECHECK_PATH if path is None else Path(path)
    if not path.exists():
        return None
    meta = json.loads(path.read_text(encoding="utf-8"))
    return Sample(
        ids=tuple(meta["ids"]),
        strata=tuple(
            Stratum(s["bucket"], s["verdict"], s["held"], tuple(s["ids"]))
            for s in meta["strata"]
        ),
        seed=meta["seed"],
    )


def sidecar(sample: Sample, cfg: Config, judged: Mapping[str, Verdict]) -> dict[str, Any]:
    """What the draw was a function of, and what it drew."""
    return {
        "config_hash": cfg.hash,
        "ids": list(sample.ids),
        "judged_rows": len(judged),
        "sample_n": len(sample.ids),
        "seed": sample.seed,
        "strata": [
            {
                "bucket": one.bucket,
                "verdict": one.verdict,
                "held": one.held,
                "drawn": len(one.ids),
                "ids": list(one.ids),
            }
            for one in sample.strata
        ],
        "target_n": SAMPLE_N,
    }


# --- saying so ------------------------------------------------------------------------------------


def table(sample: Sample) -> str:
    """The draw, one line per (bucket, verdict) cell that holds anything."""
    lines = [f"{'bucket':<14}{'verdict':<9}{'held':>6}{'drawn':>7}"]
    for one in sample.strata:
        if not one.held:
            continue
        lines.append(f"{one.bucket:<14}{one.verdict:<9}{one.held:>6}{len(one.ids):>7}")
    held = sum(one.held for one in sample.strata)
    lines.append(f"{'total':<14}{'':<9}{held:>6}{len(sample.ids):>7}")
    return "\n".join(lines)


def command(ids: Sequence[str]) -> str:
    """The re-read, copy-pasteably. One command, because thirty ids are not typed by hand."""
    return "uv run python -m eval.scripts.review --triage --redo " + " ".join(ids)


def render_result(found: Result, out: TextIO) -> int:
    """``--result``: how much of the sample has been re-read, and what it overturned."""
    if not found.reread:
        out.write(
            f"none of the {found.asked} sampled rows has been read a second time yet. The "
            "re-read is the measurement; run the command above.\n"
        )
        return 1
    lo, hi = found.interval
    out.write(
        f"re-read                {found.reread} of {found.asked} sampled rows\n"
        f"overturned             {len(found.changed)} "
        f"({found.rate:.0%}, 95% CI [{lo:.0%}, {hi:.0%}])\n"
    )
    out.writelines(f"  {one.id}  {one.first} -> {one.second}\n" for one in found.changed)
    if not found.changed:
        out.write(
            f"\nNothing overturned. At n={found.reread} that bounds the pass's error rate at "
            f"{hi:.0%}, which is the claim it supports -- not that the pass was right.\n"
        )
    if not found.done:
        out.write(f"\n{found.asked - found.reread} still to read.\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m eval.scripts.recheck",
        description="Draw a stratified sample of already-judged rows for a careful second "
        "reading, and report what that reading overturned.",
    )
    parser.add_argument(
        "--result",
        action="store_true",
        help="report the overturn rate over the drawn sample instead of drawing one",
    )
    parser.add_argument(
        "--n", type=int, default=SAMPLE_N, help=f"how many rows to draw (default: {SAMPLE_N})"
    )
    parser.add_argument("--seed", type=int, default=None, help="draw seed (default: config seed)")
    parser.add_argument("--force", action="store_true", help="redraw over an existing sample")
    parser.add_argument("--log", type=Path, default=None, help="the review log")
    parser.add_argument("--out", type=Path, default=None, help="where the sidecar goes")
    args = parser.parse_args(argv)

    cfg = load_config()
    path = RECHECK_PATH if args.out is None else args.out
    entries = read_entries(args.log)
    judged = read_verdicts(args.log)

    if args.result:
        sample = read_sample(path)
        if sample is None:
            print(f"no sample at {rel_to_repo(path)}; draw one first.", file=sys.stderr)
            return 2
        return render_result(result(sample.ids, entries), sys.stdout)

    existing = read_sample(path)
    if existing is not None and not args.force:
        found = result(existing.ids, entries)
        print(
            f"{rel_to_repo(path)} already holds a sample of {len(existing.ids)} "
            f"({found.reread} re-read). Redrawing it after a re-read has begun would report an "
            "overturn rate over rows nobody was asked to read; pass --force to redraw anyway.\n"
        )
        print(command([i for i in existing.ids]))
        return 1

    candidates, _ = load_candidates(find_drafts())
    if not judged:
        print(
            "no row has a verdict yet, so there is nothing to re-read. Run the pass first.",
            file=sys.stderr,
        )
        return 1
    from eval.scripts.triage import read_triage

    always = model_deletes(judged, read_triage())
    sample = draw(candidates, judged, cfg["seed"] if args.seed is None else args.seed,
                  args.n, always)
    write_json(path, sidecar(sample, cfg, judged))
    print(table(sample))
    print(
        f"\n{plural(len(sample.ids), 'row')} at seed {sample.seed}, including every delete a "
        f"model asked for ({plural(len(always), 'row')}).\n"
        f"wrote {rel_to_repo(path)}\n\nRead them properly -- the earlier verdict is printed "
        "above each row, and a different one replaces it:\n"
    )
    print(command(sample.ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())
