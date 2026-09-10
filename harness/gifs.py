"""Render short GIFs of generated levels for the README.

Generates a corpus, replays a known winning solution for one level of
each family (the C solver's path, or the generator's own labelled
solution for `match`), and writes an animated GIF per family.

    uv run python -m harness.gifs --out docs/media
"""
import argparse
import ctypes
import dataclasses
import sys
from pathlib import Path

from .clib import Library
from .dsl import DslGame
from .generate import (randomise_controls, sample_match, sample_rooms,
                       sample_select, sample_environment)
from .render import record
from .validate import solve_path

SAMPLERS = {"sokoban": sample_environment, "rooms": sample_rooms,
            "select": sample_select, "match": sample_match}


def _level_spec(spec, level: int):
    return dataclasses.replace(spec, layouts=spec.layouts[level:level + 1],
                               floors=spec.floors[level:level + 1],
                               budgets=None)


def _solution(spec, level: int, proposal, library, max_nodes: int):
    """A winning action list for `level`, or None."""
    meta = proposal.mechanics.get("levels", [])
    cells = meta[level].get("solution_cells") if level < len(meta) else None
    if cells is not None:
        return [(6, spec.origin_x + cx * spec.pitch + spec.pitch // 2,
                 spec.origin_y + cy * spec.pitch + spec.pitch // 2)
                for cx, cy in cells]
    one = _level_spec(spec, level)
    game = DslGame(one, library=library)
    path = solve_path(game, max_nodes)
    game.close()
    return path


def _wins(spec, level: int, actions, library) -> bool:
    game = DslGame(_level_spec(spec, level), library=library)
    game.init()
    for action in actions:
        game.act(*action)
    state = game.state
    game.close()
    return state == "WIN"


def render(family: str, out: Path, seed: int, lo: int, hi: int,
           library, scale: int, tries: int = 40):
    """Draw games until one has a level with a verified winning solution
    of length in [lo, hi], and write it as a GIF."""
    import numpy as np

    aux_size = ctypes.sizeof(library.headers.struct("arc_dsl_aux"))
    rng = np.random.default_rng(seed)
    nodes = 5_000 if family == "match" else 40_000
    for _ in range(tries):
        proposal = SAMPLERS[family](rng, library=library, aux_size=aux_size,
                                    stage=2, max_nodes=nodes)
        if proposal is None:
            continue
        spec = randomise_controls(rng, proposal.spec, family)
        # Deepest first: later levels carry more of the family's mechanics,
        # and the tutorial is the least illustrative level there is.
        for level in reversed(range(spec.num_levels)):
            actions = _solution(spec, level, proposal, library, nodes)
            if not actions or not (lo <= len(actions) <= hi):
                continue
            if not _wins(spec, level, actions, library):
                continue
            path = out / f"{family}.gif"
            frames = record(_level_spec(spec, level), path, actions,
                            scale=scale, library=library)
            kb = path.stat().st_size / 1024
            print(f"{family:8} level {level}  {len(actions):3} actions  "
                  f"{frames:3} frames  {kb:6.0f} KB  {path}")
            return path
    print(f"{family:8} no level found in [{lo}, {hi}] after {tries} draws",
          file=sys.stderr)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(prog="harness.gifs")
    parser.add_argument("--out", default="docs/media", type=Path)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--min", dest="lo", type=int, default=8)
    parser.add_argument("--max", dest="hi", type=int, default=34)
    parser.add_argument("--scale", type=int, default=6)
    parser.add_argument("--families", nargs="*", default=list(SAMPLERS))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    library = Library()
    ok = True
    for family in args.families:
        ok &= render(family, args.out, args.seed, args.lo, args.hi,
                     library, args.scale) is not None
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
