"""Demonstrations from the solver for behaviour cloning.

For every level of every generated game in a directory, replay a known
solution (the corpus label, or the C solver's path) through the DSL game
and record, per step, what the policy in relax-arc sees and what it should
do: the frame, the action ids the game declares, the previous action and
reward, a done flag at sequence starts, and the target action in the
policy's own index space (kinds 0..6, clicks as 6 + cell on a coarse grid).
"""
import dataclasses
import glob
import json
from pathlib import Path

import numpy as np

from .corpus import load
from .dsl import DslGame
from .validate import solve_path

CLICK = 6


def _policy_index(action, click_patch: int) -> int:
    aid, x, y = action
    if aid != CLICK:
        return int(aid)
    cells = 64 // click_patch
    return CLICK + (y // click_patch) * cells + (x // click_patch)


def _action_ids(spec) -> int:
    bits = 0
    for a in spec.simple_actions:
        bits |= 1 << a
    return bits | (1 << CLICK)


def level_demo(spec, level: int, solution, library, click_patch: int,
               baseline: int | None):
    """One sequence for one level: dict of arrays of length len(solution)."""
    one = dataclasses.replace(spec, layouts=spec.layouts[level:level + 1],
                              floors=spec.floors[level:level + 1],
                              budgets=None)
    game = DslGame(one, library=library)
    game.init()
    frames, targets, prev_action, prev_reward = [], [], [], []
    last_action, last_reward = 0, 0.0
    for k, action in enumerate(solution):
        frames.append(game.frame().copy())
        targets.append(_policy_index(action, click_patch))
        prev_action.append(last_action)
        prev_reward.append(last_reward)
        game.act(*action)
        last_action = targets[-1]
        last_reward = 0.0
    won = game.state == "WIN" or int(library.sym.harness_level_index(game.handle)) > 0
    game.close()
    if not won:
        return None
    n = len(solution)
    efficiency = 1.0 if baseline is None else min(1.0, baseline / max(n, 1))
    reward = np.zeros(n, np.float32)
    reward[-1] = (level + 1) * efficiency * efficiency
    return dict(obs=np.stack(frames).astype(np.uint8),
                target=np.array(targets, np.int32),
                prev_action=np.array(prev_action, np.int32),
                prev_reward=np.array(prev_reward, np.float32),
                reward=reward,
                done=np.array([True] + [False] * (n - 1)),
                action_ids=np.full(n, _action_ids(spec), np.int32))


def build(corpus: str, out: Path, library, click_patch: int = 4,
          max_nodes: int = 40_000, max_len: int = 256, limit: int | None = None,
          verbose: bool = True) -> dict:
    """Write demos for a corpus directory into out/demos.npz (ragged
    sequences concatenated, with offsets) and return summary counts."""
    files = sorted(glob.glob(str(Path(corpus) / "*.npz")))
    if limit:
        files = files[:limit]
    seqs = []
    counts = {"levels": 0, "from_label": 0, "from_solver": 0, "skipped": 0}
    for path in files:
        spec, labels = load(path)
        solutions = labels.get("solutions") or [None] * spec.num_levels
        baselines = labels.get("baselines") or [None] * spec.num_levels
        for level in range(spec.num_levels):
            solution = solutions[level]
            if solution is not None:
                solution = [tuple(a) for a in solution]
                counts["from_label"] += 1
            else:
                one = dataclasses.replace(spec, layouts=spec.layouts[level:level + 1],
                                          floors=spec.floors[level:level + 1],
                                          budgets=None)
                g = DslGame(one, library=library)
                solution = solve_path(g, max_nodes)
                g.close()
                if solution is None:
                    counts["skipped"] += 1
                    continue
                counts["from_solver"] += 1
            if len(solution) > max_len:
                counts["skipped"] += 1
                continue
            demo = level_demo(spec, level, solution, library, click_patch,
                              baselines[level])
            if demo is None:
                counts["skipped"] += 1
                continue
            demo["family"] = labels.get("family", "")
            seqs.append(demo)
            counts["levels"] += 1
        if verbose and len(seqs) % 200 == 0 and seqs:
            print(f"  {len(seqs)} sequences", flush=True)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    lengths = np.array([len(s["target"]) for s in seqs], np.int32)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    np.savez_compressed(
        out / "demos.npz",
        obs=np.concatenate([s["obs"] for s in seqs]),
        target=np.concatenate([s["target"] for s in seqs]),
        prev_action=np.concatenate([s["prev_action"] for s in seqs]),
        prev_reward=np.concatenate([s["prev_reward"] for s in seqs]),
        reward=np.concatenate([s["reward"] for s in seqs]),
        done=np.concatenate([s["done"] for s in seqs]),
        action_ids=np.concatenate([s["action_ids"] for s in seqs]),
        offsets=offsets,
        family=np.array([s["family"] for s in seqs]),
    )
    counts["steps"] = int(lengths.sum())
    (out / "summary.json").write_text(json.dumps(counts, indent=1))
    return counts
