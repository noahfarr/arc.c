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
from .tokens import tokenize
from .validate import solve_path

CLICK = 6


def _snap(v: int, origin: int, pitch: int, click_patch: int) -> int | None:
    """The coordinate of the click-patch centre that lies in the same grid
    cell as pixel `v`, or None when no patch centre falls inside that cell.
    The policy clicks patch centres, so a demo must label the patch whose
    centre hits the solver's cell, not the patch containing the solver's
    pixel (those differ whenever cells are not aligned to the patch grid)."""
    lo = origin + ((v - origin) // pitch) * pitch
    half = click_patch // 2
    first = ((lo - half + click_patch - 1) // click_patch) * click_patch + half
    return first if first < lo + pitch else None


def _policy_index(action, click_patch: int, spec=None) -> int | None:
    aid, x, y = action
    if aid != CLICK:
        return int(aid)
    if spec is not None and spec.pitch > 0:
        x = _snap(int(x), spec.origin_x, spec.pitch, click_patch)
        y = _snap(int(y), spec.origin_y, spec.pitch, click_patch)
        if x is None or y is None:
            return None
    cells = 64 // click_patch
    return CLICK + (y // click_patch) * cells + (x // click_patch)


def _action_ids(spec) -> int:
    bits = 0
    for a in spec.simple_actions:
        bits |= 1 << a
    return bits | (1 << CLICK)


def level_demo(spec, level: int, solution, library, click_patch: int,
               baseline: int | None, tokens: int = 0):
    """One sequence for one level: dict of arrays of length len(solution).
    With tokens > 0 the frame is also tokenised (that many regions) and a
    click's target is CLICK + the index of the region the solver clicked,
    replayed at that region's click pixel as the policy would."""
    one = dataclasses.replace(spec, layouts=spec.layouts[level:level + 1],
                              floors=spec.floors[level:level + 1],
                              budgets=None)
    game = DslGame(one, library=library)
    game.init()
    frames, targets, prev_action, prev_reward, toks = [], [], [], [], []
    last_action, last_reward = 0, 0.0
    for k, action in enumerate(solution):
        frame = game.frame().copy()
        frames.append(frame)
        if tokens:
            tok, labels, _ = tokenize(library, frame, cap=tokens, labels=True)
            toks.append(tok)
            if action[0] == CLICK:
                aid, x, y = action
                region = int(labels[int(y), int(x)])
                if region < 0:
                    game.close()
                    return None
                index = CLICK + region
                action = (CLICK, int(tok[region, 8]), int(tok[region, 9]))
            else:
                index = int(action[0])
        else:
            index = _policy_index(action, click_patch, spec)
        if index is None:
            game.close()
            return None
        targets.append(index)
        if not tokens and action[0] == CLICK:
            # Act where the policy would: the centre of the labelled patch.
            cell = index - CLICK
            cells = 64 // click_patch
            action = (CLICK, (cell % cells) * click_patch + click_patch // 2,
                      (cell // cells) * click_patch + click_patch // 2)
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
    extra = {"tokens": np.stack(toks)} if tokens else {}
    return dict(obs=np.stack(frames).astype(np.uint8),
                target=np.array(targets, np.int32),
                **extra,
                prev_action=np.array(prev_action, np.int32),
                prev_reward=np.array(prev_reward, np.float32),
                reward=reward,
                done=np.array([True] + [False] * (n - 1)),
                action_ids=np.full(n, _action_ids(spec), np.int32))


def build(corpus: str, out: Path, library, click_patch: int = 4,
          max_nodes: int = 40_000, max_len: int = 256, limit: int | None = None,
          verbose: bool = True, shuffle: bool = True, tokens: int = 0) -> dict:
    """Write demos for a corpus directory into out/demos.npz (ragged
    sequences concatenated, with offsets) and return summary counts."""
    files = sorted(glob.glob(str(Path(corpus) / "*.npz")))
    if limit:
        files = files[:limit]
    seqs = []
    counts = {"levels": 0, "from_label": 0, "from_solver": 0, "skipped": 0, "shuffled": 0}
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
            demo = None
            if shuffle and all(a[0] == CLICK for a in solution) and len(solution) > 1:
                # Commuting clicks (e.g. the match family): a random order
                # leaves "which cells" as the only predictable signal, so
                # the policy has to read the frame instead of learning the
                # solver's scan order. Keep the original order if the
                # shuffled replay does not win.
                rng = np.random.default_rng(hash((path, level)) & 0xFFFFFFFF)
                shuffled = [solution[i] for i in rng.permutation(len(solution))]
                demo = level_demo(spec, level, shuffled, library, click_patch,
                                  baselines[level], tokens)
                counts["shuffled"] += demo is not None
            if demo is None:
                demo = level_demo(spec, level, solution, library, click_patch,
                                  baselines[level], tokens)
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
        **({"tokens": np.concatenate([s["tokens"] for s in seqs])} if seqs and "tokens" in seqs[0] else {}),
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
