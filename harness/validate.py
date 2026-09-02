import ctypes
import dataclasses
from collections import deque

import numpy as np


@dataclasses.dataclass
class Exploration:
    nodes: int
    edges: int
    solvable: bool
    shortest: int | None
    exhausted: bool
    terminal_depth: int | None = None
    path: list | None = None


def _actions_for(game) -> list[tuple[int, int, int]]:
    spec = getattr(game, "spec", None)
    from .dsl import ON_CLICK

    if spec is None:
        # A ported game: use its declared simple actions. ACTION7 is undo,
        # which only revisits states and is useless to a graph search.
        simple = [a for a in game.levels.simple_actions if a != 7]
        out = [(a, 0, 0) for a in simple]
        if game.levels.has_click:
            for cy in range(64):
                for cx in range(64):
                    out.append((6, cx, cy))
        return out

    out = [(a, 0, 0) for a in (1, 2, 3, 4)]
    clickable = {i for i, k in enumerate(spec.kinds) if k.on_click}
    clickable |= {r.subject for r in spec.rules
                  if r.trigger == ON_CLICK and r.enabled}
    if -1 in clickable:
        cells = [(cy, cx) for cy in range(spec.grid_h)
                 for cx in range(spec.grid_w)]
    else:
        # Kinds never move in the interpreter (toggle swaps in place), so
        # only cells that start with a clickable kind can ever respond.
        cells = sorted({(int(cy), int(cx))
                        for lv in spec.layouts
                        for cy, cx in np.argwhere(np.isin(lv, list(clickable)))})
    for cy, cx in cells:
        out.append((6, spec.origin_x + cx * spec.pitch + spec.pitch // 2,
                    spec.origin_y + cy * spec.pitch + spec.pitch // 2))
    return out


def explore(game, aux_size: int, max_nodes: int = 50_000,
            actions: list[tuple[int, int, int]] | None = None) -> Exploration:
    sym = game.library.sym
    size = sym.game_state_size(game.handle, aux_size)
    actions = actions or _actions_for(game)

    def snapshot():
        buf = (ctypes.c_uint8 * size)()
        sym.game_save(game.handle, aux_size, ctypes.byref(buf))
        return buf

    def restore(buf):
        sym.game_load(game.handle, aux_size, ctypes.byref(buf))

    game.init()
    start_level = int(sym.harness_level_index(game.handle))
    root = snapshot()
    root_hash = sym.game_hash(game.handle, aux_size)
    seen = {root_hash}
    parent = {root_hash: (None, None)}
    frontier = deque([(root, 0, root_hash)])
    nodes = edges = 0
    shortest = None
    path = None

    while frontier:
        state, depth, state_hash = frontier.popleft()
        nodes += 1
        if nodes > max_nodes:
            return Exploration(nodes, edges, shortest is not None, shortest,
                               False, path=path)
        for action in actions:
            restore(state)
            game.act(*action)
            edges += 1
            level = int(sym.harness_level_index(game.handle))
            status = game.state
            if status == "WIN" or level > start_level:
                if shortest is None:
                    shortest = depth + 1
                    path = [action]
                    back = state_hash
                    while parent[back][0] is not None:
                        prev, act = parent[back]
                        path.append(act)
                        back = prev
                    path.reverse()
                continue
            if status == "GAME_OVER":
                continue
            h = sym.game_hash(game.handle, aux_size)
            if h in seen:
                continue
            seen.add(h)
            parent[h] = (state_hash, action)
            frontier.append((snapshot(), depth + 1, h))
    return Exploration(nodes, edges, shortest is not None, shortest, True,
                       path=path)


def random_solve_rate(game, trials: int = 10_000, horizon: int = 200,
                      seed: int = 0, actions=None) -> float:
    rng = np.random.default_rng(seed)
    actions = actions or _actions_for(game)
    sym = game.library.sym
    wins = 0
    for t in range(trials):
        game.init()
        start = int(sym.harness_level_index(game.handle))
        for _ in range(horizon):
            a = actions[int(rng.integers(len(actions)))]
            game.act(*a)
            if game.state == "WIN" or int(
                    sym.harness_level_index(game.handle)) > start:
                wins += 1
                break
            if game.state == "GAME_OVER":
                break
    return wins / trials


def certify(spec, trials: int = 10_000, horizon: int = 300, threads: int = 8,
            seed: int = 1, library=None, start_level: int = 0
            ) -> tuple[float, int]:
    from .dsl import DslGame

    games = [DslGame(spec, library=library) for _ in range(threads)]
    library = games[0].library
    actions = _actions_for(games[0])
    flat = np.ascontiguousarray(
        [v for a in actions for v in a], np.int32)
    handles = (ctypes.c_void_p * threads)(*[g.handle for g in games])
    steps = ctypes.c_int64()
    wins = library.sym.certify_random(
        handles, threads, trials, horizon, start_level,
        flat.ctypes.data_as(ctypes.c_void_p), len(actions), seed,
        ctypes.byref(steps))
    for g in games:
        g.close()
    return wins / trials, int(steps.value)


def _classify(base, without) -> str:
    if not without.solvable:
        return "enabling"
    if base.shortest is None or without.shortest is None:
        return "unknown"
    if without.shortest < base.shortest:
        return "constraining"
    return "decorative"


def necessity(spec, aux_size: int, max_nodes: int = 30_000,
              library=None) -> list[dict]:
    import dataclasses

    import numpy as np

    from .dsl import DslGame, EMPTY

    base_game = DslGame(spec, library=library)
    base = explore(base_game, aux_size, max_nodes=max_nodes)
    base_game.close()

    report = []
    for i in range(len(spec.rules)):
        variant = dataclasses.replace(
            spec,
            rules=[dataclasses.replace(r, enabled=0 if j == i else r.enabled)
                   for j, r in enumerate(spec.rules)])
        g = DslGame(variant, library=library)
        e = explore(g, aux_size, max_nodes=max_nodes)
        g.close()
        report.append({"kind": "rule", "index": i,
                       "solvable_without": e.solvable,
                       "shortest_without": e.shortest,
                       "role": _classify(base, e)})

    present = sorted({int(v) for v in np.unique(spec.layouts)})
    for k in present:
        if k < 0 or not spec.kinds[k].motion:
            continue
        layouts = np.array(spec.layouts)
        layouts[layouts == k] = EMPTY
        variant = dataclasses.replace(spec, layouts=layouts)
        g = DslGame(variant, library=library)
        e = explore(g, aux_size, max_nodes=max_nodes)
        g.close()
        report.append({"kind": "actor", "index": k,
                       "solvable_without": e.solvable,
                       "shortest_without": e.shortest,
                       "role": _classify(base, e)})
    return report, base
