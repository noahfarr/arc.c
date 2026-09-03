import dataclasses

import numpy as np

ARC_MAX_KINDS = 16

from .dsl import (BECOME, BLOCK, EMPTY, LOSE, NONE, PUSH, REMOVE, TOGGLE,
                  WIN_ALL_ON, WIN_NONE_LEFT, WIN_REACH, Kind, Spec)

PALETTE = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
ENTER_EFFECTS = [NONE, BLOCK, REMOVE, PUSH, LOSE]
CLICK_EFFECTS = [NONE, REMOVE, BECOME, TOGGLE]


@dataclasses.dataclass
class Proposal:
    spec: Spec
    seed: int
    mechanics: dict


def _layout(rng, w, h, kinds_present, wall, player, wall_density):
    floor = np.zeros((1, h, w), np.int8)
    obj = np.full((1, h, w), EMPTY, np.int8)
    obj[0, 0, :] = wall
    obj[0, -1, :] = wall
    obj[0, :, 0] = wall
    obj[0, :, -1] = wall
    free = [(y, x) for y in range(1, h - 1) for x in range(1, w - 1)]
    rng.shuffle(free)
    n_walls = int(len(free) * wall_density)
    for y, x in free[:n_walls]:
        obj[0, y, x] = wall
    cells = free[n_walls:]
    return floor, obj, cells


def sample(rng, seed: int) -> Proposal | None:
    w = int(rng.integers(6, 9))
    h = int(rng.integers(6, 9))
    colors = list(rng.permutation(PALETTE))

    floor_kind, wall_kind, player_kind = 0, 1, 2
    kinds = [Kind(color=int(colors[0])),
             Kind(color=int(colors[1]), on_enter=BLOCK),
             Kind(color=int(colors[2]))]
    extra = int(rng.integers(1, 4))
    mechanics = {}
    for i in range(extra):
        on_enter = int(rng.choice(ENTER_EFFECTS))
        on_click = int(rng.choice(CLICK_EFFECTS)) if rng.random() < 0.4 else NONE
        k = Kind(color=int(colors[3 + i]), on_enter=on_enter, on_click=on_click)
        if on_click in (BECOME,):
            k.click_a = floor_kind
        if on_click == TOGGLE:
            k.click_a, k.click_b = wall_kind, floor_kind
        kinds.append(k)
        mechanics[3 + i] = (on_enter, on_click)

    pushable = [i for i, k in enumerate(kinds) if k.on_enter == PUSH]
    removable = [i for i, k in enumerate(kinds) if k.on_enter == REMOVE]
    goal_kind = len(kinds)
    kinds.append(Kind(color=int(colors[3 + extra])))

    if pushable:
        win_mode, win_a, win_b = WIN_ALL_ON, pushable[0], goal_kind
    elif removable:
        win_mode, win_a, win_b = WIN_NONE_LEFT, removable[0], -1
    else:
        win_mode, win_a, win_b = WIN_REACH, goal_kind, -1
    if len(kinds) > 16:
        return None

    floor, obj, cells = _layout(rng, w, h, kinds, wall_kind, player_kind,
                                float(rng.uniform(0.0, 0.18)))
    if len(cells) < 6:
        return None

    at = 0
    py, px = cells[at]
    obj[0, py, px] = player_kind
    at += 1

    if win_mode == WIN_ALL_ON:
        n = int(rng.integers(1, 3))
        for _ in range(n):
            if at + 1 >= len(cells):
                return None
            by, bx = cells[at]
            obj[0, by, bx] = win_a
            at += 1
            gy, gx = cells[at]
            floor[0, gy, gx] = goal_kind
            at += 1
    elif win_mode == WIN_NONE_LEFT:
        for _ in range(int(rng.integers(1, 4))):
            if at >= len(cells):
                return None
            y, x = cells[at]
            obj[0, y, x] = win_a
            at += 1
    else:
        if at >= len(cells):
            return None
        y, x = cells[at]
        floor[0, y, x] = goal_kind
        at += 1

    for i, k in enumerate(kinds):
        if i in (floor_kind, wall_kind, player_kind, goal_kind):
            continue
        if k.on_enter in (PUSH, REMOVE) and i == win_a:
            continue
        for _ in range(int(rng.integers(0, 3))):
            if at >= len(cells):
                break
            y, x = cells[at]
            obj[0, y, x] = i
            at += 1

    spec = Spec(kinds=kinds, layouts=obj, floors=floor,
                player_kind=player_kind, win_mode=win_mode, win_a=win_a,
                win_b=win_b, pitch=int(min(64 // max(w, h), 8)),
                origin_x=1, origin_y=1)
    return Proposal(spec=spec, seed=seed, mechanics=mechanics)


def sample_push(rng, w=9, h=9, boxes=2, pulls=20, wall_density=0.08,
                colors=None, bias=0.0):
    """Backwards sokoban: boxes start on their goals and are pulled away.
    With bias > 0 that fraction of pulls picks the option that moves the
    boxes furthest from their goals, so solutions grow with pulls instead
    of random-walking back."""
    from .dsl import WIN_ALL_ON

    colors = colors or [11, 9, 8, 13, 7]
    floor_kind, wall_kind, player_kind, box_kind, goal_kind = 0, 1, 2, 3, 4
    kinds = [Kind(color=int(colors[0])),
             Kind(color=int(colors[1]), on_enter=BLOCK),
             Kind(color=int(colors[2])),
             Kind(color=int(colors[3]), on_enter=PUSH),
             Kind(color=int(colors[4]))]

    floor = np.full((1, h, w), EMPTY, np.int8)
    obj = np.full((1, h, w), EMPTY, np.int8)
    obj[0, 0, :] = wall_kind
    obj[0, -1, :] = wall_kind
    obj[0, :, 0] = wall_kind
    obj[0, :, -1] = wall_kind

    interior = [(y, x) for y in range(1, h - 1) for x in range(1, w - 1)]
    rng.shuffle(interior)
    n_walls = int(len(interior) * wall_density)
    for y, x in interior[:n_walls]:
        obj[0, y, x] = wall_kind
    free = interior[n_walls:]
    if len(free) < boxes + 2:
        return None

    box_cells = [tuple(c) for c in free[:boxes]]
    for y, x in box_cells:
        floor[0, y, x] = goal_kind
        obj[0, y, x] = box_kind
    py, px = free[boxes]

    def is_free(y, x):
        return 0 <= y < h and 0 <= x < w and obj[0, y, x] == EMPTY

    def reachable(sy, sx):
        seen = {(sy, sx)}
        stack = [(sy, sx)]
        while stack:
            y, x = stack.pop()
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if (ny, nx) not in seen and is_free(ny, nx):
                    seen.add((ny, nx))
                    stack.append((ny, nx))
        return seen

    def goal_distance(y, x):
        return min(abs(y - gy) + abs(x - gx) for gy, gx in box_cells)

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    applied = 0
    for _ in range(pulls):
        # Every pull the player can reach: stand at box+d, step to box+2d,
        # the box follows to box+d. Reachability decides, not a walk, so
        # the player's route between pulls is not part of the level.
        reach = reachable(py, px)
        options = []
        for by, bx in map(tuple, np.argwhere(obj[0] == box_kind)):
            for dy, dx in dirs:
                sy, sx = by + dy, bx + dx
                ty, tx = by + 2 * dy, bx + 2 * dx
                if (sy, sx) in reach and is_free(ty, tx):
                    options.append((by, bx, dy, dx))
        if not options:
            break
        if rng.random() < bias:
            gains = [goal_distance(by + dy, bx + dx)
                     for by, bx, dy, dx in options]
            best = max(gains)
            options = [o for o, g in zip(options, gains) if g == best]
        by, bx, dy, dx = options[int(rng.integers(len(options)))]
        obj[0, by, bx] = EMPTY
        obj[0, by + dy, bx + dx] = box_kind
        py, px = by + 2 * dy, bx + 2 * dx
        applied += 1

    if bias > 0 and applied:
        # Start the player far from the boxes, so walking to them is part
        # of the solution as it is in the public games.
        reach = sorted(reachable(py, px))
        boxes_now = [tuple(c) for c in np.argwhere(obj[0] == box_kind)]

        def box_distance(c):
            return min(abs(c[0] - by) + abs(c[1] - bx) for by, bx in boxes_now)
        reach.sort(key=box_distance, reverse=True)
        top = reach[:max(1, len(reach) // 4)]
        py, px = top[int(rng.integers(len(top)))]

    if applied == 0:
        return None
    obj[0, py, px] = player_kind
    pitch = int(max(1, min(64 // max(w, h), 8)))
    spec = Spec(kinds=kinds, layouts=obj, floors=floor,
                player_kind=player_kind, win_mode=WIN_ALL_ON, win_a=box_kind,
                win_b=goal_kind, pitch=pitch,
                origin_x=(64 - w * pitch) // 2,
                origin_y=(64 - h * pitch) // 2)
    return Proposal(spec=spec, seed=0, mechanics={"pulls": applied})


# Target bands for the optimal solution length of each level, by curriculum
# stage. Stage 2 is the public games: humans need a median of 30 actions on
# level 1 and are at the optimum there; later levels run 2-2.5x their
# optimum, and human baselines for the last level have a median of 113
# (docs/benchmark.md). Stages 0 and 1 are shorter so that a policy trained
# from scratch finds its first completions; a corpus mixes the stages.
STAGE_BANDS = [
    [(3, 8), (5, 12), (8, 18), (10, 24), (12, 30), (14, 36), (16, 40), (18, 45)],
    [(6, 16), (10, 24), (14, 32), (18, 42), (22, 55), (26, 65), (30, 75), (34, 85)],
    [(12, 30), (18, 40), (24, 55), (30, 70), (36, 90), (40, 120), (45, 140), (50, 160)],
]
LENGTH_BANDS = STAGE_BANDS[2]
# Random-play win rate a non-tutorial level may have, by stage; the last is
# the foundation's own bar.
STAGE_RANDOM_BAR = [2.5e-1, 5e-3, 1e-4]

LADDER = [
    dict(w=11, h=11, boxes=1, pulls=30, wall_density=0.06, bias=0.6),
    dict(w=11, h=11, boxes=1, pulls=45, wall_density=0.10, bias=0.6),
    dict(w=11, h=11, boxes=2, pulls=70, wall_density=0.08, bias=0.5),
    dict(w=13, h=13, boxes=2, pulls=120, wall_density=0.10, bias=0.5),
    dict(w=13, h=13, boxes=3, pulls=200, wall_density=0.10, bias=0.4),
    dict(w=15, h=15, boxes=3, pulls=300, wall_density=0.12, bias=0.4),
    dict(w=15, h=15, boxes=4, pulls=400, wall_density=0.12, bias=0.3),
    dict(w=15, h=15, boxes=5, pulls=500, wall_density=0.12, bias=0.3),
]

# Budgets on the public set run 1-3x the human baseline, i.e. roughly 2-6x
# the optimum; draw the multiplier per level.
BUDGET_RANGE = (2.5, 4.5)


def _shortest(spec, level, library, aux_size, max_nodes):
    import dataclasses

    from .dsl import DslGame
    from .validate import solve

    one = dataclasses.replace(spec, layouts=spec.layouts[level:level + 1],
                              floors=spec.floors[level:level + 1],
                              budgets=None)
    g = DslGame(one, library=library)
    shortest, exhausted, _ = solve(g, max_nodes=max_nodes)
    g.close()
    return shortest, exhausted


def sample_environment(rng, levels=6, library=None, aux_size=None,
                       max_nodes=40_000, attempts=30, stage=2):
    """A sokoban ladder whose levels are drawn until their optimal solution
    length falls in LENGTH_BANDS, with a per-level budget derived from it.

    Levels whose search does not finish within max_nodes are accepted on
    the pull count alone and get a budget from that instead."""
    import dataclasses

    from .dsl import Spec, WIN_ALL_ON

    colors = [int(c) for c in rng.permutation(PALETTE)[:5]]
    w = max(cfg["w"] for cfg in LADDER[:levels])
    h = max(cfg["h"] for cfg in LADDER[:levels])
    stack_obj, stack_floor, meta, budgets = [], [], [], []
    bands = STAGE_BANDS[stage]
    for i in range(levels):
        cfg = dict(LADDER[i % len(LADDER)])
        lo, hi = bands[min(i, len(bands) - 1)]
        if stage < 2:
            # Shorter pulls and smaller boards for the easy stages, so the
            # search exhausts and the optimum lands in the band.
            cfg["pulls"] = max(4, int(cfg["pulls"] * (0.25 if stage == 0 else 0.5)))
            cfg["w"] = cfg["h"] = max(7, cfg["w"] - (4 if stage == 0 else 2))
        chosen = None
        for _ in range(attempts):
            cand = sample_push(rng, colors=colors, **cfg)
            if cand is None or cand.mechanics["pulls"] < 0.5 * cfg["pulls"]:
                continue
            shortest, exhausted = (None, False)
            if library is not None:
                shortest, exhausted = _shortest(cand.spec, 0, library,
                                                aux_size, max_nodes)
            if shortest is None and exhausted:
                continue
            if shortest is not None and not (lo <= shortest <= hi):
                continue
            chosen = (cand, shortest)
            break
        if chosen is None:
            return None
        p, shortest = chosen
        # Pad with background, not wall: each level carries its own border
        # walls, and a wall-coloured slab would cover most of the frame.
        obj = np.full((h, w), EMPTY, np.int8)
        flr = np.full((h, w), EMPTY, np.int8)
        oy = (h - cfg["h"]) // 2
        ox = (w - cfg["w"]) // 2
        obj[oy:oy + cfg["h"], ox:ox + cfg["w"]] = p.spec.layouts[0]
        flr[oy:oy + cfg["h"], ox:ox + cfg["w"]] = p.spec.floors[0]
        stack_obj.append(obj)
        stack_floor.append(flr)
        base = shortest if shortest is not None else int(0.6 * p.mechanics["pulls"])
        budgets.append(int(round(base * rng.uniform(*BUDGET_RANGE))))
        meta.append({"pulls": p.mechanics["pulls"], "boxes": cfg["boxes"],
                     "shortest": shortest})
    proto = p.spec
    pitch = int(max(1, min(62 // max(w, h), 8)))
    spec = Spec(kinds=proto.kinds, layouts=np.stack(stack_obj),
                floors=np.stack(stack_floor), player_kind=proto.player_kind,
                win_mode=WIN_ALL_ON, win_a=proto.win_a, win_b=proto.win_b,
                pitch=pitch, origin_x=(64 - w * pitch) // 2,
                origin_y=(62 - h * pitch) // 2,
                budgets=np.array(budgets, np.int32))
    return Proposal(spec=spec, seed=0, mechanics={"levels": meta, "stage": stage})


MECHANICS = ("key", "switch", "collect")
HAZARDS = ("patrol", "chase")


def randomise_appearance(rng, spec, protect=()):
    import dataclasses

    pitch = spec.pitch
    kinds = []
    for i, k in enumerate(spec.kinds):
        if i in protect or pitch < 3:
            kinds.append(k)
            continue
        span = int(np.clip(round(pitch * rng.beta(1.1, 2.2)), 1, pitch))
        slack = pitch - span
        kinds.append(dataclasses.replace(
            k, size=span,
            off_x=int(rng.integers(0, slack + 1)),
            off_y=int(rng.integers(0, slack + 1))))
    return dataclasses.replace(spec, kinds=kinds)


def _rooms_kinds(rng, chosen, hazards):
    """Kinds and rules for a chain of rooms gated by the mechanics in
    `chosen` (each "key", "switch" or "collect"), plus decor kinds and
    hazard kinds. Returns everything the layout needs."""
    from .dsl import (CHASE, IF_NONE_LEFT, ON_CLICK, ON_ENTER, ON_STEP,
                      PATROL, REMOVE, Rule, TOGGLE)

    swatch = [int(c) for c in rng.permutation(PALETTE)[:int(rng.integers(5, 8))]]
    colors = [swatch[i % len(swatch)] for i in range(2 * ARC_MAX_KINDS)]
    rng.shuffle(colors)
    floor_k, wall_k, player_k, goal_k = 0, 1, 2, 3
    kinds = [Kind(color=int(colors[0])),
             Kind(color=int(colors[1]), on_enter=BLOCK),
             Kind(color=int(colors[2])),
             Kind(color=int(colors[3]))]
    rules, door_kind, trigger_kind = [], [], []
    for m, kind in enumerate(chosen):
        d = len(kinds)
        kinds.append(Kind(color=int(colors[4 + 2 * m]), on_enter=BLOCK))
        t = len(kinds)
        kinds.append(Kind(color=int(colors[5 + 2 * m]),
                          on_enter=REMOVE if kind != "switch" else NONE))
        door_kind.append(d)
        trigger_kind.append(t)
        if kind == "key":
            rules.append(Rule(trigger=ON_ENTER, subject=t, effect=TOGGLE,
                              effect_a=d, effect_b=floor_k))
        elif kind == "switch":
            rules.append(Rule(trigger=ON_CLICK, subject=t, effect=TOGGLE,
                              effect_a=d, effect_b=floor_k))
        else:
            rules.append(Rule(trigger=ON_STEP, subject=-1, effect=TOGGLE,
                              predicate=IF_NONE_LEFT, pred_a=t,
                              effect_a=d, effect_b=floor_k))
    scenery_kinds = []
    for _ in range(3):
        if len(kinds) >= 13:
            break
        scenery_kinds.append(len(kinds))
        kinds.append(Kind(color=int(colors[len(kinds)])))
    hazard_kinds = []
    for hz in range(hazards):
        if len(kinds) >= 15:
            break
        style = str(rng.choice(HAZARDS))
        a = len(kinds)
        if style == "patrol":
            kinds.append(Kind(color=int(colors[-1 - 2 * hz]), motion=PATROL,
                              motion_a=3, motion_b=a + 1, deadly=1))
            kinds.append(Kind(color=int(colors[-1 - 2 * hz]), motion=PATROL,
                              motion_a=2, motion_b=a, deadly=1))
        else:
            kinds.append(Kind(color=int(colors[-1 - 2 * hz]), motion=CHASE,
                              deadly=1))
        hazard_kinds.append(a)
    return dict(kinds=kinds, rules=rules, door_kind=door_kind,
                trigger_kind=trigger_kind, scenery_kinds=scenery_kinds,
                hazard_kinds=hazard_kinds, floor_k=floor_k, wall_k=wall_k,
                player_k=player_k, goal_k=goal_k)


def _rooms_layout(rng, parts, chosen, order, room_w, room_h, hazards,
                  scenery, collect_n):
    """One level: rooms order[0..] in a snake grid, door i between rooms i
    and i+1 opened by mechanic order[i], player in the first room, goal in
    the last. Returns (obj, floor, w, h)."""
    import math

    used = len(order)
    rooms = used + 1
    cols = max(1, math.ceil(math.sqrt(rooms)))
    grid_rows = math.ceil(rooms / cols)
    w = cols * room_w + cols + 1
    h = grid_rows * room_h + grid_rows + 1
    wall_k, player_k, goal_k = parts["wall_k"], parts["player_k"], parts["goal_k"]

    def place(i):
        r = i // cols
        c = i % cols if r % 2 == 0 else cols - 1 - (i % cols)
        return r, c

    def room_cells(i):
        r, c = place(i)
        x0 = 1 + c * (room_w + 1)
        y0 = 1 + r * (room_h + 1)
        return [(y, x) for y in range(y0, y0 + room_h)
                for x in range(x0, x0 + room_w)]

    def door_between(i, j):
        ri, ci = place(i)
        rj, cj = place(j)
        if ri == rj:
            x = 1 + min(ci, cj) * (room_w + 1) + room_w
            y = 1 + ri * (room_h + 1) + int(rng.integers(room_h))
        else:
            x = 1 + ci * (room_w + 1) + int(rng.integers(room_w))
            y = 1 + min(ri, rj) * (room_h + 1) + room_h
        return y, x

    obj = np.full((h, w), wall_k, np.int8)
    flr = np.full((h, w), EMPTY, np.int8)
    for i in range(rooms):
        for y, x in room_cells(i):
            obj[y, x] = EMPTY
    for i in range(used):
        dy, dx = door_between(i, i + 1)
        obj[dy, dx] = parts["door_kind"][order[i]]

    def free_in(i):
        spots = [c for c in room_cells(i) if obj[c] == EMPTY]
        rng.shuffle(spots)
        return spots

    obj[free_in(0)[0]] = player_k
    for i in range(used):
        m = order[i]
        spots = free_in(i)
        count = collect_n if chosen[m] == "collect" else 1
        for j in range(min(count, len(spots))):
            obj[spots[j]] = parts["trigger_kind"][m]
    for n, hk in enumerate(parts["hazard_kinds"][:hazards]):
        room = 1 + (n % max(1, used))
        spots = free_in(room)
        if len(spots) > 3:
            obj[spots[0]] = hk
    for _ in range(scenery):
        room = int(rng.integers(0, rooms))
        spots = free_in(room)
        if len(spots) > 3 and parts["scenery_kinds"]:
            obj[spots[0]] = parts["scenery_kinds"][
                int(rng.integers(len(parts["scenery_kinds"])))]
    last = free_in(used)
    flr[last[0]] = goal_k
    return obj, flr, w, h


def _rooms_spec(parts, layouts, floors, budgets=None):
    from .dsl import Spec, WIN_REACH

    h = max(o.shape[0] for o in layouts)
    w = max(o.shape[1] for o in layouts)
    stack_obj, stack_flr = [], []
    for obj, flr in zip(layouts, floors):
        o = np.full((h, w), EMPTY, np.int8)
        f = np.full((h, w), EMPTY, np.int8)
        oy, ox = (h - obj.shape[0]) // 2, (w - obj.shape[1]) // 2
        o[oy:oy + obj.shape[0], ox:ox + obj.shape[1]] = obj
        f[oy:oy + obj.shape[0], ox:ox + obj.shape[1]] = flr
        stack_obj.append(o)
        stack_flr.append(f)
    pitch = int(max(1, min(62 // max(w, h), 8)))
    return Spec(kinds=parts["kinds"], layouts=np.stack(stack_obj),
                floors=np.stack(stack_flr), player_kind=parts["player_k"],
                win_mode=WIN_REACH, win_a=parts["goal_k"], pitch=pitch,
                origin_x=(64 - w * pitch) // 2, origin_y=(62 - h * pitch) // 2,
                rules=parts["rules"],
                budgets=None if budgets is None else np.array(budgets, np.int32))


def sample_composed(rng, num_mechanics=3, room_w=3, room_h=5, levels=None,
                    hazards=0, scenery=6):
    """Rooms gated by num_mechanics mechanics, one level per number of
    mechanics in use (the single-size form novelty and necessity checks
    use)."""
    chosen = [str(rng.choice(MECHANICS)) for _ in range(num_mechanics)]
    parts = _rooms_kinds(rng, chosen, hazards)
    if len(parts["kinds"]) > 16 or len(parts["rules"]) > 8:
        return None
    plan = levels or list(range(1, num_mechanics + 1))
    layouts, floors = [], []
    for used in plan:
        obj, flr, w, h = _rooms_layout(
            rng, parts, chosen, list(range(used)), room_w, room_h, hazards,
            scenery, int(rng.integers(2, 4)))
        if w > 32 or h > 32:
            return None
        layouts.append(obj)
        floors.append(flr)
    spec = _rooms_spec(parts, layouts, floors)
    return Proposal(spec=spec, seed=0,
                    mechanics={"kinds": chosen, "hazards": parts["hazard_kinds"]})


# Six-level plan for the rooms family: mechanics arrive one per level,
# then the same mechanics recur in bigger rooms, in a different order,
# with hazards and decor. (used, room_w, room_h, hazards, scenery, collect)
ROOMS_LADDER = [
    (1, 5, 5, 0, 2, 2),
    (2, 5, 5, 0, 3, 2),
    (3, 5, 5, 0, 4, 3),
    (3, 6, 6, 1, 4, 3),
    (3, 7, 7, 1, 6, 4),
    (3, 7, 7, 2, 6, 4),
]


def sample_rooms(rng, levels=6, library=None, aux_size=None,
                 max_nodes=40_000, attempts=20, stage=2):
    """The rooms family as a ladder: each level is redrawn until BFS puts
    its optimum inside LENGTH_BANDS (or the search does not finish), with
    budgets from the optimum."""
    import dataclasses

    chosen = [str(rng.choice(MECHANICS)) for _ in range(3)]
    parts = _rooms_kinds(rng, chosen, 2)
    if len(parts["kinds"]) > 16 or len(parts["rules"]) > 8:
        return None
    layouts, floors, budgets, meta = [], [], [], []
    bands = STAGE_BANDS[stage]
    for i in range(levels):
        used, rw, rh, hz, sc, cn = ROOMS_LADDER[min(i, len(ROOMS_LADDER) - 1)]
        used = min(used, len(chosen))
        if stage < 2:
            rw, rh = max(3, rw - (2 if stage == 0 else 1)), max(3, rh - (2 if stage == 0 else 1))
            hz = 0 if stage == 0 else hz
        lo, hi = bands[min(i, len(bands) - 1)]
        chosen_level = None
        for _ in range(attempts):
            order = list(range(used))
            if i >= 3:
                rng.shuffle(order)
            obj, flr, w, h = _rooms_layout(rng, parts, chosen, order, rw, rh,
                                           hz, sc, cn)
            if w > 32 or h > 32:
                continue
            shortest, exhausted = None, False
            if library is not None:
                one = _rooms_spec(parts, [obj], [flr])
                shortest, exhausted = _shortest(one, 0, library, aux_size,
                                                max_nodes)
                if shortest is None and exhausted:
                    continue
                if shortest is not None and not (lo <= shortest <= hi):
                    continue
            chosen_level = (obj, flr, shortest, order)
            break
        if chosen_level is None:
            return None
        obj, flr, shortest, order = chosen_level
        layouts.append(obj)
        floors.append(flr)
        base = shortest if shortest is not None else (used + 1) * (rw + rh)
        budgets.append(int(round(base * rng.uniform(*BUDGET_RANGE))))
        meta.append({"used": used, "order": order, "shortest": shortest})
    spec = _rooms_spec(parts, layouts, floors, budgets)
    return Proposal(spec=spec, seed=0,
                    mechanics={"kinds": chosen, "hazards": parts["hazard_kinds"],
                               "levels": meta, "stage": stage})


def _pad(layouts, floors, fill=EMPTY):
    h = max(o.shape[0] for o in layouts)
    w = max(o.shape[1] for o in layouts)
    outs, flrs = [], []
    for obj, flr in zip(layouts, floors):
        o = np.full((h, w), fill, np.int8)
        f = np.full((h, w), EMPTY, np.int8)
        oy, ox = (h - obj.shape[0]) // 2, (w - obj.shape[1]) // 2
        o[oy:oy + obj.shape[0], ox:ox + obj.shape[1]] = obj
        f[oy:oy + obj.shape[0], ox:ox + obj.shape[1]] = flr
        outs.append(o)
        flrs.append(f)
    return np.stack(outs), np.stack(flrs), w, h


def _geometry(w, h):
    pitch = int(max(1, min(62 // max(w, h), 8)))
    return pitch, (64 - w * pitch) // 2, (62 - h * pitch) // 2


# (blocks, side, walls, decoys) per level of the selection family
# A single block is random-solvable on any board, so only the tutorial
# level has one.
SELECT_LADDER = [(1, 8, 0.04, 0), (2, 10, 0.08, 1), (2, 12, 0.10, 1),
                 (3, 12, 0.10, 2), (3, 14, 0.12, 2), (4, 14, 0.12, 3),
                 (4, 16, 0.12, 3), (5, 16, 0.12, 4)]


def sample_select(rng, levels=6, library=None, aux_size=None,
                  max_nodes=40_000, attempts=30, stage=2):
    """Selection family (the cn04 / ka59 / sk48 control scheme): no avatar.
    A click, or ACTION5, selects a block; the arrows slide it; every block
    must end on a goal tile. Blocks never push and walls stop them."""
    from .dsl import CONTROL_SELECT, Spec, WIN_ALL_ON

    swatch = [int(c) for c in rng.permutation(PALETTE)[:6]]
    floor_k, wall_k, ghost_k, block_k, goal_k, decoy_k = range(6)
    kinds = [Kind(color=swatch[0]), Kind(color=swatch[1], on_enter=BLOCK),
             Kind(color=swatch[2]),  # never placed: stands in for the player
             Kind(color=swatch[3], selectable=1), Kind(color=swatch[4]),
             Kind(color=swatch[5])]
    bands = STAGE_BANDS[stage]
    layouts, floors, budgets, meta = [], [], [], []
    for i in range(levels):
        blocks, side, walls, decoys = SELECT_LADDER[min(i, len(SELECT_LADDER) - 1)]
        side = max(6, min(20, side + {0: -3, 1: -1, 2: 6}[stage]))
        lo, hi = bands[min(i, len(bands) - 1)]
        chosen = None
        for _ in range(attempts):
            obj = np.full((side, side), EMPTY, np.int8)
            flr = np.full((side, side), EMPTY, np.int8)
            obj[0, :] = obj[-1, :] = obj[:, 0] = obj[:, -1] = wall_k
            cells = [(y, x) for y in range(1, side - 1) for x in range(1, side - 1)]
            rng.shuffle(cells)
            n_walls = int(len(cells) * walls)
            for y, x in cells[:n_walls]:
                obj[y, x] = wall_k
            free = cells[n_walls:]
            if len(free) < 2 * blocks + decoys + 2:
                continue
            for k in range(blocks):
                flr[free[k]] = goal_k
            # blocks far from the goals, so sliding them is the level
            rest = free[blocks:]
            rest.sort(key=lambda c: -min(abs(c[0] - g[0]) + abs(c[1] - g[1])
                                        for g in free[:blocks]))
            far = rest[:max(blocks, len(rest) // 3)]
            rng.shuffle(far)
            for k in range(blocks):
                obj[far[k]] = block_k
            for k in range(decoys):
                c = far[blocks + k] if blocks + k < len(far) else None
                if c is not None:
                    obj[c] = decoy_k
            one = Spec(kinds=kinds, layouts=obj[None], floors=flr[None],
                       player_kind=ghost_k, win_mode=WIN_ALL_ON, win_a=block_k,
                       win_b=goal_k, pitch=1, control=CONTROL_SELECT,
                       uses_action5=1)
            shortest, exhausted = None, False
            if library is not None:
                shortest, exhausted = _shortest(one, 0, library, aux_size,
                                                max_nodes)
                if shortest is None and exhausted:
                    continue
                if shortest is not None and not (lo <= shortest <= hi):
                    continue
            chosen = (obj, flr, shortest)
            break
        if chosen is None:
            return None
        obj, flr, shortest = chosen
        layouts.append(obj)
        floors.append(flr)
        base = shortest if shortest is not None else 3 * blocks * side // 2
        budgets.append(int(round(base * rng.uniform(*BUDGET_RANGE))))
        meta.append({"blocks": blocks, "side": side, "shortest": shortest})
    L, F, w, h = _pad(layouts, floors)
    pitch, ox, oy = _geometry(w, h)
    spec = Spec(kinds=kinds, layouts=L, floors=F, player_kind=ghost_k,
                win_mode=WIN_ALL_ON, win_a=block_k, win_b=goal_k, pitch=pitch,
                origin_x=ox, origin_y=oy, control=CONTROL_SELECT,
                uses_action5=1, budgets=np.array(budgets, np.int32))
    return Proposal(spec=spec, seed=0, mechanics={"levels": meta, "stage": stage})


# Canvas size, colour count and stencil of the match family are drawn once
# per game: one spec has one set of kinds, and the cycle must only visit
# colours the target uses or the search space explodes. Scramble clicks
# per level follow the stage's length bands.
MATCH_SIDES = {0: (2, 3), 1: (4, 6), 2: (6, 8)}
MATCH_COLOURS = {0: (2, 2), 1: (2, 3), 2: (3, 4)}
# Length bands for the match family at stage 0: a uniform clicker on a
# 2x2 or 3x3 canvas completes a one- or two-click level often enough to
# give the click head its first gradient; the shared bands start at 3
# exact clicks, which it never does.
MATCH_STAGE0_BANDS = [(1, 2), (1, 3), (2, 4), (2, 5), (3, 6), (3, 8), (4, 9), (4, 9)]


def sample_match(rng, levels=6, library=None, aux_size=None,
                 max_nodes=40_000, attempts=20, stage=2):
    """Picture-match family (the ft09 / cd82 / re86 goal): a canvas on the
    left, a target on the right. Clicking a canvas cell cycles its colour;
    with a stencil (per game) the neighbours cycle too, lights-out style.
    Levels are made backwards by scrambling the target with random clicks,
    so the optimum is at most the number of scramble clicks."""
    from .dsl import CYCLE, STENCIL_BLOCK, STENCIL_CROSS, Spec, WIN_MATCH

    swatch = [int(c) for c in rng.permutation(PALETTE)[:7]]
    floor_k, wall_k, ghost_k = 0, 1, 2
    kinds = [Kind(color=swatch[0]), Kind(color=swatch[1], on_enter=BLOCK),
             Kind(color=swatch[2])]
    lo_side, hi_side = MATCH_SIDES[stage]
    side = int(rng.integers(lo_side, hi_side + 1))
    colours = int(rng.integers(MATCH_COLOURS[stage][0], MATCH_COLOURS[stage][1] + 1))
    stencil = int(rng.choice([0, 0, 1, 2] if stage == 2 else [0, 1]))
    canvas_k = len(kinds)
    for c in range(colours):
        kinds.append(Kind(color=swatch[3 + c], on_click=CYCLE, click_a=canvas_k,
                          click_b=canvas_k + colours - 1, stencil=stencil))
    target_k = len(kinds)
    for c in range(colours):
        kinds.append(Kind(color=swatch[3 + c]))
    w, h = 2 * side + 3, side + 2
    match = (1, 1, side + 2, 1, side, side)
    bands = MATCH_STAGE0_BANDS if stage == 0 else STAGE_BANDS[stage]
    layouts, floors, budgets, meta = [], [], [], []
    capacity = side * side * (colours - 1)
    for i in range(levels):
        ncol = colours
        lo, hi = bands[min(i, len(bands) - 1)]
        chosen = None
        for _ in range(attempts):
            # Scramble with about as many clicks as the band asks for;
            # cancelling clicks make the optimum shorter, which the search
            # catches when it can finish.
            scramble = int(min(capacity, rng.integers(lo, hi + 1)))
            obj = np.full((h, w), wall_k, np.int8)
            flr = np.full((h, w), EMPTY, np.int8)
            target = rng.integers(0, ncol, (side, side))
            canvas = target.copy()
            clicked = []
            for _ in range(scramble):
                y, x = rng.integers(0, side, 2)
                clicked.append((int(y), int(x)))
                cells = [(y, x)]
                if stencil >= STENCIL_CROSS:
                    cells += [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]
                if stencil == STENCIL_BLOCK:
                    cells += [(y - 1, x - 1), (y + 1, x - 1), (y - 1, x + 1), (y + 1, x + 1)]
                for cy, cx in cells:
                    if 0 <= cy < side and 0 <= cx < side:
                        canvas[cy, cx] = (canvas[cy, cx] + 1) % ncol
            if np.array_equal(canvas, target):
                continue
            obj[1:1 + side, 1:1 + side] = canvas_k + canvas
            obj[1:1 + side, side + 2:2 * side + 2] = target_k + target
            one = Spec(kinds=kinds, layouts=obj[None], floors=flr[None],
                       player_kind=ghost_k, win_mode=WIN_MATCH, pitch=1,
                       match=match)
            shortest, exhausted = None, False
            if library is not None:
                shortest, exhausted = _shortest(one, 0, library, aux_size,
                                                max_nodes)
                if shortest is None and exhausted:
                    continue
                if shortest is not None and not (lo <= shortest <= hi):
                    continue
            chosen = (obj, flr, shortest, clicked)
            break
        if chosen is None:
            return None
        obj, flr, shortest, clicked = chosen
        layouts.append(obj)
        floors.append(flr)
        base = shortest if shortest is not None else scramble
        budgets.append(int(round(base * rng.uniform(*BUDGET_RANGE))))
        # A known solution: undo every scramble click by cycling the same
        # cell the rest of the way round (ncol - 1 more clicks). Stored as
        # grid cells of the canvas; the corpus turns them into pixels.
        solution = [(1 + x, 1 + y) for (y, x) in clicked for _ in range(ncol - 1)]
        meta.append({"side": side, "colours": ncol, "stencil": stencil,
                     "scramble": scramble, "shortest": shortest,
                     "solution_cells": solution})
    L = np.stack(layouts)
    F = np.stack(floors)
    pitch, gx, gy = _geometry(w, h)
    spec = Spec(kinds=kinds, layouts=L, floors=F, player_kind=ghost_k,
                win_mode=WIN_MATCH, pitch=pitch, origin_x=gx, origin_y=gy,
                match=match, budgets=np.array(budgets, np.int32))
    return Proposal(spec=spec, seed=0, mechanics={"levels": meta, "stage": stage})


# Key layouts: the benchmark maps ACTION1..4 to up/down/left/right; one
# public game rotates them on some levels. Identity most of the time.
KEY_LAYOUTS = [((0, 1, 2, 3), 0.85), ((2, 3, 1, 0), 0.05), ((3, 2, 0, 1), 0.05),
               ((1, 0, 3, 2), 0.05)]


def randomise_controls(rng, spec, family: str):
    """Per-game control semantics an agent cannot assume on the hidden
    set: which key moves which way, whether ACTION5 is declared and what
    it does, and a fresh colour permutation at every reset."""
    from .dsl import A5_CYCLE, A5_NONE

    layouts, weights = zip(*KEY_LAYOUTS)
    key_dir = layouts[int(rng.choice(len(layouts), p=weights))]
    if family == "select":
        uses5, action5 = 1, A5_CYCLE
    else:
        # Declared but inert half the time: probing ACTION5 must cost.
        uses5 = int(rng.random() < 0.5)
        action5 = A5_NONE
    return dataclasses.replace(spec, key_dir=key_dir, uses_action5=uses5,
                               action5=action5, palette_shuffle=1,
                               seed=int(rng.integers(1, 2**31 - 1)))


def sample_verified(rng, aux_size, library=None, attempts=12, **kwargs):
    import ctypes
    import dataclasses

    from .validate import necessity

    for _ in range(attempts):
        proposal = sample_composed(rng, **kwargs)
        if proposal is None:
            continue
        spec = proposal.spec
        final = dataclasses.replace(spec, layouts=spec.layouts[-1:],
                                    floors=spec.floors[-1:])
        report, base = necessity(final, aux_size, max_nodes=120_000,
                                 library=library)
        if not base.solvable:
            continue
        gates = [r for r in report if r["kind"] == "rule"]
        actors = [r for r in report if r["kind"] == "actor"]
        if any(r["role"] != "enabling" for r in gates):
            continue
        if actors and all(r["role"] == "decorative" for r in actors):
            continue
        proposal.mechanics["roles"] = report
        proposal.mechanics["shortest"] = base.shortest
        return proposal
    return None
