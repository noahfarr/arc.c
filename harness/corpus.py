import json
from pathlib import Path

import numpy as np

from .dsl import Kind, Rule, Spec
from .generate import sample_environment
from .validate import certify


def save(spec: Spec, labels: dict, path: Path, distances=None) -> None:
    """distances: per level, None or (hashes, dists) from
    validate.distance_table; stored flat with per-level offsets."""
    path.parent.mkdir(parents=True, exist_ok=True)
    hashes, dists, offsets = [], [], [0]
    for entry in (distances or [None] * spec.num_levels):
        if entry is not None:
            hashes.append(np.asarray(entry[0], np.uint64))
            dists.append(np.asarray(entry[1], np.int32))
        offsets.append(offsets[-1] + (0 if entry is None else len(entry[0])))
    np.savez_compressed(
        path, layouts=spec.layouts, floors=spec.floors,
        dist_hash=(np.concatenate(hashes) if hashes else np.zeros(0, np.uint64)),
        dist_val=(np.concatenate(dists) if dists else np.zeros(0, np.int32)),
        dist_offset=np.array(offsets, np.int32),
        kinds=np.array([k.packed() for k in spec.kinds], np.int32),
        params=np.array([spec.player_kind, spec.win_mode, spec.win_a,
                         spec.win_b, spec.pitch, spec.origin_x, spec.origin_y,
                         spec.background, spec.hud, spec.hud_on, spec.hud_off,
                         spec.control, spec.uses_action5, spec.select_color,
                         *spec.match], np.int32),
        rules=np.array([[r.trigger, r.subject, r.effect, r.predicate,
                         r.pred_a, r.pred_b, r.effect_a, r.effect_b,
                         r.enabled] for r in spec.rules], np.int32
                       ).reshape(-1, 9),
        budgets=(np.zeros(spec.num_levels, np.int32) if spec.budgets is None
                 else np.asarray(spec.budgets, np.int32)),
        labels=np.frombuffer(json.dumps(labels).encode("utf-8"), np.uint8))


def load(path: Path) -> tuple[Spec, dict]:
    with np.load(path) as z:
        k = z["kinds"]
        p = z["params"]
        extra = {}
        if len(p) > 8:
            extra = dict(hud=int(p[8]), hud_on=int(p[9]), hud_off=int(p[10]))
        if len(p) > 11:
            extra.update(control=int(p[11]), uses_action5=int(p[12]),
                         select_color=int(p[13]),
                         match=tuple(int(v) for v in p[14:20]))
        rules = [Rule(*[int(v) for v in row]) for row in z["rules"]] \
            if "rules" in z else []
        budgets = z["budgets"] if "budgets" in z else None
        spec = Spec(
            kinds=[Kind(*[int(v) for v in row]) for row in k],
            layouts=z["layouts"], floors=z["floors"], player_kind=int(p[0]),
            win_mode=int(p[1]), win_a=int(p[2]), win_b=int(p[3]),
            pitch=int(p[4]), origin_x=int(p[5]), origin_y=int(p[6]),
            background=int(p[7]), rules=rules, budgets=budgets, **extra)
        labels = json.loads(bytes(z["labels"]).decode("utf-8"))
        if "dist_offset" in z:
            labels["_distances"] = (z["dist_hash"], z["dist_val"],
                                    z["dist_offset"])
    return spec, labels


FAMILIES = ("sokoban", "rooms", "select", "match")


def _rekey(spec, level, library, aux_size, max_nodes):
    """Distance table for `level` computed on the full spec, so hashes
    carry the real level index."""
    from .dsl import DslGame
    from .validate import distance_table

    g = DslGame(spec, library=library)
    orig = g.init

    def init():
        orig()
        library.sym.game_set_level(g.handle, level)
    g.init = init
    try:
        return distance_table(g, aux_size, max_nodes=max_nodes)
    finally:
        g.close()

# Human first-run actions over the BFS optimum on the public games we could
# measure (tu93 levels 1-4: 1.06, 1.6, 1.8, 2.5; re86 level 1: 1.3): about
# the optimum on the tutorial level, 2-2.5x by level 4. Used to turn a
# generated level's optimum into a surrogate human baseline.
HUMAN_OVER_OPTIMAL = (1.1, 1.5, 1.8, 2.2, 2.5, 2.5, 2.5, 2.5)


def baselines_for(shortest, budgets) -> list[int]:
    out = []
    for l, (opt, budget) in enumerate(zip(shortest, budgets)):
        k = HUMAN_OVER_OPTIMAL[min(l, len(HUMAN_OVER_OPTIMAL) - 1)]
        if opt is None:
            # Unverified level: the budget was drawn at 2.5-4.5x a pull-count
            # estimate of the optimum; take the middle of that.
            opt = budget / 3.5
        out.append(max(1, int(round(k * opt))))
    return out


def build(count: int, out: Path, seed: int = 0, trials: int = 10_000,
          horizon: int = 800, threads: int = 8, verbose: bool = True,
          families=FAMILIES, library=None, distance_nodes: int = 60_000,
          stages=(2,), prefix: str = "env_", manifest: bool = True) -> list:
    """Generate `count` environments across `families` and curriculum
    `stages` (see generate.STAGE_BANDS), keep those whose non-tutorial
    levels a random policy wins no more often than the stage's bar (the
    foundation's 1 in 10,000 at stage 2), and save them with labels."""
    import ctypes

    import dataclasses

    from .clib import Library
    from .dsl import DslGame
    from .generate import (STAGE_RANDOM_BAR, sample_match, sample_rooms,
                           sample_select)
    from .validate import distance_table

    library = library or Library()
    aux_size = ctypes.sizeof(library.headers.struct("arc_dsl_aux"))
    samplers = {"sokoban": sample_environment, "rooms": sample_rooms,
                "select": sample_select, "match": sample_match}
    rng = np.random.default_rng(seed)
    out = Path(out)
    manifest = []
    made = 0
    attempts = 0
    while made < count and attempts < count * 6:
        attempts += 1
        family = families[attempts % len(families)]
        stage = stages[(attempts // len(families)) % len(stages)]
        # The match family's search branches over every canvas cell; keep
        # it short and let scramble counts stand in for the optimum.
        nodes = 5_000 if family == "match" else 40_000
        proposal = samplers[family](rng, library=library, aux_size=aux_size,
                                    stage=stage, max_nodes=nodes)
        if proposal is None:
            continue
        spec = proposal.spec
        rates = []
        for level in range(spec.num_levels):
            rate, _ = certify(spec, trials=trials, horizon=horizon,
                              threads=threads, seed=made * 97 + level + 1,
                              start_level=level, library=library)
            rates.append(rate)
        if any(r > STAGE_RANDOM_BAR[stage] for r in rates[1:]):
            continue
        levels = proposal.mechanics["levels"]
        shortest = [m.get("shortest") for m in levels]
        budgets = [int(b) for b in spec.budgets]
        labels = {"family": family,
                  "stage": stage,
                  "shortest": shortest,
                  "budgets": budgets,
                  "baselines": baselines_for(shortest, budgets),
                  "random_rate": rates,
                  "grid": [spec.grid_w, spec.grid_h],
                  "levels": spec.num_levels,
                  "mechanics": proposal.mechanics.get("kinds", [])}
        distances = []
        for level in range(spec.num_levels):
            one = dataclasses.replace(spec, layouts=spec.layouts[level:level + 1],
                                      floors=spec.floors[level:level + 1],
                                      budgets=None)
            g = DslGame(one, library=library)
            table = distance_table(g, aux_size, max_nodes=distance_nodes)
            g.close()
            if table is not None:
                # The table was built for a one-level spec (level index 0);
                # the hash includes the level index, so rebuild the keys
                # for the level's real index.
                table = _rekey(spec, level, library, aux_size, distance_nodes)
            distances.append(table)
        labels["distances"] = [None if t is None else int(len(t[0]))
                               for t in distances]
        name = f"{prefix}{made:04d}"
        save(spec, labels, out / f"{name}.npz", distances=distances)
        manifest.append({"name": name, **labels})
        made += 1
        if verbose:
            print(f"  {name} {family} stage {stage}: shortest={labels['shortest']} "
                  f"budgets={labels['budgets']} rates="
                  f"{['%.5f' % r for r in rates]}")
    if manifest:
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def _worker(args):
    count, out, seed, trials, horizon, families, stages, distance_nodes, prefix = args
    from .clib import Library

    return build(count, Path(out), seed=seed, trials=trials, horizon=horizon,
                 threads=1, verbose=False, families=families,
                 library=Library(), distance_nodes=distance_nodes,
                 stages=stages, prefix=prefix, manifest=False)


def build_parallel(count: int, out: Path, workers: int = 16, seed: int = 0,
                   trials: int = 4_000, horizon: int = 800, families=FAMILIES,
                   stages=(0, 1, 2), distance_nodes: int = 40_000) -> list:
    """build() across `workers` processes, each with its own library and
    seed, writing w<NN>_<i>.npz into `out`; one manifest at the end."""
    import multiprocessing as mp

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    per = [count // workers + (1 if i < count % workers else 0)
           for i in range(workers)]
    jobs = [(n, str(out), seed * 1000 + i, trials, horizon, families, stages,
             distance_nodes, f"w{i:02d}_") for i, n in enumerate(per) if n]
    ctx = mp.get_context("fork")
    with ctx.Pool(len(jobs)) as pool:
        parts = pool.map(_worker, jobs)
    manifest = [entry for part in parts for entry in part]
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest
