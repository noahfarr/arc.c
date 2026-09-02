import json
from pathlib import Path

import numpy as np

from .dsl import Kind, Rule, Spec
from .generate import sample_environment
from .validate import certify


def save(spec: Spec, labels: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, layouts=spec.layouts, floors=spec.floors,
        kinds=np.array([[k.color, k.motion, k.motion_a, k.motion_b, k.deadly,
                         k.gravity, k.size, k.off_x, k.off_y, k.on_enter,
                         k.enter_a, k.enter_b, k.on_click, k.click_a,
                         k.click_b]
                        for k in spec.kinds], np.int32),
        params=np.array([spec.player_kind, spec.win_mode, spec.win_a,
                         spec.win_b, spec.pitch, spec.origin_x, spec.origin_y,
                         spec.background, spec.hud, spec.hud_on, spec.hud_off],
                        np.int32),
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
    return spec, labels


FAMILIES = ("sokoban", "rooms")

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
          families=FAMILIES, library=None) -> list:
    """Generate `count` environments across `families`, keep those whose
    non-tutorial levels a random policy wins at most 1 in 10,000 times
    (the foundation's bar), and save them with their labels."""
    import ctypes

    from .clib import Library
    from .generate import sample_rooms

    library = library or Library()
    aux_size = ctypes.sizeof(library.headers.struct("arc_dsl_aux"))
    samplers = {"sokoban": sample_environment, "rooms": sample_rooms}
    rng = np.random.default_rng(seed)
    out = Path(out)
    manifest = []
    made = 0
    attempts = 0
    while made < count and attempts < count * 6:
        attempts += 1
        family = families[attempts % len(families)]
        proposal = samplers[family](rng, library=library, aux_size=aux_size)
        if proposal is None:
            continue
        spec = proposal.spec
        rates = []
        for level in range(spec.num_levels):
            rate, _ = certify(spec, trials=trials, horizon=horizon,
                              threads=threads, seed=made * 97 + level + 1,
                              start_level=level, library=library)
            rates.append(rate)
        if any(r > 1e-4 for r in rates[1:]):
            continue
        levels = proposal.mechanics["levels"]
        shortest = [m.get("shortest") for m in levels]
        budgets = [int(b) for b in spec.budgets]
        labels = {"family": family,
                  "shortest": shortest,
                  "budgets": budgets,
                  "baselines": baselines_for(shortest, budgets),
                  "random_rate": rates,
                  "grid": [spec.grid_w, spec.grid_h],
                  "levels": spec.num_levels,
                  "mechanics": proposal.mechanics.get("kinds", [])}
        name = f"env_{made:04d}"
        save(spec, labels, out / f"{name}.npz")
        manifest.append({"name": name, **labels})
        made += 1
        if verbose:
            print(f"  {name} {family}: shortest={labels['shortest']} "
                  f"budgets={labels['budgets']} rates="
                  f"{['%.5f' % r for r in rates]}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest
