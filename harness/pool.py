import ctypes

import numpy as np

from . import aux as auxdecl
from . import differ
from .clib import Library


class Spec(ctypes.Structure):
    _fields_ = [
        ("levels", ctypes.c_void_p),
        ("hooks", ctypes.c_void_p),
        ("aux_array", ctypes.c_void_p),
        ("aux_stride", ctypes.c_size_t),
        ("statics", ctypes.c_void_p),
        ("simple_actions", ctypes.c_void_p),
        ("num_simple", ctypes.c_int32),
        ("has_click", ctypes.c_int32),
        ("max_frames", ctypes.c_int32),
        ("baseline", ctypes.c_void_p),
        ("state_hash", ctypes.c_void_p),
        ("dist_hash", ctypes.c_void_p),
        ("dist_val", ctypes.c_void_p),
        ("dist_offset", ctypes.c_void_p),
    ]

REWARD_LEVELS, REWARD_RHAE = 0, 1


def signatures(lib):
    lib.arc_vecenv_new_pool.restype = ctypes.c_void_p
    lib.arc_vecenv_new_pool.argtypes = [
        ctypes.POINTER(Spec), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_uint64,
    ]
    lib.arc_vecenv_free.argtypes = [ctypes.c_void_p]
    lib.arc_vecenv_reset.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.arc_vecenv_step.argtypes = [ctypes.c_void_p] * 8
    lib.arc_vecenv_num_actions.restype = ctypes.c_int32
    lib.arc_vecenv_num_actions.argtypes = [ctypes.c_void_p]
    lib.arc_vecenv_tasks.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.arc_vecenv_action_counts.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.arc_vecenv_set_reward.argtypes = [ctypes.c_void_p, ctypes.c_int32,
                                          ctypes.c_float]
    lib.arc_vecenv_set_shaping.argtypes = [ctypes.c_void_p, ctypes.c_float]
    lib.arc_vecenv_replace_game.argtypes = [ctypes.c_void_p, ctypes.c_int32,
                                            ctypes.POINTER(Spec)]
    lib.arc_vecenv_num_games.restype = ctypes.c_int32
    lib.arc_vecenv_num_games.argtypes = [ctypes.c_void_p]
    lib.arc_vecenv_restarts.restype = ctypes.c_int64
    lib.arc_vecenv_restarts.argtypes = [ctypes.c_void_p]
    return lib


def human_baselines() -> dict:
    """Per-level human baseline actions for the public games, from the
    games.json the reference ships."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "reference" / "games.json"
    out = {}
    for entry in json.loads(path.read_text()):
        out[entry["game_id"][:4]] = list(entry["baseline_actions"])
    return out


class Pool:
    """A vector environment drawing each slot's game from a pool. Entries
    are public game ids ("ls20") or paths to generated environments
    ("corpus/env_0001.npz"). With reward=REWARD_RHAE the reward is the
    benchmark's per-level score against each game's baseline: the human
    baseline for public games, the "baselines" label for generated ones."""

    def __init__(self, games, num_envs: int = 64, num_threads: int = 16,
                 seed: int = 0, library: Library | None = None,
                 reward: int = REWARD_LEVELS, cap: float = 0.0,
                 max_frames: int = 8, shaping: float = 0.0):
        if isinstance(games, str):
            games = [games]
        games = list(games)
        if not games:
            raise ValueError("a pool needs at least one game")
        self.games = games
        self.num_envs = num_envs
        self.library = library or Library()
        self.lib = signatures(self.library.lib)
        self._keep = []
        self._retired: list = []
        self._live: dict = {}
        self._swaps = 0
        self._max_frames = max_frames
        self.num_levels: list[int] = []
        humans = human_baselines() if reward == REWARD_RHAE else {}

        specs = (Spec * len(games))()
        for index, game in enumerate(games):
            if str(game).endswith(".npz"):
                specs[index] = self._generated(game, num_envs, max_frames)
                continue
            _, proto = differ.build(game, self.library)
            self._keep.append(proto)
            self.num_levels.append(int(proto.levels.num_levels))
            kind = type(proto._aux)
            auxes = (kind * num_envs)()
            self._keep.append(auxes)
            allocate = auxdecl.ALLOC.get(game)
            for slot in range(num_envs):
                self._keep += auxdecl.allocate(game, auxes[slot],
                                               dict(proto._dims))
                if allocate:
                    call = getattr(self.lib, f"{game}_aux_alloc")
                    call.argtypes = ([ctypes.c_void_p] +
                                     [ctypes.c_int32] * len(allocate))
                    call(ctypes.byref(auxes[slot]),
                         *[proto._dims[name] for name in allocate])
            baseline = None
            if game in humans:
                arr = np.ascontiguousarray(humans[game], np.int32)
                assert len(arr) == proto.levels.num_levels, game
                self._keep.append(arr)
                baseline = arr.ctypes.data
            specs[index] = Spec(
                levels=ctypes.addressof(proto._level_data_struct),
                hooks=ctypes.addressof(proto._hooks),
                aux_array=ctypes.addressof(auxes),
                aux_stride=ctypes.sizeof(kind),
                statics=ctypes.addressof(proto._static),
                simple_actions=proto._simple.ctypes.data,
                num_simple=len(proto._simple),
                has_click=int(proto.levels.has_click),
                max_frames=proto.max_frames,
                baseline=baseline,
            )
        self._keep.append(specs)
        self.handle = self.lib.arc_vecenv_new_pool(
            specs, len(games), num_envs, num_threads, seed)
        self.lib.arc_vecenv_set_reward(ctypes.c_void_p(self.handle), reward,
                                       float(cap))
        if shaping:
            self.lib.arc_vecenv_set_shaping(ctypes.c_void_p(self.handle),
                                            float(shaping))
        self.num_actions = int(self.lib.arc_vecenv_num_actions(self.handle))

    def set_trial_budget(self, multiple: float) -> None:
        """Trials end when the game is won or after multiple x the summed
        per-level baseline actions (0 turns it off)."""
        self.library.sym.vecenv_set_trial_budget(self.handle, float(multiple))

    def restarts(self) -> int:
        return int(self.lib.arc_vecenv_restarts(ctypes.c_void_p(self.handle)))

    def replace(self, k: int, path) -> None:
        """Swap generated game `path` into pool slot k. The buffers of the
        game it replaces stay alive until every environment has restarted
        since (see gc), because an environment mid-game keeps playing the
        old game until then (arc_vecenv_replace_game)."""
        assert str(path).endswith(".npz"), "only generated games can be swapped in"
        keep_before = len(self._keep)
        spec = self._generated(path, self.num_envs, self._max_frames,
                               level_index=k)
        buffers = self._keep[keep_before:]
        self._keep = self._keep[:keep_before]
        if k in self._live:
            self._retired.append((self.restarts(), self._live[k]))
        self._live[k] = buffers
        self.lib.arc_vecenv_replace_game(ctypes.c_void_p(self.handle), int(k),
                                         ctypes.byref(spec))
        self.games[k] = str(path)
        self._swaps += 1
        self.gc()

    def gc(self, margin: float = 2.0) -> int:
        """Free retired buffers once the environment has restarted at least
        margin * num_envs times since they were retired. In trial mode all
        environments restart at each boundary, so two boundaries suffice;
        with terminations only, budgets bound every game's length, and a
        larger margin covers it."""
        now = self.restarts()
        keep, freed = [], 0
        for tag, buffers in self._retired:
            if now - tag >= margin * self.num_envs:
                freed += 1
            else:
                keep.append((tag, buffers))
        self._retired = keep
        return freed

    def _generated(self, path, num_envs, max_frames, level_index=None):
        from .corpus import load
        from .dsl import DslNative

        spec, labels = load(path)
        if level_index is None:
            self.num_levels.append(int(spec.num_levels))
        else:
            self.num_levels[level_index] = int(spec.num_levels)
        native = DslNative(spec, self.library)
        self._keep.append(native)
        auxes = (native.aux_t * num_envs)()
        self._keep.append(auxes)
        baseline = None
        if labels.get("baselines"):
            arr = np.ascontiguousarray(labels["baselines"], np.int32)
            self._keep.append(arr)
            baseline = arr.ctypes.data
        state_hash = dist_hash = dist_val = dist_offset = None
        if labels.get("_distances") is not None:
            dh, dv, do = (np.ascontiguousarray(a) for a in labels["_distances"])
            self._keep += [dh, dv, do]
            fn = getattr(self.library.lib, "arc_dsl_state_hash")
            state_hash = ctypes.cast(fn, ctypes.c_void_p).value
            dist_hash, dist_val, dist_offset = (dh.ctypes.data, dv.ctypes.data,
                                                do.ctypes.data)
        spec = Spec(
            levels=ctypes.addressof(native.level_data),
            hooks=ctypes.addressof(native.hooks),
            aux_array=ctypes.addressof(auxes),
            aux_stride=ctypes.sizeof(native.aux_t),
            statics=ctypes.addressof(native.native),
            simple_actions=native.simple.ctypes.data,
            num_simple=len(native.simple),
            has_click=1,
            max_frames=max_frames,
            baseline=baseline,
            state_hash=state_hash,
            dist_hash=dist_hash,
            dist_val=dist_val,
            dist_offset=dist_offset,
        )
        self._keep.append(spec)
        return spec

    def tasks(self, out):
        self.lib.arc_vecenv_tasks(self.handle, out.ctypes.data)

    def action_counts(self, out):
        self.lib.arc_vecenv_action_counts(self.handle, out.ctypes.data)

    def close(self):
        if self.handle is not None:
            self.lib.arc_vecenv_free(ctypes.c_void_p(self.handle))
            self.handle = None


def make(games, **kwargs) -> Pool:
    return Pool(games, **kwargs)
