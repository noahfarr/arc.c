import ctypes
import os
from pathlib import Path

import numpy as np

from . import aux as auxdecl
from . import statics
from .cdecl import CHeaders
from .reference import Levels

LEVEL_FIELDS = (
    "pixels", "h", "w", "x", "y", "layer", "order",
    "interaction", "blocking", "alive", "tags", "grid_size",
)
LEVEL_DTYPES = {
    "pixels": np.int8, "interaction": np.int8, "blocking": np.int8,
    "alive": np.uint8, "tags": np.uint8,
}


def as_pointer(array: np.ndarray, pointer_type, label: str):
    element = pointer_type._type_
    wanted = np.dtype(element)
    if array.dtype != wanted:
        raise TypeError(
            f"{label}: the C header declares {element.__name__} "
            f"but the harness built a {array.dtype} array"
        )
    return array.ctypes.data_as(pointer_type)


def default_root() -> Path:
    override = os.environ.get("ARC_CSRC")
    if override:
        return Path(override)
    repo = Path(__file__).resolve().parent.parent
    for candidate in (repo / "csrc", repo / "include" / "arc", repo):
        if (candidate / "engine.h").exists():
            return candidate
    return repo / "csrc"


def default_library(root: Path) -> Path:
    override = os.environ.get("ARC_LIBRARY")
    if override:
        return Path(override)
    bases = [root, root / "build"]
    for parent in list(root.parents)[:3]:
        bases += [parent, parent / "build"]
    for name in ("libarc_harness", "libarc"):
        for suffix in (".dylib", ".so"):
            for base in bases:
                candidate = base / f"{name}{suffix}"
                if candidate.exists():
                    return candidate
    raise FileNotFoundError(f"no arc library found under {root}; set ARC_LIBRARY")


class Symbols:
    def __init__(self, lib: ctypes.CDLL) -> None:
        self.lib = lib

    def __getattr__(self, name: str):
        for candidate in (f"arc_{name}", name):
            try:
                return getattr(self.lib, candidate)
            except AttributeError:
                continue
        raise AttributeError(f"neither arc_{name} nor {name} is exported")


class Library:
    def __init__(self, root: Path | None = None, path: Path | None = None) -> None:
        self.root = Path(root) if root else default_root()
        self.path = Path(path) if path else default_library(self.root)
        games = self.root / "games"
        if not games.exists():
            games = self.root.parent.parent / "src" / "games"
        self.games_dir = games
        self.include_root = self.root.parent if self.root.name == "arc" else self.root
        self.arc_prefix = "arc/" if self.root.name == "arc" else ""
        self.headers = CHeaders().load(
            self.root / "engine.h", self.root / "game.h", *sorted(games.glob("*.h"))
        )
        for optional in ("scene.h", "scene_game.h", "dsl.h"):
            if (self.root / optional).exists():
                self.headers.load(self.root / optional)
        self.lib = ctypes.CDLL(str(self.path))
        self.sym = Symbols(self.lib)
        self._declare()

    def _declare(self) -> None:
        s, v, i = self.sym, ctypes.c_void_p, ctypes.c_int32
        s.game_new.restype = v
        s.game_new.argtypes = [v] * 5 + [i, i, i]
        s.game_init.argtypes = [v]
        s.game_free.argtypes = [v]
        s.game_set_level.argtypes = [v, i]
        s.game_perform_action_frames.restype = i
        s.game_perform_action_frames.argtypes = [v, i, i, i, v, i]
        s.game_frame.argtypes = [v, v]
        s.game_state_size.restype = ctypes.c_size_t
        s.game_state_size.argtypes = [v, ctypes.c_size_t]
        s.game_save.argtypes = [v, ctypes.c_size_t, v]
        s.game_load.argtypes = [v, ctypes.c_size_t, v]
        s.certify_random.restype = ctypes.c_int64
        s.certify_random.argtypes = [v, i, i, i, i, v, i, ctypes.c_uint32, v]
        s.game_hash.restype = ctypes.c_uint64
        s.game_hash.argtypes = [v, ctypes.c_size_t]
        s.dsl_state_hash.restype = ctypes.c_uint64
        s.dsl_state_hash.argtypes = [v]
        s.dsl_solve.restype = i
        s.dsl_solve.argtypes = [v, i, ctypes.POINTER(i), ctypes.POINTER(i)]
        s.vecenv_set_trial_budget.argtypes = [v, ctypes.c_float]
        s.frame_tokens.restype = i
        s.frame_tokens.argtypes = [v, v, i, v]
        s.dsl_solve_path.restype = i
        s.dsl_solve_path.argtypes = [v, i, v, i, ctypes.POINTER(i),
                                     ctypes.POINTER(i)]
        for name in ("harness_score", "harness_status", "harness_level_index"):
            fn = getattr(s, name)
            fn.restype, fn.argtypes = i, [v]

    def status_names(self) -> dict[int, str]:
        return {
            self.headers.constants[n]: n
            for n in ("NOT_PLAYED", "NOT_FINISHED", "WIN", "GAME_OVER")
        }


class Game:
    def __init__(self, library: Library, game_id: str, levels: Levels, config,
                 max_frames: int) -> None:
        self.library = library
        self.game_id = game_id
        self.levels = levels
        self.max_frames = max_frames
        self._keep: list = []
        self._aux_free = None

        headers = library.headers
        level_data = self._level_data(headers.struct("LevelData"), levels)
        static_type = headers.struct(f"{game_id.capitalize()}Static")
        static, owned = statics.pack(static_type, config.static_fields)
        self._keep += owned + [static, level_data]

        aux_type = headers.struct(f"{game_id.capitalize()}Aux")
        aux = aux_type()
        ph, pw = levels.patch_shape
        dims = {"num_slots": levels.num_slots, "ph": ph, "pw": pw}
        self._keep += auxdecl.allocate(game_id, aux, dims)
        self._keep.append(aux)

        args = auxdecl.ALLOC.get(game_id)
        if args:
            alloc = getattr(library.lib, f"{game_id}_aux_alloc")
            alloc.argtypes = [ctypes.c_void_p] + [ctypes.c_int32] * len(args)
            alloc(ctypes.byref(aux), *[dims[a] for a in args])
            free = getattr(library.lib, f"{game_id}_aux_free", None)
            if free is not None:
                free.argtypes = [ctypes.c_void_p]
                self._aux_free = lambda: free(ctypes.byref(aux))

        simple = np.ascontiguousarray(levels.simple_actions, np.int32)
        self._keep.append(simple)
        hooks = ctypes.c_void_p.in_dll(library.lib, f"{game_id}_hooks")
        self._aux, self._dims, self._static = aux, dims, static
        self._level_data_struct, self._hooks, self._simple = level_data, hooks, simple
        self.handle = library.sym.game_new(
            ctypes.byref(level_data), ctypes.addressof(hooks), ctypes.byref(aux),
            ctypes.byref(static), simple.ctypes.data_as(ctypes.c_void_p),
            len(simple), int(levels.has_click), max_frames,
        )
        self.frames = np.zeros((max_frames, 64, 64), np.int8)
        self._frames_ptr = self.frames.ctypes.data_as(ctypes.c_void_p)

    def _level_data(self, struct_type, levels: Levels):
        declared = dict(struct_type._fields_)
        values = {}
        for name in LEVEL_FIELDS:
            array = np.ascontiguousarray(getattr(levels, name), LEVEL_DTYPES.get(name, np.int32))
            self._keep.append(array)
            values[name] = as_pointer(array, declared[name], name)
        ph, pw = levels.patch_shape
        values.update(
            num_levels=levels.num_levels, num_slots=levels.num_slots,
            num_tags=len(levels.tag_names), ph=ph, pw=pw,
            win_score=levels.win_score, background=levels.background,
            letter_box=levels.letter_box,
        )
        level_data, owned = statics.pack(struct_type, values)
        self._keep += owned
        return level_data

    def init(self) -> None:
        self.library.sym.game_init(self.handle)

    def set_level(self, index: int) -> None:
        self.library.sym.game_set_level(self.handle, int(index))

    def act(self, action_id: int, x: int, y: int) -> list[np.ndarray]:
        count = self.library.sym.game_perform_action_frames(
            self.handle, int(action_id), int(x), int(y), self._frames_ptr, self.max_frames
        )
        return [self.frames[i].copy() for i in range(min(count, self.max_frames))]

    @property
    def score(self) -> int:
        return int(self.library.sym.harness_score(self.handle))

    @property
    def state(self) -> str:
        return self.library.status_names()[int(self.library.sym.harness_status(self.handle))]

    @property
    def level_index(self) -> int:
        return int(self.library.sym.harness_level_index(self.handle))

    def clone(self):
        aux_type = type(self._aux)
        aux = aux_type()
        self._keep.append(aux)
        dims = dict(self._dims)
        self._keep += auxdecl.allocate(self.game_id, aux, dims)
        args = auxdecl.ALLOC.get(self.game_id)
        if args:
            alloc = getattr(self.library.lib, f"{self.game_id}_aux_alloc")
            alloc.argtypes = [ctypes.c_void_p] + [ctypes.c_int32] * len(args)
            alloc(ctypes.byref(aux), *[dims[a] for a in args])
        return self.library.sym.game_new(
            ctypes.byref(self._level_data_struct), ctypes.addressof(self._hooks),
            ctypes.byref(aux), ctypes.byref(self._static),
            self._simple.ctypes.data_as(ctypes.c_void_p), len(self._simple),
            int(self.levels.has_click), self.max_frames,
        )

    def close(self) -> None:
        if self.handle is not None:
            self.library.sym.game_free(self.handle)
            self.handle = None
        if self._aux_free is not None:
            self._aux_free()
            self._aux_free = None
