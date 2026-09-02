import ctypes
import dataclasses

import numpy as np

from .clib import Library
from .reference import Levels

MAX_KINDS = 16
NONE, BLOCK, REMOVE, PUSH, BECOME, TOGGLE, WIN, LOSE, CYCLE = range(9)
A5_NONE, A5_CYCLE = range(2)
WIN_NONE_LEFT, WIN_ALL_ON, WIN_REACH, WIN_MATCH = range(4)
CONTROL_AVATAR, CONTROL_SELECT = range(2)
STENCIL_CELL, STENCIL_CROSS, STENCIL_BLOCK = range(3)
ON_ENTER, ON_CLICK, ON_STEP = range(3)
ALWAYS, IF_COUNT_LE, IF_NONE_LEFT, IF_ADJACENT = range(4)
MAX_RULES = 8


@dataclasses.dataclass
class Rule:
    trigger: int
    subject: int
    effect: int
    predicate: int = ALWAYS
    pred_a: int = -1
    pred_b: int = -1
    effect_a: int = -1
    effect_b: int = -1
    enabled: int = 1
EMPTY = -1


STATIC, CHASE, FLEE, PATROL = range(4)
HUD_NONE, HUD_BOTTOM, HUD_LEFT, HUD_TOP, HUD_RIGHT = range(5)


@dataclasses.dataclass
class Kind:
    color: int
    motion: int = STATIC
    motion_a: int = 0
    motion_b: int = -1
    deadly: int = 0
    gravity: int = 0
    size: int = 0
    off_x: int = 0
    off_y: int = 0
    on_enter: int = NONE
    enter_a: int = -1
    enter_b: int = -1
    on_click: int = NONE
    click_a: int = -1
    click_b: int = -1
    selectable: int = 0
    stencil: int = STENCIL_CELL

    def packed(self) -> list[int]:
        return [self.color, self.motion, self.motion_a, self.motion_b,
                self.deadly, self.gravity, self.size, self.off_x, self.off_y,
                self.on_enter, self.enter_a, self.enter_b, self.on_click,
                self.click_a, self.click_b, self.selectable, self.stencil]


@dataclasses.dataclass
class Spec:
    kinds: list[Kind]
    layouts: np.ndarray
    floors: np.ndarray
    player_kind: int
    win_mode: int
    win_a: int = -1
    win_b: int = -1
    pitch: int = 4
    origin_x: int = 0
    origin_y: int = 0
    background: int = 0
    rules: list = dataclasses.field(default_factory=list)
    budgets: np.ndarray | None = None
    hud: int = HUD_BOTTOM
    hud_on: int = 12
    hud_off: int = 11
    control: int = CONTROL_AVATAR
    uses_action5: int = 0
    action5: int = 1  # A5_CYCLE; 0 = declared but does nothing
    key_dir: tuple = (0, 1, 2, 3)  # direction of ACTION1..4
    palette_shuffle: int = 0
    seed: int = 0
    select_color: int = 15
    # WIN_MATCH rectangles: canvas (x0, y0) must equal target (x1, y1).
    match: tuple = (0, 0, 0, 0, 0, 0)

    @property
    def simple_actions(self) -> list[int]:
        return [1, 2, 3, 4, 5] if self.uses_action5 else [1, 2, 3, 4]

    @property
    def num_levels(self) -> int:
        return self.layouts.shape[0]

    @property
    def grid_h(self) -> int:
        return self.layouts.shape[1]

    @property
    def grid_w(self) -> int:
        return self.layouts.shape[2]


class DslNative:
    """The C-side objects for a Spec: the arc_dsl_spec, its level data and
    the buffers they point into. Shared by DslGame and the pool."""

    def __init__(self, spec: Spec, library: Library) -> None:
        self.library = library
        self.spec = spec
        self._keep: list = []
        h = library.headers

        kind_t = h.struct("arc_dsl_kind")
        spec_t = h.struct("arc_dsl_spec")
        self.aux_t = h.struct("arc_dsl_aux")

        layout = np.ascontiguousarray(spec.layouts, np.int8)
        floor = np.ascontiguousarray(spec.floors, np.int8)
        self._keep += [layout, floor]

        native = spec_t()
        native.num_kinds = len(spec.kinds)
        native.num_levels = spec.num_levels
        native.grid_w = spec.grid_w
        native.grid_h = spec.grid_h
        native.pitch = spec.pitch
        native.origin_x = spec.origin_x
        native.origin_y = spec.origin_y
        native.player_kind = spec.player_kind
        native.win_mode = spec.win_mode
        native.win_a = spec.win_a
        native.win_b = spec.win_b
        native.background = spec.background
        native.hud = spec.hud
        native.hud_on = spec.hud_on
        native.hud_off = spec.hud_off
        native.control = spec.control
        native.uses_action5 = spec.uses_action5
        native.action5 = spec.action5
        for i in range(4):
            native.key_dir[i] = int(spec.key_dir[i])
        native.palette_shuffle = spec.palette_shuffle
        native.seed = int(spec.seed) & 0xFFFFFFFF
        native.select_color = spec.select_color
        (native.match_x0, native.match_y0, native.match_x1, native.match_y1,
         native.match_w, native.match_h) = [int(v) for v in spec.match]
        rule_t = h.struct("arc_dsl_rule")
        native.num_rules = len(spec.rules)
        for i, r in enumerate(spec.rules):
            native.rules[i] = rule_t(r.trigger, r.subject, r.predicate,
                                     r.pred_a, r.pred_b, r.effect, r.effect_a,
                                     r.effect_b, r.enabled)
        for i, k in enumerate(spec.kinds):
            native.kinds[i] = kind_t(*k.packed())
        native.layout = layout.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
        native.floor = floor.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
        if spec.budgets is None:
            budget = np.zeros(spec.num_levels, np.int32)
        else:
            budget = np.ascontiguousarray(spec.budgets, np.int32)
            assert budget.shape == (spec.num_levels,)
        self._keep.append(budget)
        native.budget = budget.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        self._keep.append(native)
        self.native = native

        self.levels = self._blank_levels()
        self.level_data = self._level_data()
        self.simple = np.ascontiguousarray(spec.simple_actions, np.int32)
        self._keep.append(self.simple)
        self.hooks = ctypes.c_void_p.in_dll(library.lib, "arc_dsl_hooks")

    def _blank_levels(self) -> Levels:
        n, s = self.spec.num_levels, 1
        return Levels(
            game_id="dsl", tag_names=[],
            pixels=np.full((n, s, 1, 1), -1, np.int8),
            h=np.zeros((n, s), np.int32), w=np.zeros((n, s), np.int32),
            x=np.zeros((n, s), np.int32), y=np.zeros((n, s), np.int32),
            layer=np.zeros((n, s), np.int32),
            order=np.zeros((n, s), np.int32),
            interaction=np.full((n, s), 3, np.int32),
            blocking=np.zeros((n, s), np.int32),
            alive=np.zeros((n, s), bool),
            tags=np.zeros((n, s, 0), bool),
            grid_size=np.tile(np.array([64, 64], np.int32), (n, 1)),
            names=[[""] for _ in range(n)], level_data=[{} for _ in range(n)],
            background=self.spec.background, letter_box=0,
            win_score=n, available_actions=self.spec.simple_actions + [6],
        )

    def _level_data(self):
        from .clib import LEVEL_DTYPES, LEVEL_FIELDS, as_pointer
        from . import statics

        t = self.library.headers.struct("LevelData")
        declared = dict(t._fields_)
        values = {}
        for name in LEVEL_FIELDS:
            arr = np.ascontiguousarray(getattr(self.levels, name),
                                       LEVEL_DTYPES.get(name, np.int32))
            self._keep.append(arr)
            values[name] = as_pointer(arr, declared[name], name)
        values.update(num_levels=self.spec.num_levels, num_slots=1, num_tags=0,
                      ph=1, pw=1, win_score=self.spec.num_levels,
                      background=self.spec.background, letter_box=0)
        data, owned = statics.pack(t, values)
        self._keep += owned + [data]
        return data


class DslGame:
    def __init__(self, spec: Spec, library: Library | None = None,
                 max_frames: int = 8) -> None:
        self.library = library or Library()
        self.spec = spec
        self._native = DslNative(spec, self.library)
        self.levels = self._native.levels
        aux = self._native.aux_t()
        self._aux = aux
        simple = self._native.simple
        self.handle = self.library.sym.game_new(
            ctypes.byref(self._native.level_data),
            ctypes.addressof(self._native.hooks),
            ctypes.byref(aux), ctypes.byref(self._native.native),
            simple.ctypes.data_as(ctypes.c_void_p), len(simple), 1, max_frames,
        )
        self.max_frames = max_frames
        self.frames = np.zeros((max_frames, 64, 64), np.int8)
        self._ptr = self.frames.ctypes.data_as(ctypes.c_void_p)

    def init(self):
        self.library.sym.game_init(self.handle)

    def frame(self):
        buf = np.zeros((64, 64), np.int8)
        self.library.sym.game_frame(self.handle,
                                    buf.ctypes.data_as(ctypes.c_void_p))
        return buf

    def act(self, action_id, x=0, y=0):
        n = self.library.sym.game_perform_action_frames(
            self.handle, int(action_id), int(x), int(y), self._ptr,
            self.max_frames)
        return [self.frames[i].copy() for i in range(min(n, self.max_frames))]

    @property
    def score(self):
        return int(self.library.sym.harness_score(self.handle))

    @property
    def state(self):
        return self.library.status_names()[
            int(self.library.sym.harness_status(self.handle))]

    def close(self):
        if self.handle is not None:
            self.library.sym.game_free(self.handle)
            self.handle = None
