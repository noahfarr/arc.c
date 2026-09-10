"""Frames as images, and generated levels as GIFs.

`record` replays a solution through a DSL game and writes the frames as
an animated GIF. With `chrome`, the 64x64 frame is matted into a panel
styled after the ARC-AGI-3 player - action count against the level's
budget, score, and where the level sits in its ladder. The chrome is
drawn *around* the frame and never into it: what an agent observes is
the 64x64 grid alone.
"""
import numpy as np

PALETTE = np.array([
    (0, 0, 0), (0, 116, 217), (255, 65, 54), (46, 204, 64), (255, 220, 0),
    (170, 170, 170), (240, 18, 190), (255, 133, 27), (127, 219, 255),
    (135, 12, 37), (60, 60, 60), (100, 200, 160), (200, 120, 255),
    (90, 90, 200), (30, 160, 90), (250, 250, 250),
], np.uint8)

# Panel geometry and colours.
PAD_X, HEADER_H, FOOTER_H = 18, 42, 62
INK = (232, 232, 236)
DIM = (122, 122, 134)
RULE = (38, 38, 44)
TRACK = (44, 44, 52)
GROUND = (13, 13, 15)
ACCENT = (255, 133, 27)

FONTS = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
         "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
         "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
         "/System/Library/Fonts/Supplemental/Arial Bold.ttf")


def to_image(frame, scale: int = 8):
    from PIL import Image

    idx = np.clip(np.asarray(frame, np.int16), 0, 15)
    rgb = PALETTE[idx]
    return Image.fromarray(np.kron(rgb, np.ones((scale, scale, 1), np.uint8)))


def _font(size: int):
    from PIL import ImageFont

    for path in FONTS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _panel(image, chrome: dict, taken: int, note, won: bool):
    """The frame matted into the player panel. `note` is the level's own
    objective as (label, done, total), or None."""
    from PIL import Image, ImageDraw

    fw, fh = image.size
    w, h = fw + 2 * PAD_X, HEADER_H + fh + FOOTER_H
    panel = Image.new("RGB", (w, h), GROUND)
    panel.paste(image, (PAD_X, HEADER_H))
    d = ImageDraw.Draw(panel)
    big, small = _font(14), _font(12)

    # Header: the mark, the wordmark, and what is being played.
    d.rectangle([PAD_X, 16, PAD_X + 9, 25], fill=ACCENT)
    d.text((PAD_X + 15, 15), "ARC PRIZE", font=big, fill=INK)
    title = chrome.get("title", "")
    d.text((w - PAD_X - d.textlength(title, font=small), 17), title,
           font=small, fill=DIM)
    d.line([0, HEADER_H - 1, w, HEADER_H - 1], fill=RULE)

    foot = HEADER_H + fh
    d.line([0, foot, w, foot], fill=RULE)
    budget = int(chrome.get("budget") or 0)
    levels = int(chrome.get("levels") or 0)
    level = int(chrome.get("level") or 0)

    # Row one: actions against the budget, and the level's place in the
    # ladder as one dot per level.
    d.text((PAD_X, foot + 12), "ACTIONS", font=small, fill=DIM)
    count = f"{taken}/{budget}" if budget else str(taken)
    d.text((PAD_X + 62, foot + 12), count, font=small, fill=INK)
    if levels:
        d.text((w - PAD_X - 11 * levels - 44, foot + 12), "LEVEL",
               font=small, fill=DIM)
        for i in range(levels):
            cx = w - PAD_X - 11 * (levels - i) + 4
            cy = foot + 18
            on = i <= level
            d.ellipse([cx - 3, cy - 3, cx + 3, cy + 3],
                      fill=ACCENT if i == level else (INK if on else TRACK))

    # Row two: the level's own objective - what "progress" means differs
    # by game type, so the label and the count come from the win mode.
    if won:
        d.text((PAD_X, foot + 34), "SOLVED", font=small, fill=ACCENT)
    elif note is not None:
        label, done, total = note
        d.text((PAD_X, foot + 34), label, font=small, fill=DIM)
        d.text((PAD_X + d.textlength(label, font=small) + 7, foot + 34),
               f"{done}/{total}", font=small, fill=INK)
    bw, bx, by = 132, w - PAD_X - 132, foot + 39
    d.rectangle([bx, by, bx + bw, by + 6], fill=TRACK)
    if budget:
        fill = int(round(bw * min(1.0, taken / budget)))
        if fill:
            d.rectangle([bx, by, bx + fill, by + 6], fill=INK)
    return panel


def _quantize(frames, colors: int = 96):
    """One palette for the whole animation, so text does not shimmer as
    each frame is quantised on its own."""
    from PIL import Image

    picks = frames[:: max(1, len(frames) // 12)] or frames[:1]
    w, h = frames[0].size
    strip = Image.new("RGB", (w, h * len(picks)))
    for i, f in enumerate(picks):
        strip.paste(f, (0, i * h))
    base = strip.quantize(colors=colors, method=Image.MEDIANCUT)
    return [f.quantize(palette=base, dither=Image.Dither.NONE)
            for f in frames]


def _gate_kinds(spec):
    """Kinds that block and that some rule acts on: the doors a level
    opens, as opposed to its static walls."""
    touched = set()
    for r in spec.rules:
        touched.update(v for v in (r.effect_a, r.effect_b) if v >= 0)
    return {k for k in touched
            if 0 <= k < len(spec.kinds) and spec.kinds[k].on_enter == 1}


def _progress(spec, grid, floor, start):
    """(label, done, total) for the level's own objective, mirroring the
    win check in src/dsl.c so the readout cannot drift from the engine.
    Returns None when the mode has nothing to count."""
    from .dsl import (CONTROL_SELECT, WIN_ALL_ON, WIN_MATCH, WIN_NONE_LEFT,
                      WIN_REACH)

    if spec.win_mode == WIN_ALL_ON:
        goals = floor == spec.win_b
        total = int(goals.sum())
        done = int((goals & (grid == spec.win_a)).sum())
        label = "PLACED" if spec.control == CONTROL_SELECT else "ON TARGET"
        return label, done, total
    if spec.win_mode == WIN_NONE_LEFT:
        total = start.get("targets", 0)
        return "COLLECTED", total - int((grid == spec.win_a).sum()), total
    if spec.win_mode == WIN_MATCH:
        x0, y0, x1, y1, mw, mh = spec.match
        a = grid[y0:y0 + mh, x0:x0 + mw]
        b = grid[y1:y1 + mh, x1:x1 + mw]
        # Compared by colour, as the engine does: the target may use its
        # own unclickable kinds that merely look like the canvas kinds.
        col = np.array([k.color for k in spec.kinds], np.int16)
        ca = np.where(a < 0, spec.background, col[np.clip(a, 0, None)])
        cb = np.where(b < 0, spec.background, col[np.clip(b, 0, None)])
        return "CELLS", int((ca == cb).sum()), int(mw * mh)
    if spec.win_mode == WIN_REACH:
        gates = start.get("gates")
        if not gates:
            return None
        total = start.get("gate_cells", 0)
        left = int(np.isin(grid, list(gates)).sum())
        return "GATES", total - left, total
    return None


def record(spec, path, actions, scale: int = 8, ms: int = 260,
           hold: int = 2200, library=None, chrome: dict | None = None):
    """Replay `actions` through `spec` and write the frames to `path`.
    `chrome`, when given, takes title / budget / levels / level and mats
    every frame into the player panel."""
    from .dsl import DslGame

    game = DslGame(spec, library=library, max_frames=64)
    game.init()
    grid, floor = game.cells()
    # Totals the objective is measured against, taken once at the start:
    # collectibles vanish as they are taken, and gates as they open.
    start = {"targets": int((grid == spec.win_a).sum()),
             "gates": _gate_kinds(spec)}
    start["gate_cells"] = int(np.isin(grid, list(start["gates"])).sum()) \
        if start["gates"] else 0

    def shot(frame):
        g, f = game.cells()
        return frame, _progress(spec, g, f, start), game.state == "WIN"

    shots = [shot(game.frame())]
    takes = [0]
    for k, action in enumerate(actions):
        for f in game.act(*action):
            shots.append(shot(f))
            takes.append(k + 1)
    game.close()

    frames = [to_image(f, scale) for f, _, _ in shots]
    if chrome is not None:
        frames = [_panel(im, chrome, taken, note, won)
                  for im, taken, (_, note, won) in zip(frames, takes, shots)]
        frames = _quantize(frames)

    durations = [ms] * len(frames)
    durations[0] = 1200
    durations[-1] = hold
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, disposal=1, optimize=True)
    return len(frames)
