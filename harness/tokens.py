"""Frames as objects: uniform-colour regions from the C tokeniser."""
import ctypes

import numpy as np

FIELDS = 10  # color, cx, cy, x0, y0, x1, y1, area, px, py
COLOR, CX, CY, X0, Y0, X1, Y1, AREA, PX, PY = range(FIELDS)


def tokenize(library, frame, cap: int = 192, labels: bool = False):
    """Tokens [cap, FIELDS] int16 (area 0 = padding) for a 64x64 int8
    frame, largest regions first; with labels also the per-pixel token
    index map (-1 for dropped regions)."""
    frame = np.ascontiguousarray(frame, dtype=np.int8)
    out = np.zeros((cap, FIELDS), np.int16)
    lab = np.zeros((64, 64), np.int16) if labels else None
    n = library.sym.frame_tokens(
        frame.ctypes.data_as(ctypes.c_void_p), out.ctypes.data_as(ctypes.c_void_p),
        int(cap), lab.ctypes.data_as(ctypes.c_void_p) if labels else None)
    return (out, lab, int(n)) if labels else (out, int(n))
