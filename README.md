<div align="center">

# ⚡ arc.c

**The ARC-AGI-3 interactive environments, in C.**

A dependency-free C implementation of the [ARC Prize](https://arcprize.org/arc-agi/3) reasoning environments: 25 games, two rendering backends, a batched multi-threaded vector environment, and an optional XLA custom call so JAX can drive it without leaving the compiled program.

Environments are held to **frame-exact** parity with the official
[ARC-AGI Toolkit](https://github.com/arcprize/ARC-AGI), enforced by differential
tests that replay both implementations side by side against the *reference*,
not against a sibling port.

[![C11](https://img.shields.io/badge/C-11-blue.svg)](https://en.wikipedia.org/wiki/C11_(C_standard_revision))
[![No dependencies](https://img.shields.io/badge/dependencies-none-brightgreen.svg)](#quickstart)
[![Parity](https://img.shields.io/badge/parity-23%2F25%20frame--exact-yellow.svg)](#status)

</div>

---

## Status

All 25 public ARC-AGI-3 environments are implemented. 23 of them are verified
frame-exact against the official Python, every level, 120 actions each:

```
23 games, 164 runs, 19,772 actions and 37,658 frames verified in 19s
```

Two gaps. `g50t` level 5 emits seven frames where the reference emits eight,
the missing one a duplicate of the last. `bp35` and `lf52` run on the scene
backend, which needs a separate construction path in the harness and is not
wired up yet.

`lf52` also cannot be fully verified against the reference in any language: it
calls `np.random.shuffle` on numpy's unseeded global `RandomState`, so that
region is not reproducible between processes.

[docs/benchmark.md](docs/benchmark.md) is the reference for what the
generator in `src/dsl.c` is aiming at: how the Kaggle evaluation scores, what
is known about the hidden set, a mechanic-by-mechanic reading of the 25 public
games, and the DSL's coverage against them.

## Quickstart

**Requirements:** a C11 compiler.

```bash
git clone git@github.com:noahfarr/arc.c.git
cd arc.c
make -j
```

That builds `libarc`, which has no dependencies at all. `make pgo` does a
profile-guided build.

To check it against the official Python:

```bash
uv run python -m harness           # every game, every level
uv run python -m harness --layout  # struct mirrors vs the compiler
```

The harness needs `arc-agi` and `numpy`, and nothing else. See
[harness/README.md](harness/README.md) for what it checks and why it compares
only against the official implementation.

`make ffi` builds the XLA custom call so JAX can drive the library without
leaving the compiled program. That is the only part of this repo that involves
JAX, and it needs the jax headers: `uv sync --extra ffi`.
