# M1/M2: modelx integration branch `r4/modelx-integration`

Base 3a26454. Upstream main fec516d is an ancestor of 636877b: merge no-op.

## Two-way difference 636877b <-> e1fb7c0 (merge base 175ee3f)

Only in 636877b (7 commits, +532/-67):

| Topic | Commit | Non-test | Tests |
|---|---|---|---|
| Collect: ordered numeric rows for compiled models | c098af3 | +52/-6 | - |
| Collect: named series callable with stored keys | 9cdf04d | +39/-14 | +21 |
| Collect: numeric series folded in the serial loop | fb3f9f0 | +64/-2 | +42 |
| Collect: packed sums published after reduction | d165664 | +158/-11 | +36 |
| Collect: layout reuse for equal ranges | 8613960 | +12/-6 | +4 |
| lifelib test: trailing comments in the asttokens comparison | f0e91ec | - | +21/-2 |
| Executor: Intel macOS 3.10 stack, restart after fork | 636877b | +43/-38 | +56/-4 |

Only in e1fb7c0 (2 commits, +16/-6), brought in by merge 3524cec: CI action
versions and a lifelib pin for Python < 3.10 (d309095: workflow +2/-2,
requirements +3/-1), Python 3.9 on Ubuntu 24.04 (e1fb7c0: workflow +2/-1),
and the same lifelib test fix as f0e91ec, written differently (+9/-2).

Conflict, `modelx/tests/lifelib/test_deserializer.py`: I kept 636877b's
version (f0e91ec). Both versions pass against lifelib main 20d4dae, and
neither fixes the other. f0e91ec is the later one: it removes a trailing
inline comment from both sides, while d309095 drops only a final comment token
from the modelx side. Upstream main fails this test against current lifelib.

## Export determinism (2814cc9)

`DataManager` keyed `pickle_data` and `io_data` by `id()`. Keys are now
running numbers: IOSpecs in `Model.iospecs` order, and pickled values in the
order the translators first ask for them (model traversal).
`ModuleClassPickler` uses the same IOSpec numbers. The loader treats keys as
opaque, so what an exported model loads is unchanged. New test
`tests/export/test_export_determinism.py` exports one saved model in two
processes (PYTHONHASHSEED 1 and 2) and compares every file. With the old
exporter 5 files differ; now none do. An ad hoc check on lifelib BasicTerm_S,
CashValue_ME_EX1, BasicBonds, BasicHullWhite and IntegratedLife: all `.py`
files match. Only `.xlsx` differ, because openpyxl writes the time into
`docProps/core.xml`. That comes from the `write_ios` path that saving also
uses, so it is out of scope here; the test uses CSV and the release note says so.

## Branch diff against upstream main (+4,153/-105, 43 files, before this report)

| Area | Non-test | Tests | Purpose | For (a/b/c) |
|---|---|---|---|---|
| core: api, collect, model, parent, space, itemspaces | +684/-24 | +1,303 | `collect()`, `collect_over`/`collect_outputs`, `set_collect`, `collect_workers` | b |
| serialize: `__init__`, serializer_8 | +48 | (above) | Save and read the collect declaration | b |
| export: _mx_sys, _mx_collect_semantics | +757 | (above) | Collect runtime of exported models, protocol 2 hooks | a b |
| exporter: dynamic references (2097f7b) | +138/-4 | +86 | Export statically named dynamic Space refs | a |
| exporter: collect | +93/-19 | - | Emit collect classes and options | a b |
| moduleio, custom_pickle, exporter/_mx_sys modules | +90/-4 | +136 | Classes in stored modules, saved and exported | c |
| exporter: deterministic keys | +22/-8 | +125 | Byte-identical exports | a |
| docs: reference, relnotes | +35 | - | User-facing notes | all |
| core/execution/executor, core/system (636877b) | +43/-38 | +56/-4 | ThreadedExecutor on Intel macOS 3.10 | none |
| CI workflow, requirements-travis | +7/-4 | +21/-2 | CI matrix, lifelib pin, deserializer test | CI |
| devnotes/ | +511 (+ this file) | - | Project notes | none |
| modelx/__init__.py | +1/-1 | - | Version 0.33.1.dev0 | none |

Not needed for (a) export, (b) collect or (c) stored enum classes: devnotes
(leave out of the upstream series); the executor change (its own small PR); the version bump (maintainer's call); the `mx.new_space(..., refs=)`
pass-through in `core/api.py` (175ee3f); CI/requirements and the deserializer
test (needed for green CI; could be a first PR). Inside (b), the numeric fast
paths c098af3/fb3f9f0/d165664/8613960 (~311 lines) run only when a compiled
class offers `_mx_prepare_ordered_rows_v1` or `_mx_collect_numeric_output_v1`,
so they could be split off. The inherited commits have tool or placeholder
author identities; re-cut the final series with the maintainer's identity.

## Test counts (all of `modelx/tests`, pytest -n 4, lifelib 20d4dae editable)

| Tree | Python 3.13.18 | Python 3.14.6t, PYTHON_GIL=0 |
|---|---|---|
| upstream main fec516d | 1872 passed, 1 failed (deserializer), 7 skipped | not run |
| 636877b | 1956 passed, 7 skipped | 1956 passed, 6 skipped, 1 failed* |
| branch 2814cc9 | 1962 passed, 7 skipped | 1962 passed, 6 skipped, 1 failed* |

\* `test_lifelib.py::test_assets` needs QuantLib (no free-threaded wheel; CI
deselects it). No new failures; all upstream tests pass on the branch. Test
edits: the deserializer conflict only, plus one new file. None weakened.

## Executor default (not changed)

- e1fb7c0 (= upstream main): `NonThreadedExecutor`, except `ThreadedExecutor`
  (256 MB worker stack) on Windows with Python < 3.12.
- 636877b: the same, plus `ThreadedExecutor` (256 MB) on CPython 3.10 on
  Intel macOS, whose main-thread stack is too small for deep formulas. The
  worker also restarts after a fork, and FormulaError keeps `__cause__`.
- Normal Python: only those two platforms get a worker thread. Everything
  else evaluates on the calling thread, with the recursion limit modelx sets.
- Free-threaded 3.13t/3.14t: both use `NonThreadedExecutor`, so there is one
  shared call stack, and live modelx must not be called from several threads.
  For that reason live `collect()` is serial, and only exported or compiled
  models use `collect_workers` threads (default None, i.e. 1 worker or the
  compiled `parallel_workers`, on both sides).
- Recommendation: keep 636877b's selection. It matches upstream except on
  Intel-macOS 3.10, where it avoids running out of native stack. Submit it as
  a PR separate from export and collect, and keep `NonThreadedExecutor` on
  free-threaded builds.
