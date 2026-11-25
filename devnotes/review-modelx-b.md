# Review B: modelx integration branch `r4/modelx-integration` (df3ced9)

Independent review against upstream main fec516d. Environment: CPython 3.13.18 and
free-threaded 3.14.6t (`PYTHON_GIL=0`), lifelib main 20d4dae, 4 cores. Repro scripts
were run from a scratch directory. "Run" means executed here; "read" means by reading
the code only. Line numbers refer to df3ced9.

Suites: 3.13, whole suite, 1962 passed, 7 skipped (QuantLib installed). 3.14t, QuantLib test
deselected: 1962 passed, 6 skipped. Both run.

## Blocking or fix-now findings (all run unless marked)

| # | Piece | Defect (repro, expected, actual) | Fix (lines) |
|---|---|---|---|
| F1 | collect | `pickle.dumps(r)`, `copy.copy(r)`, `copy.deepcopy(r)` of a `CollectResult` (live and exported) raise `RecursionError`. Expected: a copy. Cause: `__getattr__` reads `self._fields` before `__slots__` is set. | `collect.py:67-71`, `_mx_sys.py:180-184`: read `object.__getattribute__(self, '_fields')` and add `__reduce__` returning `(type(self), (self._fields,))`. Run as a patch: pickle, copy and deepcopy then work. |
| F2 | collect | Live `collect()` is O(N x graph). 8,000 items: 1.5 s cold, 16.1 s when the 8,000 ItemSpaces already exist (4,000: 0.9 s vs 6.2 s). With 20,000 unrelated cached nodes, 2,000 items: collect 2.2 s, plain loop 0.35 s. Causes: `collect.py:406` scans every tracegraph node per item; `collect.py:221-223` scans all of `parent.dynamic_cache` per item. | Find nodes from the transient Cells' `data` keys and pop the root's own `dynamic_key` only. Run as a monkeypatch: 8,000 items with existing ItemSpaces 16.1 s -> 2.6 s; `tests/core/space` 550 passed. About 15 lines. |
| F3 | dyn refs | Export now refuses Space formulas it used to export. Upstream fec516d exports `lambda i: {'base': B}` (silently wrong: live 7, exported 1), `{'refs': d}` with `d` built by item assignment, and `{'refs': {k: i for k in ...}}`; the branch raises `ValueError` in `exporter.py:197-267`. Not in the release notes. | List it under Backward Incompatible Changes (the `base` case is a fix). Optionally fall back to the old export with a warning when no returned Reference is read by a Cells (not run). |
| F4 | collect | `hasattr(space, 'collect')` is now True for every Space, ItemSpace and DynamicSpace (`space.py:79,86`); `getattr(s, 'collect', None)` returns a method that raises `TypeError`. Not listed. | Return `_mx_collect` only when `getattr(self._impl, 'collect_runner', None) is not None`. 2 lines, not run. |
| F5 | collect | A collect declared over a plain iterable (accepted by `new_space`, `collect.py:187-194`) makes `write_model` raise at `serializer_8.py:419` after `P/`, `__init__.py` and `_system.json` are already written. Export rejects it too (`exporter.py:1080-1082`). | Reject a domain that is neither a Cells nor a Reference of the Space at declaration time, in `validate_collect_contract`; drop the identity match `collect.py:187-193`. |
| F6 | collect | A stale declaration survives a Space formula change: after `s.formula = lambda x: None`, `collect()` raises `TypeError: missing a required argument: 'x'`; `write_model` saves the stale `_collect.json`, and `read_model` restores it unchecked (`serialize/__init__.py:162-181`). | In `UserSpaceImpl.set_formula` (`space.py:2348`) and on restore, re-run `validate_collect_contract` or drop the declaration with a warning. |
| F7 | determ | Release note "writes the same pickle file" is too strong. A `frozenset` of str gives different `_mx_pickled` bytes under PYTHONHASHSEED 1 and 2 (run). 3.13 pickles with protocol 4 and 3.14 with protocol 5 (run), so the bytes differ across versions. On Windows `write_str_utf8(newline=None)` writes CRLF (read). | Pass `protocol=4` in `DataManager.pickle` (`exporter.py:431-439`, not run); narrow the note to keys and `.py` files of one platform and Python version. |

Not blocking: errors match the plain loop (`ZeroDivisionError` inside `FormulaError`,
traceback `P.collect() -> P[2].collect_outputs() -> P[2].bad()`, run). Values match
the loop (scalars, dict leaves, individual order; run). Note: a `sum` leaf is folded
left to right, so it equals a `+=` loop but not builtin `sum()`, which compensates on
3.12+ (890.0887461408802 vs ...805, run). A list `sum` leaf concatenates. The relnote
should say so.

## (1) collect: `collect()`, declaration, `set_collect`, `collect_workers`, save, export runtime, protocol-2 hooks

1. **Mechanism.** `new_space(collect_over=, collect_outputs=)` (`parent.py:100-146`)
   validates (`collect.py:81-122`), creates a `collect_outputs` Cells and a `CollectRunner`
   (`space.py:2368`). `Space.collect` is answered by `__getattr__` (`space.py:75-88`) ->
   `UserSpaceImpl.collect` -> `executor.eval_node((runner, key))`, so the result is a cached
   node. `on_eval_formula` (`collect.py:449-468`) makes one `_TransientItemSpaceImpl` per
   coordinate, evaluates `collect_outputs`, detaches the outputs (`_mx_collect_semantics.py:219-265`),
   folds them (`Accumulator`, 312-340), moves the item's external dependencies onto the
   collect node (`collect.py:394-447`), then deletes the item. Invalidation:
   `itemspaces.py:76-81,104-105` and template IDs (`collect.py:293-299`). Save:
   `_collect.json` (`serializer_8.py:406-426`, restore `serialize/__init__.py:162-184`).
   Export: a `collect()` method calling `_mx_sys.collect_from_space` (`exporter.py:1065-1114`).
   That function runs serially, with ordered batches on threads (`_mx_sys.py:422-455`), or
   through compiled hooks (`_mx_sys.py:230-302`).
   Measured, memory (run): 300 items, live collect peak 0.5 MB vs plain loop 16.9 MB retained.
   Speed: live collect is 2.7x slower than the plain loop at 500 items, from F2.
   Invalidation is correct after a ref, a domain Cells, a Cells formula and the Space formula change (run).
2. **Complexity.**

| Cut | Lines | Why |
|---|---|---|
| Compiled-only fast paths: `_OrderedNumericRows`, `_PendingCollectSum`, `_publish_collect_result`, private and evaluate-serial branches, `_mx_collect_chunk`, `shared_root`/attach/release/gather | `_mx_sys.py:193-228, 251-302, 305-367, 375-419, 428-449, 476-529` (~185) | No modelx caller. `_OrderedNumericRows` and the `_v1` hooks are not exercised by any modelx test. |
| `numeric_keys`, `PackedFloatSeries`, `NativeCollectedSeries` and its views, `public_numeric_series`, `detached_numeric_sum`, Packed branches of `_add` | `_mx_collect_semantics.py:43-200, 268-274, 293-299` (~170) | Same. The "callable month mapping" and the 4096-key int64 limit fit time-indexed lifelib models. |
| `_mx_collect_set_workers_v2`, `evaluate=` parameter, 4 module flags | `_mx_sys.py:96-100, 214-219, 230` | Alias "for packages built before": no such release exists. `evaluate` is never passed. |
| Second `CollectResult` | `collect.py:52-78` = `_mx_sys.py:171-190` | Move one copy to `_mx_collect_semantics` and import it in both. |
| `reset_signature`/`bind_key` | `collect.py:268-291, 324-340` | Duplicates `Formula._set_signature` (`formula.py:375-411`) and `trace.get_node`. Give the runner a `Formula`-like with `_bind_tails`. |
| `_IdentityArgument` | `collect.py:25-49, 336-349` | Lets `collect()` take unhashable arguments that `Space[...]` rejects. That is a divergence; drop it. |
| `product()` over domains | `collect.py:358-362, 456-457` | Exactly one domain is allowed (`collect.py:101`). |
| Plain-iterable domains | `collect.py:145-148, 187-193` | Cannot be saved or exported (F5). |
| `refs=` pass-through in `mx.new_space` | `api.py:259,320` | Unrelated to collect. |

   Total: about 360 lines serve only the compiler, more than the report's 311. Without
   commits c098af3, fb3f9f0, d165664 and 8613960 the suite still passes (run, below), but 175ee3f
   already contains most of protocol 2.
3. **Generic.** The declaration (one loop parameter, `individual`/`sum`, a zero-argument
   output formula) is generic, and nothing in live collect is lifelib-specific. The hook
   contract is not written anywhere in `doc/` (searched). Its names are mixed: `_v1`, `_v2`
   and unversioned names (`_mx_collect_new_root`, `_mx_collect_attach`, `_mx_assign_params`).
   It has 4 capability flags plus `_mx_collect_adapter_version = 2`; the `options()` tuple
   takes a 5th element by length (`_mx_sys.py:252-260`); the emitted `_mx_collect_contract_v2`
   is read only by the compiler. That is neither minimal nor versioned in one place.
   Proposal: one documented hook, `_mx_collect_evaluate_v1(coords, args) -> CollectResult | None`,
   plus `collect_workers`. The compiler would ship its numeric paths in its own runtime.
4. **Reliable.** F1, F2, F4, F5, F6. Exported collect on 3.14t gives the same values,
   individual order and error type as a plain loop for 1 and 4 workers. 4 concurrent
   `collect()` calls plus 2 threads creating ItemSpaces give one distinct result and no
   error (run). Fast paths switch off under `sys.gettrace()`/`getprofile()`
   (`_mx_sys.py:270,288,431,475`): 7 collect tests fail under `coverage run`, and results
   may take another code path under a debugger. Older modelx (fec516d) reads a collect
   model and keeps `collect_outputs` as an ordinary Cells; `collect` is absent (run).
   Free-threaded scaling (run): with `.pyc` present, 1 worker 0.83 s, 4 workers 0.34 s. In the
   first run after export (no `.pyc`), 4 workers take 2.2 s against 1.1 s for 1 worker. A plain threaded
   loop over the same exported ItemSpaces slows down the same way (1.26 -> 2.04 s), so the
   cause is CPython 3.14t with in-process compiled code, not collect. Suggestion, not run: call
   `compileall.compile_dir` at the end of `export_model`.
5. **Independence.** It applies to fec516d only after a textual resolution in `exporter.py`.
   9 export tests then fail (`test_collect_parallel_export`, `test_collect_persistent_items`)
   because they use Space formulas that return `refs`, which needs (2). Runtime dependency:
   `_mx_sys.py:502` calls `_mx_apply_formula` from (2) when present.
6. **Bottom line.** Live `collect()`, the declaration, `set_collect`, save/restore and a
   pure-Python exported runtime (serial plus ordered threads): about 900 non-test lines
   instead of about 1,450. Fix F1, F2, F4, F5 and F6 first. Move the protocol-2 hooks and
   numeric series to a separate compiler PR with a written contract, or into the compiler's
   own runtime.

## (2) Export of statically named dynamic Space references (2097f7b)

1. **Mechanism.** `_dynamic_ref_names` (`exporter.py:197-267`) parses the Space formula.
   It accepts `return None`, `{...}` or `dict(...)` literals, and names bound once by
   assignment (checked with `symtable`), and collects the `refs` keys. `dynamic_ref_formula`
   (270-286) emits `_mx_space_formula`. `_mx_apply_formula` (`exporter.py:1046-1063`) sets
   those attributes on each new ItemSpace; the slots default to `None` (1125-1128).
2. **Complexity.** ~140 lines, compact for what it checks. Keep it.
3. **Generic.** Yes: any model that returns `{'refs': {...}}` with literal keys.
4. **Reliable.** F3. A key returned only on some branches is `None` in the exported item,
   where live modelx raises `NameError` (read). The exported model now gives the live value for
   `{'refs': {'r': i*2}}`, where upstream raised `AttributeError` (run).
5. **Independence.** It applies alone on fec516d; suite 1884 passed, 1 failed (the known upstream
   deserializer test), 7 skipped (run). Submit it before (1).
6. **Bottom line.** Ship as is, with F3 in the release notes.

## (3) Classes defined in stored modules: save and export (5f26ea7, 0883979)

1. **Mechanism.** `module_classes` (`moduleio.py:141-160`) maps top-level classes of
   `ModuleData` modules. `ModelPickler.persistent_id` writes `("ModuleAttr", spec, name)`
   (`custom_pickle.py:105-108`), and the unpickler reads it back (186-188). On export, `ModuleClassPickler` (`exporter.py:442-455`),
   `DataUnpickler`, `MiniModuleIO`/`MiniModuleData` (`_mx_sys.py:536-547, 622-638`) do the same.
2. **Complexity.** ~90 lines, no duplication worth removing.
3. **Generic.** Yes, for any class at module top level; functions and nested classes still
   fail as before. Extending it to functions is a one-line `isinstance` change (not run).
4. **Reliable.** Upstream cannot save or export such a model (`PicklingError`, run), so
   nothing regresses. The branch round-trips the values, and `m.k is m.enums.Kind.A` holds (run). Upstream reading a
   branch-saved model warns and sets the References to `None` (run): state this in the note.
5. **Independence.** Only a textual conflict in `_mx_sys.py` (append position). Resolved
   alone on fec516d: 1877 passed, 1 failed (deserializer), 7 skipped (run).
6. **Bottom line.** Ship as is.

## (4) Deterministic export keys (2814cc9)

1. **Mechanism.** `DataManager` numbers IOSpecs in `Model.iospecs` order and pickled
   values in first-request order (`exporter.py:375-407`).
2. **Complexity.** 22 lines, minimal.
3. **Generic.** Yes.
4. **Reliable.** For BasicTerm_SE and CashValue_ME_EX1 under seeds 1 and 2, the `.py` files and `_mx_pickled` are identical
   (run); only `.xlsx` differ. See F7 for the limits.
5. **Independence.** The code conflicts textually with (3) in `DataManager.pickle`. Resolved
   alone, `test_export_determinism` fails with `PicklingError` because its model uses an enum
   from a stored module (run). Either order (3) before (4), or drop the enum from that test.
6. **Bottom line.** Ship after (3), with F7.

## (5) Executor on Intel macOS 3.10 (636877b), and CI (d309095, e1fb7c0, f0e91ec)

1. **Mechanism.** `system.py:80-88` selects `ThreadedExecutor` with a 256 MB stack on CPython 3.10
   x86_64 macOS. `executor.py:149,174` restarts the worker after a fork, and `_formula_error`
   (108-115) now gives the threaded path `Caused by` too.
2. **Complexity.** Small. Python 3.10 reached end of life in October 2026, so the platform
   branch targets an unsupported Python. Keep the fork restart and `_formula_error`; drop or
   justify the 3.10 macOS branch.
3. **Generic.** Yes.
4. **Reliable.** Applied alone: 1874 passed, 1 failed (deserializer), 7 skipped (run). Not run on macOS.
5. **Independence.** Independent. CI: d309095 and f0e91ec both rewrite `test_deserializer.py`
   and conflict. The CI PR must carry one version (HEAD's).
6. **Bottom line.** Make CI the first PR, then executor (fork + cause only).

## Order and independence summary (all run on 3.13 against fec516d)

| Piece alone | Applies | Suite |
|---|---|---|
| CI | conflicts between its own two test fixes | take HEAD's version |
| executor | yes | 1 upstream failure only |
| dyn refs | yes | 1 upstream failure only |
| enum | textual conflict | 1 upstream failure only |
| determ | textual conflict | + `test_export_determinism` fails (needs enum) |
| collect (175ee3f, 9cdf04d) | textual conflict | + 9 export tests fail (need dyn refs) |
| dyn refs + collect without the numeric commits | yes | 1 upstream failure only |

Order: CI -> executor -> dyn refs -> enum -> determ -> collect -> (compiler hooks, separate).

## Smallest version of each piece

| Piece | Keep | Cut or move |
|---|---|---|
| collect | live runner, declaration, save, generic export runtime | ~360 lines of compiled-only paths, `_IdentityArgument`, binding duplicate, plain domains |
| dyn refs | all | - |
| enum | all | - |
| determ | keys | overstated note |
| executor/CI | CI, fork restart, cause | 3.10 macOS branch (EOL) |

