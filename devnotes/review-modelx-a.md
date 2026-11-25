# Review A: `r4/modelx-integration` (df3ced9) against upstream main (fec516d)

Independent review. Lines are those of df3ced9. "Run" = executed here on Linux x86-64, 4 CPUs,
CPython 3.13.18 and free-threaded 3.14.6 (`PYTHON_GIL=0`), lifelib main; "not run" = by reading only.
Baseline reproduced: 3.13 1962 passed / 7 skipped; 3.14t 1962 passed, 1 failed (`test_assets`, no QuantLib wheel).
Tested fixes for F1, F2, F4 and F5 are in `devnotes/review-modelx-a-fixes.patch` (+37/-7, 7 files); with it the
suite result is the same on both interpreters (1962 passed; on 3.14t `test_assets` still fails).

## 0. Findings to act on

| Id | PR | Kind | Finding (reproduction -> expected / actual) | Fix | Run |
|---|---|---|---|---|---|
| F1 | 1 | regression, users without collect | Every edit walks all Spaces looking for collect runners (`core/itemspaces.py:78-81`). 800 Spaces x 20 Cells built by `new_cells`: upstream 9.7-10.2 s, branch 11.3-12.2 s (+15-25%); profile: 88M -> 127M calls | Walk only if the model ever declared collect (`ModelImpl.has_collect`, set in `space.py:2370`) | yes: 9.1-10.2 s |
| F2 | 2 | regression, existing exports | `formula=lambda i: {'base': _space}` (also `'bases'`, base + refs): upstream exports it and the values equal live; branch raises `ValueError` (`export/exporter.py:249-250`, call at `:934`) | Accept `base`/`bases` naming `_space`/`_self` at `:249` and in the emitted check `:1061` | yes |
| F3 | 1 | behaviour change, not listed | `hasattr(space, 'collect')` is now True for every UserSpace and ItemSpace; the call raises `TypeError` (`core/space.py:79-80, 86-87`). Before: `AttributeError` | Return `_mx_collect` only when `collect_runner` is set, or list it | no |
| F4 | 1 | defect | `pickle.dumps(space.collect())`, `copy.copy`, `copy.deepcopy`: `RecursionError`, live and exported (`core/collect.py:67`, `export/_mx_sys.py:180`: `__getattr__` reads `_fields` on a bare instance) | Guard `_fields`, add `__reduce__` | yes |
| F5 | 1 | defect, data loss | Saved model, then `set_collect(over={'i': [1, 2, 3]})`, then `write` to the same path: `ValueError` raised at `serialize/serializer_8.py:419`, after the backup rotation and after part of the tree is written; the path now holds a model without the declaration | Same check in `validate_model` (`serializer_8.py:316-331`), which exists for this purpose | yes: old save intact |
| F6 | 1 | defect | A Space derived from, or copied from (`Space.copy`), a collect Space has `collect_outputs` but no declaration: `TypeError: this Space has no collect declaration` | Carry the declaration, or document | no |
| F7 | 1 | difference to a loop | Empty domain: loop gives 0, `collect()` raises `ValueError` (`_mx_collect_semantics.py:20-21`). `'sum'` of list, tuple or str leaves concatenates (`:307`): three items of `[1.0, 2.0]` give six elements | List both; reject non-numeric `sum` leaves | no |
| F8 | 3 | old reader | upstream modelx reading a model that stores a module class: loads, References become `None` with warnings (no serializer version change). A collect model reads cleanly there (`_collect.json` ignored) | Say so in the proposal | read: yes |
| F9 | 4 | limit of the claim | `_mx_pickled` still differs between hash seeds when a pickled Reference holds a `set`; text files are written with `newline=None` (`serialize/ziputil.py:165-179`), so CRLF on Windows | State the limits; `newline='\n'` | set: yes; Windows: no |
| F10 | all | docs | Release notes (`doc/source/releases/relnotes_v0_34_0.rst`) have no entry for `collect()` itself, dynamic References, module classes, F2/F3, or the model lock taken by ItemSpace creation in collect exports; `collect` is in no reference page; the example uses `mgsv_pp(72)` | Write per PR | - |

No finding blocks PR 3, 4 or 5. F1, F2, F4, F5 should be fixed before PR 1 and PR 2 are proposed.

## PR 1: collect

**1. Mechanism.**

| Part | Lines | What it does |
|---|---|---|
| Declaration | `core/parent.py:100-143`, `core/space.py:792-840, 2368-2371`, `core/api.py:259-322`, `core/collect.py:81-122, 135-195` | `new_space(collect_over=, collect_outputs=)` or `set_collect(over, outputs)` validates one loop parameter, creates the Cells `collect_outputs`, binds the domain to a Cells, a Reference or a plain object, creates a `CollectRunner` |
| Call | `core/space.py:76-87, 642-652, 2373-2378` | `space.collect(*other_params)` reaches `_mx_collect` through `__getattr__`, so a member named `collect` wins |
| Live evaluation | `core/collect.py:253-468` | The runner is a cached node. Per item: build a `_TransientItemSpaceImpl` (`:198-224`), evaluate `collect_outputs`, detach and accumulate, rewire trace-graph and reference-graph edges from the item's nodes to the runner node (`:394-447`), delete the item |
| Invalidation | `core/itemspaces.py:76-81, 104-105`, `core/space.py:2348-2366`, `core/collect.py:293-299` | The runner keeps the ids of the templates it used; any edit that dirties one clears it |
| Result | `core/collect.py:52-78`, `_mx_collect_semantics.py:11-41, 202-340` | `CollectResult` with `individual` (dict by item) and `sum`; dict leaves are `CollectedSeries` (`x[t]` and `x(t)`); outputs are deep-copied unless flat scalars |
| Saving | `serialize/serializer_8.py:406-426`, `serialize/__init__.py:157-184` | `<Space>/_collect.json` = `{"parameter", "domain": {"kind": "cell" or "ref", "name"}}`; read back for every serializer version |
| Export | `export/exporter.py:310-352, 635-669, 1034-1039, 1065-1114` | Emits `collect()` (or `_mx_collect`), `_mx_collect_contract_v2`, a model `_CollectLock`, and routes every parameterized Space of the model through the locked ItemSpace template |
| Exported runtime | `export/_mx_sys.py:21-46, 80-100, 171-534` | `collect_from_space`: cache, worker count (1 with the GIL), serial loop over transient roots, or ordered batches of at most 8 on a thread pool (`:422-456`) |
| Compiler hooks | `_mx_sys.py:193-219, 226-228, 247-271, 305-368, 386-419, 477-515`; `_mx_collect_semantics.py:43-200, 268-274` | Looked up with `getattr` on the Space class; never defined by modelx itself |

**2. Complexity.**

| Cut | Lines | Size | Evidence |
|---|---|---|---|
| C1 Compiler-only paths: `_PendingCollectSum`, `_publish_collect_result`, three flags, `_mx_collect_options_v2`, `_mx_collect_parallel`, `_mx_collect_chunk`, `_OrderedNumericRows`, numeric gather, shared root, attach/release; `numeric_keys`, `PackedFloatSeries`, `NativeCollectedSeries` and views, `public_numeric_series`, `detached_numeric_sum` | as in the last row above | -450/+45 (`_mx_sys.py` 651 -> 425, semantics 340 -> 161) | Run: only tests that drive fake hooks fail (8 tests in `test_collect_parallel_export.py` and `test_collect_export_cleanup.py`, plus module `test_collect_numeric_series.py`); 1949 others pass; same on 3.14t. The reported 311 lines understate it. Serial exported collect, 20,000 items: 0.90 s before, 0.71 s after |
| C2 Worker threads for the pure-Python export: `_CollectLock`, `collect_workers` (live and exported), `_mx_collect_ordered`, `_mx_collect_batch`, dispatch state, lock emission | `_mx_sys.py:21-46, 80-100, 370-456`, `core/model.py:606-646, 1245, 1275`, `exporter.py:635-669, 1034-1039`, `core/api.py:1531-1561` | about -190, and 425 test lines | Not run as a cut. Run on 3.14t, 4 CPUs, 20,000 items: policy-only model 0.47 s serial, 0.52 / 0.48 / 0.30 s with 2 / 4 / 8 workers; model reading shared Spaces 0.90 s serial, 1.37 / 1.35 / 1.32 s. No reliable gain without a compiler. Live modelx accepts the setting and ignores it |
| C3 Live collect as a traced node | `core/collect.py:198-251, 293-299, 386-447`, `itemspaces.py:76-81, 104-105`, `space.py:2348-2366` | about -200 | Run: 1,200 items x 60 steps: `collect()` 5.05 s, 0.8 MB peak; loop keeping ItemSpaces 2.62 s, 99.7 MB; loop with `del space[i]` 2.73 s, 0.7 MB. The existing idiom gives the same memory at half the time. `_contract_dependencies` is 43% of collect time and scans the whole trace graph per item (`:406`). A loop over ordinary ItemSpaces with `clear_itemspace_at` was not written |
| C4 Declaration surface: three entry points; identity matching of a plain object to a Reference (`collect.py:188-193`); `_IdentityArgument` for unhashable arguments (`:25-49`); `_bind_tails`, a copy of the Formula bind fast path, for a call made once (`:280-291, 328-337`); a mapping and `product()` for exactly one parameter (`:101, 456`) | `parent.py:100-143`, `api.py:259-322`, `collect.py` as given | about -110 | Not run. Plain-object domains cannot be saved or exported anyway (F5) |
| C5 Duplicates and dead code: `CollectResult` twice (`collect.py:52-78`, `_mx_sys.py:171-190`); alias `_mx_collect_set_workers_v2` "for packages built before" a protocol that was never released (`_mx_sys.py:96-100`); parameter `evaluate=` never passed (`:230, 297-298`) | as given | about -35 | Not run |

**3. Generic.** Domain: any sized, re-iterable, duplicate-free sequence (tuple, list, range, numpy, Index,
Series, dict keys: run). Outputs: scalars, nested dicts, numpy arrays, Series (run). Limits that come from
the lifelib shape: exactly one loop parameter; the domain must be a Cells or Reference *of the same Space*
to be saved or exported (a model-level Reference or a parent's Cells: `ValueError`, run); fixed name
`collect_outputs`; the words "policy" and "month" in messages and docstrings; constants 4096 keys, 16
layouts, batch 8 (`_mx_collect_semantics.py:61, 67`, `_mx_sys.py:428`); a compiler design reference in a
comment (`_mx_sys.py:478`).
Hook contract: **not written** (no document; the names appear only in code and in one sentence of the
report), **versioned three ways** (`_v1`/`_v2` suffixes, `_mx_collect_adapter_version`, three capability
flags), **not minimal**: 8 methods looked up on compiled classes, 3 emitted methods, 4 module constants,
3 keyword parameters, 6 helper names imported by the compiler. The fast paths switch off under
`sys.settrace`/`setprofile` (`_mx_sys.py:270, 288, 431, 475`), so a debugger or coverage run takes other code.

**4. Reliable.**

| Check | Result |
|---|---|
| Live `collect()` vs loop, values | Identical bits in 3 models (scalars, dict series, second parameter, positional and keyword). Dependencies: input on another Space, Reference change, formula change, domain change, call from a Cells: recomputed correctly each time. Existing `space[i]` kept |
| Live, errors | Same `FormulaError` and cause, trace has the extra `collect_outputs` frame; no transient Space left; next call works. Differences: F7 |
| Exported vs live vs loop | Identical bits, serial and 2, 4, 7, 64 workers on 3.14t; `individual` in domain order; item 5 raising: the same `ZeroDivisionError` serial and parallel. The docstring "can differ in the last bits" (`core/model.py:621-623`) holds only for a compiled model |
| Free-threaded 3.14 | 40 cold models x 8 workers x 400 items reading a shared parameterized Space, and 10 x 6 threads calling `collect()` at once: 0 mismatches, one cached result object. Live modelx stays serial; it is not thread-safe, as before |
| Save and read | Directory and zip round trip: run. Older reader: F8. Old models read by this branch: lifelib suite passes. Failed save: F5 |
| Pickling | F4. A cached result is returned by reference: `r.sum.cf[0] = -1` changes later `collect()` calls (as with any cached modelx value) |

## PR 2: dynamic References in exports

1. `dynamic_ref_formula` (`exporter.py:190-287`) reads the Space formula with `ast` and `symtable`, accepts
   `return None`, `return {'refs': {...}}`, `dict(refs=...)` and one level of local names, emits
   `_mx_space_formula` and `_mx_apply_formula` (`:1047-1066`), which set the named References on the ItemSpace root.
2. 138 lines; nothing to cut. The `dict`-is-builtin and symbol checks (`:211-218, 257-266`) are defensive but short.
3. Fits lifelib `nestedlife`/`ifrs17sim` (`refs = {...}; return {'refs': refs}`). Not accepted: conditional
   expressions (`{...} if c else None`), computed keys, a call, names starting with `_`, any `base` (F2).
4. Run, 10 formula shapes, live vs exported: upstream exports all 10 and is **silently wrong** for the 4 that
   set References; the branch is right for static names and raises for the rest. Raising where upstream was
   wrong is justified and must be listed; raising for base-only formulas is F2.

## PR 3: classes in stored modules

1. `module_classes` (`io/moduleio.py:141-160`) maps top-level classes of each `ModuleData`; the picklers write
   `("ModuleAttr", spec id, name)` (`serialize/custom_pickle.py:86, 105-108, 186-188`; `exporter.py:431-455`);
   the export gets `MiniModuleIO`/`MiniModuleData`/`DataUnpickler` (`_mx_sys.py:536-547, 622-651`).
2. 90 lines; `MiniModuleData.load_value` repeats `ModuleIO._load_module` (needed: the export is standalone).
3. Generic for any top-level class. Run: enum class, enum member, alias, dataclass instance, list of them:
   saved, re-read (class identity with the re-read module holds) and exported. Functions and nested classes
   of the module still fail with `PicklingError`, as upstream; the same persistent id would cover functions (not run).
4. upstream cannot write such a model at all (`PicklingError`), so no saved model changes meaning. F8.

## PR 4: deterministic export keys

1. `DataManager` numbers IOSpecs in `Model.iospecs` order and pickled values in first-use order (`exporter.py:375-405`).
2. 22 lines. `self.ios` is still keyed by `id()` (`:389`) but the ids do not reach the output.
3. Generic. 4. Run: a model with collect, dynamic References, a stored enum, a DataFrame, an array: all files
   equal for `PYTHONHASHSEED` 1 and 2; the `.py` files also equal between 3.13 and 3.14t. Limits: F9.

## PR 5: executor on Intel macOS, CI

1. `core/system.py:80-88` selects `ThreadedExecutor(256 MB)` on CPython 3.10/x86-64/macOS; the worker restarts
   after a fork (`executor.py:140-149, 174-176`); one `_formula_error` for both executors (`:108-115`).
2. Fine. The `_formula_error` merge changes the Windows < 3.12 message (adds "Caused by"): list it.
3. `== (3, 10)` fits the CI matrix; 3.7-3.9 have the same native-stack recursion: use `< (3, 11)` (not run).
4. Selection not runnable here; `test_system.py` repeats the condition, so it cannot fail. Fork test: run, passes.
   CI and requirements changes (+7/-4): fine, propose first.

## 5. Independence (run: piece removed from df3ced9, full suite on 3.13)

| Removed | Result | Meaning |
|---|---|---|
| 5 executor (636877b reverted) | clean revert, 1960 passed | independent |
| 4 deterministic keys (2814cc9) | clean revert, 1961 passed | code independent |
| 3 module classes, export half (0883979) | one conflict in `DataManager.pickle`; 1 failed: `test_export_determinism.py` uses a stored enum | PR 4's test needs PR 3, and both edit `exporter.py:431-455` |
| 2 dynamic References (cut by hand) | 9 failed: `test_collect_persistent_items.py` (6), `test_collect_parallel_export.py` (3) | PR 1's tests need PR 2; the runtime only probes `_mx_apply_formula` |
| 1 compiler paths (C1) | see C1 | separable from collect |

Order: CI -> 5 -> 3 -> 4 (or 4 first with a test that stores no enum) -> 2 -> 1 -> compiler hooks.
The save half of PR 3 (5f26ea7) and each piece alone on upstream were not run.

## 6. Bottom line: smallest version of each piece

| PR | Smallest version | Non-test lines |
|---|---|---|
| 1 collect | `set_collect` only; domain = Cells or Reference of the Space; live loop over ordinary ItemSpaces (C3) or today's runner with F1, F3-F6 fixed; `_collect.json` checked in `validate_model`; exported serial loop and cache (about 60 lines of runtime); one `CollectResult`. Follow-up PRs: worker threads with a measurement that justifies them; compiler hooks with a written, single-version contract | about 550 instead of about 1,580; C1 alone gives about 1,175 |
| 2 dynamic References | As is, with F2 and a release note for the forms that now raise | 138 + 8 |
| 3 module classes | As is, with a note on F8 | 90 |
| 4 deterministic keys | As is; test without the enum; note F9 | 22 |
| 5 executor, CI | As is; `< (3, 11)`; CI as its own first PR | 43 |

Also for the proposal: leave `devnotes/`, the version bump and the `refs=` pass-through in `mx.new_space`
(`core/api.py:259`) out of the series; re-cut the commits with the maintainer's identity (the report says so too).
