# P2: thin collect, `r4/p2-modelx-collect`

Base: upstream main fec516d + `r4/p1-modelx-dynrefs` (6df5f81), used as it is.
Five commits by topic, then this report and the patch series of step 5 (drop
`devnotes/` from the pull request). Collect's export test of a Space formula
that returns `refs` needs P1 (the exported item calls `_mx_apply_formula`),
so the cut stays on P1.

## Lines (non-test, against P1)

| File | +/- | What |
|---|---|---|
| `modelx/core/collect.py` | +295 | declaration checks, `CollectRunner` (trace node), `CollectNode` (inspection view) |
| `modelx/core/space.py` | +133/-1 | `set_collect` (docstring ~75 lines), `_mx_collect`, `collect` in `__getattr__` only when declared, `UserSpaceImpl` slot and 4 methods |
| `modelx/export/_mx_collect.py` | +185 | shared rules: domain, `CollectResult`, `prepare_output`, `Accumulator` |
| `modelx/export/exporter.py` | +62 | `collect()`/`_mx_collect()` and `_mx_collect_item()` for declaring Spaces |
| `modelx/export/_mx_sys.py` | +50 | `collect_from_space`, the hook dispatch, `COLLECT_HOOK_VERSIONS` |
| `modelx/serialize/serializer_8.py` | +35 | `_collect.json` write, check in `validate_model`, read |
| `modelx/core/itemspaces.py` | +29/-10 | clear collect nodes; `_mros` factored out of `_closure_parts` |
| `modelx/core/parent.py` | +21/-1 | `new_space(collect_over=, collect_outputs=)` |
| `modelx/core/model.py` | +2 | `ModelImpl.collect_spaces` |
| Python total | **+812/-12** | integration branch: about 1,580 |
| docs | +156 | `collect_hook.rst` 93, release notes 61, two index lines |
| tests | +647 | 62 tests |

## What changed against the integration branch

- Live runner: kept as a traced node, re-measured (below): 1.3x a loop with `del`,
  not 2x. It no longer builds private transient ItemSpaces. It takes each item as
  `space[...]` does: an existing ItemSpace is used and kept; one it creates is
  deleted before the next, after its outside dependencies (trace and reference
  graph predecessors of the item's nodes, found from the cells' `data` keys, never
  by scanning the graph) are moved onto the collect node. ItemSpaceManager clears
  the node with the same closure rule it applies to ItemSpaces, visiting only
  `ModelImpl.collect_spaces`.
- Declaration: one parameter; the domain is a Cells of the Space or the name of a
  Cells or Reference of the Space, stored by name. Plain iterables, identity
  matching, `_IdentityArgument`, `_bind_tails`, `product()` are gone; binding goes
  through `trace.get_node`. `set_collect(None)` removes a declaration.
- Out: compiler paths and numeric series, worker threads and `collect_workers`,
  the model `_CollectLock` and locked ItemSpace creation, the second
  `CollectResult`, `refs=` of `mx.new_space`, `collect_over` of `mx.new_space`
  (unusable: no `refs`), the individual-shape check.

## Defects fixed

| Review | Defect | Test (fails on integration code) |
|---|---|---|
| a F4, b F1 | pickle/copy/deepcopy of CollectResult: RecursionError | `test_result_pickles_and_copies` (3) |
| b F2 | O(N x graph) when the ItemSpaces exist: 8,000 items 96 s here | `test_existing_itemspaces_are_used_not_recomputed` (no graph scan, items reused) |
| a F1 | walk over all Spaces on every edit | `test_edits_of_a_model_without_collect_never_read_a_declaration` |
| a F3, b F4 | `hasattr(space, 'collect')` True everywhere | `test_collect_is_defined_only_where_declared` |
| a F5, b F5 | plain-iterable domain; save fails half-written | `test_declaration_is_checked`, `test_a_stale_declaration_is_not_written` |
| b F6 | stale declaration after a formula change, saved and read back unchecked | `test_a_stale_declaration_raises_until_set_again`, `..._is_not_written`, `test_an_invalid_declaration_fails_the_read`, `test_a_stale_declaration_is_not_exported` |
| a F7 | list/str `sum` concatenated; empty domain | list/str rejected (`test_invalid_outputs`); empty domain raises and is in the release note |
| a F6 | derived/copied Space: outputs but no declaration | declaration not inherited or copied; `collect` undefined there (documented) |

## Measurements (CPython 3.13, 4 CPUs, best of 3)

| Case | This branch | Integration |
|---|---|---|
| live collect, 8,000 items x 10 steps | 2.05 s (loop with `del` 1.60 s, loop keeping 1.51 s) | 4.99 s |
| same, all 8,000 ItemSpaces exist | 0.36 s (loop 0.07 s) | 96.2 s |
| 1,200 items x 60 steps | 1.14 s (loop with `del` 0.90 s) | 1.97 s (loop 1.03 s) |
| peak memory, 8,000 x 10 (tracemalloc) | 0.7 MB (loop with `del` 0.5 MB, keeping 157.9 MB) | - |
| 800 Spaces x 20 Cells build | 8.9-9.5 s; 60,103,840 calls | 9.6-10.5 s; 80,324,843 calls |
| same, upstream main | 8.6-9.4 s; 60,103,782 calls | |

## Tests

3.13: 1962 passed, 7 skipped, 1 failed. 3.14.6t (`PYTHON_GIL=0`, `test_assets`
deselected, no QuantLib wheel): 1962 passed, 6 skipped, 1 failed. The failure is
`lifelib/test_deserializer.py`, which fails the same way on upstream main against
lifelib main 20d4dae; its fix belongs to the CI pull request. New: 62 tests.

## Old readers

Saved here, read by upstream main and by modelx 0.33.0 from PyPI, directory and
zip: no warning, `collect_outputs` is an ordinary Cells, `hasattr(P, 'collect')`
is False. Clean degradation; the serializer version stays 8.
