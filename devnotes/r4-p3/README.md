# P3 patch series: stored module classes, then deterministic export keys

Only `r4/p2-modelx-collect` can be pushed, so the two P3 branches are
delivered here as `git format-patch` series. Both apply to upstream main
fec516d with `git am` and give exactly the trees of the local branches they
were made from (checked):

    git checkout -b r4/p3-modelx-stored-classes fec516d
    git am devnotes/r4-p3/stored-classes/*.patch
    git checkout -b r4/p3-modelx-deterministic-export
    git am devnotes/r4-p3/deterministic-export/*.patch

Neither contains collect or dynamic-reference code.

## `r4/p3-modelx-stored-classes` (3 patches, from upstream main)

| Patch | Source | Non-test | Tests |
|---|---|---|---|
| 0001 Save References to classes defined in user modules | 5f26ea7 | `io/moduleio.py` +22, `serialize/custom_pickle.py` +10 | `test_serializer_8.py` +93 |
| 0002 Export user modules and the classes defined in them | 0883979, conflict in `_mx_sys.py` resolved by dropping the collect runtime it sat next to | `export/_mx_sys.py` +38/-3, `export/exporter.py` +20/-1 | `test_export.py` +43 |
| 0003 Release note | new | relnotes +11 | - |

Release note (Enhancements): a Reference to a top-level class of a model
module (`new_module`), or to a value holding one such as an enum member, is
saved and exported, where both raised `PicklingError`; functions and nested
classes still raise; modelx 0.33 reads such a model with a warning and sets
those References to `None` (reviews a F8, b (3)4; run here with modelx 0.33.0
from PyPI: 3 warnings, `Plan` and `k` are `None`).

## `r4/p3-modelx-deterministic-export` (1 patch, on top of the above)

| Patch | Source | Non-test | Tests |
|---|---|---|---|
| 0001 Write deterministic keys for pickled and IOSpec values in exports | 2814cc9, release note narrowed | `export/exporter.py` +22/-8, relnotes +12 | `test_export_determinism.py` +125 |

It sits on the stored-classes series because its test model stores an enum
from a model module and `ModuleClassPickler` takes the new keys (reviews a
section 5, b (4)5). The release note no longer says "the same pickle file"
without limits (reviews a F9, b F7): it names the same Python version and
platform, and a `set`/`frozenset` of str or bytes, whose pickle follows the
hash seed. No pickle protocol was pinned, so nothing an existing user sees
changes besides the keys.

## Tests

| Tree | Python 3.13 | Python 3.14.6t, `PYTHON_GIL=0`, `test_assets` deselected |
|---|---|---|
| stored-classes | 1877 passed, 7 skipped, 1 failed | 1877 passed, 6 skipped, 1 failed |
| + deterministic-export | 1878 passed, 7 skipped, 1 failed | 1878 passed, 6 skipped, 1 failed |

The one failure in every tree is `lifelib/test_deserializer.py`, which fails
the same way on upstream main against lifelib main 20d4dae (the CI pull
request fixes it).
