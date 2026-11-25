# R4 / S4: enum classes stored with the model

Branch `r4/s4-module-export`, from pin `636877b`. Lifelib fork `04d69db`,
compiler fork `633c9be`.

## Result in one paragraph

Two small modelx changes are enough. They implement candidates (1) and (2)
together, and the same persistent id serves both. A Reference to a class
defined in a module stored with `new_module` is now pickled as
`("ModuleAttr", <spec id>, <class name>)` instead of by import path. On load
it is taken from the module the reader has just loaded from the model folder.
The export now reads `ModuleIO`/`ModuleData`, and `_mx_pickled` uses the same
persistent id. KLV_DE_S was converted with its formulas unchanged, and
`result_cf()` (including `net_cf`), `result_cf_annual()`, `result_surplus()`
and 7 `check_*()` cells are **bit-identical** to the shipped model on all 14
model points in all three runtime situations. The compiler's translator
recognises every enum Reference and enum argument exactly as it does for the
shipped model. A compiled build gives the same results, but its 4
enum-valued table cells fail when called from Python. That gap is on the
compiler side (see "Compiler").

## What changed and why

| file | +/- | what |
|---|---|---|
| `modelx/io/moduleio.py` | +22 | `module_classes(specs)`: `{id(cls): (ModuleData, name)}` for every top-level class of each `ModuleData` module (`cls.__module__ == module.__name__ and cls.__qualname__ == name`, so aliases and imported classes are skipped). |
| `modelx/serialize/custom_pickle.py` | +10 | `ModelPickler` builds the map from `writer.model.iospecs`. `persistent_id` emits `("ModuleAttr", spec_id(spec), name)`. `ModelUnpickler.persistent_load` returns `getattr(_find_iospec(id).value, name)`. The tolerant unpickler reuses `ModelUnpickler.persistent_load`, so it is covered without changes. |
| `modelx/export/exporter.py` | +20 −1 | `DataManager.pickle` uses `ModuleClassPickler` (same persistent id, keyed by the `io_data` key `id(spec.value)`) **only when** the model has such classes. Otherwise it still calls `pickle.dump` exactly as before. |
| `modelx/export/_mx_sys.py` | +38 −3 | `MiniModuleIO`/`MiniModuleData`: loads the exported module file as `ModuleIO._load_module` does. This fixes `KeyError: 'ModuleIO'`. `DataUnpickler` resolves `ModuleAttr` against `io_data`. |
| tests | +136 | see "Tests" |

Library code: **90 lines added, 4 removed** (26 of the added lines are
docstrings and comments). The serializer version (8), the source grammar of
`__init__.py` and `_data/iospecs.py` are unchanged. A model without classes
from user modules gives byte-identical output: no persistent id is emitted,
and the export takes the `pickle.dump` path.

Why this is the smallest design. The class is already reachable from data
that the model stores as text: the module's source in the model folder, plus
its `iospecs.py` entry. `read_pickledata` restores the specs (and loads the
module) before it unpickles `data.pickle`. The only missing piece was a way
for the pickle to point at "class X of spec k" rather than at an import
path, and a persistent id is the existing mechanism for exactly that
(`DataValue`, `BaseIOSpec`, `BaseSharedIO`). Because it lives in the
pickler and not at the ref site, it also covers classes and members nested
in pickled values, for example a `{Plan.T10: ...}` dict Reference. That
case is tested.

### Decisions taken (conservative) and alternatives not taken

- **New persistent-id shape, no serializer version bump.** Before this
  change such a model could not be saved at all (`PicklingError`), so no
  file that exists today changes. An older modelx reading the new file
  degrades the same way the shipped form does without lifelib. Measured
  with the unmodified pin on the converted KLV_DE_S: warnings, the 11 class
  References load as `None`, and the `enums` module Reference loads.
  Alternative: bump to serializer 9 so that older readers refuse the model
  cleanly. Not done, because it would change the format of every model to
  serve a case that previously failed.
- **Top-level classes only**, matched by identity in the module. Not done:
  functions or nested classes (a qualname walk would be about 3 more lines).
- Alternative ref-site form `PolsTiming = ("ModuleAttr", 1, "PolsTiming")`
  in `__init__.py`. It is more readable, but it changes the source grammar,
  an older reader's `LiteralDecoder` would silently take the tuple as the
  value, and it does not cover nested values. Rejected.
- Alternative: register stored modules in `sys.modules` under unique names
  so that plain pickle works. It adds global state, risks name collisions
  across models, and changes behaviour for every `new_module` user. Rejected.
- Alternative for the export: emit `self.PolsTiming = io_data[k].PolsTiming`
  in `_mx_classes.py` instead of pickling. It is more readable but covers
  only direct References. Rejected for one mechanism in both places.
- `update_module` (live) rebinds References that hold the old *module*, but
  not References that hold its *classes*. Those keep the old class until
  re-read. This is unchanged here; it is a possible follow-up.

## Four situations — KLV_DE_S

Reference values come from the shipped model (lifelib importable) and are
compared with `pd.testing.assert_frame_equal(check_exact=True)` and `==` on
the checks. Sum of `net_cf` over the book is −148382.620382 in every row.

| situation | how | result |
|---|---|---|
| live | `convert.py`: read shipped model, `new_module`, rebind 11 References, recompute | 14/14 bit-identical |
| saved, copied, reloaded **without lifelib** | patched folder copied out of the lifelib tree, package `enums.py` deleted, `find_spec("lifelib") is None` asserted (`copied.py`) | 14/14 bit-identical. `Projection.PolsTiming is model.enums.PolsTiming` |
| … edit to the copy's `enums.py` | added `EDITED = 7` to `PolsTiming` in the copy, reloaded | `Projection.PolsTiming.EDITED == <PolsTiming.EDITED: 7>`, members now `BEF_DECR, AFT_MORT, AFT_LAPSE, EDITED`; results still 14/14 identical |
| exported, imported as pure Python | `model.export("exported")`, imported with neither modelx nor lifelib importable (`exported.py`) | 14/14 bit-identical. `_mx_io.py` lists `enums.py` as `ModuleIO`/`ModuleData` |
| compiler-recognisable | translate-only run on the export (below) | all enum References and enum arguments recognised, same as the shipped model. A build also ran: results 14/14 identical; the 4 enum-valued table cells fail when called from Python (compiler gap) |

Contrast, the shipped form copied without lifelib: `Reference 'PolsTiming' …
could not be restored and is set to None` (for all 11), and `result_cf()`
raises `FormulaError`.

lifelib's own `delib/tests/test_kapitallebensversicherung_de.py` gives
100 passed on the patched tree and 100 passed on the pristine tree.

### Compiler

Exported `_mx_classes.py` (Projection):

```
        self.enums = io_data[140554989594368]
        self.PolsTiming = pickle_data[768859840]
        self.Kind = pickle_data[781322272]
...
        if timing == self.PolsTiming.BEF_DECR:
        if kind == self.Kind.DEATH:
```

This is the same form as an export of the shipped model. A diff of the two
`_mx_classes.py` files, with object ids masked, differs only in the added
`enums` Reference (assignment, copy, slot). The compiler's discovery that
these lines satisfy:

- `modelx_cython/tracer.py` `RuntimeValueInfo.__init__`: `enum_safe =
  isinstance(value, type) and issubclass(value, IntEnum) and type(value) is
  type(IntEnum) and value.__eq__ is int.__eq__ ...`, with `enum =
  (value.__module__, value.__qualname__, {members})`. The traced value of
  `self.PolsTiming` is the IntEnum class, which holds here.
- `modelx_cython/enums.py` `prepare()` selects References with `ref.enum and
  ref.enum_safe`. `Constants.member_ref()` accepts `self.<Ref>.<member>`
  when `(cls, Ref, member)` is in the symbol table.
- `tracer.py` `_widen_to_enum`: an argument whose traced type is an
  `IntEnum` subclass (qualname without `<locals>`) gets the member domain.

Translate-only run (`python -m modelx_cython exported --sample
sample_nocollect.py --spec spec.py --translate-only`), converted and
shipped export side by side:

| | converted | shipped |
|---|---|---|
| exit | 0 | 0 |
| `cdef enum` constants in `_mx_classes.pxd` | 32 | 32 |
| `_mx_classes.pxd` | identical except `cdef public object enums` (×2) | |
| `_mx_check_enum_schema` bindings | Data `('Sex','PremFreq')`, Projection 9 names | same |
| `_mx_admission.json` cells with `enum_domains` | 4: `pols_if_at.timing` → `<unnamed module>.PolsTiming (3 members)`, `res_pp_at.timing` → `ResTiming`, `av_sur_pp_at.timing` → `SurTiming`, `claims.kind` → `Kind (4 members)` | the same 4, named `lifelib….enums.X` |
| native enum constants / comparisons | 32 / 19 | 32 / 19 |

`sample.py` at 04d69db calls `item.collect_outputs()`, which this
KLV_DE_S does not define. That is true for the shipped model too, and it is
unrelated to this change. I ran a copy of the sample without that line
(`sample_nocollect.py`).

**Compiler gap (not fixed; the compiler fork is out of scope):** the
translated code returns enum-valued table cells to Python through
`_mx_table_cells.enum_member('<unnamed module>', 'Sex', ...)`.
`runtime/table_cells.py:enum_member` resolves the class with
`importlib.import_module(module)`, and `'<unnamed module>'` cannot be
imported. Calling it directly raises `ModuleNotFoundError: No module named
'<unnamed module>'`. The shipped model has the mirror dependency: its
compiled form imports `lifelib….enums` at run time. The fix belongs in the
compiler: resolve the class through the enum Reference
(`self._mx_definition_<Ref>`, already stored) rather than by import path.
Also, `enums._enum_key` keys enums by `(module, qualname)`, which would
collide for two stored modules that define the same class name, since all
stored modules are named `<unnamed module>`.

**Build (done after all):** `--compile-only` on the converted export built
(exit 0, 4 extension modules). The compiled package was run with lifelib
not importable (`r4-s4/compiled.py`). `result_cf`, `result_cf_annual`,
`result_surplus` and the 7 checks are **14/14 bit-identical** to the
shipped live model. As predicted, `Projection[1].sex()`, `smoker()`,
`prem_freq()` and `surplus_use()` raise `ModuleNotFoundError: No module
named '<unnamed module>'`. These four cells return an enum member to a
Python caller; nothing inside the projection calls them that way.

## Tests

`python -m pytest modelx/tests/serialize modelx/tests/export modelx/tests/io -n 8`,
run with scipy installed and upstream lifelib 0.17.3 on the path, which
`tests/export/test_lifelib.py` needs:

| | passed | failed | skipped |
|---|---|---|---|
| before (636877b) | 543 | 1 | 3 |
| after (this branch) | 548 | 1 | 3 |

The one failure in both runs is `test_lifelib.py::test_assets`
(`ModuleNotFoundError: QuantLib`, an environment dependency that is not
installed). The 5 new tests fail on the pin and pass on the branch:

- `serialize/test_serializer_8.py::test_module_class_ref_roundtrip[write|zip]`:
  class References and a dict keyed by members survive write → read →
  write → read, and stay identical to the module's classes.
- `…::test_module_class_ref_follows_edited_module`: a copied model with an
  edited module file gets the edited class.
- `…::test_module_class_ref_missing_class`: a class removed from the file
  degrades like any lost pickled value (a warning, the Reference becomes
  `None`, the rest loads).
- `export/test_export.py::test_module_and_module_class`: the module source
  is exported, `self.Plan.T20` appears in `_mx_classes.py`, and the package
  imports with the classes and members resolved.

## Recipe: convert one lifelib model's enums to the stored form

With lifelib importable (the shipped References are pickled by lifelib
import path), see `r4-s4/convert.py`:

```python
import enum, modelx as mx
m = mx.read_model(folder / "KLV_DE_S")
m.new_module(name="enums", path="enums.py", module=folder / "enums.py")   # copy of package enums.py
for space in m.spaces.values():                          # recurse if the model nests spaces
    for name, value in list(space._own_refs.items()):
        if isinstance(value, type) and issubclass(value, enum.Enum) and value.__module__.endswith(".enums"):
            setattr(space, name, getattr(m.enums, value.__qualname__))
m.write(out / "KLV_DE_S")
```

Formulas are untouched and keep `PolsTiming.BEF_DECR`. Check that no
Reference still holds a class from the package module (the loop rebinds all
of them). Then, to keep the hand-written section comments that a modelx save
drops from space sources:

1. Copy `KLV_DE_S/enums.py`, `_data/iospecs.py`, `_data/data.pickle` and
   the model `__init__.py` from the saved output. The new `__init__.py`
   gains `enums = ("IOSpec", 1)  # ModuleData path='enums.py'`.
2. In `Data/__init__.py` and `Projection/__init__.py`, replace only the
   `X = ("Pickle", n)` lines with the saved ones. The ids shift by one
   because the spec takes id 1.

`r4-s4/KLV_DE_S-enums-stored.patch` is the result for KLV_DE_S (a `git diff
--binary` against lifelib `04d69db`, 6 files, +75 −11; it applies cleanly
with `git apply`). It keeps the package `kapitallebensversicherung/enums.py`,
because `delib/tests/test_kapitallebensversicherung_de.py` imports from it
(the tests pass either way: IntEnum members compare and hash as their ints).
Deleting it is a follow-up for once the tests take `model.enums`. Until
then there are two copies of the source, and the model folder's copy is the
one that counts.

Scripts in `r4-s4/`: `results.py` (collect and compare), `shipped.py`,
`convert.py`, `copied.py`, `exported.py`, `compiled.py`.

## Unverified

- A compiled build of the **shipped** form, for comparison. Not built; its
  `enum_member` imports `lifelib….enums`, so it needs lifelib at run time.
- Compiled, stored form: results verified. The 4 enum-valued table cells
  fail when called from Python until the compiler resolves enum classes
  through the Reference (see "Compiler").
- Other lifelib models. Only KLV_DE_S was converted. Models with nested
  spaces need the loop to recurse.
- `update_module` on a model with class References (live): the class
  References keep the old class until the model is re-read.
- Classes defined in a module in a non-top-level position (nested classes),
  and functions: not covered by `module_classes`.
