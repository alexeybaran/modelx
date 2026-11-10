# Explicit expression streams (development contract)

These helpers are ordinary Python functions with direct Cython `.pxd` provider
declarations. Providers never inspect generator or callback bodies. The compiler
uses its existing CST/type facts for explicitly registered numeric reductions;
other generators compile through ordinary Cython. Regional qualification belongs
to exact source pins, workloads and evidence, rather than this API declaration.

```python
sum_float(net_cf(t) * disc(t) for t in range(proj_len()))

# Optional shorthand for deliberately authored iterator operations.
sum_float(periods(proj_len()), lambda t: net_cf(t) * disc(t))
sum_float(periods(proj_len()), lambda t: claims(t),
          where=lambda t: pols_if(t) > 0.0)

# Same map/filter engine, composed explicitly.
sum_float(over(periods(proj_len()), lambda t: net_cf(t) * disc(t)))

all(over(periods(proj_len()), lambda t: check_balance(t)))
next(over(zip(ages[:-1], ages[1:]), lambda p, q: (p, q),
          where=lambda p, q: p <= age <= q, unpack=2))

# One flat contribution sequence; no sum of inner sums.
sum(flatten(over(periods(1, 6), lambda g:
    over(periods(1, karenz_months() + 1), lambda z: pols_karenz(t, g, z)))))
```


Use `from modelx_helpers import expressions as e` for one shared expression namespace.
The facade re-exports the exact registered function objects; it adds no wrappers,
registry or runtime. Preparation remains explicit in `modelx_helpers.pandas_inputs`.

```python
e.sum_float(net_cf(t) * disc(t) for t in range(proj_len()))
e.sum_float(e.periods(proj_len()),
            lambda t: claims(t) * e.lookup_float(rates, age(t), "discount"))
all(check_state(label) for label in e.keys(prepared_states()))
```

Typed lookup variants describe one lookup operation with distinct public result
contracts. `is_missing(predicate, value)` preserves the current predicate binding.
Standard Python consumers keep their ordinary semantics.

## Identity traversal (ABI 2 development extension)

The optional-mapper extension passed provider/protocol and normal-CLI checks.
It does not qualify new all-model source variants until their financial, native
coverage and matched performance checks pass. Rebuild
native packages after this `.pxd` change; cross-revision binary ABI is not promised.

```python
sum(e.over(net_cf(t) * discount(t) for t in e.periods(proj_len())))
all(e.over(abs(check_resid(t)) <= tolerance for t in e.periods(proj_len())))
tuple(e.over(e.keys(rates), where=lambda age: age <= attained_age))
```

`over(source)` creates its own fresh identity traversal. It acquires `iter(source)`
at construction and yields the original objects lazily, without a mapper callback.
A source may already be suspended; the wrapper still starts fresh. Sent values
are ignored. Throw/close affect the wrapper and release its source reference;
they are not forwarded to an independently retained source.

With only `where=`, test each source value before yielding that same object.
Unpacking requires an explicit mapper; `over(source, unpack=2)` raises before
iterator acquisition. Explicit `function=None` remains a non-callable mapper
and fails lazily on the first selected contribution. These distinctions preserve
existing callback calls and make omission an intentional identity operation.
Cython compiles an authored generator expression normally; the provider never
examines its body or introduces a per-item mapper for this form.

## Small vocabulary

| Helper | Contract |
|---|---|
| `periods(start, stop=omitted, step=omitted)` | Reusable `range` domain; one, two or three supplied arguments have `range` values, integer conversion order and zero-step behavior. Optional arguments may be named. |
| `over(source, function=omitted, where=None, *, unpack=None)` | One-shot lazy stream with standard send/throw/close protocol. Acquire `iter(source)` at construction. For each candidate, unpack if requested, evaluate the predicate, then the payload once. |
| `flatten(sources)` | One-shot lazy concatenation. Acquire the outer iterator at construction; acquire each inner iterator only when reached. |
| `zip_values(sources, *, strict=False)` | Zip an authored exact tuple of source iterables, e.g. `zip_values((ages[:-1], ages[1:]))`. It has the values, acquisition order, strict behavior, retries and retained-row lifetime of `zip(*sources, strict=strict)`. |
| `sum_float(source, function=omitted, where=None) -> float` | Lazily convert each passing contribution with standard `float`, then sum the resulting doubles using float-start summation. Empty input is float `0.0`. |
| `sum_values(source)` | Compatibility spelling for ordinary builtin `sum(source)`, including empty integer zero, arbitrary integers and custom/NumPy arithmetic. Prefer ordinary `sum` in authored formulas. |

`sum_float` is precisely `sum((float(value) for value in source), 0.0)` with
the standard builtin bindings. Its shorthand first creates the canonical
map/filter stream. It does not consult a replaced public `over` binding.
An explicitly authored `sum_float(over(...))` does consult that current binding.
`sum_float` differs from ordinary `sum` and from `sum(values, 0.0)` for some
integer prefixes, NumPy scalars, Decimal and custom numeric objects. Migration
requires proof of the original supported numeric domain; a float-looking sample
does not justify replacing an ordinary sum.

The compiled `sum_values` helper also declares a conditional `ordinary_sum`
reduction. At an already-double Cell return boundary, a compiler may reuse the
existing compensated double reducer if every contribution has a proved double
ABI. Empty integer zero then crosses the same existing return conversion.
Assignments, object-result Cells and unproved integer/mixed/custom streams retain
ordinary summation. Canonical identity/version checks and lazy replacement/error
fallbacks remain mandatory. This changes neither the public helper contract nor
the model's author-facing settings.

With only `where=`, `sum_float(source, where=eligible)` filters source values
before float conversion. Explicit `function=None` supplies a non-callable; it
does not mean omission. `periods(start=5)` is the one-bound domain 0 through 4;
`periods(stop=5)` lacks its required first bound. The named three-bound form is
`periods(start=0, stop=5, step=2)`.

Existing Python consumers remain useful: `all`, `any`, `min`, `max`, `next`,
`sorted`, `tuple`, `list`, `set`, and `dict`. Their original short circuit,
identity, tie, ordering and materialization contracts apply. Keep eager lists
eager before passing them to pandas. Their native coverage is proved separately
from the stream provider; the presence of these expressions is not a receipt.

`zip_values` is a dedicated resumable iterator, separate from `over` and
`flatten`. A transient source `StopIteration` or other exception does not close
it permanently; strict mismatch probes and retries match builtin zip. It
converts `strict` to truth once before acquiring source iterators from left to
right. Passing the same iterator twice shares its state; passing the same
sequence twice creates independent cursors. It retains acquired iterators and
one reusable result row, as builtin zip does, rather than the source-group tuple.
Exact ranges within signed 64-bit bounds and exact tuple/list sources advance
through native cursor state. Wider ranges and arbitrary iterators preserve
compatibility and require separate coverage admission. The exact tuple group is
an explicit public restriction, enforced equally in live and native execution.
Native cached-row ownership uses ordinary GIL ownership on CPython 3.12 and
the public unique-reference operation on CPython 3.14 and later. Free-threaded
CPython 3.13 is explicitly rejected at build time. The 3.14 free-threaded path
is qualified by separate native Windows exact-pin receipts. The original 3.12
provider checkpoint is historical evidence, not qualification of a later source.

```python
sum(e.over(e.zip_values((loss_days, probabilities)),
           lambda days, probability: days * probability, unpack=2))
```

Ordinary generator syntax remains valid. For a qualified regional adjustment,
`over` and `flatten` use native cursors behind standard generator execution and
a small Cython protocol compatibility adapter. The adapter preserves rejected
pre-start sends and invalid throws without closing the producer; valid execution,
throw normalization and close use the standard runtime. Legacy throw warning
origin, metaclass hooks, payload lifetime and unpacked binding lifetime are
covered by differential contracts. Upstream iterators do not receive close/throw.

The source adjustment declines nested generator objects, escaping lambda
captures, assignment expressions, asynchronous forms and unsupported target
bindings. None appears in the 520 current regional generator expressions.
Independent for-clauses retain lazy dependent inner bounds and flat value order.
The public helper binding is an explicit authoring choice; arbitrary edits to
helper implementation globals require rebuilding native artifacts.

Generator implementation objects and Cython introspection differ from CPython:
exact generator type, code/frame locals and `gi_suspended` are not promised.
These operations are outside the inventoried pricing operations; the documented
iteration/send/throw/close behavior is tested separately.

`unpack=N` is explicit because a tuple may itself be one intended callback
argument. It accepts a nonnegative exact Python integer. Unpacking runs before
the predicate, requires exactly N elements and reads at most N+1 values. Too
few/many elements produce `ValueError`, rather than callback arity errors.
Payload and predicate receive the unpacked arguments in the same order.

## Laziness, exceptions and mutable bindings

Bounds are evaluated at the authored call position. Inner dependent bounds
remain inside the outer callback. Preserve original nesting and filter order;
do not prebuild a Cartesian product or change flat summation into inner sums.
Use lambdas for on-demand Cell/Reference lookup. Passing a Cell itself as the
callback captures its callable earlier and is not always an equivalent edit.

Natural source exhaustion finishes with plain `StopIteration`, value None,
without source exception context. A payload, predicate, unpack acquisition, or
inner-iterator acquisition that raises `StopIteration` is inside the generator
body boundary: raise `RuntimeError('generator raised StopIteration')` with the
original exception as cause/context. Other exceptions preserve identity and
close that producer. Streams stay exhausted after a body failure.

Consumer-side failures do not close an externally retained suspended producer.
In particular, `sum_float` conversion belongs to its private normalization
layer. Conversion failure leaves a retained upstream `over` stream resumable;
a conversion `StopIteration` is wrapped at the normalization boundary.

All registered public dispatchers return objects and receive the current
callable first. Replacement callables receive the actual original positional
and keyword argument layout and may return any object. Only the canonical
`sum_float_kernel` has a double ABI. Neither Reference replacement nor callback
creation grants permission to freeze future Cell/Reference values.

## Actual Cython typing

An ordinary lambda calling declared typed Cells can use native Cell entries and
C numeric arithmetic, with boxing/unboxing at its callable boundary. A compiled
lambda alone is not proof. For direct scalar arithmetic needing an explicit
input type, ordinary named Cython-annotated callbacks are available:

```python
def contribution(t: cython.longlong) -> cython.double:
    return 1.0 + t * 0.5
return sum_float(periods(n), contribution)
```

The normal model exporter must preserve these declarations and the compiler's
Cell override checks. Stock Cython does not turn an arbitrary capturing lambda
into a fully unboxed C function pointer through a helper wrapper. Generated
Cython/C and runtime binary identities must substantiate each native path.

The range cursor has a header-only int64 fast domain with checked admission and
overflow-safe termination. Wide range endpoints retain their exact Python range
semantics through compatibility iteration. Arbitrary source iterators and Python
callbacks also retain compatibility behavior. These paths are tested, but do
not complete a native coverage row.

Provider/binding and generator-protocol evidence does not alone establish full
regional financial coverage or performance. Those require the separate exact-pin
census, transitive payload/domain review and matched timing receipts.
The surrounding compiler Cell return/cache ABI also remains a gate: a traced
double Cell can still coerce an object returned by a replaced helper after the
provider has preserved it. Established numeric inference may widen mixed observed
int/float results to double when their values are preserved. Explicit object
Cell declarations retain exact scalar types and empty-zero/report contracts.
Provider replacement behavior does not expand a traced numeric Cell's ABI;
provider tests alone do not qualify every composition.

Regional formulas retain their original power operators. The compiler preserves
Python result/error semantics outside its proved real-power domain, so no new
public `pow` binding or power-spelling source workaround is required. The retired
adjustment receipt remains in Git for reproducibility.
