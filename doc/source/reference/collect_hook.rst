.. _collect-hook:

The collect hook of exported models
===================================

A Space that declares ``collect()``
(:meth:`~modelx.core.space.UserSpace.set_collect`) is exported with a
``collect()`` method. That method evaluates temporary ItemSpaces one at a
time in Python. Direct access and Collect use the explicitly assigned Cells
inputs captured at export. A compiler that builds an extension from the
exported package can take over the evaluation, for example to run the items in parallel,
through one hook described on this page. modelx itself never defines the
hook. Live modelx has no hook: only exported models call it.

Version 1 is the only version.

What the compiled class provides
--------------------------------

The hook is looked up on the class of the exported Space object, the
class that the package's ``_mx_classes.py`` defines for that Space, or a
class that a compiler puts in its place. The class defines two
attributes:

``_mx_collect_hook_version``
    The int ``1``.

``_mx_collect_hook(self, coords, args)``
    A method. ``self`` is the Space object on which ``collect()`` was
    called. ``coords`` is the domain as a tuple of coordinates, in domain
    order. ``args`` is the tuple of the other parameters of the Space
    formula, in signature order, with defaults applied: for a Space
    formula ``lambda point_id, scen=1: ...`` collected over ``point_id``,
    ``collect()`` passes ``(1,)`` and ``collect(scen=2)`` passes ``(2,)``.

    It returns either an instance of ``CollectResult`` from the package's
    ``_mx_collect`` module, holding the values that modelx would compute
    (see below), or :obj:`None` to leave the evaluation to modelx. An
    exception propagates to the caller of ``collect()``.

The generated ``_mx_collect_item(*full_args)`` constructs a private item
with its exported inputs. Supply the arguments in the original Space
signature order. Release the returned item with ``_mx_sys.release_collect_item``
after detaching its output, including when evaluation or output preparation
raises.

What the result holds
---------------------

The result is the one that ``_mx_collect.Accumulator`` builds when the
item at each coordinate ``c`` is read in domain order, as modelx does::

    accumulator = Accumulator()
    for c in coords:
        outputs = prepare_output(item_at(c).collect_outputs(), (BaseMxObject,))
        accumulator.add(c, outputs)
    result = accumulator.result()

That is, ``result.individual`` (when ``collect_outputs`` returns an
``'individual'`` section) maps each coordinate, in domain order, to a
``CollectResult`` of that item's individual outputs, and ``result.sum``
(when it returns a ``'sum'`` section) is a ``CollectResult`` of the sums.
modelx adds a sum as ``total = total + value``, item after item, so a
floating-point sum that a hook adds in another order can differ in its
last bits; whether that is acceptable is the compiler's decision to
document.

What modelx guarantees
----------------------

* ``collect()`` calls the hook only when its cache has no result for
  ``(coords, args)``, after it has evaluated the domain and checked that
  it is a non-empty sequence of distinct, hashable coordinates. A domain
  that fails the check raises before the hook is called.
* It calls the hook at most once per such call, on the thread that
  called ``collect()``. modelx starts no threads. Calls from several
  threads at once can call the hook at once; the first result stored
  wins and is returned to every later call.
* A ``CollectResult`` that the hook returns is stored in the cache and
  returned as it is. :obj:`None` makes modelx evaluate the items itself,
  and that result is stored. Nothing is stored when the hook raises.
* An ItemSpace has no ``collect``: reading it raises
  :obj:`AttributeError`, as in live modelx.

When the hook is absent or unknown
----------------------------------

* No ``_mx_collect_hook`` on the class: modelx evaluates the items, as
  for any exported model.
* ``_mx_collect_hook`` with a ``_mx_collect_hook_version`` that is missing
  or not in ``_mx_sys.COLLECT_HOOK_VERSIONS`` (``(1,)`` in this version of
  modelx): ``collect()`` raises :obj:`RuntimeError` naming the class and
  both versions. modelx does not fall back to its own evaluation, because
  a hook of another version may expect other arguments or return
  other values.
* Any other return value than :obj:`None` or a ``CollectResult``:
  ``collect()`` raises :obj:`TypeError` and stores nothing.

A later version of the hook gets a new number; a runtime that can call it
lists that number in ``COLLECT_HOOK_VERSIONS``.
