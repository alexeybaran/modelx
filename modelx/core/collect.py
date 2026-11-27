# Copyright (c) 2017-2026 Fumito Hamamura <fumito.ham@gmail.com>

# This library is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation version 3.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library.  If not, see <http://www.gnu.org/licenses/>.

"""Live ``collect()``: one Space parameter iterated over a domain.

A Space declares the parameter it is collected over, the domain (a
zero-argument Cells or a Reference of the Space, by name) and the
``collect_outputs`` Cells, a zero-argument formula that returns the
``individual`` and ``sum`` outputs of one ItemSpace.

``Space.collect(*args)`` is a cached node of the trace graph. Each
coordinate is evaluated in a private ItemSpace that shares authored
inputs with direct access. External dependencies are recorded on the
collect node. :class:`~modelx.core.itemspaces.ItemSpaceManager` also
invalidates the node when the item's templates change.
"""

from collections import OrderedDict
from inspect import BoundArguments, Parameter, Signature
from types import MethodType

from modelx.core.base import Interface, Impl
from modelx.core.binding.namespace import BaseNamespace
from modelx.core.cells import Cells, CellsImpl
from modelx.core.execution.trace import get_node, key_to_node, node_get_args, KEY
from modelx.core.formula import Formula
from modelx.core.node import NodeFactory, NodeFactoryImpl
from modelx.core.reference import ReferenceImpl, ReferenceProxy
from modelx.export._mx_collect import Accumulator, normalize_domain, prepare_output

OUTPUTS = 'collect_outputs'
_LIVE_TYPES = (Interface, Impl, BaseNamespace, ReferenceProxy)
_VARIADIC = (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD)


def check_space_formula(signature, parameter):
    """Raise unless ``parameter`` can be collected over in ``signature``."""
    params = signature.parameters
    if parameter not in params:
        raise ValueError(
            "the Space formula has no parameter %r to collect over" % parameter)
    if any(p.kind in _VARIADIC for p in params.values()):
        raise TypeError("a collected Space formula cannot take *args or **kwargs")


def parse_declaration(space, over, outputs):
    """Check ``over`` and ``outputs`` for ``space``.

    Returns ``(parameter, domain name, outputs Formula or None)``.
    ``over`` maps the one parameter to the domain: a Cells of ``space``
    or the name of a Cells or Reference of ``space``.
    """
    if not isinstance(over, dict) or len(over) != 1:
        raise TypeError("collect_over must be a dict of one parameter to its domain")
    (parameter, domain), = over.items()
    if space.formula is None:
        raise TypeError("a collected Space must have a parameter formula")
    check_space_formula(space.formula.signature, parameter)
    if isinstance(domain, Cells):
        if domain._impl.parent is not space:
            raise ValueError("the collect domain must be a Cells of %s" % space.name)
        domain = domain.name
    elif not isinstance(domain, str):
        raise TypeError(
            "the collect domain must be a Cells of the Space, or the name "
            "of a Cells or a Reference of the Space")
    formula = None
    if outputs is not None:
        formula = Formula(outputs)
        if formula.parameters:
            raise TypeError("collect_outputs must take no arguments")
    return parameter, domain, formula


class _RunnerFormula:
    """What the trace functions read from an object's ``formula``."""

    __slots__ = ('signature', 'parameters', 'source', '_bind_tails')

    def __init__(self, signature):
        self.signature = signature
        self.parameters = tuple(signature.parameters)
        self.source = "collect%s" % signature
        self._bind_tails = None     # get_node binds with the signature


class CollectNode(NodeFactory):
    """The interface of a collect node in dependency inspection."""

    __slots__ = ('_impl',)

    def __init__(self, runner):
        self._impl = runner

    def _get_repr(self, fullname=True, add_params=True):
        return self._impl.get_repr(fullname=fullname, add_params=add_params)

    def __repr__(self):
        return self._get_repr()

    @property
    def _baseattrs(self):
        return {"type": "Collect", "name": "collect",
                "parameters": self._impl.formula.parameters,
                "repr": self._get_repr()}

    def _get_attrdict(self, extattrs=None, recursive=True):
        return self._baseattrs


class _CollectFrame:
    """Private item tree and the external dependencies it observes."""

    def __init__(self, caller, nodes, refs, enclosing, names_by_code):
        self.caller = caller
        self.nodes = nodes
        self.refs = refs
        self.enclosing = enclosing
        self.names_by_code = names_by_code
        self.dynamic_cache = {}
        self.spaces = {}    # private Space -> matching Direct Space, if any
        self._visited = set()

    def owns(self, obj):
        return obj in self.spaces or getattr(obj, 'parent', None) in self.spaces

    def owner_of(self, obj):
        frame = self
        while frame is not None:
            if frame.owns(obj):
                return frame
            frame = frame.enclosing

    def owns_cache_key(self, key):
        pending = list(key)
        seen = set()
        while pending:
            value = pending.pop()
            if isinstance(value, (Interface, BaseNamespace)):
                if self.owner_of(value._impl) is not None:
                    return True
            elif isinstance(value, MethodType) and isinstance(value.__self__, CellsImpl):
                if self.owner_of(value.__self__) is not None:
                    return True
            elif type(value) is tuple or type(value) is frozenset:
                identity = id(value)
                if identity not in seen:
                    seen.add(identity)
                    pending.extend(value)
        return False

    def direct_item(self, parent, key):
        direct = self.spaces.get(parent)
        return direct.param_spaces.get(key) if direct is not None else None

    def input_source(self, obj, key):
        if isinstance(obj, CellsImpl):
            direct = self.spaces[obj.parent]
            source = direct.cells.get(obj.name) if direct is not None else None
            if source is not None and key in source.input_keys:
                return source

    def record_formula_refs(self, obj):
        # A bare global in a modelx formula doesn't pass through add_reference.
        if obj not in self._visited and hasattr(obj, '_get_referents'):
            self._visited.add(obj)
            code = obj.altfunc.__code__
            names = self.names_by_code.get(code)
            if names is None:
                names = tuple(obj._extract_globals(code))
                self.names_by_code[code] = names
            # The names are common to each copy of the formula. Their
            # Reference bindings still come from this item's namespace.
            for ref in obj._get_referents(global_names=names)['refs'].values():
                if isinstance(ref, ReferenceImpl) and self.owner_of(ref) is None:
                    self.refs.add(ref)

    def new_item(self, space, key):
        from modelx.core.dynamic import ItemSpaceImpl

        self.record_formula_refs(space)
        base, refs = space._resolve_itemspace_params(key)
        return ItemSpaceImpl(
            parent=space, base=base, name=None, refs=refs,
            arguments=node_get_args(key_to_node(space, key)), cache=None,
            frame=self, direct=space.param_spaces.get(key))

    def release(self):
        for space in reversed(self.spaces):
            space._release_collect_space()
        self.spaces.clear()
        self.dynamic_cache.clear()
        self._visited.clear()
        self.names_by_code = None


class CollectRunner(NodeFactoryImpl):
    """The declaration of a collected Space and its ``collect()`` node."""

    __slots__ = ('space', 'parameter', 'domain', 'data', 'bases',
                 'interface', '_formula_of', '_formula')

    is_cached = True

    def __init__(self, space, parameter, domain):
        self.space = space
        self.parameter = parameter
        self.domain = domain
        self.data = {}
        self.bases = set()      # idstrs of the dynbases of deleted ItemSpaces
        self.interface = CollectNode(self)
        self._formula_of = None
        self._formula = None

    @property
    def model(self):
        return self.space.model

    @property
    def formula(self):
        space_formula = self.space.formula
        if self._formula_of is not space_formula:
            params = [] if space_formula is None else [
                p for p in space_formula.signature.parameters.values()
                if p.name != self.parameter]
            self._formula = _RunnerFormula(Signature(params))
            self._formula_of = space_formula
        return self._formula

    # ----------------------------------------------------------------------
    # Declaration

    def check(self):
        """Raise if the declaration no longer fits the Space."""
        space = self.space
        prefix = "the collect declaration of %s " % space.get_repr(fullname=True)
        fix = "; call set_collect again, or set_collect(None) to remove it"
        try:
            if space.formula is None:
                raise ValueError("the Space has no parameter formula")
            check_space_formula(space.formula.signature, self.parameter)
            outputs = space.cells.get(OUTPUTS)
            if outputs is None or outputs.formula.parameters:
                raise ValueError("the Space has no zero-argument %s Cells" % OUTPUTS)
            cells = space.cells.get(self.domain)
            if cells is not None:
                if cells.formula.parameters:
                    raise ValueError("the domain Cells %s takes arguments" % self.domain)
            elif self.domain not in space.own_refs:
                raise ValueError(
                    "the domain %s is no Cells or Reference of the Space" % self.domain)
        except (TypeError, ValueError) as exc:
            raise type(exc)(prefix + "is stale: " + str(exc) + fix) from None

    def domain_value(self):
        space = self.space
        cells = space.cells.get(self.domain)
        if cells is not None:
            return cells.interface()
        ref = space.system.executor.add_reference(space.own_refs[self.domain])
        return ref.interface

    # ----------------------------------------------------------------------
    # Trace object

    def get_repr(self, fullname=True, add_params=True):
        name = self.space.get_repr(fullname=fullname, add_params=False) + ".collect"
        if add_params:
            name += "(" + ", ".join(self.formula.parameters) + ")"
        return name

    def repr_parent(self):
        return self.space.get_repr(fullname=True, add_params=False)

    def get_valuerefs(self):
        return []

    def has_node(self, key):
        return key in self.data

    def get_value_from_key(self, key):
        return self.data[key]

    def on_clear_trace(self, key):
        self.data.pop(key, None)
        self.model.refgraph.remove_with_referred((self, key))
        if not self.data:
            self.bases.clear()

    def clear(self):
        if self.data:
            self.model.clear_obj(self)

    def call(self, args, kwargs):
        self.check()
        node = get_node(self, args, kwargs)
        return self.space.system.executor.eval_node(node)

    def on_eval_formula(self, key):
        space = self.space
        caller = (self, key)
        executor = space.system.executor
        signature = space.formula.signature
        values = dict(zip(self.formula.parameters, key))
        accumulator = Accumulator()
        nodes, refs, bases = set(), set(), set()
        names_by_code = {}
        for coordinate in normalize_domain(self.domain_value()):
            values[self.parameter] = coordinate
            bound = BoundArguments(
                signature, OrderedDict((name, values[name]) for name in signature.parameters))
            item_key = get_node(space, bound.args, bound.kwargs)[KEY]
            previous_frame = executor.collect_frame
            frame = _CollectFrame(caller, nodes, refs, previous_frame, names_by_code)
            executor.collect_frame = frame
            try:
                item = frame.new_item(space, item_key)
                outputs = prepare_output(item.cells[OUTPUTS].interface(), _LIVE_TYPES)
                accumulator.add(coordinate, outputs)
                bases.update(sp._dynbase.idstr for sp in frame.spaces)
            finally:
                executor.collect_frame = previous_frame
                frame.release()
        result = accumulator.result()
        for node in nodes - {caller}:
            self.model.tracegraph.add_edge(node, caller)
        for ref in refs:
            self.model.refgraph.add_edge(ref, caller)
        self.bases.update(bases)
        self.data[key] = result
        return result
