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

``Space.collect(*args)`` is a cached node of the trace graph. For each
coordinate it takes the ItemSpace as ``space[...]`` does, reads its
``collect_outputs`` and detaches the values. An ItemSpace that already
existed is kept as it is. An ItemSpace that collect created is deleted
before the next one is created; the node first takes over the
dependencies of the deleted ItemSpace on everything outside it, and
:class:`~modelx.core.itemspaces.ItemSpaceManager` clears the node for
the edits that would have cleared that ItemSpace.
"""

from collections import OrderedDict
from inspect import BoundArguments, Parameter, Signature

from modelx.core.base import Interface, Impl
from modelx.core.binding.namespace import BaseNamespace
from modelx.core.cells import Cells
from modelx.core.execution.trace import get_node, KEY
from modelx.core.formula import Formula
from modelx.core.node import NodeFactory, NodeFactoryImpl
from modelx.core.reference import ReferenceProxy
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


def _tree(space):
    """``space`` and the Spaces and ItemSpaces in it."""
    yield space
    for child in space.named_spaces.values():
        yield from _tree(child)
    for item in getattr(space, 'param_spaces', {}).values():
        yield from _tree(item)


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
        signature = space.formula.signature
        values = dict(zip(self.formula.parameters, key))
        accumulator = Accumulator()
        for coordinate in normalize_domain(self.domain_value()):
            values[self.parameter] = coordinate
            bound = BoundArguments(
                signature, OrderedDict((name, values[name]) for name in signature.parameters))
            node = get_node(space, bound.args, bound.kwargs)
            created = not space.has_node(node[KEY])
            item = space.system.executor.eval_node(node)
            done = False
            try:
                outputs = item.cells[OUTPUTS].interface()
                accumulator.add(coordinate, prepare_output(outputs, _LIVE_TYPES))
                done = True
            finally:
                if created:
                    self._release(item, node, caller, keep_dependencies=done)
        result = accumulator.result()
        self.data[key] = result
        return result

    def _release(self, item, item_node, caller, keep_dependencies):
        """Delete an ItemSpace that collect created, keeping its dependencies."""
        graph, refgraph = self.model.tracegraph, self.model.refgraph
        spaces = list(_tree(item))
        inside = set(spaces)
        nodes = {item_node}
        for space in spaces:
            for cells in space.cells.values():
                nodes.update((cells, k) for k in cells.data)
            nodes.update((space, k) for k in getattr(space, 'param_spaces', ()))

        outer_nodes, outer_refs = set(), set()
        for node in nodes:
            if node in graph:
                if keep_dependencies:
                    outer_nodes.update(
                        p for p in graph.predecessors(node) if p not in nodes)
                if graph.has_edge(node, caller):
                    graph.remove_edge(node, caller)
            if keep_dependencies and node in refgraph:
                outer_refs.update(
                    r for r in refgraph.predecessors(node)
                    if getattr(r, 'parent', None) not in inside)
        if caller in refgraph:
            for obj in list(refgraph.predecessors(caller)):
                if getattr(obj, 'parent', None) in inside:
                    refgraph.remove_edge(obj, caller)

        self.space.clear_itemspace_at(item_node[KEY])

        if keep_dependencies:
            outer_nodes.discard(caller)
            for node in outer_nodes:
                graph.add_edge(node, caller)
            for ref in outer_refs:
                refgraph.add_edge(ref, caller)
            self.bases.update(space._dynbase.idstr for space in spaces)
