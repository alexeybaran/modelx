"""Opt-in, inspectable shared-helper adaptations of ordinary model formulas.

Apply to a separately loaded adjusted model. Original serialized model files
are never written. This overlay changes authoring syntax; its receipt is not a
claim that every changed expression has obtained native compiler coverage.
"""
import builtins
import hashlib

import libcst as cst
from libcst.metadata import ScopeProvider

from .typed_inputs import all_true, any_true
from . import typed_inputs as _operations


_HELPERS = {'all': ('all_true', all_true), 'any': ('any_true', any_true)}
_NAMESPACE = 'generators'


def _digest(source):
    return hashlib.sha256(source.encode('utf-8')).hexdigest()


def _rewrite(source, available):
    wrapper, bound, functions, unsupported = _lexical_source(source)
    if unsupported:
        return source, ()
    available = set(available) - bound
    if _NAMESPACE in bound:
        return source, ()
    # A nested function may receive a different global environment on export.
    if functions != 1:
        return source, ()
    used = set()

    class Rewrite(cst.CSTTransformer):
        def leave_Call(self, before, node):
            if (isinstance(node.func, cst.Name) and node.func.value in available
                    and len(node.args) == 1 and not node.args[0].keyword
                    and not node.args[0].star):
                used.add(node.func.value)
                return node.with_changes(func=cst.Attribute(cst.Name(_NAMESPACE),
                                          cst.Name(_HELPERS[node.func.value][0])))
            return node

    result = wrapper.module.visit(Rewrite()).code
    return (result, tuple(sorted(used))) if used else (source, ())


def _lexical_source(source):
    wrapper = cst.MetadataWrapper(cst.parse_module(source))
    scopes = set(wrapper.resolve(ScopeProvider).values())
    bound = {assignment.name for scope in scopes for assignment in scope.assignments}
    class Declarations(cst.CSTVisitor):
        functions = 0
        unsupported = False
        def visit_FunctionDef(self, node):
            self.functions += 1
            self.unsupported |= node.asynchronous is not None
        def visit_Lambda(self, node): self.functions += 1
        def visit_ClassDef(self, node): self.unsupported = True
    declarations = Declarations()
    wrapper.module.visit(declarations)
    return wrapper, bound, declarations.functions, declarations.unsupported


def apply_generators(model):
    """Bind existing all_true/any_true helpers and adapt builtin boolean calls.

    Return per-Cell before/after sources and hashes for review and provenance.
    Existing Cells/References named all/any prevent that adaptation; a conflicting
    generators namespace also prevents it. The module Reference serializes as an
    import, preserving formulas-only folders without embedded function pickles.
    Do not infer immutability of other
    model References or report native coverage from this authoring adjustment.
    """
    receipt = []

    def visit(space):
        available = []
        namespace = getattr(space, _NAMESPACE, _operations)
        if namespace is not _operations:
            return
        for name, (helper, function) in _HELPERS.items():
            if name in space.cells:
                continue
            current = getattr(space, name, getattr(builtins, name))
            if current is getattr(builtins, name) and getattr(_operations, helper) is function:
                available.append(name)
        for cell in list(space.cells.values()):
            if cell.formula is None:
                continue
            before = cell.formula.source
            after, used = _rewrite(before, available)
            if not used:
                continue
            setattr(space, _NAMESPACE, _operations)
            doc = cell.doc
            cell.formula = after
            if doc:
                cell.doc = doc
            receipt.append({'cell': cell.fullname, 'before': before, 'after': after,
                            'before_sha256': _digest(before), 'after_sha256': _digest(after),
                            'helpers': [_NAMESPACE + '.' + _HELPERS[name][0] for name in used],
                            'native_coverage': 'UNQUALIFIED'})
        for child in list(space.spaces.values()):
            visit(child)

    for space in list(model.spaces.values()):
        visit(space)
    return receipt


def rewrite_native_generators(source, namespace='expressions'):
    """Offline, reviewable authoring change; never a compiler recognizer.

    Supported generator clauses become the shared over/flatten vocabulary.
    Standard consumers and eager comprehensions stay authored. Escaping closure
    cells, assignment expressions and forward target bindings are declined.
    """
    from libcst.metadata import ExpressionContext, ExpressionContextProvider, PositionProvider
    from libcst.metadata.scope_provider import BuiltinAssignment
    wrapper = cst.MetadataWrapper(cst.parse_module(source))
    scopes = wrapper.resolve(ScopeProvider)
    contexts = wrapper.resolve(ExpressionContextProvider)
    positions = wrapper.resolve(PositionProvider)
    allowed_namespace_nodes = set()
    for statement in wrapper.module.body:
        if not isinstance(statement, cst.SimpleStatementLine):
            continue
        for assignment in statement.body:
            if (isinstance(assignment, cst.Assign) and len(assignment.targets) == 1
                    and isinstance(assignment.targets[0].target, cst.Name)
                    and assignment.targets[0].target.value == namespace
                    and isinstance(assignment.value, cst.Tuple)
                    and len(assignment.value.elements) == 2
                    and all(isinstance(item, cst.Element) and isinstance(item.value, cst.SimpleString)
                            for item in assignment.value.elements)
                    and [item.value.raw_value for item in assignment.value.elements]
                        == ['Module', 'modelx_helpers.expressions']):
                allowed_namespace_nodes.update((assignment, assignment.targets[0], assignment.targets[0].target))
    changes = []

    def names_read(expression):
        result = set()
        class Reads(cst.CSTVisitor):
            def visit_Name(self, node):
                if contexts.get(node) is ExpressionContext.LOAD:
                    result.add(node.value)
        expression.visit(Reads())
        return result

    def target_names(target):
        if isinstance(target, cst.Name):
            return [target.value], False
        if (isinstance(target, (cst.Tuple, cst.List))
                and len(target.elements) in (2, 3, 4)
                and all(isinstance(item, cst.Element) and isinstance(item.value, cst.Name)
                        for item in target.elements)):
            return [item.value.value for item in target.elements], True
        raise ValueError('unsupported target')

    class Rewrite(cst.CSTTransformer):
        def __init__(self):
            self.starts = []
        def visit_GeneratorExp(self, node):
            self.starts.append(len(changes))
        def leave_GeneratorExp(self, before, after):
            start = self.starts.pop()
            def decline():
                del changes[start:]
                return before

            class Safety(cst.CSTVisitor):
                generators = 0
                unsafe = False
                def visit_GeneratorExp(self, node): self.generators += 1
                def visit_Lambda(self, node): self.unsafe = True
                visit_NamedExpr = visit_Await = visit_Yield = visit_Lambda
                def visit_CompFor(self, node): self.unsafe |= node.asynchronous is not None
            if any(getattr(binding, 'node', None) not in allowed_namespace_nodes
                   for binding in scopes[before][namespace]):
                return decline()
            safety = Safety()
            before.visit(safety)
            if safety.unsafe or safety.generators != 1:
                return decline()
            parts = []
            part = before.for_in
            while part is not None:
                parts.append(part)
                part = part.inner_for_in
            zipped_sources = []
            for part in parts:
                source = part.iter
                bindings = scopes[source.func]['zip'] if (isinstance(source, cst.Call)
                    and isinstance(source.func, cst.Name) and source.func.value == 'zip') else ()
                zipped_sources.append(bool(bindings) and all(isinstance(binding, BuiltinAssignment) for binding in bindings)
                    and not any(arg.star or arg.keyword and arg.keyword.value != 'strict' for arg in source.args))
            try:
                targets = [target_names(part.target)[0] for part in parts]
                flattened = [name for group in targets for name in group]
                if len(flattened) != len(set(flattened)) or namespace in flattened:
                    return decline()
                all_targets, assigned = set(flattened), set()
                for i, part in enumerate(parts):
                    if i and names_read(part.iter) & (all_targets - assigned):
                        return decline()
                    assigned.update(targets[i])
                    if any(names_read(item.test) & (all_targets - assigned) for item in part.ifs):
                        return decline()

                def produce(part, payload, index=0):
                    names, unpack = target_names(part.target)
                    nested = part.inner_for_in is not None
                    if nested:
                        payload = produce(part.inner_for_in, payload, index + 1)
                    parameters = cst.Parameters(params=tuple(cst.Param(cst.Name(name)) for name in names))
                    source = part.iter
                    if zipped_sources[index]:
                        sources = cst.Tuple(tuple(cst.Element(arg.value) for arg in source.args if arg.keyword is None))
                        source = cst.Call(cst.Attribute(cst.Name(namespace), cst.Name('zip_values')),
                            (cst.Arg(sources), *(arg for arg in source.args if arg.keyword is not None)))
                    arguments = [cst.Arg(source), cst.Arg(cst.Lambda(parameters, payload))]
                    if part.ifs:
                        condition = cst.Name('True')
                        for item in reversed(part.ifs):
                            if isinstance(condition, cst.IfExp):
                                condition = condition.with_changes(lpar=(cst.LeftParen(),), rpar=(cst.RightParen(),))
                            condition = cst.IfExp(test=item.test, body=condition, orelse=cst.Name('False'))
                        arguments.append(cst.Arg(cst.Lambda(parameters, condition), keyword=cst.Name('where')))
                    if unpack:
                        arguments.append(cst.Arg(cst.Integer(str(len(names))), keyword=cst.Name('unpack')))
                    result = cst.Call(cst.Attribute(cst.Name(namespace), cst.Name('over')), tuple(arguments))
                    if nested:
                        result = cst.Call(cst.Attribute(cst.Name(namespace), cst.Name('flatten')), (cst.Arg(result),))
                    return result
                result = produce(after.for_in, after.elt)
            except ValueError:
                return decline()
            position = positions[before]
            changes.append({'line': position.start.line, 'column': position.start.column,
                            'before': wrapper.module.code_for_node(before),
                            'after': wrapper.module.code_for_node(result),
                            'clauses': len(parts), 'targets': targets,
                            'namespace_preexisting': bool(allowed_namespace_nodes),
                            'helpers': (['over', 'flatten'] if len(parts) > 1 else ['over'])
                                       + (['zip_values'] if any(zipped_sources) else [])})
            return result

    result = wrapper.module.visit(Rewrite()).code
    cst.parse_module(result)  # Check emitted syntax; no formula execution.
    return result, changes


def apply_native_generators(model):
    """Apply the qualified source vocabulary on a separately loaded model.

    Original consumers remain unchanged. Receipts retain complete source/hashes;
    they are authoring evidence, not a native or financial qualification claim.
    """
    from . import expressions
    receipt = []
    def visit(space):
        if 'expressions' in space.cells or getattr(space, 'expressions', expressions) is not expressions:
            return
        for cell in list(space.cells.values()):
            if cell.formula is None:
                continue
            before = cell.formula.source
            after, operations = rewrite_native_generators(before)
            if not operations:
                continue
            space.expressions = expressions
            doc = cell.doc
            cell.formula = after
            if doc:
                cell.doc = doc
            receipt.append({'cell': cell.fullname, 'before': before, 'after': after,
                            'before_sha256': _digest(before), 'after_sha256': _digest(after),
                            'operations': operations, 'native_coverage': 'UNQUALIFIED'})
        for child in list(space.spaces.values()):
            visit(child)
    for space in list(model.spaces.values()):
        visit(space)
    return receipt
