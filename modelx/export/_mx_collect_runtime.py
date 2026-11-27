"""Source sections added to the standalone runtime for Collect exports."""

COLLECT_IMPORTS = b'from types import BuiltinMethodType, MethodType\n'

COLLECT_HELPERS = br'''# The versions of the collect hook this runtime calls. See the page
# "collect hook" of the modelx reference: a compiled Space class takes
# over collect() by defining _mx_collect_hook and _mx_collect_hook_version.
COLLECT_HOOK_VERSIONS = (1,)

# The cache key, for the same arguments, of the domain tuple a result was last read with
_SAME_DOMAIN = object()


def release_collect_item(item):
    """Break the links that keep a private ItemSpace tree alive."""
    pending = [item]
    owned = set()
    candidates = []
    while pending:
        space: BaseSpace = pending.pop()
        owned.add(id(space))
        names = getattr(sys.modules[type(space).__module__], '_v_collect_refs_' + space._name, ())
        if names:
            candidates.append((space, names))
        pending.extend(space._mx_spaces.values())
        space._mx_spaces.clear()
        try:
            nested = space._mx_itemspaces
        except AttributeError:
            nested = None
        if nested is not None:
            pending.extend(nested.values())
            nested.clear()
        space._mx_cells.clear()
        space._mx_roots.clear()
        space._space = None
        space._parent = None

    for space, names in candidates:
        for name, is_cell in names:
            value = getattr(space, name, None)
            if is_cell:
                if not isinstance(value, (MethodType, BuiltinMethodType)):
                    continue
                value = value.__self__
            if id(value) in owned:
                setattr(space, name, None)


def collect_from_space(space, domain, item, args, cache):
    """Evaluate ``collect()`` of an exported Space.

    ``domain`` is the value of the domain, ``item(c)`` a private ItemSpace
    at coordinate ``c``, ``args`` the other Space parameters in signature
    order, and ``cache`` the Space's dict of results.

    A result is read again without normalizing the domain when the arguments
    are equal and the domain is the very tuple object it was last read with:
    a tuple cannot change, so its coordinates are those of the result.
    """
    from ._mx_collect import Accumulator, CollectResult, normalize_domain, prepare_output

    same = (_SAME_DOMAIN, args)
    if type(domain) is tuple:
        held = cache.get(same)
        if held is not None and held[0] is domain:
            return held[1]
    coords = normalize_domain(domain)
    key = (coords, args)
    result = cache.get(key)
    if result is not None:
        if type(domain) is tuple:
            cache[same] = (domain, result)
        return result

    cls = type(space)
    hook = getattr(cls, '_mx_collect_hook', None)
    if hook is not None:
        version = getattr(cls, '_mx_collect_hook_version', None)
        if type(version) is not int or version not in COLLECT_HOOK_VERSIONS:
            raise RuntimeError(
                "%s.%s defines _mx_collect_hook version %r; this exported "
                "model calls version %s" % (
                    cls.__module__, cls.__qualname__, version,
                    ', '.join(map(str, COLLECT_HOOK_VERSIONS))))
        result = hook(space, coords, args)
        if result is not None and type(result) is not CollectResult:
            raise TypeError(
                "_mx_collect_hook must return a CollectResult of _mx_collect "
                "or None, not %s" % type(result).__name__)

    if result is None and hook is None:
        workers = getattr(space._model, '_mx_collect_workers', 1)
        if type(workers) is not int or workers < 1:
            raise ValueError('_mx_collect_workers must be a positive int, not %r' % (workers,))
        if workers > 1 and not getattr(sys, '_is_gil_enabled', lambda: True)():
            from ._mx_collect import _collect_once
            result = _collect_once(space, coords, item, cache, key, workers)

    if result is None:
        accumulator = Accumulator()
        for coordinate in coords:
            root = item(coordinate)
            try:
                outputs = prepare_output(root.collect_outputs(), (BaseMxObject,))
            finally:
                release_collect_item(root)
                del root
            accumulator.add(coordinate, outputs)
        result = accumulator.result()
    result = cache.setdefault(key, result)
    if type(domain) is tuple:
        cache[same] = (domain, result)
    return result


'''
