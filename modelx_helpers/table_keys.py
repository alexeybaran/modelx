"""Ordered key traversal over the existing owned table representation."""
from .immutable_inputs import Record


def keys(table):
    """Return keys in input order; Table/Index owners retain tuple identity.

    Tables return row keys; ordinary Records return field names. Explicit owned
    Index Records return labels, preserving duplicates without confusing a real
    Record field named ``keys`` with an Index snapshot.
    Other representations retain the supplied function's ordinary semantics.
    """
    if isinstance(table, Record) and table._metadata != ('index',):
        return tuple(table)
    return table['keys']


keys.__modelx_native_operation__ = 'keys'
keys.__modelx_cython_helper__ = ('modelx_cython.runtime._native_indices', 'keys')
keys.__modelx_cython_call_layout__ = 'keywords'
keys.__modelx_cython_scalar_result__ = 'object'
