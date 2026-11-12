Model helpers
=============

The modelx distribution ships the standalone ``modelx_helpers`` package.
Import its modules explicitly, for example::

    from modelx_helpers import pandas_inputs as inputs
    from modelx_helpers import expressions as e

These are ordinary Python functions in live modelx and exported Python models.
Importing the expression API does not load the modelx engine, pandas, NumPy,
lifelib or modelx-cython. Input preparation through ``pandas_inputs`` requires
the optional dependencies installed with ``pip install modelx[helpers]``.

When compiling, modelx-cython recognizes the same functions' declarative
metadata and installs its native source resources from
``modelx_cython.runtime`` into each generated package. No lifelib installation
is required to use these generic helpers.

* :download:`Table API contracts <NATIVE_TABLES.md>`
* :download:`Expression and generator contracts <EXPRESSION_STREAMS.md>`

Prototype models that reference ``lifelib`` helper modules must update their
Module References to ``modelx_helpers`` and be exported and compiled again.
Previously serialized helper objects also require recreation under the new
canonical module identities; duplicate compatibility classes are not installed.
