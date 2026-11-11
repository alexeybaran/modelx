"""Unary scalar math operations with ordinary Python reference behavior."""
from math import erf as _erf, exp as _exp, isinf as _isinf, log as _log, sqrt as _sqrt


def erf(value):
    return _erf(value)


def exp(value):
    return _exp(value)


def isinf(value):
    return _isinf(value)


def log(value):
    return _log(value)


def sqrt(value):
    return _sqrt(value)


for _function, _result in ((erf, 'float64'), (exp, 'float64'), (isinf, 'bool'),
                           (log, 'float64'), (sqrt, 'float64')):
    _function.__modelx_cython_helper__ = (
        'modelx_cython.runtime._native_math', _function.__name__)
    _function.__modelx_cython_scalar_result__ = _result
