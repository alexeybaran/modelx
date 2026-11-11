"""Explicit lazy expression helpers; Cython providers never inspect formulas."""

from .typed_inputs import all_true, sum_f64


_DOMAIN_MISSING = object()
_NO_CALLBACK = object()


def periods(start, stop=_DOMAIN_MISSING, step=_DOMAIN_MISSING):
    """A reusable period domain, with range's values and index conversions."""
    if step is not _DOMAIN_MISSING:
        return range(start, stop, step)
    if stop is not _DOMAIN_MISSING:
        return range(start, stop)
    return range(start)


def zip_values(sources, *, strict=False):
    """Zip an authored tuple of iterables, preserving ordinary zip semantics.

    The tuple groups the sources; it does not materialize their values. Iterator
    acquisition is eager and ordered. Advancement remains lazy and ordered.
    """
    if type(sources) is not tuple:
        raise TypeError('zip_values requires a tuple of source iterables')
    return zip(*sources, strict=strict)


def _validate_unpack(count):
    if count is not None and (type(count) is not int or count < 0):
        raise ValueError('unpack must be None or a nonnegative integer')


def _unpack(value, count):
    # Fixed arities use Python's unpack primitive, including exact errors and
    # bounded consumption. Other explicit arities retain the same N+1 limit.
    if count == 2:
        first, second = value
        return first, second
    if count == 3:
        first, second, third = value
        return first, second, third
    if count == 4:
        first, second, third, fourth = value
        return first, second, third, fourth
    iterator = iter(value)
    result = []
    for position in range(count):
        exhausted = False
        try:
            result.append(next(iterator))
        except StopIteration:
            exhausted = True
        if exhausted:
            raise ValueError(f'not enough values to unpack (expected {count}, got {position})')
    exhausted = False
    try:
        next(iterator)
    except StopIteration:
        exhausted = True
    if not exhausted:
        raise ValueError(f'too many values to unpack (expected {count})')
    return tuple(result)



def _stream_generator(cursor):
    try:
        while True:
            try:
                cursor.pending = next(cursor)
            except StopIteration:
                return
            yield _take_pending(cursor)
    finally:
        cursor.close()


def _take_pending(cursor):
    value = cursor.pending
    cursor.pending = None
    return value



class _InputCursor:
    def __init__(self, source, unpack):
        _validate_unpack(unpack)
        self.source = iter(source)
        self.unpack = unpack
        self.closed = False

    def __iter__(self):
        return self

    def close(self):
        self.closed = True
        self.source = None

    def __next__(self):
        if self.closed:
            raise StopIteration
        exhausted = False
        try:
            value = next(self.source)
        except StopIteration:
            exhausted = True
        except BaseException:
            self.close()
            raise
        if exhausted:
            self.close()
            raise StopIteration
        if self.unpack is None:
            return value
        try:
            return _unpack(value, self.unpack)
        except StopIteration as error:
            self.close()
            raise RuntimeError('generator raised StopIteration') from error
        except BaseException:
            self.close()
            raise



# Private compatibility name used by cross-provider profiling oracles.
_Over = _InputCursor


class _ProtocolGenerator:
    def __init__(self, generator):
        self.generator = generator
        self.started = False

    def __iter__(self):
        return self

    def __next__(self):
        self.started = True
        return next(self.generator)

    @property
    def send(self):
        return self.generator.send

    @property
    def throw(self):
        return self.generator.throw

    @property
    def close(self):
        return self.generator.close



def _mapped_generator(cursor, function, where, unpack):
    if unpack is None:
        return (function(value) for value in cursor if where is None or where(value))
    if unpack == 2:
        return (function(a, b) for a, b in cursor if where is None or where(a, b))
    if unpack == 3:
        return (function(a, b, c) for a, b, c in cursor if where is None or where(a, b, c))
    if unpack == 4:
        return (function(a, b, c, d) for a, b, c, d in cursor if where is None or where(a, b, c, d))
    return (function(*arguments) for arguments in cursor if where is None or where(*arguments))


def _identity_generator(cursor, where):
    try:
        while True:
            try:
                cursor.pending = next(cursor)
            except StopIteration:
                return
            if where(cursor.pending):
                yield _take_pending(cursor)
            else:
                _take_pending(cursor)
    finally:
        cursor.close()


def over(source, function=_NO_CALLBACK, where=None, *, unpack=None):
    """Traverse lazily, optionally filtering and mapping each source value.

    Omitting function retains source values, including their identities. The
    stream owns its fresh generator protocol; it does not forward send, throw
    or close to an externally retained source. Unpacking requires a mapper.
    """
    if function is _NO_CALLBACK:
        _validate_unpack(unpack)
        if unpack is not None:
            raise TypeError('unpack requires a mapping function')
        cursor = _InputCursor(source, None)
        generator = (_stream_generator(cursor) if where is None else
                     _identity_generator(cursor, where))
        return _ProtocolGenerator(generator)
    return _ProtocolGenerator(_mapped_generator(_InputCursor(source, unpack), function, where, unpack))


class _Flatten:
    def __init__(self, sources):
        self.outer, self.inner = iter(sources), None
        self.closed = False

    def __iter__(self):
        return self

    def close(self):
        self.closed = True
        self.outer = self.inner = None

    def __next__(self):
        if self.closed:
            raise StopIteration
        while True:
            if self.inner is not None:
                exhausted = False
                try:
                    return next(self.inner)
                except StopIteration:
                    exhausted = True
                except BaseException:
                    self.close()
                    raise
                if exhausted:
                    self.inner = None
            exhausted = False
            try:
                source = next(self.outer)
            except StopIteration:
                exhausted = True
            except BaseException:
                self.close()
                raise
            if exhausted:
                self.close()
                raise StopIteration
            try:
                self.inner = iter(source)
            except StopIteration as error:
                self.close()
                raise RuntimeError('generator raised StopIteration') from error
            except BaseException:
                self.close()
                raise


def flatten(sources):
    """Lazily traverse each inner iterable in one flat, ordered sequence."""
    return _ProtocolGenerator(_stream_generator(_Flatten(sources)))


def sum_values(source):
    """Ordinary sum: empty integer zero and Python's numeric transitions."""
    return sum(source)


def _FloatTerms(source):
    converter = _FLOAT
    return (converter(value) for value in source)



def sum_float(source, function=_NO_CALLBACK, where=None) -> float:
    """Sum lazily float-converted contributions with float-start arithmetic.

    With a callback, this is the shorthand for sum_float(over(source, function,
    where)). It deliberately differs from ordinary mixed-type sum; migration
    must preserve the original source's numeric and empty-result contract.
    """
    if function is not _NO_CALLBACK:
        source = _OVER(source, function, where)
    elif where is not None:
        source = _OVER(source, lambda value: value, where)
    return sum(_FloatTerms(source), 0.0)


# Private original identities survive replacement of public helper bindings.
_FLOAT = float
_PERIODS, _OVER, _FLATTEN = periods, over, flatten
_ZIP_VALUES = zip_values
_SUM_VALUES, _SUM_FLOAT = sum_values, sum_float
for _function in (_PERIODS, _OVER, _FLATTEN, _SUM_VALUES, _SUM_FLOAT, _ZIP_VALUES):
    _function.__modelx_cython_helper__ = ('modelx_cython.runtime._native_streams', _function.__name__)
    _function.__modelx_native_operation__ = _function.__name__
    _function.__modelx_cython_scalar_result__ = 'object'
    _function.__modelx_cython_call_layout__ = 'keywords'

# The compiler may reuse its double reducer only at an already-double return
# boundary with proven double contributions. The general helper remains ordinary
# Python sum, including integer zero, arbitrary integers and mixed arithmetic.
_SUM_VALUES.__modelx_cython_reduction__ = 'ordinary_sum'
