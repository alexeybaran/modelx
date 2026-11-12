"""Qualified generator source forms preserve order, laziness and public protocol."""
import pytest
from modelx_helpers import expressions
from modelx_helpers.generator_models import rewrite_native_generators


def compile_pair(source,globals):
    adjusted,operations=rewrite_native_generators(source)
    left=dict(globals);right=dict(globals,expressions=expressions)
    exec(source,left);exec(adjusted,right)
    assert operations
    return left['formula'],right['formula']


def test_filter_truth_is_evaluated_once_per_original_clause():
    source='def formula(n):\n    return tuple(x for x in range(n) if first(x) if second(x))\n'
    events=[]
    class Truth:
        def __init__(self,name,value):self.name,self.value=name,value
        def __bool__(self):events.append(self.name);return self.value
    def first(x):return Truth(('first',x),x>0)
    def second(x):return Truth(('second',x),x==2)
    left,right=compile_pair(source,{'first':first,'second':second})
    assert left(4)==(2,);expected=events.copy();events.clear()
    assert right(4)==(2,) and events==expected


def test_nested_bounds_are_lazy_and_flat_order_is_preserved():
    source='def formula(n):\n    return ((x,y) for x in range(n) for y in inner(x))\n'
    events=[]
    def inner(x):events.append(x);return range(x)
    left,right=compile_pair(source,{'inner':inner})
    for formula in [left,right]:
        events.clear();stream=formula(4)
        assert events==[] and next(stream)==(1,0) and events==[0,1]
        assert list(stream)==[(2,0),(2,1),(3,0),(3,1),(3,2)]


def test_mutable_consumer_keeps_send_throw_and_exception_identity():
    source='def formula(n):\n    return consume(x for x in range(n))\n'
    failure=LookupError('injected')
    def consume(stream):
        assert stream.send(None)==0
        assert stream.send('ignored')==1
        with pytest.raises(LookupError) as error:stream.throw(failure)
        assert error.value is failure
        assert list(stream)==[]
        return 17
    left,right=compile_pair(source,{'consume':consume})
    assert left(4)==right(4)==17


def test_zipped_sources_keep_evaluation_acquisition_and_keyword_order():
    source='def formula(n):\n    return tuple(x+y for x,y in zip(left(n),right(n),strict=flag()))\n'
    events=[]
    class Source:
        def __init__(self,name,n):self.name,self.n=name,n
        def __iter__(self):events.append(('iter',self.name));return iter(range(self.n))
    class Truth:
        def __bool__(self):events.append('strict truth');return True
    def left(n):events.append('left');return Source('left',n)
    def right(n):events.append('right');return Source('right',n)
    def flag():events.append('flag');return Truth()
    original,adjusted=compile_pair(source,locals())
    assert original(3)==(0,2,4);expected=events.copy();events.clear()
    assert adjusted(3)==(0,2,4) and events==expected
    rewritten,operations=rewrite_native_generators(source)
    assert 'expressions.zip_values' in rewritten and 'zip_values' in operations[0]['helpers']


def test_shadowed_zip_is_called_at_its_original_position():
    source='def formula(zip,n):\n    return tuple(x+y for x,y in zip(range(n),range(n)))\n'
    rewritten,operations=rewrite_native_generators(source)
    assert operations and 'zip_values' not in rewritten
    original,adjusted=compile_pair(source,{})
    def replacement(a,b):return [(4,5)]
    assert original(replacement,3)==adjusted(replacement,3)==(9,)


@pytest.mark.parametrize('source',[
 'def formula(n):\n    return (lambda: x for x in range(n))\n',
 'def formula(n):\n    return ((last := x) for x in range(n))\n',
 'def formula(n):\n    return (x+y for x in range(n) for y in range(y))\n',
 'def formula(n):\n    return (x for x,x in ((1,2),))\n',
 'def formula(n):\n    return ((x+y for y in range(2)) for x in range(n))\n',
 'def formula(expressions,n):\n    return (x for x in range(n))\n',
 'def formula(n):\n    return [x for x in range(n)]\n',
])
def test_unsupported_scope_forms_and_eager_comprehensions_stay_authored(source):
    assert rewrite_native_generators(source)==(source,[])
