"""Clearing a shared input may destroy a dynamic Space during graph traversal."""
import modelx as mx


def test_shared_input_clear_and_dynamic_space_clear_overlap():
    model=mx.new_model('SharedInputInvalidation')
    try:
        data=model.new_space('Data',refs={'values':{1:100.0}})
        data.new_cells('value', 'lambda policy: values[policy]')
        initial=model.new_space('Initial',formula=lambda policy: None,refs={'Data':data})
        initial.new_cells('state','lambda: Data.value(policy)')
        policy=model.new_space('Policy',refs={'Data':data,'Initial':initial})
        policy.formula="""def formula(policy):
    return {'refs': {'opening': Initial[policy].state()}}"""
        policy.new_cells('premium','lambda: Data.value(policy)')
        assert policy[1].premium()==100.0
        data.values={1:175.0}
        assert policy[1].premium()==175.0
        assert policy[1].opening==175.0
        data.values={1:250.0}
        assert policy[1].premium()==250.0
    finally:
        model.close()
