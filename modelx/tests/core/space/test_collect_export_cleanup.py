"""Compiler lifecycle hooks must run even when exported collection fails."""
import importlib
import sys

import modelx as mx
import pytest


def test_exported_collect_releases_on_output_and_binding_failure(tmp_path):
    model = mx.new_model('ExportCollectCleanup')
    domain = (1, 2, 3)
    model.new_space(
        'Policies', formula='lambda policy_id: None',
        refs={'policy_ids': domain}, collect_over={'policy_id': domain},
        collect_outputs='lambda: {"sum": {"v": 1 / (policy_id - 2)}}',
    )
    try:
        model.export(tmp_path / 'cleanup_export')
    finally:
        model.close()
    sys.path.insert(0, str(tmp_path))
    try:
        exported = importlib.import_module('cleanup_export').mx_model
        cls = type(exported.Policies)
        attached, released = [], []

        def attach(self, context, template):
            attached.append(self)

        def release(self, context, template):
            released.append(self)

        cls._mx_collect_attach = attach
        cls._mx_collect_release = release
        for _ in range(2):
            with pytest.raises(ZeroDivisionError):
                exported.Policies.collect()
            assert attached == released
            assert not exported.Policies._mx_collect_cache
        assert len(attached) == 4

        def fail_binding(self, **values):
            raise ValueError('binding failed')

        cls._mx_apply_formula = fail_binding
        with pytest.raises(ValueError, match='binding failed'):
            exported.Policies.collect()
        assert attached == released and len(attached) == 5
        del cls._mx_apply_formula

        def outputs(self):
            return {'sum': {'v': float(self.policy_id)}}

        cls.collect_outputs = outputs
        result = exported.Policies.collect()
        assert result.sum.v == 6.0
        assert attached == released and len(attached) == 8
        assert exported.Policies.collect() is result
        assert len(attached) == 8
    finally:
        sys.path.pop(0)
        for name in tuple(sys.modules):
            if name == 'cleanup_export' or name.startswith('cleanup_export.'):
                del sys.modules[name]
