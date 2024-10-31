import pytest


class TestImport:
    def test_import_plugin(self):
        try:
            from molfeat_kpgt.trans.kpgt_nfp import KPGTModel as KPGT
        except ImportError:
            pytest.fail("Failed to import KPGTModel from molfeat_kpgt")

        from molfeat.plugins import load_registered_plugins

        load_registered_plugins(add_submodules=True, plugins=["kpgt"])
        try:
            from molfeat.trans.pretrained import KPGTModel
        except ImportError:
            pytest.fail("Failed to import KPGTModel from molfeat")

        assert KPGT == KPGTModel, "KPGTModel is not the same class"