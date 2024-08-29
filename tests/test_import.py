import unittest as ut
import pytest



class TestImport(ut.TestCase):
    def test_import_plugin(self):
        try:
            from molfeat_kpgt.calc import KPGTDescriptors as KPGT
        except ImportError:
            pytest.fail("Failed to import KPGTDescriptors from molfeat_kpgt")

        # load_registered_plugins()
        # try:
        #     from molfeat.calc import KPGTDescriptors
        # except ImportError:
        #     pytest.fail("Failed to import KPGTDescriptors from molfeat")

        # assert KPGT == KPGTDescriptors, "KPGTDescriptors is not the same class"

if __name__ == "__main__":
    ut.main()