import unittest
import numpy as np
from simsopt.core.optimizable import Optimizable
from simsopt.core.functions import Adder, Rosenbrock

class DOFTests(unittest.TestCase):
    pass

class DOFsTests(unittest.TestCase):
    pass

class OptimizableTests(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_name(self):
        pass

    def test_hash(self):
        pass

    def test_dof_size(self):
        pass

    def test_full_dof_size(self):
        pass

    def test_local_dof_size(self):
        pass

    def test_local_full_dof_size(self):
        pass

    def test_dofs(self):
        pass

    def test_local_dofs(self):
        pass

    def test_transitive_relation(self):
        pass

    def test_parent_dof_fix(self):
        pass

    def test_call_with_arguments(self):
        pass

    def test_fix(self):
        pass

    def test_unfix(self):
        pass

    def test_fix_all(self):
        pass

    def test_unfix_all(self):
        pass

    def test_get_ancestors(self):
        pass

    def test_is_free(self):
        pass

    def test_is_fixed(self):
        pass

    def test_bounds(self):
        pass

    def test_lower_bounds(self):
        pass

    def test_upper_bounds(self):
        pass

    def test_local_lower_bounds(self):
        pass

    def test_local_upper_bounds(self):
        pass


class OldOptimizableTests(unittest.TestCase, Optimizable):
    def get_dofs(self):
        pass

    def set_dofs(self, x):
        pass

    def test_instantiation(self):
        """
        Test Optimizable.index()
        """
        with self.assertRaises(TypeError):
            o = Optimizable()

    def test_index(self):
        """
        Test Optimizable.index()
        """
        o = Adder(4)
        o.dof_names = ['foo', 'bar', 'gee', 'whiz'] # Not OK
        self.assertEqual(o.index('foo'), 0)
        self.assertEqual(o.index('bar'), 1)
        # If the string does not match any name, raise an exception
        with self.assertRaises(ValueError):
            o.index('zig')

    def test_get_set(self):
        """
        Test Optimizable.set() and Optimizable.get()
        """
        o = Adder(4)
        o.dof_names = ['foo', 'bar', 'gee', 'whiz'] # Not OK
        o.set_dof('gee', 42)
        self.assertEqual(o.get_dof('gee'), 42)
        o.set_dof('foo', -12)
        self.assertEqual(o.get_dof('foo'), -12)
        np.testing.assert_allclose(o.get_dofs(), [-12, 0, 42, 0])
        
    def test_get_set_fixed(self):
        """
        Test Optimizable.set_fixed() and Optimizable.is_dof_fixed()
        """
        o = Adder(5)
        o.dof_names = ['foo', 'bar', 'gee', 'whiz', 'arf']
        self.assertFalse(o.is_dof_fixed('gee'))
        o.fix_dof('gee')
        self.assertTrue(o.is_dof_fixed('gee'))
        o.unfix_dof('gee')
        self.assertFalse(o.is_dof_fixed('gee'))
        
if __name__ == "__main__":
    unittest.main()
