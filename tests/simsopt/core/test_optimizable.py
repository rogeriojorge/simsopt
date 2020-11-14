import unittest
import numpy as np
from simsopt.core.optimizable import Optimizable
from simsopt.core.functions import Adder

class OptimizableTests(unittest.TestCase, Optimizable):
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
