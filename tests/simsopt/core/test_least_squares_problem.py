import unittest
import logging
from simsopt.core.new_functions import Identity, Rosenbrock
from simsopt.core.optimizable import Target
from simsopt.core.least_squares_problem import LeastSquaresProblem, LeastSquaresTerm

#logging.basicConfig(level=logging.DEBUG)

class LeastSquaresTermTests(unittest.TestCase):


    def test_single_value_func_in(self):
        iden = Identity()
        lst = LeastSquaresTerm.from_sigma(iden, 3, 0.1)

        iden.x = [17]
        correct_value = ((17 - 3) / 0.1) ** 2
        self.assertAlmostEqual(lst(), correct_value, places=11)

        iden.x = [0]
        term1 = LeastSquaresTerm.from_sigma(iden, 3, 2)
        self.assertAlmostEqual(term1(), 2.25)

        term1.x = [10]
        self.assertAlmostEqual(term1(), 12.25)
        self.assertAlmostEqual(term1(x=[0]), 2.25)
        self.assertAlmostEqual(term1(x=[10]), 12.25)

    def test_exceptions(self):
        """
        Test that exceptions are thrown when invalid inputs are
        provided.
        """
        iden = Identity()

        # sigma cannot be zero
        with self.assertRaises(ValueError):
            lst = LeastSquaresTerm.from_sigma(iden, 3, 0)

        # Weight cannot be negative
        with self.assertRaises(ValueError):
            lst = LeastSquaresTerm(iden, 3, -1.0)

    def test_multiple_funcs_single_input(self):
        iden1 = Identity(x=10)
        iden2 = Identity()
        # Objective function
        # f(x,y) = ((x - 3) / 2) ** 2 + ((y + 4) / 5) ** 2
        term = LeastSquaresTerm.from_sigma([iden1, iden2], [3, -4], [2, 5])
        self.assertAlmostEqual(term(), 12.89)
        term.x = [5, -7]
        self.assertAlmostEqual(term(), 1.36)
        self.assertAlmostEqual(term([10, 0]), 12.89)
        self.assertAlmostEqual(term([5, -7]), 1.36)

    def test_parent_dof_transitive_behavior(self):
        iden1 = Identity()
        iden2 = Identity()
        term = LeastSquaresTerm.from_sigma([iden1, iden2], [3, -4], [2, 5])
        iden1.x = [10]
        self.assertAlmostEqual(term(), 12.89)

    def test_least_squares_combination(self):
        iden1 = Identity()
        iden2 = Identity()
        term1 = LeastSquaresTerm.from_sigma(iden1, 3, 2)
        term2 = LeastSquaresTerm.from_sigma(iden2, -4, 5)
        term = term1 + term2
        iden1.x = [10]
        self.assertAlmostEqual(term(), 12.89)

class LeastSquaresProblemTests(unittest.TestCase):

    def test_supply_LeastSquaresTerm(self):
        """
        Test basic usage
        """
        # Objective function f(x) = ((x - 3) / 2) ** 2
        iden1 = Identity()
        term1 = LeastSquaresTerm.from_sigma(iden1.J, 3, sigma=2)
        prob = LeastSquaresProblem([term1])
        self.assertAlmostEqual(prob.objective(), 2.25)
        iden1.set_dofs([10])
        self.assertAlmostEqual(prob.objective(), 12.25)
        self.assertAlmostEqual(prob.objective([0]), 2.25)
        self.assertAlmostEqual(prob.objective([10]), 12.25)
        self.assertEqual(prob.dofs.all_owners, [iden1])
        self.assertEqual(prob.dofs.dof_owners, [iden1])

        # Objective function
        # f(x,y) = ((x - 3) / 2) ** 2 + ((y + 4) / 5) ** 2
        iden2 = Identity()
        term2 = LeastSquaresTerm.from_sigma(iden2.J, -4, sigma=5)
        prob = LeastSquaresProblem([term1, term2])
        self.assertAlmostEqual(prob.objective(), 12.89)
        iden1.set_dofs([5])
        iden2.set_dofs([-7])
        self.assertAlmostEqual(prob.objective(), 1.36)
        self.assertAlmostEqual(prob.objective([10, 0]), 12.89)
        self.assertAlmostEqual(prob.objective([5, -7]), 1.36)
        self.assertEqual(prob.dofs.dof_owners, [iden1, iden2])
        self.assertEqual(prob.dofs.all_owners, [iden1, iden2])

    def test_supply_tuples(self):
        """
        Test basic usage
        """
        # Objective function f(x) = ((x - 3) / 2) ** 2
        iden1 = Identity()
        term1 = (iden1.J, 3, 0.25)
        prob = LeastSquaresProblem([term1])
        self.assertAlmostEqual(prob.objective(), 2.25)
        iden1.set_dofs([10])
        self.assertAlmostEqual(prob.objective(), 12.25)
        self.assertAlmostEqual(prob.objective([0]), 2.25)
        self.assertAlmostEqual(prob.objective([10]), 12.25)
        self.assertEqual(prob.dofs.all_owners, [iden1])
        self.assertEqual(prob.dofs.dof_owners, [iden1])

        # Objective function
        # f(x,y) = ((x - 3) / 2) ** 2 + ((y + 4) / 5) ** 2
        iden2 = Identity()
        term2 = (iden2.J, -4, 0.04)
        prob = LeastSquaresProblem([term1, term2])
        self.assertAlmostEqual(prob.objective(), 12.89)
        iden1.set_dofs([5])
        iden2.set_dofs([-7])
        self.assertAlmostEqual(prob.objective(), 1.36)
        self.assertAlmostEqual(prob.objective([10, 0]), 12.89)
        self.assertAlmostEqual(prob.objective([5, -7]), 1.36)
        self.assertEqual(prob.dofs.dof_owners, [iden1, iden2])
        self.assertEqual(prob.dofs.all_owners, [iden1, iden2])

    def test_exceptions(self):
        """
        Verify that exceptions are raised when invalid inputs are
        provided.
        """
        # Argument must be a list in which each element is a
        # LeastSquaresTerm or a 3- or 4-element tuple/list.
        with self.assertRaises(TypeError):
            prob = LeastSquaresProblem(7)
        with self.assertRaises(ValueError):
            prob = LeastSquaresProblem([])
        with self.assertRaises(TypeError):
            prob = LeastSquaresProblem([7, 1])

if __name__ == "__main__":
    unittest.main()
