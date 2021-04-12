import unittest
import logging

import numpy as np

from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo.surfacegarabedian import SurfaceGarabedian
from simsopt.core.optimizable import optimizable
from simsopt.util.mpi import MpiPartition
from simsopt.solve.least_squares_problem import LeastSquaresProblem
from simsopt.solve.serial_solve import least_squares_serial_solve
from simsopt.solve.mpi_solve import least_squares_mpi_solve

def mpi_solve_1group(prob, **kwargs):
    least_squares_mpi_solve(prob, MpiPartition(ngroups=1), **kwargs)

solvers = [least_squares_serial_solve, mpi_solve_1group]

#logging.basicConfig(level=logging.DEBUG)

class IntegratedTests(unittest.TestCase):
    def test_2dof_surface_opt(self):
        """
        Optimize the minor radius and elongation of an axisymmetric torus to
        obtain a desired volume and area.
        """

        for solver in solvers:
            desired_volume = 0.6
            desired_area = 8.0

            # Start with a default surface, which is axisymmetric with major
            # radius 1 and minor radius 0.1.
            surf = SurfaceRZFourier(quadpoints_phi=62, quadpoints_theta=63)

            # Set initial surface shape. It helps to make zs(1,0) larger
            # than rc(1,0) since there are two solutions to this
            # optimization problem, and for testing we want to find one
            # rather than the other.
            surf.set_zs(1, 0, 0.2)

            # Parameters are all non-fixed by default, meaning they will be
            # optimized.  You can choose to exclude any subset of the variables
            # from the space of independent variables by setting their 'fixed'
            # property to True.
            surf.set_fixed('rc(0,0)')

            # Each function you want in the objective function is then
            # equipped with a shift and weight, to become a term in a
            # least-squares objective function. A list of terms are
            # combined to form a nonlinear-least-squares problem.
            prob = LeastSquaresProblem([(surf.volume, desired_volume, 1),
                                        (surf.area,   desired_area,   1)])

            # Verify the state vector and names are what we expect
            np.testing.assert_allclose(prob.x, [0.1, 0.2])
            self.assertEqual(prob.dofs.names[0][:28], 'rc(1,0) of SurfaceRZFourier ')
            self.assertEqual(prob.dofs.names[1][:28], 'zs(1,0) of SurfaceRZFourier ')

            # Solve the minimization problem:
            solver(prob)

            # Check results
            self.assertAlmostEqual(surf.get_rc(0, 0), 1.0, places=13)
            self.assertAlmostEqual(surf.get_rc(1, 0), 0.10962565115956417, places=13)
            self.assertAlmostEqual(surf.get_zs(0, 0), 0.0, places=13)
            self.assertAlmostEqual(surf.get_zs(1, 0), 0.27727411213693337, places=13)
            self.assertAlmostEqual(surf.volume(), desired_volume, places=8)
            self.assertAlmostEqual(surf.area(), desired_area, places=8)
            self.assertLess(np.abs(prob.objective()), 1.0e-15)

    def test_2dof_surface_Garabedian_opt(self):
        """
        Optimize the minor radius and elongation of an axisymmetric torus
        to obtain a desired volume and area, optimizing in the space
        of Garabedian coefficients.
        """

        for solver in solvers:
            desired_volume = 0.6
            desired_area = 8.0

            # Start with a default surface, which is axisymmetric with
            # major radius 1 and minor radius 0.1. Setting mmax=2
            # allows elongation to be added.
            surf = SurfaceGarabedian(mmax=2)

            # Set initial surface shape. It helps to make the initial
            # surface shape slightly different from the default one
            # since there are two solutions to this optimization
            # problem, and for testing we want to find one rather than
            # the other.
            surf.set_Delta(2, 0, -0.1)

            # Parameters are all non-fixed by default, meaning they will be
            # optimized.  You can choose to exclude any subset of the variables
            # from the space of independent variables by setting their 'fixed'
            # property to True.
            surf.all_fixed()
            surf.set_fixed('Delta(0,0)', False) # Minor radius
            surf.set_fixed('Delta(2,0)', False) # Elongation

            # Each function you want in the objective function is then
            # equipped with a shift and weight, to become a term in a
            # least-squares objective function. A list of terms are
            # combined to form a nonlinear-least-squares problem.
            prob = LeastSquaresProblem([(surf.volume, desired_volume, 1),
                                        (surf.area,   desired_area,   1)])

            # Verify the state vector and names are what we expect
            np.testing.assert_allclose(prob.x, [0.1, -0.1])
            self.assertEqual(prob.dofs.names[0][:31], 'Delta(0,0) of SurfaceGarabedian')
            self.assertEqual(prob.dofs.names[1][:31], 'Delta(2,0) of SurfaceGarabedian')

            # Solve the minimization problem:
            solver(prob)

            # Check results
            self.assertAlmostEqual(surf.get_Delta(0, 0), 0.193449881648249, places=11)
            self.assertAlmostEqual(surf.get_Delta(1, 0), 1.0, places=13)
            self.assertAlmostEqual(surf.get_Delta(2, 0), -0.083824230488685, places=11)
            self.assertAlmostEqual(surf.volume(), desired_volume, places=8)
            self.assertAlmostEqual(surf.area(), desired_area, places=8)

            # Convert the SurfaceGarabedian to a SurfaceRZFourier and
            # make sure its properties match those of the optimization
            # direction in RZFourier-space.
            surfRZ = surf.to_RZFourier()
            self.assertAlmostEqual(surfRZ.get_rc(0, 0), 1.0, places=13)
            self.assertAlmostEqual(surfRZ.get_rc(1, 0), 0.10962565115956417, places=11)
            self.assertAlmostEqual(surfRZ.get_zs(0, 0), 0.0, places=13)
            self.assertAlmostEqual(surfRZ.get_zs(1, 0), 0.27727411213693337, places=11)
            self.assertAlmostEqual(surfRZ.volume(), desired_volume, places=8)
            self.assertAlmostEqual(surfRZ.area(), desired_area, places=8)
            self.assertLess(np.abs(prob.objective()), 1.0e-15)

if __name__ == "__main__":
    unittest.main()
