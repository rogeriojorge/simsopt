#!/usr/bin/env python

import os

import numpy as np

from simsopt._core.optimizable import Optimizable
from simsopt.mhd import VmecJax, QuasisymmetryRatioResidualJax
from simsopt.solve import least_squares_mpi_solve
from simsopt.util import MpiPartition, proc0_print

"""
Optimize a VMEC-JAX equilibrium for quasi-helical symmetry (M=1, N=1)
throughout the volume using the classic MPI finite-difference solver path.
Run this example with mpirun -n 2 python QH_fixed_resolution_jaxfd_mpi.py
"""

max_nfev = 10  # Maximum number of function evaluations
max_mode = 1  # Maximum poloidal and toroidal mode numbers to vary


class VmecJaxLeastSquaresProblem(Optimizable):
    """Minimal Optimizable wrapper for least_squares_mpi_solve."""

    return_fn_map = {}

    def __init__(self, vmec, qs, *, aspect_target: float = 7.0):
        self.vmec = vmec
        self.qs = qs
        self.aspect_target = float(aspect_target)
        x0 = np.asarray(self.vmec.boundary.get_free_params(), dtype=float)
        names = [name for name, free in zip(self.vmec.boundary.dof_names, self.vmec.boundary.free) if free]
        super().__init__(x0=x0, names=names, external_dof_setter=VmecJaxLeastSquaresProblem._set_x)

    @staticmethod
    def _set_x(self, val):
        self.vmec.boundary.set_free_params(np.asarray(val, dtype=float))

    def unweighted_residuals(self, x=None):
        if x is not None:
            self.x = x
        x_free = self.x
        aspect_residual = np.asarray([self.vmec.aspect(x_free) - self.aspect_target], dtype=float)
        qs_residual = np.asarray(self.qs.residuals(x_free), dtype=float)
        return np.concatenate([aspect_residual, qs_residual])

    def residuals(self):
        return self.unweighted_residuals()

    def objective(self):
        r = self.unweighted_residuals()
        return float(np.dot(r, r))


VmecJaxLeastSquaresProblem.return_fn_map = {
    "residuals": VmecJaxLeastSquaresProblem.residuals,
    "objective": VmecJaxLeastSquaresProblem.objective,
}

proc0_print("Running 2_Intermediate/QH_fixed_resolution_jaxfd_mpi.py")
proc0_print("=======================================================")

mpi = MpiPartition()

# For forming filenames for VMEC, pathlib sometimes does not work, so use os.path.join instead.
filename = os.path.join(os.path.dirname(__file__), "inputs", "input.nfp4_QH_warm_start")
vmec = VmecJax(filename, verbose=False)
vmec.use_residual_autodiff_defaults(
    outer_method="scipy",
    residual_adjoint_mode="chunked",
    stateless_evaluations=False,
    optimization_profile="qh",
)
vmec.set_solver_options(
    residual_derivative_backend="implicit",
    residual_adjoint_mode="auto",
    residual_tangent_mode="opaque",
)
vmec.indata.mpol = max_mode + 2
vmec.indata.ntor = vmec.indata.mpol

# Define parameter space:
surf = vmec.boundary
surf.fix_all()
surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
surf.fix("rc(0,0)")  # Major radius

proc0_print("Parameter space:", [name for name, free in zip(surf.dof_names, surf.free) if free])

# Configure quasisymmetry objective:
qs = QuasisymmetryRatioResidualJax(
    vmec,
    [i / 10 for i in range(11)],  # Radii to target
    helicity_m=1,
    helicity_n=-1,
)

# Define objective function.
prob = VmecJaxLeastSquaresProblem(vmec, qs, aspect_target=7.0)

# Make sure all procs participate in computing the objective.
prob.objective()

proc0_print("Quasisymmetry objective before optimization:", qs.total(prob.x))
proc0_print("Total objective before optimization:", prob.objective())

# This follows the classic MPI finite-difference workflow, but all forward
# solves are vmec_jax runs inside a warm Python process.
least_squares_mpi_solve(
    prob,
    mpi,
    grad=True,
    rel_step=1e-5,
    abs_step=1e-8,
    max_nfev=max_nfev,
)

# Make sure all procs participate in computing the objective.
prob.objective()

proc0_print("Final aspect ratio:", vmec.aspect())
proc0_print("Quasisymmetry objective after optimization:", qs.total(prob.x))
proc0_print("Total objective after optimization:", prob.objective())

proc0_print("End of 2_Intermediate/QH_fixed_resolution_jaxfd_mpi.py")
proc0_print("=====================================================")
