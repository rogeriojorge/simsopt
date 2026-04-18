#!/usr/bin/env python

import os
import tempfile
from pathlib import Path

import numpy as np
import vmec_jax as vj

from simsopt._core.optimizable import Optimizable
from simsopt.mhd import VmecJax, QuasisymmetryRatioResidualJax
from simsopt.solve import least_squares_mpi_solve
from simsopt.util import MpiPartition, proc0_print

"""
Optimize a VMEC-JAX equilibrium for quasi-helical symmetry (M=1, N=1)
throughout the volume using the classic MPI finite-difference solver path,
but with accelerated forward vmec_jax solves.
Run this example with mpirun -n 2 python QH_fixed_resolution_jaxfd_mpi.py
"""

max_nfev = int(os.environ.get("SIMSOPT_VMEC_JAXFD_MPI_MAX_NFEV", "10"))
max_mode = int(os.environ.get("SIMSOPT_VMEC_JAXFD_MPI_MAX_MODE", "1"))
solver_mode = os.environ.get("SIMSOPT_VMEC_JAXFD_MPI_SOLVER_MODE", "accelerated").strip().lower()
use_scan = os.environ.get("SIMSOPT_VMEC_JAXFD_MPI_USE_SCAN", "1").strip().lower() not in ("0", "false", "no", "off")


class VmecJaxAcceleratedLeastSquaresProblem(Optimizable):
    """Minimal Optimizable wrapper for least_squares_mpi_solve."""

    return_fn_map = {}

    def __init__(self, vmec, qs, *, input_path: str, aspect_target: float = 7.0):
        self.vmec = vmec
        self.qs = qs
        self.input_path = str(input_path)
        self._input_text = Path(input_path).read_text()
        self.aspect_target = float(aspect_target)
        self._tmpdir = Path(tempfile.mkdtemp(prefix="vmec_jaxfd_mpi_"))
        self._run_input_path = self._tmpdir / f"input.rank{os.getpid()}.vmec_jax"
        self._last_x = None
        self._last_state = None
        self._last_residuals = None
        self._last_objective = None
        x0 = np.asarray(self.vmec.boundary.get_free_params(), dtype=float)
        names = [name for name, free in zip(self.vmec.boundary.dof_names, self.vmec.boundary.free) if free]
        super().__init__(x0=x0, names=names, external_dof_setter=VmecJaxAcceleratedLeastSquaresProblem._set_x)

    @staticmethod
    def _set_x(self, val):
        self.vmec.boundary.set_free_params(np.asarray(val, dtype=float))
        self._last_x = None
        self._last_state = None
        self._last_residuals = None
        self._last_objective = None

    def _format_boundary_overrides(self, x_free):
        boundary = self.vmec.boundary
        x_full = np.asarray(boundary.expand_free(np.asarray(x_free, dtype=float)), dtype=float)
        boundary_input = boundary.apply_params(x_full)
        modes = boundary._modes
        lasym = bool(self.vmec._cfg.lasym)

        lines = [
            "",
            "! vmec_jax accelerated FD overrides",
            f"MPOL = {int(self.vmec.indata.mpol)}",
            f"NTOR = {int(self.vmec.indata.ntor)}",
        ]
        for idx, (m, n) in enumerate(zip(np.asarray(modes.m, dtype=int), np.asarray(modes.n, dtype=int))):
            rc = float(np.asarray(boundary_input.R_cos)[idx])
            zs = float(np.asarray(boundary_input.Z_sin)[idx])
            lines.append(f"RBC({m},{n}) = {rc:.16e}")
            lines.append(f"ZBS({m},{n}) = {zs:.16e}")
            if lasym:
                rs = float(np.asarray(boundary_input.R_sin)[idx])
                zc = float(np.asarray(boundary_input.Z_cos)[idx])
                lines.append(f"RBS({m},{n}) = {rs:.16e}")
                lines.append(f"ZBC({m},{n}) = {zc:.16e}")
        lines.append("")
        return "\n".join(lines)

    def _write_input_for_x(self, x_free):
        text = self._input_text
        slash = text.rfind("/")
        if slash < 0:
            raise ValueError("Input file is missing terminating '/' for &INDATA.")
        override = self._format_boundary_overrides(x_free)
        rewritten = text[:slash] + override + "\n/\n" + text[slash + 1 :]
        self._run_input_path.write_text(rewritten)
        return self._run_input_path

    def _solve_state(self, x_free):
        x_key = tuple(np.asarray(x_free, dtype=float).tolist())
        if self._last_x == x_key and self._last_state is not None:
            return self._last_state

        input_path = self._write_input_for_x(x_free)
        run = vj.run_fixed_boundary(
            input_path,
            solver="vmec2000_iter",
            solver_mode=solver_mode,
            performance_mode=(solver_mode != "parity"),
            use_scan=use_scan,
            verbose=False,
            jit_forces=True,
            cli_fixed_boundary_mode=False,
            _auto_cli_fixed_boundary_mode=False,
        )
        self._last_x = x_key
        self._last_state = run.state
        self._last_residuals = None
        self._last_objective = None
        return self._last_state

    def _compute_residuals(self, x_free):
        x_key = tuple(np.asarray(x_free, dtype=float).tolist())
        if self._last_x == x_key and self._last_residuals is not None:
            return self._last_residuals

        state = self._solve_state(x_free)
        aspect_residual = np.asarray(
            [float(np.asarray(self.vmec.aspect_equilibrium_from_state_jax(state))) - self.aspect_target],
            dtype=float,
        )
        qs_residual = np.asarray(self.qs.residuals_from_state(state), dtype=float)
        residuals = np.concatenate([aspect_residual, qs_residual])
        self._last_x = x_key
        self._last_residuals = residuals
        self._last_objective = float(np.dot(residuals, residuals))
        return residuals

    def unweighted_residuals(self, x=None):
        if x is not None:
            self.x = x
        return self._compute_residuals(self.x)

    def residuals(self):
        return self.unweighted_residuals()

    def objective(self):
        x_key = tuple(np.asarray(self.x, dtype=float).tolist())
        if self._last_x == x_key and self._last_objective is not None:
            return self._last_objective
        r = self.unweighted_residuals()
        self._last_objective = float(np.dot(r, r))
        return self._last_objective


VmecJaxAcceleratedLeastSquaresProblem.return_fn_map = {
    "residuals": VmecJaxAcceleratedLeastSquaresProblem.residuals,
    "objective": VmecJaxAcceleratedLeastSquaresProblem.objective,
}

proc0_print("Running 2_Intermediate/QH_fixed_resolution_jaxfd_mpi.py")
proc0_print("=======================================================")
proc0_print("Forward solver mode:", solver_mode, "| use_scan:", use_scan)

mpi = MpiPartition()

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

surf = vmec.boundary
surf.fix_all()
surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
surf.fix("rc(0,0)")

proc0_print("Parameter space:", [name for name, free in zip(surf.dof_names, surf.free) if free])

qs = QuasisymmetryRatioResidualJax(
    vmec,
    [i / 10 for i in range(11)],
    helicity_m=1,
    helicity_n=-1,
)

prob = VmecJaxAcceleratedLeastSquaresProblem(vmec, qs, input_path=filename, aspect_target=7.0)

# Warm the accelerated forward path once per process before MPI FD starts.
prob.objective()

proc0_print("Quasisymmetry objective before optimization:", qs.total_from_state(prob._last_state))
proc0_print("Total objective before optimization:", prob.objective())

least_squares_mpi_solve(
    prob,
    mpi,
    grad=True,
    rel_step=1e-5,
    abs_step=1e-8,
    max_nfev=max_nfev,
)

prob.objective()

proc0_print("Final aspect ratio:", float(np.asarray(vmec.aspect_equilibrium_from_state_jax(prob._last_state))))
proc0_print("Quasisymmetry objective after optimization:", qs.total_from_state(prob._last_state))
proc0_print("Total objective after optimization:", prob.objective())

proc0_print("End of 2_Intermediate/QH_fixed_resolution_jaxfd_mpi.py")
proc0_print("=====================================================")
