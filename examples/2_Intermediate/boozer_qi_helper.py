"""Utilities shared by the Boozer QI example entrypoint.

The goal of this module is to keep boozerQI.py close to the structure of
boozerQA.py while centralizing the environment parsing, restart handling,
optimizer bookkeeping, and artifact export logic that are specific to the QI
workflow.
"""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import f90nml
import numpy as np
from scipy.optimize import BFGS as TrustRegionBFGS, minimize

from simsopt._core.optimizable import load
from simsopt.configs import get_data
from simsopt.field import BiotSavart
from simsopt.geo import (
    BoozerSurface,
    Iotas,
    MajorRadius,
    SurfaceRZFourier,
    SurfaceXYZTensorFourier,
    ToroidalFlux,
    Volume,
    boozer_surface_residual,
)
from simsopt.objectives import QuadraticPenalty


def _env_int(name, default):
    return int(os.environ.get(name, str(default)))


def _env_bool(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _env_float(name, default):
    return float(os.environ.get(name, str(default)))


def _env_str_list(name, default):
    raw = os.environ.get(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _sorted_optimizables(keys, class_name):
    return sorted(
        [opt for opt in keys if opt.__class__.__name__ == class_name],
        key=lambda opt: getattr(opt, "name", class_name),
    )


@dataclass
class QIConfig:
    """Runtime configuration for the Boozer QI example.

    The fields mirror the supported environment overrides so that the example
    can stay script-like while still exposing the knobs needed for debugging,
    validation, and exporting restart artifacts.
    """

    out_dir: str
    write_vtk: bool
    skip_taylor: bool
    vmec_geometry: str | None
    vmec_input: str | None
    coils_json: str | None
    boozer_verbose: bool
    boozer_type: str
    exact_ls_fallback: bool
    exact_ls_weight: float
    exact_ls_maxiter: int
    opt_method: str
    step_rms_limit: float
    step_penalty: float
    maxls: int
    maxcor: int
    history_path: str
    export_coils_json: str
    export_surface_restart: str
    export_vmec_input: str
    export_vmec_diagnostics: str
    export_vmec_cross_section_plot: str
    export_vmec_surface_plot: str
    vmec_template_input: str
    surface_restart: str | None
    compare_fd: bool
    compare_fd_dirs: int
    compare_fd_eps: float
    compare_fd_subspace: int
    compare_fd_opt: bool
    compare_fd_opt_maxiter: int
    compare_optimizers: bool
    compare_optimizers_list: list[str]
    compare_optimizers_subspace: int
    compare_optimizers_maxiter: int
    exact_report: bool
    exact_report_maxiter: int

    @classmethod
    def from_environment(cls, repo_root: Path):
        """Build a configuration object from the supported environment variables."""
        out_dir = os.environ.get("SIMSOPT_BOOZER_QI_OUT_DIR", "./output/")
        history_path = os.environ.get("SIMSOPT_BOOZER_QI_HISTORY_PATH")
        if history_path is None:
            history_path = os.path.join(out_dir, "optimization_history.json")
        return cls(
            out_dir=out_dir,
            write_vtk=_env_bool("SIMSOPT_BOOZER_QI_WRITE_VTK", True),
            skip_taylor=_env_bool("SIMSOPT_BOOZER_QI_SKIP_TAYLOR", True),
            vmec_geometry=os.environ.get("SIMSOPT_BOOZER_QI_VMEC"),
            vmec_input=os.environ.get("SIMSOPT_BOOZER_QI_VMEC_INPUT"),
            coils_json=os.environ.get("SIMSOPT_BOOZER_QI_COILS_JSON"),
            boozer_verbose=_env_bool("SIMSOPT_BOOZER_QI_BOOZER_VERBOSE", False),
            boozer_type=os.environ.get("SIMSOPT_BOOZER_QI_BOOZER_TYPE", "ls").strip().lower(),
            exact_ls_fallback=_env_bool("SIMSOPT_BOOZER_QI_EXACT_LS_FALLBACK", True),
            exact_ls_weight=_env_float("SIMSOPT_BOOZER_QI_EXACT_LS_WEIGHT", 1.0),
            exact_ls_maxiter=_env_int("SIMSOPT_BOOZER_QI_EXACT_LS_MAXITER", 160),
            opt_method=os.environ.get("SIMSOPT_BOOZER_QI_OPT_METHOD", "trust-constr"),
            step_rms_limit=_env_float("SIMSOPT_BOOZER_QI_STEP_RMS_LIMIT", 2.0e-2),
            step_penalty=_env_float("SIMSOPT_BOOZER_QI_STEP_PENALTY", 1.0e6),
            maxls=_env_int("SIMSOPT_BOOZER_QI_MAXLS", 10),
            maxcor=_env_int("SIMSOPT_BOOZER_QI_MAXCOR", 20),
            history_path=history_path,
            export_coils_json=os.environ.get(
                "SIMSOPT_BOOZER_QI_EXPORT_COILS_JSON",
                os.path.join(out_dir, "biot_savart_opt.json"),
            ),
            export_surface_restart=os.environ.get(
                "SIMSOPT_BOOZER_QI_EXPORT_SURFACE_RESTART",
                os.path.join(out_dir, "boozer_surface_restart.json"),
            ),
            export_vmec_input=os.environ.get(
                "SIMSOPT_BOOZER_QI_EXPORT_VMEC_INPUT",
                os.path.join(out_dir, "input.boozer_qi"),
            ),
            export_vmec_diagnostics=os.environ.get(
                "SIMSOPT_BOOZER_QI_VMEC_EXPORT_DIAGNOSTICS",
                os.path.join(out_dir, "vmec_export_surface_diagnostics.json"),
            ),
            export_vmec_cross_section_plot=os.environ.get(
                "SIMSOPT_BOOZER_QI_VMEC_EXPORT_CROSS_SECTION_PLOT",
                os.path.join(out_dir, "vmec_export_cross_sections.png"),
            ),
            export_vmec_surface_plot=os.environ.get(
                "SIMSOPT_BOOZER_QI_VMEC_EXPORT_SURFACE_PLOT",
                os.path.join(out_dir, "vmec_export_surface_3d.png"),
            ),
            vmec_template_input=os.environ.get(
                "SIMSOPT_BOOZER_QI_VMEC_TEMPLATE_INPUT",
                str(repo_root / "src" / "simsopt" / "mhd" / "input.default"),
            ),
            surface_restart=os.environ.get("SIMSOPT_BOOZER_QI_SURFACE_RESTART"),
            compare_fd=_env_bool("SIMSOPT_BOOZER_QI_COMPARE_FD", False),
            compare_fd_dirs=_env_int("SIMSOPT_BOOZER_QI_COMPARE_FD_DIRS", 2),
            compare_fd_eps=_env_float("SIMSOPT_BOOZER_QI_COMPARE_FD_EPS", 2.0 ** -18),
            compare_fd_subspace=_env_int("SIMSOPT_BOOZER_QI_COMPARE_FD_SUBSPACE", 6),
            compare_fd_opt=_env_bool("SIMSOPT_BOOZER_QI_COMPARE_FD_OPT", False),
            compare_fd_opt_maxiter=_env_int("SIMSOPT_BOOZER_QI_COMPARE_FD_OPT_MAXITER", 1),
            compare_optimizers=_env_bool("SIMSOPT_BOOZER_QI_COMPARE_OPTIMIZERS", False),
            compare_optimizers_list=_env_str_list(
                "SIMSOPT_BOOZER_QI_COMPARE_OPTIMIZERS_LIST",
                "L-BFGS-B,BFGS,trust-constr",
            ),
            compare_optimizers_subspace=_env_int(
                "SIMSOPT_BOOZER_QI_COMPARE_OPTIMIZERS_SUBSPACE",
                _env_int("SIMSOPT_BOOZER_QI_COMPARE_FD_SUBSPACE", 6),
            ),
            compare_optimizers_maxiter=_env_int(
                "SIMSOPT_BOOZER_QI_COMPARE_OPTIMIZERS_MAXITER",
                max(2, _env_int("SIMSOPT_BOOZER_QI_COMPARE_FD_OPT_MAXITER", 1)),
            ),
            exact_report=_env_bool("SIMSOPT_BOOZER_QI_EXACT_REPORT", False),
            exact_report_maxiter=_env_int("SIMSOPT_BOOZER_QI_EXACT_REPORT_MAXITER", 20),
        )

    def validate(self):
        """Validate combinations of user-provided runtime overrides."""
        if self.boozer_type not in {"exact", "ls"}:
            raise ValueError(
                f"SIMSOPT_BOOZER_QI_BOOZER_TYPE must be 'exact' or 'ls', got {self.boozer_type!r}."
            )
        if sum(value is not None for value in [self.surface_restart, self.vmec_geometry, self.vmec_input]) > 1:
            raise ValueError(
                "Specify at most one of SIMSOPT_BOOZER_QI_SURFACE_RESTART, "
                "SIMSOPT_BOOZER_QI_VMEC, and SIMSOPT_BOOZER_QI_VMEC_INPUT."
            )

    def prepare_output_dir(self):
        """Create the output directory when file exports are enabled."""
        if self.write_vtk:
            os.makedirs(self.out_dir, exist_ok=True)

    def history_settings(self):
        """Return the run settings that should be embedded in optimization history."""
        return {
            "boozer_type": self.boozer_type,
            "optimizer": self.opt_method,
            "step_rms_limit": self.step_rms_limit,
            "step_penalty": self.step_penalty,
            "maxls": self.maxls,
            "maxcor": self.maxcor,
            "skip_taylor": self.skip_taylor,
            "compare_fd": self.compare_fd,
            "compare_optimizers": self.compare_optimizers,
            "compare_optimizers_list": self.compare_optimizers_list,
            "compare_optimizers_subspace": self.compare_optimizers_subspace,
            "compare_optimizers_maxiter": self.compare_optimizers_maxiter,
            "exact_report": self.exact_report,
            "exact_report_maxiter": self.exact_report_maxiter,
        }


@dataclass
class SurfaceInitializer:
    """Description of the surface data used to seed the Boozer solve."""

    reference: Path | None
    label: str | None
    nfp: int | None
    restart_state: dict | None
    is_vmec_geometry: bool


@dataclass
class CoilSeed:
    """Coil and field objects used by the single-surface QI optimization."""

    base_curves: list
    base_currents: list
    ma: object | None
    nfp: int
    bs: BiotSavart
    all_curves: list
    bs_qi: BiotSavart
    G0: float


def load_seed_biot_savart(filename):
    """Load an exported Biot-Savart object and recover its base curves/currents."""
    path = Path(filename).expanduser().resolve()
    bs_seed = load(str(path))
    if not isinstance(bs_seed, BiotSavart):
        raise TypeError(f"Expected a BiotSavart JSON seed in {path}, got {type(bs_seed)!r}.")
    dof_keys = list(bs_seed.dof_indices.keys())
    base_curves = _sorted_optimizables(dof_keys, "CurveXYZFourier")
    base_currents = _sorted_optimizables(dof_keys, "Current")
    if not base_curves:
        raise ValueError(f"No free base curves were found in {path}.")
    if not base_currents:
        raise ValueError(f"No base currents were found in {path}.")
    return path, base_curves, base_currents, bs_seed


def resolve_surface_initializer(config: QIConfig):
    """Resolve which surface source, if any, should seed the optimization surface."""
    if config.surface_restart:
        restart_state = load_surface_restart(config.surface_restart)
        reference = Path(restart_state["path"])
        return SurfaceInitializer(
            reference=reference,
            label=f"Boozer surface restart from {reference}",
            nfp=int(restart_state["nfp"]),
            restart_state=restart_state,
            is_vmec_geometry=False,
        )
    if config.vmec_geometry:
        vmec_path = Path(config.vmec_geometry).expanduser().resolve()
        surface_nfp = SurfaceRZFourier.from_wout(
            str(vmec_path), nphi=16, ntheta=16, range="half period"
        ).nfp
        return SurfaceInitializer(
            reference=vmec_path,
            label=f"VMEC boundary from {vmec_path}",
            nfp=surface_nfp,
            restart_state=None,
            is_vmec_geometry=True,
        )
    if config.vmec_input:
        vmec_input_path = Path(config.vmec_input).expanduser().resolve()
        surface_nfp = SurfaceRZFourier.from_vmec_input(
            str(vmec_input_path), range="half period", nphi=16, ntheta=16
        ).nfp
        return SurfaceInitializer(
            reference=vmec_input_path,
            label=f"VMEC input boundary from {vmec_input_path}",
            nfp=surface_nfp,
            restart_state=None,
            is_vmec_geometry=False,
        )
    return SurfaceInitializer(None, None, None, None, False)


def load_coil_seed(config: QIConfig, initializer: SurfaceInitializer):
    """Load either the default NCSX coils or a user-provided Biot-Savart seed."""
    ma = None
    if config.coils_json:
        coils_path, base_curves, base_currents, bs = load_seed_biot_savart(config.coils_json)
        if initializer.nfp is None:
            raise ValueError(
                "SIMSOPT_BOOZER_QI_COILS_JSON requires a matching surface initializer via "
                "SIMSOPT_BOOZER_QI_VMEC or SIMSOPT_BOOZER_QI_VMEC_INPUT."
            )
        nfp = initializer.nfp
        print(f"Using Biot-Savart seed from {coils_path}")
    else:
        base_curves, base_currents, ma, nfp, bs = get_data("ncsx")
        if initializer.nfp is not None and initializer.nfp != nfp:
            raise ValueError(
                f"Surface initializer nfp={initializer.nfp} does not match the default coil configuration nfp={nfp}."
            )

    all_curves = [coil.curve for coil in bs.coils]
    bs_qi = BiotSavart(bs.coils)
    current_sum = nfp * sum(abs(current.get_value()) for current in base_currents)
    G0 = 2.0 * np.pi * current_sum * (4 * np.pi * 10 ** (-7) / (2 * np.pi))
    return CoilSeed(base_curves, base_currents, ma, nfp, bs, all_curves, bs_qi, G0)


def make_surface(nfp, mpol, ntor):
    """Create the tensor-Fourier surface and its quadrature grids."""
    phis = np.linspace(0.0, 1.0 / nfp, 2 * ntor + 1, endpoint=False)
    thetas = np.linspace(0.0, 1.0, 2 * mpol + 1, endpoint=False)
    surface = SurfaceXYZTensorFourier(
        mpol=mpol,
        ntor=ntor,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=phis,
        quadpoints_theta=thetas,
    )
    return surface, phis, thetas


def initialize_surface(surface, initializer: SurfaceInitializer, ma, nfp, phis, thetas):
    """Populate the optimization surface from a restart, VMEC source, or magnetic axis."""
    if initializer.restart_state is not None:
        if initializer.nfp != nfp:
            raise ValueError(
                f"Surface restart nfp={initializer.nfp} does not match the coil configuration nfp={nfp}."
            )
        print(f"Using surface initializer from {initializer.label}")
        apply_surface_restart(surface, initializer.restart_state)
        return
    if initializer.reference is not None:
        if initializer.nfp != nfp:
            raise ValueError(
                f"Surface initializer nfp={initializer.nfp} does not match the coil configuration nfp={nfp}."
            )
        print(f"Using surface initializer from {initializer.label}")
        if initializer.is_vmec_geometry:
            reference_surface = SurfaceRZFourier.from_wout(
                str(initializer.reference),
                nphi=len(phis),
                ntheta=len(thetas),
                range="half period",
            )
        else:
            reference_surface = SurfaceRZFourier.from_vmec_input(
                str(initializer.reference),
                range="half period",
                nphi=len(phis),
                ntheta=len(thetas),
            )
        surface.least_squares_fit(reference_surface.gamma())
        return
    surface.fit_to_curve(ma, 0.1, flip_theta=True)


def solve_boozer_surface_with_seed_fallback(boozer_surface, iota, G, tol, maxiter, verbose):
    """Run the configured Boozer solve using the current surface state as the seed."""
    del tol, maxiter, verbose
    boozer_surface.need_to_run_code = True
    return boozer_surface.run_code(iota=iota, G=G)


def qi_resolution_defaults(in_github_actions):
    """Return default NonQuasiIsodynamicRatio resolution settings."""
    return {
        "sDIM": 8 if in_github_actions else 20,
        "nphi": 31 if in_github_actions else 151,
        "nalpha": 5 if in_github_actions else 31,
        "nBj": 7 if in_github_actions else 51,
        "nphi_out": 61 if in_github_actions else 2000,
    }


def qi_resolution_from_environment(defaults):
    """Apply environment overrides to the QI objective resolution settings."""
    return {
        "sDIM": _env_int("SIMSOPT_BOOZER_QI_SDIM", defaults["sDIM"]),
        "nphi": _env_int("SIMSOPT_BOOZER_QI_NPHI", defaults["nphi"]),
        "nalpha": _env_int("SIMSOPT_BOOZER_QI_NALPHA", defaults["nalpha"]),
        "nBj": _env_int("SIMSOPT_BOOZER_QI_NBJ", defaults["nBj"]),
        "nphi_out": _env_int("SIMSOPT_BOOZER_QI_NPHI_OUT", defaults["nphi_out"]),
    }


def maxiter_from_environment(in_github_actions):
    """Return the optimizer iteration cap for the current environment."""
    default = 3 if in_github_actions else 1000
    return _env_int("SIMSOPT_BOOZER_QI_MAXITER", default)


def clone_surface_tensor_fourier(surface):
    """Clone a tensor-Fourier surface without relying on dof-wrapper internals."""
    cloned = SurfaceXYZTensorFourier(
        nfp=surface.nfp,
        stellsym=surface.stellsym,
        mpol=surface.mpol,
        ntor=surface.ntor,
        quadpoints_phi=np.asarray(surface.quadpoints_phi),
        quadpoints_theta=np.asarray(surface.quadpoints_theta),
    )
    cloned.set_dofs(surface.get_dofs().copy())
    return cloned


def normalize_optimizer_method(method):
    """Normalize optimizer method names to the spellings expected by scipy."""
    normalized = method.strip()
    if normalized.lower() == "trust-constr":
        return "trust-constr"
    upper = normalized.upper()
    if upper in {"L-BFGS-B", "BFGS"}:
        return upper
    return normalized


def build_minimize_config(method, maxiter, ndofs, step_rms_limit, maxls, maxcor):
    """Assemble scipy.optimize.minimize options for the supported optimizers."""
    method = normalize_optimizer_method(method)
    options = {"maxiter": maxiter}
    kwargs = {}
    if method == "L-BFGS-B":
        options.update({
            "maxls": maxls,
            "maxcor": min(maxcor, max(3, ndofs)),
            "ftol": 1e-16,
            "gtol": 1e-12,
        })
    elif method == "BFGS":
        options.update({"gtol": 1e-12})
    elif method == "trust-constr":
        options.update({
            "gtol": 1e-12,
            "xtol": 1e-12,
            "initial_tr_radius": max(step_rms_limit * np.sqrt(ndofs), 1e-3),
            "verbose": 0,
        })
        kwargs["hess"] = TrustRegionBFGS()
    return method, options, kwargs


def run_taylor_test(fun, dofs, eps_values=(1e-3, 1e-4, 1e-5, 1e-6)):
    """Run a fourth-order centered Taylor test for the objective gradient."""
    np.random.seed(1)
    h = np.random.uniform(size=dofs.shape)
    _, dJ0 = fun(dofs)
    dJh = float(np.dot(dJ0, h))
    for eps in eps_values:
        J1, _ = fun(dofs + 2 * eps * h)
        J2, _ = fun(dofs + eps * h)
        J3, _ = fun(dofs - eps * h)
        J4, _ = fun(dofs - 2 * eps * h)
        error = ((J1 * (-1 / 12) + J2 * (8 / 12) + J3 * (-8 / 12) + J4 * (1 / 12)) / eps - dJh)
        print("err", error / max(np.linalg.norm(dJh), 1e-15))


class QIOptimizationDriver:
    """Manage robust objective evaluations and optimization diagnostics.

    The driver owns the policy for keeping the last valid Boozer state, applying
    trust-region-like step penalties, and recording enough metadata to diagnose
    optimizer behavior after a run.
    """

    def __init__(
        self,
        config,
        JF,
        boozer_surface,
        J_non_qi_ratio,
        J_iotas,
        J_major_radius,
        J_length,
        major_radius,
        length_terms,
        bs,
        bs_qi,
        vol_target,
        initial_iota_target,
        initial_major_radius_target,
        qi_resolution,
    ):
        self.config = config
        self.JF = JF
        self.boozer_surface = boozer_surface
        self.J_non_qi_ratio = J_non_qi_ratio
        self.J_iotas = J_iotas
        self.J_major_radius = J_major_radius
        self.J_length = J_length
        self.major_radius = major_radius
        self.length_terms = length_terms
        self.bs = bs
        self.bs_qi = bs_qi
        self.vol_target = vol_target
        self.initial_iota_target = initial_iota_target
        self.initial_major_radius_target = initial_major_radius_target
        self.qi_resolution = qi_resolution
        self.history = {
            "settings": config.history_settings(),
            "evaluations": [],
            "iterations": [],
        }
        self.evaluation_state = {"count": 0}
        self.iteration_state = {"previous": JF.x.copy()}
        self.last_good = {
            "dofs": JF.x.copy(),
            "surface": boozer_surface.surface.x.copy(),
            "iota": boozer_surface.res["iota"],
            "G": boozer_surface.res["G"],
            "J": None,
            "grad": None,
        }
        self.best_good = {
            "dofs": JF.x.copy(),
            "surface": boozer_surface.surface.x.copy(),
            "iota": boozer_surface.res["iota"],
            "G": boozer_surface.res["G"],
            "J": None,
            "grad": None,
        }

    def snapshot(self):
        """Return a copy of the latest physically valid optimization state."""
        return self._clone_good_state(self.last_good)

    def restore_best(self):
        """Restore the best physically valid state encountered so far."""
        if self.best_good["J"] is not None:
            self._restore_boozer_state(self.best_good)

    def _clone_good_state(self, state):
        return {
            "dofs": state["dofs"].copy(),
            "surface": state["surface"].copy(),
            "iota": state["iota"],
            "G": state["G"],
            "J": state["J"],
            "grad": None if state["grad"] is None else state["grad"].copy(),
        }

    def _saved_state(self):
        return {
            "dofs": self.JF.x.copy(),
            "surface": self.boozer_surface.surface.x.copy(),
            "iota": self.boozer_surface.res["iota"],
            "G": self.boozer_surface.res["G"],
            "J": self.last_good["J"],
            "grad": None if self.last_good["grad"] is None else self.last_good["grad"].copy(),
        }

    def _restore_boozer_state(self, saved):
        self.JF.x = saved["dofs"]
        self.boozer_surface.surface.x = saved["surface"].copy()
        self.boozer_surface.res["iota"] = saved["iota"]
        self.boozer_surface.res["G"] = saved["G"]
        self.boozer_surface.res["success"] = True
        self.boozer_surface.need_to_run_code = True
        self.J_non_qi_ratio.recompute_bell()
        self.J_iotas.recompute_bell()
        self.J_major_radius.recompute_bell()
        self.J_length.recompute_bell()

    def _physical_summary(self):
        curve_lengths = [float(term.J()) for term in self.length_terms]
        return {
            "J_nonQIRatio": float(self.J_non_qi_ratio.J()),
            "J_iotas": float(self.J_iotas.J()),
            "J_major_radius": float(self.J_major_radius.J()),
            "J_length": float(self.J_length.J()),
            "iota": float(self.boozer_surface.res["iota"]),
            "G": float(self.boozer_surface.res["G"]),
            "major_radius": float(self.major_radius.J()),
            "curve_lengths": curve_lengths,
            "curve_length_total": float(sum(curve_lengths)),
        }

    def _print_physical_summary(self, prefix, J, grad, eval_seconds):
        summary = self._physical_summary()
        cl_string = ", ".join([f"{length:.1f}" for length in summary["curve_lengths"]])
        outstr = (
            f"{prefix}J={J:.1e}, J_nonQIRatio={summary['J_nonQIRatio']:.2e}, "
            f"iota={summary['iota']:.2e}, mr={summary['major_radius']:.2e}"
        )
        outstr += f", Len=sum([{cl_string}])={summary['curve_length_total']:.1f}"
        outstr += f", ||grad J||={np.linalg.norm(grad):.1e}, eval={eval_seconds:.2f}s"
        print(outstr)

    def _record_history(self, kind, dofs, J, grad, step_rms, physical, note=None):
        grad = np.asarray(grad, dtype=float)
        entry = {
            "kind": kind,
            "evaluation": len(self.history["evaluations"]) if kind == "evaluation" else len(self.history["iterations"]),
            "J": float(J),
            "grad_norm": float(np.linalg.norm(grad)),
            "step_rms": float(step_rms),
            "note": note,
        }
        if physical:
            entry.update(self._physical_summary())
        else:
            entry.update({
                "J_nonQIRatio": None,
                "J_iotas": None,
                "J_major_radius": None,
                "J_length": None,
                "iota": None,
                "G": None,
                "major_radius": None,
                "curve_lengths": [],
                "curve_length_total": None,
            })
        self.history["evaluations" if kind == "evaluation" else "iterations"].append(entry)

    def _step_penalty(self, dofs, reference_dofs, base_value):
        scales = np.maximum(1.0, np.abs(reference_dofs))
        step_scaled = (dofs - reference_dofs) / scales
        norm_sq = float(np.dot(step_scaled, step_scaled))
        limit_sq = dofs.size * self.config.step_rms_limit ** 2
        excess = norm_sq - limit_sq
        if excess <= 0:
            return None
        J = base_value + self.config.step_penalty * excess ** 2
        grad = 4.0 * self.config.step_penalty * excess * step_scaled / scales
        return J, grad, np.sqrt(norm_sq / dofs.size)

    def evaluate(self, dofs, reference_state=None, allow_step_penalty=True, update_good_state=False, record_history=False, verbose=False):
        """Evaluate the objective while preserving a valid Boozer state on failures."""
        step_rms = 0.0
        if reference_state is not None:
            step_rms = float(np.linalg.norm(dofs - reference_state["dofs"]) / np.sqrt(dofs.size))
        if allow_step_penalty and reference_state is not None:
            penalty = self._step_penalty(
                dofs,
                reference_state["dofs"],
                reference_state["J"] if reference_state["J"] is not None else 0.0,
            )
            if penalty is not None:
                self._restore_boozer_state(reference_state)
                J, grad, limited_step_rms = penalty
                metadata = {
                    "physical": False,
                    "step_rms": limited_step_rms,
                    "note": "step_limit",
                    "eval_seconds": 0.0,
                }
                if record_history:
                    self._record_history("evaluation", dofs, J, grad, limited_step_rms, physical=False, note="step_limit")
                if verbose:
                    print(
                        f"Trial step exceeded trust region; applying step penalty. "
                        f"J={J:.1e}, ||grad J||={np.linalg.norm(grad):.1e}"
                    )
                return J, grad, metadata

        saved = self._saved_state()
        self.JF.x = dofs
        started = time.perf_counter()
        try:
            J = float(self.JF.J())
            grad = np.asarray(self.JF.dJ(), dtype=float)
            if not self.boozer_surface.res["success"]:
                raise RuntimeError("Boozer surface solve failed during trial evaluation.")
            metadata = {
                "physical": True,
                "step_rms": step_rms,
                "note": None,
                "eval_seconds": time.perf_counter() - started,
            }
        except (np.linalg.LinAlgError, RuntimeError, ValueError) as err:
            fallback = reference_state if reference_state is not None else saved
            self._restore_boozer_state(fallback)
            delta = dofs - fallback["dofs"]
            fallback_J = fallback.get("J")
            J = (fallback_J if fallback_J is not None else 0.0) + 1e3 + float(np.dot(delta, delta))
            grad = 2.0 * delta
            metadata = {
                "physical": False,
                "step_rms": step_rms,
                "note": type(err).__name__,
                "eval_seconds": time.perf_counter() - started,
            }
            if record_history:
                self._record_history("evaluation", dofs, J, grad, step_rms, physical=False, note=type(err).__name__)
            if verbose:
                print(f"Trial evaluation failed ({type(err).__name__}); applying fallback penalty.")
                print(f"J={J:.1e}, ||grad J||={np.linalg.norm(grad):.1e}, eval={metadata['eval_seconds']:.2f}s")
            return J, grad, metadata

        if update_good_state and metadata["physical"]:
            self.last_good.update({
                "dofs": dofs.copy(),
                "surface": self.boozer_surface.surface.x.copy(),
                "iota": self.boozer_surface.res["iota"],
                "G": self.boozer_surface.res["G"],
                "J": J,
                "grad": grad.copy(),
            })
            if self.best_good["J"] is None or J < self.best_good["J"]:
                self.best_good = self._clone_good_state(self.last_good)

        if record_history:
            self._record_history("evaluation", dofs, J, grad, metadata["step_rms"], physical=True)
        if verbose:
            prefix = f"eval {self.evaluation_state['count']}: " if record_history else ""
            self._print_physical_summary(prefix, J, grad, metadata["eval_seconds"])
        return J, grad, metadata

    def fun(self, dofs):
        """scipy-compatible objective wrapper returning value and gradient."""
        self.evaluation_state["count"] += 1
        J, grad, _ = self.evaluate(
            dofs,
            reference_state=self.last_good,
            allow_step_penalty=True,
            update_good_state=True,
            record_history=True,
            verbose=True,
        )
        return J, grad

    def callback(self, *args):
        """Record accepted optimizer iterates in the run history."""
        if not args:
            xk = self.last_good["dofs"]
        elif hasattr(args[0], "x"):
            xk = np.asarray(args[0].x, dtype=float)
        else:
            xk = np.asarray(args[0], dtype=float)
        accepted_step_rms = float(np.linalg.norm(xk - self.iteration_state["previous"]) / np.sqrt(xk.size))
        self.iteration_state["previous"] = xk.copy()
        grad = self.last_good["grad"] if self.last_good["grad"] is not None else np.zeros_like(xk)
        self._record_history(
            "iteration",
            xk,
            self.last_good["J"] if self.last_good["J"] is not None else np.nan,
            grad,
            accepted_step_rms,
            physical=True,
        )

    def _run_reduced_subspace_optimization(self, base_state, active, method, maxiter, jac):
        saved = self._saved_state()
        base_dofs = base_state["dofs"].copy()
        x0 = base_dofs[active].copy()

        def scalar_eval(xsub):
            dofs = base_dofs.copy()
            dofs[active] = xsub
            value, _, _ = self.evaluate(dofs, allow_step_penalty=False, update_good_state=False, record_history=False, verbose=False)
            return value

        def grad_eval(xsub):
            dofs = base_dofs.copy()
            dofs[active] = xsub
            _, grad, _ = self.evaluate(dofs, allow_step_penalty=False, update_good_state=False, record_history=False, verbose=False)
            return grad[active]

        opt_method, options, kwargs = build_minimize_config(
            method,
            maxiter,
            active.size,
            self.config.step_rms_limit,
            self.config.maxls,
            self.config.maxcor,
        )
        res = minimize(
            scalar_eval,
            x0,
            jac=jac if isinstance(jac, str) else grad_eval,
            method=opt_method,
            options=options,
            tol=1e-15,
            **kwargs,
        )
        self._restore_boozer_state(saved)
        x_final = np.asarray(res.x, dtype=float)
        return {
            "method": opt_method,
            "success": bool(res.success),
            "status": int(res.status),
            "message": str(res.message),
            "nit": int(res.nit),
            "nfev": int(res.nfev),
            "njev": int(res.njev) if getattr(res, "njev", None) is not None else None,
            "fun": float(res.fun),
            "active_step_norm": float(np.linalg.norm(x_final - x0)),
            "active_x": x_final.tolist(),
        }

    def _select_reduced_subspace(self, base_state, subspace_size):
        if subspace_size <= 0:
            return np.zeros((0,), dtype=int)
        ordering = np.argsort(-np.abs(base_state["grad"]))
        return ordering[:min(subspace_size, ordering.size)]

    def run_directional_fd_comparison(self, base_state, ndirs, eps):
        rng = np.random.default_rng(1)
        results = []
        saved = self._saved_state()
        base_dofs = base_state["dofs"].copy()
        base_grad = base_state["grad"].copy()
        for index in range(ndirs):
            direction = rng.uniform(-0.5, 0.5, size=base_dofs.shape)
            direction /= max(np.linalg.norm(direction), 1e-15)
            plus_value, _, _ = self.evaluate(base_dofs + eps * direction, allow_step_penalty=False, update_good_state=False, record_history=False, verbose=False)
            minus_value, _, _ = self.evaluate(base_dofs - eps * direction, allow_step_penalty=False, update_good_state=False, record_history=False, verbose=False)
            adjoint = float(base_grad @ direction)
            finite_difference = (plus_value - minus_value) / (2 * eps)
            relative_error = abs(finite_difference - adjoint) / max(1.0, abs(finite_difference), abs(adjoint))
            results.append({
                "direction": index,
                "adjoint": adjoint,
                "finite_difference": finite_difference,
                "relative_error": relative_error,
            })
        self._restore_boozer_state(saved)
        return results

    def run_fd_subspace_optimization(self, base_state, subspace_size, maxiter):
        if subspace_size <= 0:
            return None
        active = self._select_reduced_subspace(base_state, subspace_size)
        analytic_res = self._run_reduced_subspace_optimization(base_state, active, "L-BFGS-B", maxiter, jac=True)
        fd_res = self._run_reduced_subspace_optimization(base_state, active, "L-BFGS-B", maxiter, jac="2-point")
        return {
            "subspace_size": int(active.size),
            "indices": active.tolist(),
            "analytic_fun": analytic_res["fun"],
            "analytic_nit": analytic_res["nit"],
            "fd_fun": fd_res["fun"],
            "fd_nit": fd_res["nit"],
            "step_difference_norm": float(np.linalg.norm(np.asarray(analytic_res["active_x"]) - np.asarray(fd_res["active_x"]))),
            "fun_difference": float(abs(analytic_res["fun"] - fd_res["fun"])),
        }

    def run_reduced_optimizer_comparison(self, base_state, subspace_size, methods, maxiter):
        active = self._select_reduced_subspace(base_state, subspace_size)
        comparison = {
            "subspace_size": int(active.size),
            "indices": active.tolist(),
            "results": [],
        }
        if active.size == 0:
            return comparison
        for method in methods:
            comparison["results"].append(
                self._run_reduced_subspace_optimization(base_state, active, method, maxiter, jac=True)
            )
        return comparison

    def run_exact_report(self):
        """Re-solve the best surface with the exact Boozer solve for a final report."""
        ls_state = self.best_good
        exact_surface = clone_surface_tensor_fourier(self.boozer_surface.surface)
        exact_surface.set_dofs(ls_state["surface"].copy())
        exact_vol = Volume(exact_surface)
        exact_boozer_surface = BoozerSurface(
            self.bs,
            exact_surface,
            exact_vol,
            self.vol_target,
            constraint_weight=None,
            options={
                "verbose": self.config.boozer_verbose,
                "newton_maxiter": self.config.exact_report_maxiter,
                "exact_ls_fallback": self.config.exact_ls_fallback,
                "exact_ls_constraint_weight": self.config.exact_ls_weight,
                "exact_ls_maxiter": self.config.exact_ls_maxiter,
                "exact_ls_tol": 1e-10,
            },
        )
        started = time.perf_counter()
        exact_res = solve_boozer_surface_with_seed_fallback(
            exact_boozer_surface,
            iota=ls_state["iota"],
            G=ls_state["G"],
            tol=1e-13,
            maxiter=self.config.exact_report_maxiter,
            verbose=self.config.boozer_verbose,
        )
        report = {
            "success": bool(exact_res.get("success", False)),
            "approximate": bool(exact_res.get("approximate", False)),
            "solver": "ls-fallback" if exact_res.get("approximate", False) else "exact",
            "iter": int(exact_res["iter"]) if exact_res.get("iter") is not None else None,
            "iota": float(exact_res.get("iota", ls_state["iota"])),
            "G": float(exact_res.get("G", ls_state["G"])),
            "elapsed_seconds": time.perf_counter() - started,
            "message": None if exact_res.get("message") is None else str(exact_res.get("message")),
        }
        if not report["success"]:
            return report

        exact_out_res = boozer_surface_residual(exact_surface, exact_res["iota"], exact_res["G"], self.bs, derivatives=0)[0]
        exact_major_radius = MajorRadius(exact_boozer_surface)
        exact_J_iotas = QuadraticPenalty(Iotas(exact_boozer_surface), self.initial_iota_target, "identity")
        exact_J_major_radius = QuadraticPenalty(exact_major_radius, self.initial_major_radius_target, "identity")
        exact_J_non_qi_ratio = self.J_non_qi_ratio.__class__(
            exact_boozer_surface,
            self.bs_qi,
            sDIM=self.qi_resolution["sDIM"],
            nphi=self.qi_resolution["nphi"],
            nalpha=self.qi_resolution["nalpha"],
            nBj=self.qi_resolution["nBj"],
            nphi_out=self.qi_resolution["nphi_out"],
        )
        report.update({
            "residual_norm": float(np.linalg.norm(exact_out_res)),
            "major_radius": float(exact_major_radius.J()),
            "J_nonQIRatio": float(exact_J_non_qi_ratio.J()),
            "J_iotas": float(exact_J_iotas.J()),
            "J_major_radius": float(exact_J_major_radius.J()),
            "J_length": float(self.J_length.J()),
        })
        report["total_J"] = float(
            report["J_nonQIRatio"] + report["J_iotas"] + report["J_major_radius"] + report["J_length"]
        )
        if self.config.write_vtk:
            exact_surface.to_vtk(os.path.join(self.config.out_dir, "surf_exact_report"))
        return report


def surface_gamma_on_grid(surface, quadpoints_phi, quadpoints_theta):
    """Evaluate a surface in Cartesian coordinates on a tensor product grid."""
    gamma = np.zeros((len(quadpoints_phi), len(quadpoints_theta), 3))
    for index, phi in enumerate(quadpoints_phi):
        gamma[index, :, :] = surface.cross_section(float(phi), thetas=quadpoints_theta)
    return gamma


def relative_l2(values_a, values_b):
    """Return the relative $L^2$ mismatch between two arrays."""
    denominator = np.linalg.norm(values_b)
    if denominator == 0.0:
        return float("nan")
    return float(np.linalg.norm(values_a - values_b) / denominator)


def surface_geometry_diagnostics(reference_surface, candidate_surface, quadpoints_phi, quadpoints_theta):
    """Compute geometry mismatch metrics between two surface parameterizations."""
    reference_gamma = surface_gamma_on_grid(reference_surface, quadpoints_phi, quadpoints_theta)
    candidate_gamma = surface_gamma_on_grid(candidate_surface, quadpoints_phi, quadpoints_theta)
    displacement = candidate_gamma - reference_gamma
    distances = np.linalg.norm(displacement, axis=2)
    reference_r = np.linalg.norm(reference_gamma[:, :, :2], axis=2)
    candidate_r = np.linalg.norm(candidate_gamma[:, :, :2], axis=2)
    return {
        "nphi": int(len(quadpoints_phi)),
        "ntheta": int(len(quadpoints_theta)),
        "rms_distance": float(np.sqrt(np.mean(distances ** 2))),
        "mean_distance": float(np.mean(distances)),
        "max_distance": float(np.max(distances)),
        "relative_l2": relative_l2(candidate_gamma, reference_gamma),
        "max_abs_R_diff": float(np.max(np.abs(candidate_r - reference_r))),
        "max_abs_Z_diff": float(np.max(np.abs(candidate_gamma[:, :, 2] - reference_gamma[:, :, 2]))),
    }


def surface_restart_payload(surface, iota, G):
    """Serialize a Boozer surface state into a JSON-friendly restart payload."""
    return {
        "surface_class": surface.__class__.__name__,
        "nfp": int(surface.nfp),
        "stellsym": bool(surface.stellsym),
        "mpol": int(surface.mpol),
        "ntor": int(surface.ntor),
        "quadpoints_phi": np.asarray(surface.quadpoints_phi, dtype=float).tolist(),
        "quadpoints_theta": np.asarray(surface.quadpoints_theta, dtype=float).tolist(),
        "dofs": np.asarray(surface.get_dofs(), dtype=float).tolist(),
        "iota": float(iota),
        "G": None if G is None else float(G),
    }


def load_surface_restart(path):
    """Load a saved surface restart payload from disk."""
    restart_path = Path(path).expanduser().resolve()
    with open(restart_path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("surface_class") != "SurfaceXYZTensorFourier":
        raise ValueError(
            f"Unsupported surface restart class {payload.get('surface_class')!r} in {restart_path}."
        )
    payload["path"] = str(restart_path)
    return payload


def surface_from_restart(payload):
    """Rebuild a tensor-Fourier surface from a restart payload."""
    surface = SurfaceXYZTensorFourier(
        nfp=int(payload["nfp"]),
        stellsym=bool(payload["stellsym"]),
        mpol=int(payload["mpol"]),
        ntor=int(payload["ntor"]),
        quadpoints_phi=np.asarray(payload["quadpoints_phi"], dtype=float),
        quadpoints_theta=np.asarray(payload["quadpoints_theta"], dtype=float),
    )
    surface.set_dofs(np.asarray(payload["dofs"], dtype=float))
    return surface


def apply_surface_restart(target_surface, payload):
    """Map a saved surface state onto the target resolution if needed."""
    source_surface = surface_from_restart(payload)
    same_metadata = (
        target_surface.nfp == source_surface.nfp
        and target_surface.stellsym == source_surface.stellsym
        and target_surface.mpol == source_surface.mpol
        and target_surface.ntor == source_surface.ntor
        and np.array_equal(np.asarray(target_surface.quadpoints_phi), np.asarray(source_surface.quadpoints_phi))
        and np.array_equal(np.asarray(target_surface.quadpoints_theta), np.asarray(source_surface.quadpoints_theta))
    )
    if same_metadata:
        target_surface.set_dofs(source_surface.get_dofs().copy())
        return
    gamma = surface_gamma_on_grid(
        source_surface,
        np.asarray(target_surface.quadpoints_phi, dtype=float),
        np.asarray(target_surface.quadpoints_theta, dtype=float),
    )
    target_surface.least_squares_fit(gamma)


def surface_to_vmec_arrays(surface):
    """Convert a Fourier surface into the boundary arrays used by VMEC namelists."""
    n_values = list(range(-surface.ntor, surface.ntor + 1))
    rbc = [[float(surface.get_rc(m, n)) for n in n_values] for m in range(surface.mpol + 1)]
    zbs = [[float(surface.get_zs(m, n)) for n in n_values] for m in range(surface.mpol + 1)]
    arrays = {
        "rbc": rbc,
        "zbs": zbs,
        "start_index": [-surface.ntor, 0],
    }
    if not surface.stellsym:
        arrays["rbs"] = [[float(surface.get_rs(m, n)) for n in n_values] for m in range(surface.mpol + 1)]
        arrays["zbc"] = [[float(surface.get_zc(m, n)) for n in n_values] for m in range(surface.mpol + 1)]
    return arrays


def write_vmec_input_from_template(surface, filename, template_filename, bs):
    """Write a VMEC input file for the supplied RZ surface."""
    template_path = Path(template_filename).expanduser().resolve()
    vmec_input_path = Path(filename).expanduser().resolve()
    vmec_input_path.parent.mkdir(parents=True, exist_ok=True)

    namelist = f90nml.read(str(template_path))
    indata = namelist["indata"]
    boundary = surface_to_vmec_arrays(surface)

    indata["lasym"] = not surface.stellsym
    indata["nfp"] = int(surface.nfp)
    indata["mpol"] = int(surface.mpol)
    indata["ntor"] = int(surface.ntor)
    indata["phiedge"] = float(ToroidalFlux(surface, bs).J())
    indata["rbc"] = boundary["rbc"]
    indata["zbs"] = boundary["zbs"]
    indata.start_index["rbc"] = boundary["start_index"]
    indata.start_index["zbs"] = boundary["start_index"]
    if not surface.stellsym:
        indata["rbs"] = boundary["rbs"]
        indata["zbc"] = boundary["zbc"]
        indata.start_index["rbs"] = boundary["start_index"]
        indata.start_index["zbc"] = boundary["start_index"]

    f90nml.write(namelist, str(vmec_input_path), force=True)
    return {
        "path": str(vmec_input_path),
        "template": str(template_path),
        "phiedge": float(indata["phiedge"]),
    }


def fit_vmec_export_surface_candidate(source_surface, export_mpol, export_ntor, fit_nphi, fit_ntheta, check_nphi, check_ntheta):
    """Fit one candidate RZ export surface and measure its geometry error."""
    fit_phi = np.linspace(0.0, 1.0 / source_surface.nfp, fit_nphi, endpoint=False)
    fit_theta = np.linspace(0.0, 1.0, fit_ntheta, endpoint=False)
    export_surface = SurfaceRZFourier(
        nfp=source_surface.nfp,
        stellsym=source_surface.stellsym,
        mpol=export_mpol,
        ntor=export_ntor,
        quadpoints_phi=fit_phi,
        quadpoints_theta=fit_theta,
    )
    export_surface.least_squares_fit(surface_gamma_on_grid(source_surface, fit_phi, fit_theta))

    check_phi = np.linspace(0.0, 1.0 / source_surface.nfp, check_nphi, endpoint=False)
    check_theta = np.linspace(0.0, 1.0, check_ntheta, endpoint=False)
    diagnostics = surface_geometry_diagnostics(source_surface, export_surface, check_phi, check_theta)
    diagnostics.update({
        "export_mpol": int(export_mpol),
        "export_ntor": int(export_ntor),
        "fit_nphi": int(fit_nphi),
        "fit_ntheta": int(fit_ntheta),
    })
    return export_surface, diagnostics


def prepare_vmec_export_surface(source_surface):
    """Choose an RZ export surface that best matches the optimized XYZ surface."""
    explicit_mpol = _env_int("SIMSOPT_BOOZER_QI_VMEC_EXPORT_MPOL", 0)
    explicit_ntor = _env_int("SIMSOPT_BOOZER_QI_VMEC_EXPORT_NTOR", 0)
    mode_scale_max = max(1, _env_int("SIMSOPT_BOOZER_QI_VMEC_EXPORT_MODE_SCALE_MAX", 3))
    grid_scale = max(2, _env_int("SIMSOPT_BOOZER_QI_VMEC_EXPORT_GRID_SCALE", 4))
    check_nphi = _env_int(
        "SIMSOPT_BOOZER_QI_VMEC_EXPORT_CHECK_NPHI",
        max(61, 2 * grid_scale * (2 * source_surface.ntor + 1)),
    )
    check_ntheta = _env_int(
        "SIMSOPT_BOOZER_QI_VMEC_EXPORT_CHECK_NTHETA",
        max(91, 2 * grid_scale * (2 * source_surface.mpol + 1)),
    )
    target_relative_l2 = _env_float("SIMSOPT_BOOZER_QI_VMEC_EXPORT_MAX_RELATIVE_L2", 1.0e-3)
    strict = _env_bool("SIMSOPT_BOOZER_QI_VMEC_EXPORT_STRICT", True)

    if explicit_mpol < 0 or explicit_ntor < 0:
        raise ValueError("SIMSOPT_BOOZER_QI_VMEC_EXPORT_MPOL and _NTOR must be non-negative.")

    if explicit_mpol > 0 or explicit_ntor > 0:
        candidate_resolutions = [(
            explicit_mpol if explicit_mpol > 0 else source_surface.mpol,
            explicit_ntor if explicit_ntor > 0 else source_surface.ntor,
        )]
    else:
        candidate_resolutions = [
            (scale * source_surface.mpol, scale * source_surface.ntor)
            for scale in range(1, mode_scale_max + 1)
        ]

    selected_surface = None
    selected_diagnostics = None
    candidate_diagnostics = []
    for export_mpol, export_ntor in candidate_resolutions:
        fit_nphi = max(2 * export_ntor + 1, grid_scale * (2 * source_surface.ntor + 1))
        fit_ntheta = max(2 * export_mpol + 1, grid_scale * (2 * source_surface.mpol + 1))
        candidate_surface, diagnostics = fit_vmec_export_surface_candidate(
            source_surface,
            export_mpol,
            export_ntor,
            fit_nphi,
            fit_ntheta,
            check_nphi,
            check_ntheta,
        )
        candidate_diagnostics.append(diagnostics)
        if selected_diagnostics is None or diagnostics["relative_l2"] < selected_diagnostics["relative_l2"]:
            selected_surface = candidate_surface
            selected_diagnostics = diagnostics

    if selected_surface is None or selected_diagnostics is None:
        raise RuntimeError("Failed to construct a VMEC export surface candidate.")

    report = {
        "source_surface": {
            "surface_class": source_surface.__class__.__name__,
            "nfp": int(source_surface.nfp),
            "stellsym": bool(source_surface.stellsym),
            "mpol": int(source_surface.mpol),
            "ntor": int(source_surface.ntor),
            "quadpoints_phi": int(len(source_surface.quadpoints_phi)),
            "quadpoints_theta": int(len(source_surface.quadpoints_theta)),
        },
        "target_relative_l2": float(target_relative_l2),
        "strict": bool(strict),
        "selected": selected_diagnostics,
        "candidates": candidate_diagnostics,
    }
    if strict and selected_diagnostics["relative_l2"] > target_relative_l2:
        raise RuntimeError(
            "VMEC export surface fit did not meet the requested gamma agreement: "
            f"relative_l2={selected_diagnostics['relative_l2']:.3e}, target={target_relative_l2:.3e}."
        )
    return selected_surface, report


def set_equal_3d_limits(ax, xyz_points):
    """Apply equal axis limits to a 3D matplotlib axis."""
    mins = np.min(xyz_points, axis=0)
    maxs = np.max(xyz_points, axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    if radius <= 0.0:
        radius = 1.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def plot_vmec_export_surfaces(source_surface, export_surface, diagnostics, cross_section_path, surface_plot_path):
    """Render diagnostics plots comparing the XYZ and exported RZ surfaces."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors

    cross_section_path = Path(cross_section_path)
    surface_plot_path = Path(surface_plot_path)
    cross_section_path.parent.mkdir(parents=True, exist_ok=True)
    surface_plot_path.parent.mkdir(parents=True, exist_ok=True)

    cross_section_theta = np.linspace(0.0, 1.0, 361, endpoint=False)
    cross_section_phi = np.linspace(0.0, 1.0 / source_surface.nfp, 6, endpoint=False)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for index, (ax, phi) in enumerate(zip(axes.flat, cross_section_phi)):
        source_section = source_surface.cross_section(float(phi), thetas=cross_section_theta)
        export_section = export_surface.cross_section(float(phi), thetas=cross_section_theta)
        source_r = np.linalg.norm(source_section[:, :2], axis=1)
        export_r = np.linalg.norm(export_section[:, :2], axis=1)
        ax.plot(np.append(source_r, source_r[0]), np.append(source_section[:, 2], source_section[0, 2]), label="XYZ source", lw=2.0)
        ax.plot(np.append(export_r, export_r[0]), np.append(export_section[:, 2], export_section[0, 2]), label="RZ export", lw=1.6, linestyle="--")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_title(f"phi={phi * source_surface.nfp:.3f} field periods")
        ax.set_xlabel("R")
        ax.set_ylabel("Z")
        if index == 0:
            ax.legend(frameon=False)
    fig.suptitle(
        "VMEC export cross sections\n"
        f"selected relative_l2={diagnostics['selected']['relative_l2']:.3e}",
        fontsize=14,
    )
    fig.savefig(cross_section_path, dpi=180)
    plt.close(fig)

    surface_nphi = max(41, diagnostics["selected"]["nphi"])
    surface_ntheta = max(61, diagnostics["selected"]["ntheta"])
    plot_phi = np.linspace(0.0, 1.0 / source_surface.nfp, surface_nphi, endpoint=False)
    plot_theta = np.linspace(0.0, 1.0, surface_ntheta, endpoint=False)
    source_gamma = surface_gamma_on_grid(source_surface, plot_phi, plot_theta)
    export_gamma = surface_gamma_on_grid(export_surface, plot_phi, plot_theta)
    distance = np.linalg.norm(export_gamma - source_gamma, axis=2)
    z_values = np.concatenate((source_gamma[:, :, 2].ravel(), export_gamma[:, :, 2].ravel()))
    z_norm = colors.Normalize(vmin=float(np.min(z_values)), vmax=float(np.max(z_values)))
    max_distance = float(np.max(distance))
    min_distance = float(np.min(distance))
    d_norm = colors.Normalize(vmin=min_distance, vmax=max_distance if max_distance > min_distance else min_distance + 1.0)

    fig = plt.figure(figsize=(18, 6), constrained_layout=True)
    ax_source = fig.add_subplot(1, 3, 1, projection="3d")
    ax_export = fig.add_subplot(1, 3, 2, projection="3d")
    ax_distance = fig.add_subplot(1, 3, 3, projection="3d")
    ax_source.plot_surface(
        source_gamma[:, :, 0],
        source_gamma[:, :, 1],
        source_gamma[:, :, 2],
        facecolors=cm.cividis(z_norm(source_gamma[:, :, 2])),
        linewidth=0,
        antialiased=True,
        shade=False,
    )
    ax_export.plot_surface(
        export_gamma[:, :, 0],
        export_gamma[:, :, 1],
        export_gamma[:, :, 2],
        facecolors=cm.cividis(z_norm(export_gamma[:, :, 2])),
        linewidth=0,
        antialiased=True,
        shade=False,
    )
    ax_distance.plot_surface(
        export_gamma[:, :, 0],
        export_gamma[:, :, 1],
        export_gamma[:, :, 2],
        facecolors=cm.inferno(d_norm(distance)),
        linewidth=0,
        antialiased=True,
        shade=False,
    )
    all_points = np.vstack((source_gamma.reshape(-1, 3), export_gamma.reshape(-1, 3)))
    for ax, title in [
        (ax_source, "XYZ source surface"),
        (ax_export, "RZ surface exported to VMEC"),
        (ax_distance, "Export pointwise distance"),
    ]:
        set_equal_3d_limits(ax, all_points)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=25, azim=40)
        ax.set_title(title)
    fig.colorbar(cm.ScalarMappable(norm=d_norm, cmap=cm.inferno), ax=ax_distance, shrink=0.75, pad=0.08, label="|delta gamma|")
    fig.savefig(surface_plot_path, dpi=180)
    plt.close(fig)


def export_final_artifacts(config, bs, boozer_surface):
    """Write the optimized coils, surface restart, and VMEC export artifacts."""
    exported = {}

    coils_path = Path(config.export_coils_json)
    coils_path.parent.mkdir(parents=True, exist_ok=True)
    bs.save(str(coils_path))
    exported["coils_json"] = str(coils_path)

    restart_path = Path(config.export_surface_restart)
    restart_path.parent.mkdir(parents=True, exist_ok=True)
    restart_payload = surface_restart_payload(
        boozer_surface.surface,
        iota=boozer_surface.res["iota"],
        G=boozer_surface.res["G"],
    )
    with open(restart_path, "w", encoding="utf-8") as stream:
        json.dump(restart_payload, stream, indent=2)
    exported["surface_restart"] = str(restart_path)

    vmec_input_path = Path(config.export_vmec_input)
    rz_surface, export_diagnostics = prepare_vmec_export_surface(boozer_surface.surface)
    diagnostics_path = Path(config.export_vmec_diagnostics)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(diagnostics_path, "w", encoding="utf-8") as stream:
        json.dump(export_diagnostics, stream, indent=2)
    plot_vmec_export_surfaces(
        boozer_surface.surface,
        rz_surface,
        export_diagnostics,
        config.export_vmec_cross_section_plot,
        config.export_vmec_surface_plot,
    )
    vmec_export = write_vmec_input_from_template(rz_surface, vmec_input_path, config.vmec_template_input, bs)
    exported["vmec_input"] = vmec_export["path"]
    exported["vmec_template"] = vmec_export["template"]
    exported["vmec_phiedge"] = vmec_export["phiedge"]
    exported["vmec_export_diagnostics"] = str(diagnostics_path)
    exported["vmec_export_relative_l2"] = float(export_diagnostics["selected"]["relative_l2"])
    exported["vmec_export_cross_sections_plot"] = str(Path(config.export_vmec_cross_section_plot))
    exported["vmec_export_surface_plot"] = str(Path(config.export_vmec_surface_plot))
    return exported