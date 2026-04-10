#!/usr/bin/env python3

import json
import os
import sys
import time
from pathlib import Path

import f90nml

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
filtered_meta_path = []
for finder in sys.meta_path:
    known_source_files = getattr(finder, "known_source_files", None)
    if isinstance(known_source_files, dict):
        simsopt_init = known_source_files.get("simsopt")
        if isinstance(simsopt_init, str) and not simsopt_init.startswith(str(REPO_ROOT)):
            continue
    filtered_meta_path.append(finder)
sys.meta_path[:] = filtered_meta_path
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
from scipy.optimize import BFGS as TrustRegionBFGS, minimize

from simsopt._core.optimizable import load
from simsopt.configs import get_data
from simsopt.field import BiotSavart
from simsopt.geo import SurfaceXYZTensorFourier, SurfaceRZFourier, BoozerSurface, curves_to_vtk, boozer_surface_residual, \
    Volume, MajorRadius, CurveLength, NonQuasiIsodynamicRatio, Iotas, ToroidalFlux
from simsopt.mhd import Vmec
from simsopt.objectives import QuadraticPenalty
from simsopt.util import in_github_actions

r"""
This example optimizes the NCSX coils and currents for QI on a single surface. The objective is

    J = J_QI
        + 0.5*(iota - iota_0)**2
        + 0.5*(major_radius - target_major_radius)**2
        + 0.5*max(sum(CurveLength) - CurveLengthTarget, 0)**2

The workflow intentionally stays close to boozerQA.py. We first compute a
single Boozer surface close to the magnetic axis, then optimize the coils using
the milestone-1 quasi-isodynamic objective NonQuasiIsodynamicRatio on that
surface. The rotational transform, major radius, and total coil length are kept
close to their initial values with quadratic penalties.

Environment variables can be used to reduce runtime for testing:

    SIMSOPT_BOOZER_QI_MPOL
    SIMSOPT_BOOZER_QI_NTOR
    SIMSOPT_BOOZER_QI_SDIM
    SIMSOPT_BOOZER_QI_NPHI
    SIMSOPT_BOOZER_QI_NALPHA
    SIMSOPT_BOOZER_QI_NBJ
    SIMSOPT_BOOZER_QI_NPHI_OUT
    SIMSOPT_BOOZER_QI_MAXITER
    SIMSOPT_BOOZER_QI_SKIP_TAYLOR
    SIMSOPT_BOOZER_QI_BOOZER_TYPE
    SIMSOPT_BOOZER_QI_COMPARE_FD
    SIMSOPT_BOOZER_QI_COMPARE_FD_DIRS
    SIMSOPT_BOOZER_QI_COMPARE_FD_EPS
    SIMSOPT_BOOZER_QI_COMPARE_FD_SUBSPACE
    SIMSOPT_BOOZER_QI_COMPARE_FD_OPT
    SIMSOPT_BOOZER_QI_COMPARE_FD_OPT_MAXITER
    SIMSOPT_BOOZER_QI_COMPARE_OPTIMIZERS
    SIMSOPT_BOOZER_QI_COMPARE_OPTIMIZERS_LIST
    SIMSOPT_BOOZER_QI_COMPARE_OPTIMIZERS_SUBSPACE
    SIMSOPT_BOOZER_QI_COMPARE_OPTIMIZERS_MAXITER
    SIMSOPT_BOOZER_QI_EXACT_REPORT
    SIMSOPT_BOOZER_QI_EXACT_REPORT_MAXITER
    SIMSOPT_BOOZER_QI_WRITE_VTK
    SIMSOPT_BOOZER_QI_OUT_DIR
    SIMSOPT_BOOZER_QI_EXPORT_COILS_JSON
    SIMSOPT_BOOZER_QI_EXPORT_SURFACE_RESTART
    SIMSOPT_BOOZER_QI_EXPORT_VMEC_INPUT
    SIMSOPT_BOOZER_QI_VMEC_TEMPLATE_INPUT
    SIMSOPT_BOOZER_QI_VMEC_EXPORT_MPOL
    SIMSOPT_BOOZER_QI_VMEC_EXPORT_NTOR
    SIMSOPT_BOOZER_QI_VMEC_EXPORT_MODE_SCALE_MAX
    SIMSOPT_BOOZER_QI_VMEC_EXPORT_GRID_SCALE
    SIMSOPT_BOOZER_QI_VMEC_EXPORT_CHECK_NPHI
    SIMSOPT_BOOZER_QI_VMEC_EXPORT_CHECK_NTHETA
    SIMSOPT_BOOZER_QI_VMEC_EXPORT_MAX_RELATIVE_L2
    SIMSOPT_BOOZER_QI_VMEC_EXPORT_STRICT
    SIMSOPT_BOOZER_QI_VMEC_EXPORT_DIAGNOSTICS
    SIMSOPT_BOOZER_QI_VMEC_EXPORT_CROSS_SECTION_PLOT
    SIMSOPT_BOOZER_QI_VMEC_EXPORT_SURFACE_PLOT
    SIMSOPT_BOOZER_QI_COILS_JSON
    SIMSOPT_BOOZER_QI_SURFACE_RESTART
    SIMSOPT_BOOZER_QI_VMEC
    SIMSOPT_BOOZER_QI_VMEC_INPUT
"""


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
    raw = os.environ.get(name)
    if raw is None:
        raw = default
    return [item.strip() for item in raw.split(",") if item.strip()]


def _sorted_optimizables(keys, class_name):
    return sorted(
        [opt for opt in keys if opt.__class__.__name__ == class_name],
        key=lambda opt: getattr(opt, "name", class_name),
    )


def _load_seed_biot_savart(filename):
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


def _solve_boozer_surface_with_seed_fallback(boozer_surface, iota, G, tol, maxiter, verbose):
    boozer_surface.need_to_run_code = True
    return boozer_surface.run_code(iota=iota, G=G)


def _surface_gamma_on_grid(surface, quadpoints_phi, quadpoints_theta):
    gamma = np.zeros((len(quadpoints_phi), len(quadpoints_theta), 3))
    for index, phi in enumerate(quadpoints_phi):
        gamma[index, :, :] = surface.cross_section(float(phi), thetas=quadpoints_theta)
    return gamma


def _relative_l2(values_a, values_b):
    denominator = np.linalg.norm(values_b)
    if denominator == 0.0:
        return float("nan")
    return float(np.linalg.norm(values_a - values_b) / denominator)


def _surface_geometry_diagnostics(reference_surface, candidate_surface, quadpoints_phi, quadpoints_theta):
    reference_gamma = _surface_gamma_on_grid(reference_surface, quadpoints_phi, quadpoints_theta)
    candidate_gamma = _surface_gamma_on_grid(candidate_surface, quadpoints_phi, quadpoints_theta)
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
        "relative_l2": _relative_l2(candidate_gamma, reference_gamma),
        "max_abs_R_diff": float(np.max(np.abs(candidate_r - reference_r))),
        "max_abs_Z_diff": float(np.max(np.abs(candidate_gamma[:, :, 2] - reference_gamma[:, :, 2]))),
    }


def _surface_restart_payload(surface, iota, G):
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


def _load_surface_restart(path):
    restart_path = Path(path).expanduser().resolve()
    with open(restart_path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("surface_class") != "SurfaceXYZTensorFourier":
        raise ValueError(
            f"Unsupported surface restart class {payload.get('surface_class')!r} in {restart_path}."
        )
    payload["path"] = str(restart_path)
    return payload


def _surface_from_restart(payload):
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


def _apply_surface_restart(target_surface, payload):
    source_surface = _surface_from_restart(payload)
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

    gamma = _surface_gamma_on_grid(
        source_surface,
        np.asarray(target_surface.quadpoints_phi, dtype=float),
        np.asarray(target_surface.quadpoints_theta, dtype=float),
    )
    target_surface.least_squares_fit(gamma)


def _surface_to_vmec_arrays(surface):
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


def _write_vmec_input_from_template(surface, filename, template_filename):
    template_path = Path(template_filename).expanduser().resolve()
    vmec_input_path = Path(filename).expanduser().resolve()
    vmec_input_path.parent.mkdir(parents=True, exist_ok=True)

    namelist = f90nml.read(str(template_path))
    indata = namelist["indata"]
    boundary = _surface_to_vmec_arrays(surface)

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


def _fit_vmec_export_surface_candidate(source_surface, export_mpol, export_ntor, fit_nphi, fit_ntheta, check_nphi, check_ntheta):
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
    export_surface.least_squares_fit(_surface_gamma_on_grid(source_surface, fit_phi, fit_theta))

    check_phi = np.linspace(0.0, 1.0 / source_surface.nfp, check_nphi, endpoint=False)
    check_theta = np.linspace(0.0, 1.0, check_ntheta, endpoint=False)
    diagnostics = _surface_geometry_diagnostics(source_surface, export_surface, check_phi, check_theta)
    diagnostics.update({
        "export_mpol": int(export_mpol),
        "export_ntor": int(export_ntor),
        "fit_nphi": int(fit_nphi),
        "fit_ntheta": int(fit_ntheta),
    })
    return export_surface, diagnostics


def _prepare_vmec_export_surface(source_surface):
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
        candidate_surface, diagnostics = _fit_vmec_export_surface_candidate(
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
            f"relative_l2={selected_diagnostics['relative_l2']:.3e}, "
            f"target={target_relative_l2:.3e}."
        )

    return selected_surface, report


def _set_equal_3d_limits(ax, xyz_points):
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


def _plot_vmec_export_surfaces(source_surface, export_surface, diagnostics, cross_section_path, surface_plot_path):
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
    source_gamma = _surface_gamma_on_grid(source_surface, plot_phi, plot_theta)
    export_gamma = _surface_gamma_on_grid(export_surface, plot_phi, plot_theta)
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
        _set_equal_3d_limits(ax, all_points)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=25, azim=40)
        ax.set_title(title)
    fig.colorbar(cm.ScalarMappable(norm=d_norm, cmap=cm.inferno), ax=ax_distance, shrink=0.75, pad=0.08, label="|delta gamma|")
    fig.savefig(surface_plot_path, dpi=180)
    plt.close(fig)


OUT_DIR = os.environ.get("SIMSOPT_BOOZER_QI_OUT_DIR", "./output/")
WRITE_VTK = _env_bool("SIMSOPT_BOOZER_QI_WRITE_VTK", True)
SKIP_TAYLOR = _env_bool("SIMSOPT_BOOZER_QI_SKIP_TAYLOR", True)
VMEC_GEOMETRY = os.environ.get("SIMSOPT_BOOZER_QI_VMEC")
VMEC_INPUT = os.environ.get("SIMSOPT_BOOZER_QI_VMEC_INPUT")
COILS_JSON = os.environ.get("SIMSOPT_BOOZER_QI_COILS_JSON")
BOOZER_VERBOSE = _env_bool("SIMSOPT_BOOZER_QI_BOOZER_VERBOSE", False)
BOOZER_TYPE = os.environ.get("SIMSOPT_BOOZER_QI_BOOZER_TYPE", "ls").strip().lower()
EXACT_LS_FALLBACK = _env_bool("SIMSOPT_BOOZER_QI_EXACT_LS_FALLBACK", True)
EXACT_LS_WEIGHT = _env_float("SIMSOPT_BOOZER_QI_EXACT_LS_WEIGHT", 1.0)
EXACT_LS_MAXITER = _env_int("SIMSOPT_BOOZER_QI_EXACT_LS_MAXITER", 160)
OPT_METHOD = os.environ.get("SIMSOPT_BOOZER_QI_OPT_METHOD", "trust-constr")
STEP_RMS_LIMIT = _env_float("SIMSOPT_BOOZER_QI_STEP_RMS_LIMIT", 2.0e-2)
STEP_PENALTY = _env_float("SIMSOPT_BOOZER_QI_STEP_PENALTY", 1.0e6)
MAXLS = _env_int("SIMSOPT_BOOZER_QI_MAXLS", 10)
MAXCOR = _env_int("SIMSOPT_BOOZER_QI_MAXCOR", 20)
HISTORY_PATH = os.environ.get("SIMSOPT_BOOZER_QI_HISTORY_PATH")
EXPORT_COILS_JSON = os.environ.get(
    "SIMSOPT_BOOZER_QI_EXPORT_COILS_JSON",
    os.path.join(OUT_DIR, "biot_savart_opt.json"),
)
EXPORT_SURFACE_RESTART = os.environ.get(
    "SIMSOPT_BOOZER_QI_EXPORT_SURFACE_RESTART",
    os.path.join(OUT_DIR, "boozer_surface_restart.json"),
)
EXPORT_VMEC_INPUT = os.environ.get(
    "SIMSOPT_BOOZER_QI_EXPORT_VMEC_INPUT",
    os.path.join(OUT_DIR, "input.boozer_qi"),
)
EXPORT_VMEC_DIAGNOSTICS = os.environ.get(
    "SIMSOPT_BOOZER_QI_VMEC_EXPORT_DIAGNOSTICS",
    os.path.join(OUT_DIR, "vmec_export_surface_diagnostics.json"),
)
EXPORT_VMEC_CROSS_SECTION_PLOT = os.environ.get(
    "SIMSOPT_BOOZER_QI_VMEC_EXPORT_CROSS_SECTION_PLOT",
    os.path.join(OUT_DIR, "vmec_export_cross_sections.png"),
)
EXPORT_VMEC_SURFACE_PLOT = os.environ.get(
    "SIMSOPT_BOOZER_QI_VMEC_EXPORT_SURFACE_PLOT",
    os.path.join(OUT_DIR, "vmec_export_surface_3d.png"),
)
VMEC_TEMPLATE_INPUT = os.environ.get(
    "SIMSOPT_BOOZER_QI_VMEC_TEMPLATE_INPUT",
    str(REPO_ROOT / "src" / "simsopt" / "mhd" / "input.default"),
)
SURFACE_RESTART = os.environ.get("SIMSOPT_BOOZER_QI_SURFACE_RESTART")
COMPARE_FD = _env_bool("SIMSOPT_BOOZER_QI_COMPARE_FD", False)
COMPARE_FD_DIRS = _env_int("SIMSOPT_BOOZER_QI_COMPARE_FD_DIRS", 2)
COMPARE_FD_EPS = _env_float("SIMSOPT_BOOZER_QI_COMPARE_FD_EPS", 2.0 ** -18)
COMPARE_FD_SUBSPACE = _env_int("SIMSOPT_BOOZER_QI_COMPARE_FD_SUBSPACE", 6)
COMPARE_FD_OPT = _env_bool("SIMSOPT_BOOZER_QI_COMPARE_FD_OPT", False)
COMPARE_FD_OPT_MAXITER = _env_int("SIMSOPT_BOOZER_QI_COMPARE_FD_OPT_MAXITER", 1)
COMPARE_OPTIMIZERS = _env_bool("SIMSOPT_BOOZER_QI_COMPARE_OPTIMIZERS", False)
COMPARE_OPTIMIZERS_LIST = _env_str_list(
    "SIMSOPT_BOOZER_QI_COMPARE_OPTIMIZERS_LIST",
    "L-BFGS-B,BFGS,trust-constr",
)
COMPARE_OPTIMIZERS_SUBSPACE = _env_int(
    "SIMSOPT_BOOZER_QI_COMPARE_OPTIMIZERS_SUBSPACE",
    COMPARE_FD_SUBSPACE,
)
COMPARE_OPTIMIZERS_MAXITER = _env_int(
    "SIMSOPT_BOOZER_QI_COMPARE_OPTIMIZERS_MAXITER",
    max(2, COMPARE_FD_OPT_MAXITER),
)
EXACT_REPORT = _env_bool("SIMSOPT_BOOZER_QI_EXACT_REPORT", False)
EXACT_REPORT_MAXITER = _env_int("SIMSOPT_BOOZER_QI_EXACT_REPORT_MAXITER", 20)

if BOOZER_TYPE not in {"exact", "ls"}:
    raise ValueError(f"SIMSOPT_BOOZER_QI_BOOZER_TYPE must be 'exact' or 'ls', got {BOOZER_TYPE!r}.")

if WRITE_VTK:
    os.makedirs(OUT_DIR, exist_ok=True)
if HISTORY_PATH is None:
    HISTORY_PATH = os.path.join(OUT_DIR, "optimization_history.json")

print("Running 2_Intermediate/boozerQI.py")
print("================================")
print(f"Optimizer method={OPT_METHOD}, step_rms_limit={STEP_RMS_LIMIT:.2e}")
print(
    f"Boozer surface mode={BOOZER_TYPE}, skip_taylor={SKIP_TAYLOR}, compare_fd={COMPARE_FD}, "
    f"compare_optimizers={COMPARE_OPTIMIZERS}, exact_report={EXACT_REPORT}"
)

if sum(value is not None for value in [SURFACE_RESTART, VMEC_GEOMETRY, VMEC_INPUT]) > 1:
    raise ValueError(
        "Specify at most one of SIMSOPT_BOOZER_QI_SURFACE_RESTART, "
        "SIMSOPT_BOOZER_QI_VMEC, and SIMSOPT_BOOZER_QI_VMEC_INPUT."
    )

surface_reference = None
surface_reference_label = None
surface_restart_state = None

if SURFACE_RESTART:
    surface_restart_state = _load_surface_restart(SURFACE_RESTART)
    surface_reference = Path(surface_restart_state["path"])
    surface_nfp = int(surface_restart_state["nfp"])
    surface_reference_label = f"Boozer surface restart from {surface_reference}"
elif VMEC_GEOMETRY:
    vmec_path = Path(VMEC_GEOMETRY).expanduser().resolve()
    surface_reference = vmec_path
    surface_nfp = SurfaceRZFourier.from_wout(str(vmec_path), nphi=16, ntheta=16, range="half period").nfp
    surface_reference_label = f"VMEC boundary from {vmec_path}"
elif VMEC_INPUT:
    vmec_input_path = Path(VMEC_INPUT).expanduser().resolve()
    surface_reference = vmec_input_path
    surface_nfp = SurfaceRZFourier.from_vmec_input(str(vmec_input_path), range="half period", nphi=16, ntheta=16).nfp
    surface_reference_label = f"VMEC input boundary from {vmec_input_path}"
else:
    surface_nfp = None

if COILS_JSON:
    coils_path, base_curves, base_currents, bs = _load_seed_biot_savart(COILS_JSON)
    if surface_nfp is None:
        raise ValueError(
            "SIMSOPT_BOOZER_QI_COILS_JSON requires a matching surface initializer via "
            "SIMSOPT_BOOZER_QI_VMEC or SIMSOPT_BOOZER_QI_VMEC_INPUT."
        )
    nfp = surface_nfp
    print(f"Using Biot-Savart seed from {coils_path}")
else:
    base_curves, base_currents, ma, nfp, bs = get_data("ncsx")
    if surface_nfp is not None and surface_nfp != nfp:
        raise ValueError(
            f"Surface initializer nfp={surface_nfp} does not match the default coil configuration nfp={nfp}."
        )
    ma = ma

all_curves = [coil.curve for coil in bs.coils]
bs_qi = BiotSavart(bs.coils)
current_sum = nfp * sum(abs(c.get_value()) for c in base_currents)
G0 = 2. * np.pi * current_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))

mpol_default = 3 if in_github_actions else 6
ntor_default = 3 if in_github_actions else 6
mpol = _env_int("SIMSOPT_BOOZER_QI_MPOL", mpol_default)
ntor = _env_int("SIMSOPT_BOOZER_QI_NTOR", ntor_default)
stellsym = True

phis = np.linspace(0, 1/nfp, 2*ntor+1, endpoint=False)
thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
s = SurfaceXYZTensorFourier(
    mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)

if surface_restart_state is not None:
    if surface_nfp != nfp:
        raise ValueError(
            f"Surface restart nfp={surface_nfp} does not match the coil configuration nfp={nfp}."
        )
    print(f"Using surface initializer from {surface_reference_label}")
    _apply_surface_restart(s, surface_restart_state)
elif surface_reference is not None:
    if surface_nfp != nfp:
        raise ValueError(
            f"Surface initializer nfp={surface_nfp} does not match the coil configuration nfp={nfp}."
        )
    print(f"Using surface initializer from {surface_reference_label}")
    if VMEC_GEOMETRY:
        reference_surface = SurfaceRZFourier.from_wout(
            str(surface_reference),
            nphi=len(phis),
            ntheta=len(thetas),
            range="half period",
        )
    else:
        reference_surface = SurfaceRZFourier.from_vmec_input(
            str(surface_reference),
            range="half period",
            nphi=len(phis),
            ntheta=len(thetas),
        )
    s.least_squares_fit(reference_surface.gamma())
else:
    s.fit_to_curve(ma, 0.1, flip_theta=True)

iota = float(surface_restart_state["iota"]) if surface_restart_state is not None else -0.406

vol = Volume(s)
vol_target = vol.J()

boozer_surface = BoozerSurface(
    bs,
    s,
    vol,
    vol_target,
    constraint_weight=EXACT_LS_WEIGHT if BOOZER_TYPE == "ls" else None,
    options={
        "verbose": BOOZER_VERBOSE,
        "exact_ls_fallback": EXACT_LS_FALLBACK,
        "exact_ls_constraint_weight": EXACT_LS_WEIGHT,
        "exact_ls_maxiter": EXACT_LS_MAXITER,
        "exact_ls_tol": 1e-10,
    },
)
res = _solve_boozer_surface_with_seed_fallback(
    boozer_surface,
    iota=iota,
    G=(float(surface_restart_state["G"]) if surface_restart_state is not None and surface_restart_state.get("G") is not None else G0),
    tol=1e-13,
    maxiter=20,
    verbose=BOOZER_VERBOSE,
)

out_res = boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)[0]
initial_iter = res.get("iter")
if initial_iter is None and "info" in res:
    initial_iter = getattr(res["info"], "nfev", None)
if BOOZER_TYPE == "ls":
    initial_solver = "LS"
elif "iter" in res:
    initial_solver = "NEWTON"
else:
    initial_solver = "SEEDED"
print(
    f"{initial_solver} {res['success']}: iter={initial_iter}, iota={res['iota']:.3f}, "
    f"vol={s.volume():.3f}, ||residual||={np.linalg.norm(out_res):.3e}"
)
initial_iota_target = float(res['iota'])

sDIM_default = 8 if in_github_actions else 20
nphi_default = 31 if in_github_actions else 151
nalpha_default = 5 if in_github_actions else 31
nBj_default = 7 if in_github_actions else 51
nphi_out_default = 61 if in_github_actions else 2000

sDIM = _env_int("SIMSOPT_BOOZER_QI_SDIM", sDIM_default)
nphi_qi = _env_int("SIMSOPT_BOOZER_QI_NPHI", nphi_default)
nalpha = _env_int("SIMSOPT_BOOZER_QI_NALPHA", nalpha_default)
nBj = _env_int("SIMSOPT_BOOZER_QI_NBJ", nBj_default)
nphi_out = _env_int("SIMSOPT_BOOZER_QI_NPHI_OUT", nphi_out_default)

mr = MajorRadius(boozer_surface)
ls = [CurveLength(c) for c in base_curves]
initial_major_radius_target = float(mr.J())
initial_curve_length_target = float(sum(ls).J())

J_major_radius = QuadraticPenalty(mr, initial_major_radius_target, 'identity')
J_iotas = QuadraticPenalty(Iotas(boozer_surface), initial_iota_target, 'identity')
J_nonQIRatio = NonQuasiIsodynamicRatio(
    boozer_surface,
    bs_qi,
    sDIM=sDIM,
    nphi=nphi_qi,
    nalpha=nalpha,
    nBj=nBj,
    nphi_out=nphi_out,
)
Jls = QuadraticPenalty(sum(ls), initial_curve_length_target, 'max')

JF = J_nonQIRatio + J_iotas + J_major_radius + Jls

if WRITE_VTK:
    curves_to_vtk(all_curves, os.path.join(OUT_DIR, "curves_init"))
    boozer_surface.surface.to_vtk(os.path.join(OUT_DIR, "surf_init"))

base_currents[0].fix_all()

last_good = {
    "dofs": JF.x.copy(),
    "surface": boozer_surface.surface.x.copy(),
    "iota": boozer_surface.res['iota'],
    "G": boozer_surface.res['G'],
    "J": None,
    "grad": None,
}
best_good = {
    "dofs": JF.x.copy(),
    "surface": boozer_surface.surface.x.copy(),
    "iota": boozer_surface.res['iota'],
    "G": boozer_surface.res['G'],
    "J": None,
    "grad": None,
}
history = {
    "settings": {
        "boozer_type": BOOZER_TYPE,
        "optimizer": OPT_METHOD,
        "step_rms_limit": STEP_RMS_LIMIT,
        "step_penalty": STEP_PENALTY,
        "maxls": MAXLS,
        "maxcor": MAXCOR,
        "skip_taylor": SKIP_TAYLOR,
        "compare_fd": COMPARE_FD,
        "compare_optimizers": COMPARE_OPTIMIZERS,
        "compare_optimizers_list": COMPARE_OPTIMIZERS_LIST,
        "compare_optimizers_subspace": COMPARE_OPTIMIZERS_SUBSPACE,
        "compare_optimizers_maxiter": COMPARE_OPTIMIZERS_MAXITER,
        "exact_report": EXACT_REPORT,
        "exact_report_maxiter": EXACT_REPORT_MAXITER,
    },
    "evaluations": [],
    "iterations": [],
}
iteration_state = {"previous": JF.x.copy()}
evaluation_state = {"count": 0}


def _clone_good_state(state):
    return {
        "dofs": state["dofs"].copy(),
        "surface": state["surface"].copy(),
        "iota": state["iota"],
        "G": state["G"],
        "J": state["J"],
        "grad": None if state["grad"] is None else state["grad"].copy(),
    }


def _saved_state():
    return {
        "dofs": JF.x.copy(),
        "surface": boozer_surface.surface.x.copy(),
        "iota": boozer_surface.res['iota'],
        "G": boozer_surface.res['G'],
        "J": last_good["J"],
        "grad": None if last_good["grad"] is None else last_good["grad"].copy(),
    }


def _clone_surface_tensor_fourier(surface):
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


def _normalize_optimizer_method(method):
    normalized = method.strip()
    if normalized.lower() == "trust-constr":
        return "trust-constr"
    upper = normalized.upper()
    if upper in {"L-BFGS-B", "BFGS"}:
        return upper
    return normalized


def _build_minimize_config(method, maxiter, ndofs):
    method = _normalize_optimizer_method(method)
    options = {"maxiter": maxiter}
    kwargs = {}
    if method == "L-BFGS-B":
        options.update({
            "maxls": MAXLS,
            "maxcor": min(MAXCOR, max(3, ndofs)),
            "ftol": 1e-16,
            "gtol": 1e-12,
        })
    elif method == "BFGS":
        options.update({"gtol": 1e-12})
    elif method == "trust-constr":
        options.update({
            "gtol": 1e-12,
            "xtol": 1e-12,
            "initial_tr_radius": max(STEP_RMS_LIMIT * np.sqrt(ndofs), 1e-3),
            "verbose": 0,
        })
        kwargs["hess"] = TrustRegionBFGS()
    return method, options, kwargs


def _restore_boozer_state(saved):
    JF.x = saved["dofs"]
    boozer_surface.surface.x = saved["surface"].copy()
    boozer_surface.res['iota'] = saved["iota"]
    boozer_surface.res['G'] = saved["G"]
    boozer_surface.res['success'] = True
    boozer_surface.need_to_run_code = True
    J_nonQIRatio.recompute_bell()
    J_iotas.recompute_bell()
    J_major_radius.recompute_bell()
    Jls.recompute_bell()


def _record_history(kind, dofs, J, grad, step_rms, physical, note=None):
    grad = np.asarray(grad, dtype=float)
    entry = {
        "kind": kind,
        "evaluation": len(history["evaluations"]) if kind == "evaluation" else len(history["iterations"]),
        "J": float(J),
        "grad_norm": float(np.linalg.norm(grad)),
        "step_rms": float(step_rms),
        "note": note,
    }
    if physical:
        curve_lengths = [float(term.J()) for term in ls]
        entry.update({
            "J_nonQIRatio": float(J_nonQIRatio.J()),
            "J_iotas": float(J_iotas.J()),
            "J_major_radius": float(J_major_radius.J()),
            "J_length": float(Jls.J()),
            "iota": float(boozer_surface.res['iota']),
            "G": float(boozer_surface.res['G']),
            "major_radius": float(mr.J()),
            "curve_lengths": curve_lengths,
            "curve_length_total": float(sum(curve_lengths)),
        })
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
    history["evaluations" if kind == "evaluation" else "iterations"].append(entry)


def _physical_summary():
    curve_lengths = [float(term.J()) for term in ls]
    return {
        "J_nonQIRatio": float(J_nonQIRatio.J()),
        "J_iotas": float(J_iotas.J()),
        "J_major_radius": float(J_major_radius.J()),
        "J_length": float(Jls.J()),
        "iota": float(boozer_surface.res['iota']),
        "G": float(boozer_surface.res['G']),
        "major_radius": float(mr.J()),
        "curve_lengths": curve_lengths,
        "curve_length_total": float(sum(curve_lengths)),
    }


def _print_physical_summary(prefix, J, grad, eval_seconds):
    summary = _physical_summary()
    cl_string = ", ".join([f"{length:.1f}" for length in summary["curve_lengths"]])
    outstr = f"{prefix}J={J:.1e}, J_nonQIRatio={summary['J_nonQIRatio']:.2e}, iota={summary['iota']:.2e}, mr={summary['major_radius']:.2e}"
    outstr += f", Len=sum([{cl_string}])={summary['curve_length_total']:.1f}"
    outstr += f", ||grad J||={np.linalg.norm(grad):.1e}, eval={eval_seconds:.2f}s"
    print(outstr)


def _step_penalty(dofs, reference_dofs, base_value):
    scales = np.maximum(1.0, np.abs(reference_dofs))
    step_scaled = (dofs - reference_dofs) / scales
    norm_sq = float(np.dot(step_scaled, step_scaled))
    limit_sq = dofs.size * STEP_RMS_LIMIT**2
    excess = norm_sq - limit_sq
    if excess <= 0:
        return None
    J = base_value + STEP_PENALTY * excess**2
    grad = 4.0 * STEP_PENALTY * excess * step_scaled / scales
    return J, grad, np.sqrt(norm_sq / dofs.size)


def _evaluate(dofs, *, reference_state=None, allow_step_penalty=True, update_good_state=False, record_history=False, verbose=False):
    global best_good

    step_rms = 0.0
    if reference_state is not None:
        step_rms = float(np.linalg.norm(dofs - reference_state["dofs"]) / np.sqrt(dofs.size))
    if allow_step_penalty and reference_state is not None:
        penalty = _step_penalty(dofs, reference_state["dofs"], reference_state["J"] if reference_state["J"] is not None else 0.0)
        if penalty is not None:
            _restore_boozer_state(reference_state)
            J, grad, limited_step_rms = penalty
            metadata = {
                "physical": False,
                "step_rms": limited_step_rms,
                "note": "step_limit",
                "eval_seconds": 0.0,
            }
            if record_history:
                _record_history("evaluation", dofs, J, grad, limited_step_rms, physical=False, note="step_limit")
            if verbose:
                print(f"Trial step exceeded trust region; applying step penalty. J={J:.1e}, ||grad J||={np.linalg.norm(grad):.1e}")
            return J, grad, metadata

    saved = {
        "dofs": JF.x.copy(),
        "surface": boozer_surface.surface.x.copy(),
        "iota": boozer_surface.res['iota'],
        "G": boozer_surface.res['G'],
        "J": None,
        "grad": None,
    }

    JF.x = dofs
    started = time.perf_counter()
    try:
        J = float(JF.J())
        grad = np.asarray(JF.dJ(), dtype=float)
        if not boozer_surface.res['success']:
            raise RuntimeError("Boozer surface solve failed during trial evaluation.")
        metadata = {
            "physical": True,
            "step_rms": step_rms,
            "note": None,
            "eval_seconds": time.perf_counter() - started,
        }
    except (np.linalg.LinAlgError, RuntimeError, ValueError) as err:
        fallback = reference_state if reference_state is not None else saved
        _restore_boozer_state(fallback)
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
            _record_history("evaluation", dofs, J, grad, step_rms, physical=False, note=type(err).__name__)
        if verbose:
            print(f"Trial evaluation failed ({type(err).__name__}); applying fallback penalty.")
            print(f"J={J:.1e}, ||grad J||={np.linalg.norm(grad):.1e}, eval={metadata['eval_seconds']:.2f}s")
        return J, grad, metadata

    if update_good_state and metadata["physical"]:
        last_good["dofs"] = dofs.copy()
        last_good["surface"] = boozer_surface.surface.x.copy()
        last_good["iota"] = boozer_surface.res['iota']
        last_good["G"] = boozer_surface.res['G']
        last_good["J"] = J
        last_good["grad"] = grad.copy()
        if best_good["J"] is None or J < best_good["J"]:
            best_good = _clone_good_state(last_good)

    if record_history:
        _record_history("evaluation", dofs, J, grad, metadata["step_rms"], physical=True)
    if verbose:
        prefix = f"eval {evaluation_state['count']}: " if record_history else ""
        _print_physical_summary(prefix, J, grad, metadata["eval_seconds"])
    return J, grad, metadata


def fun(dofs):
    evaluation_state["count"] += 1
    J, grad, _ = _evaluate(
        dofs,
        reference_state=last_good,
        allow_step_penalty=True,
        update_good_state=True,
        record_history=True,
        verbose=True,
    )
    return J, grad


def _run_directional_fd_comparison(base_state, ndirs, eps):
    rng = np.random.default_rng(1)
    results = []
    saved = _saved_state()
    base_dofs = base_state["dofs"].copy()
    base_grad = base_state["grad"].copy()
    for index in range(ndirs):
        direction = rng.uniform(-0.5, 0.5, size=base_dofs.shape)
        direction /= max(np.linalg.norm(direction), 1e-15)
        plus_value, _, _ = _evaluate(base_dofs + eps * direction, allow_step_penalty=False, update_good_state=False, record_history=False, verbose=False)
        minus_value, _, _ = _evaluate(base_dofs - eps * direction, allow_step_penalty=False, update_good_state=False, record_history=False, verbose=False)
        adjoint = float(base_grad @ direction)
        finite_difference = (plus_value - minus_value) / (2 * eps)
        relative_error = abs(finite_difference - adjoint) / max(1.0, abs(finite_difference), abs(adjoint))
        results.append({
            "direction": index,
            "adjoint": adjoint,
            "finite_difference": finite_difference,
            "relative_error": relative_error,
        })
    _restore_boozer_state(saved)
    return results


def _select_reduced_subspace(base_state, subspace_size):
    if subspace_size <= 0:
        return np.zeros((0,), dtype=int)
    ordering = np.argsort(-np.abs(base_state["grad"]))
    return ordering[:min(subspace_size, ordering.size)]


def _run_reduced_subspace_optimization(base_state, active, method, maxiter, jac):
    saved = _saved_state()
    base_dofs = base_state["dofs"].copy()
    x0 = base_dofs[active].copy()

    def scalar_eval(xsub):
        dofs = base_dofs.copy()
        dofs[active] = xsub
        value, _, _ = _evaluate(dofs, allow_step_penalty=False, update_good_state=False, record_history=False, verbose=False)
        return value

    def grad_eval(xsub):
        dofs = base_dofs.copy()
        dofs[active] = xsub
        _, grad, _ = _evaluate(dofs, allow_step_penalty=False, update_good_state=False, record_history=False, verbose=False)
        return grad[active]

    method, options, kwargs = _build_minimize_config(method, maxiter, active.size)
    res = minimize(
        scalar_eval,
        x0,
        jac=jac if isinstance(jac, str) else grad_eval,
        method=method,
        options=options,
        tol=1e-15,
        **kwargs,
    )
    _restore_boozer_state(saved)
    x_final = np.asarray(res.x, dtype=float)
    return {
        "method": method,
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


def _run_fd_subspace_optimization(base_state, subspace_size, maxiter):
    if subspace_size <= 0:
        return None
    active = _select_reduced_subspace(base_state, subspace_size)
    analytic_res = _run_reduced_subspace_optimization(base_state, active, "L-BFGS-B", maxiter, jac=True)
    fd_res = _run_reduced_subspace_optimization(base_state, active, "L-BFGS-B", maxiter, jac="2-point")
    return {
        "subspace_size": int(active.size),
        "indices": active.tolist(),
        "analytic_fun": analytic_res["fun"],
        "analytic_nit": analytic_res["nit"],
        "fd_fun": fd_res["fun"],
        "fd_nit": fd_res["nit"],
        "step_difference_norm": float(
            np.linalg.norm(np.asarray(analytic_res["active_x"]) - np.asarray(fd_res["active_x"]))
        ),
        "fun_difference": float(abs(analytic_res["fun"] - fd_res["fun"])),
    }


def _run_reduced_optimizer_comparison(base_state, subspace_size, methods, maxiter):
    active = _select_reduced_subspace(base_state, subspace_size)
    comparison = {
        "subspace_size": int(active.size),
        "indices": active.tolist(),
        "results": [],
    }
    if active.size == 0:
        return comparison
    for method in methods:
        comparison["results"].append(
            _run_reduced_subspace_optimization(base_state, active, method, maxiter, jac=True)
        )
    return comparison


def _run_exact_report(ls_state):
    exact_surface = _clone_surface_tensor_fourier(boozer_surface.surface)
    exact_surface.set_dofs(ls_state["surface"].copy())
    exact_vol = Volume(exact_surface)
    exact_boozer_surface = BoozerSurface(
        bs,
        exact_surface,
        exact_vol,
        vol_target,
        constraint_weight=None,
        options={
            "verbose": BOOZER_VERBOSE,
            "newton_maxiter": EXACT_REPORT_MAXITER,
            "exact_ls_fallback": EXACT_LS_FALLBACK,
            "exact_ls_constraint_weight": EXACT_LS_WEIGHT,
            "exact_ls_maxiter": EXACT_LS_MAXITER,
            "exact_ls_tol": 1e-10,
        },
    )
    started = time.perf_counter()
    exact_res = _solve_boozer_surface_with_seed_fallback(
        exact_boozer_surface,
        iota=ls_state["iota"],
        G=ls_state["G"],
        tol=1e-13,
        maxiter=EXACT_REPORT_MAXITER,
        verbose=BOOZER_VERBOSE,
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

    exact_out_res = boozer_surface_residual(exact_surface, exact_res["iota"], exact_res["G"], bs, derivatives=0)[0]
    exact_mr = MajorRadius(exact_boozer_surface)
    exact_J_iotas = QuadraticPenalty(Iotas(exact_boozer_surface), initial_iota_target, 'identity')
    exact_J_major_radius = QuadraticPenalty(exact_mr, initial_major_radius_target, 'identity')
    exact_J_nonQIRatio = NonQuasiIsodynamicRatio(
        exact_boozer_surface,
        bs_qi,
        sDIM=sDIM,
        nphi=nphi_qi,
        nalpha=nalpha,
        nBj=nBj,
        nphi_out=nphi_out,
    )
    report.update({
        "residual_norm": float(np.linalg.norm(exact_out_res)),
        "major_radius": float(exact_mr.J()),
        "J_nonQIRatio": float(exact_J_nonQIRatio.J()),
        "J_iotas": float(exact_J_iotas.J()),
        "J_major_radius": float(exact_J_major_radius.J()),
        "J_length": float(Jls.J()),
    })
    report["total_J"] = float(
        report["J_nonQIRatio"] + report["J_iotas"] + report["J_major_radius"] + report["J_length"]
    )
    if WRITE_VTK:
        exact_surface.to_vtk(os.path.join(OUT_DIR, "surf_exact_report"))
    return report


def _export_final_artifacts():
    exported = {}

    coils_path = Path(EXPORT_COILS_JSON)
    coils_path.parent.mkdir(parents=True, exist_ok=True)
    bs.save(str(coils_path))
    exported["coils_json"] = str(coils_path)

    restart_path = Path(EXPORT_SURFACE_RESTART)
    restart_path.parent.mkdir(parents=True, exist_ok=True)
    restart_payload = _surface_restart_payload(
        boozer_surface.surface,
        iota=boozer_surface.res['iota'],
        G=boozer_surface.res['G'],
    )
    with open(restart_path, "w", encoding="utf-8") as stream:
        json.dump(restart_payload, stream, indent=2)
    exported["surface_restart"] = str(restart_path)

    vmec_input_path = Path(EXPORT_VMEC_INPUT)
    rz_surface, export_diagnostics = _prepare_vmec_export_surface(boozer_surface.surface)
    diagnostics_path = Path(EXPORT_VMEC_DIAGNOSTICS)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(diagnostics_path, "w", encoding="utf-8") as stream:
        json.dump(export_diagnostics, stream, indent=2)
    _plot_vmec_export_surfaces(
        boozer_surface.surface,
        rz_surface,
        export_diagnostics,
        EXPORT_VMEC_CROSS_SECTION_PLOT,
        EXPORT_VMEC_SURFACE_PLOT,
    )
    vmec_export = _write_vmec_input_from_template(rz_surface, vmec_input_path, VMEC_TEMPLATE_INPUT)
    exported["vmec_input"] = vmec_export["path"]
    exported["vmec_template"] = vmec_export["template"]
    exported["vmec_phiedge"] = vmec_export["phiedge"]
    exported["vmec_export_diagnostics"] = str(diagnostics_path)
    exported["vmec_export_relative_l2"] = float(export_diagnostics["selected"]["relative_l2"])
    exported["vmec_export_cross_sections_plot"] = str(Path(EXPORT_VMEC_CROSS_SECTION_PLOT))
    exported["vmec_export_surface_plot"] = str(Path(EXPORT_VMEC_SURFACE_PLOT))

    return exported


if not SKIP_TAYLOR:
    print("""
################################################################################
### Perform a Taylor test ######################################################
################################################################################
""")
    f = fun
    dofs = JF.x
    np.random.seed(1)
    h = np.random.uniform(size=dofs.shape)
    J0, dJ0 = f(dofs)
    dJh = sum(dJ0 * h)
    for eps in [1e-3, 1e-4, 1e-5, 1e-6]:
        J1, _ = f(dofs + 2*eps*h)
        J2, _ = f(dofs + eps*h)
        J3, _ = f(dofs - eps*h)
        J4, _ = f(dofs - 2*eps*h)
        print("err", ((J1*(-1/12) + J2*(8/12) + J3*(-8/12) + J4*(1/12))/eps - dJh)/max(np.linalg.norm(dJh), 1e-15))

print("""
################################################################################
### Run the optimization #######################################################
################################################################################
""")

maxiter_default = 3 if in_github_actions else 1000
maxiter = _env_int("SIMSOPT_BOOZER_QI_MAXITER", maxiter_default)

initial_dofs = JF.x.copy()
initial_value, initial_grad = fun(initial_dofs)
print(f"Initial gradient norm = {np.linalg.norm(initial_grad):.3e}")
base_snapshot = _clone_good_state(last_good)

if COMPARE_FD:
    print("""
################################################################################
### Compare against finite differences #########################################
################################################################################
""")
    fd_results = _run_directional_fd_comparison(base_snapshot, COMPARE_FD_DIRS, COMPARE_FD_EPS)
    history["fd_comparison"] = {
        "eps": COMPARE_FD_EPS,
        "directional": fd_results,
    }
    max_rel = max(result["relative_error"] for result in fd_results) if fd_results else 0.0
    for result in fd_results:
        print(
            f"FD dir {result['direction']}: adjoint={result['adjoint']:.6e}, "
            f"fd={result['finite_difference']:.6e}, rel_err={result['relative_error']:.3e}"
        )
    print(f"Max FD relative error={max_rel:.3e}")
    if COMPARE_FD_OPT:
        fd_opt = _run_fd_subspace_optimization(base_snapshot, COMPARE_FD_SUBSPACE, COMPARE_FD_OPT_MAXITER)
        history["fd_comparison"]["subspace_optimization"] = fd_opt
        print(
            f"FD reduced-subspace comparison: analytic_fun={fd_opt['analytic_fun']:.6e}, "
            f"fd_fun={fd_opt['fd_fun']:.6e}, step_diff={fd_opt['step_difference_norm']:.3e}, "
            f"fun_diff={fd_opt['fun_difference']:.3e}, subspace={fd_opt['subspace_size']}"
        )

if COMPARE_OPTIMIZERS:
    print("""
################################################################################
### Compare reduced optimizer paths ############################################
################################################################################
""")
    optimizer_comparison = _run_reduced_optimizer_comparison(
        base_snapshot,
        COMPARE_OPTIMIZERS_SUBSPACE,
        COMPARE_OPTIMIZERS_LIST,
        COMPARE_OPTIMIZERS_MAXITER,
    )
    history["optimizer_comparison"] = optimizer_comparison
    for result in optimizer_comparison["results"]:
        print(
            f"Optimizer {result['method']}: success={result['success']}, nit={result['nit']}, "
            f"fun={result['fun']:.6e}, active_step={result['active_step_norm']:.3e}"
        )


def _callback(*args):
    if not args:
        xk = last_good["dofs"]
    elif hasattr(args[0], "x"):
        xk = np.asarray(args[0].x, dtype=float)
    else:
        xk = np.asarray(args[0], dtype=float)
    accepted_step_rms = float(np.linalg.norm(xk - iteration_state["previous"]) / np.sqrt(xk.size))
    iteration_state["previous"] = xk.copy()
    grad = last_good["grad"] if last_good["grad"] is not None else np.zeros_like(xk)
    _record_history("iteration", xk, last_good["J"] if last_good["J"] is not None else np.nan, grad, accepted_step_rms, physical=True)


opt_method, minimize_options, minimize_kwargs = _build_minimize_config(OPT_METHOD, maxiter, initial_dofs.size)
res = minimize(fun, initial_dofs, jac=True, method=opt_method, options=minimize_options, callback=_callback, tol=1e-15, **minimize_kwargs)

if best_good["J"] is not None:
    _restore_boozer_state(best_good)

if EXACT_REPORT and best_good["J"] is not None:
    history["exact_report"] = _run_exact_report(best_good)
    exact_report = history["exact_report"]
    if exact_report["success"]:
        print(
            f"Exact final report ({exact_report['solver']}): total_J={exact_report['total_J']:.6e}, "
            f"residual={exact_report['residual_norm']:.3e}, iota={exact_report['iota']:.6f}, "
            f"eval={exact_report['elapsed_seconds']:.2f}s"
        )
    else:
        print(
            f"Exact final report failed: message={exact_report['message']}, "
            f"eval={exact_report['elapsed_seconds']:.2f}s"
        )

history["result"] = {
    "success": bool(res.success),
    "status": int(res.status),
    "message": str(res.message),
    "nit": int(res.nit),
    "nfev": int(res.nfev),
    "njev": int(res.njev) if getattr(res, "njev", None) is not None else None,
    "initial_J": float(initial_value),
    "final_J": float(res.fun),
    "best_J": float(best_good["J"] if best_good["J"] is not None else res.fun),
}
if WRITE_VTK:
    curves_to_vtk(all_curves, os.path.join(OUT_DIR, "curves_opt"))
    boozer_surface.surface.to_vtk(os.path.join(OUT_DIR, "surf_opt"))

history["artifacts"] = _export_final_artifacts()

history_path = Path(HISTORY_PATH)
history_path.parent.mkdir(parents=True, exist_ok=True)
with open(history_path, "w", encoding="utf-8") as stream:
    json.dump(history, stream, indent=2)

best_J = best_good["J"] if best_good["J"] is not None else res.fun
print(f"Optimization success={res.success}, status={res.status}, nit={res.nit}, initial_J={initial_value:.6e}, final_J={res.fun:.6e}, best_J={best_J:.6e}")
print(f"Wrote optimization history to {history_path}")
print(f"Wrote optimized Biot-Savart JSON to {history['artifacts']['coils_json']}")
print(f"Wrote Boozer surface restart to {history['artifacts']['surface_restart']}")
print(
    f"Wrote VMEC input to {history['artifacts']['vmec_input']} "
    f"using phiedge={history['artifacts']['vmec_phiedge']:.6e}"
)
print(
    f"Wrote VMEC export diagnostics to {history['artifacts']['vmec_export_diagnostics']} "
    f"with relative_l2={history['artifacts']['vmec_export_relative_l2']:.3e}"
)
print(f"Wrote VMEC export cross-section plot to {history['artifacts']['vmec_export_cross_sections_plot']}")
print(f"Wrote VMEC export 3D plot to {history['artifacts']['vmec_export_surface_plot']}")
print("End of 2_Intermediate/boozerQI.py")
print("================================")