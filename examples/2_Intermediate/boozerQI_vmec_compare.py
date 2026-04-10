#!/usr/bin/env python3

import argparse
import os
import json
import shutil
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import f90nml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import netcdf_file
from scipy.interpolate import UnivariateSpline
from scipy.spatial import cKDTree

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

from simsopt._core.optimizable import load
from simsopt.field import BoozerRadialInterpolant
from simsopt.geo import SurfaceRZFourier, SurfaceXYZTensorFourier
from simsopt.geo.surfaceobjectives import (
    _build_template_well,
    _make_qi_sampling_grid,
    _make_shuffled_target_values,
    _normalize_modB_global,
    _sample_modB_on_boozer_lines,
)
from simsopt.mhd import Vmec

try:
    import vmec  # noqa: F401
except ImportError:
    vmec = None

try:
    import booz_xform  # noqa: F401
except ImportError:
    booz_xform = None


def _surface_gamma_on_grid(surface, quadpoints_phi, quadpoints_theta):
    gamma = np.zeros((len(quadpoints_phi), len(quadpoints_theta), 3))
    for index, phi in enumerate(quadpoints_phi):
        gamma[index, :, :] = surface.cross_section(float(phi), thetas=quadpoints_theta)
    return gamma


def _load_surface_restart(path):
    restart_path = Path(path).expanduser().resolve()
    with open(restart_path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    surface = SurfaceXYZTensorFourier(
        nfp=int(payload["nfp"]),
        stellsym=bool(payload["stellsym"]),
        mpol=int(payload["mpol"]),
        ntor=int(payload["ntor"]),
        quadpoints_phi=np.asarray(payload["quadpoints_phi"], dtype=float),
        quadpoints_theta=np.asarray(payload["quadpoints_theta"], dtype=float),
    )
    surface.set_dofs(np.asarray(payload["dofs"], dtype=float))
    return payload, surface


def _stats(values):
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def _vmec_wout_name(input_path):
    input_name = Path(input_path).name
    if input_name.startswith("input."):
        return input_name.replace("input.", "wout_", 1) + ".nc"
    return f"wout_{input_name}.nc"


def _relative_l2(values_a, values_b):
    denominator = np.linalg.norm(values_b)
    if denominator == 0.0:
        return float("nan")
    return float(np.linalg.norm(values_a - values_b) / denominator)


def _normalized_field_comparison(values_a, values_b):
    normalized_a = values_a / np.mean(values_a)
    normalized_b = values_b / np.mean(values_b)
    return {
        "absolute_rel_l2": _relative_l2(values_a, values_b),
        "normalized_rel_l2": _relative_l2(normalized_a, normalized_b),
        "normalized_max_abs_diff": float(np.max(np.abs(normalized_a - normalized_b))),
        "mean_ratio": float(np.mean(values_a) / np.mean(values_b)),
    }


def _distance_diagnostics(distances):
    absolute_distances = np.abs(distances)
    return {
        "distance_stats": _stats(absolute_distances),
        "rms_distance": float(np.sqrt(np.mean(absolute_distances ** 2))),
        "max_abs_distance": float(np.max(absolute_distances)),
    }


def _surface_shape_diagnostics(surface_a, surface_b, nphi, ntheta):
    quadpoints_phi = np.linspace(0.0, 1.0 / surface_a.nfp, nphi, endpoint=False)
    quadpoints_theta = np.linspace(0.0, 1.0, ntheta, endpoint=False)
    surface_a_points = _surface_gamma_on_grid(surface_a, quadpoints_phi, quadpoints_theta).reshape((-1, 3))
    surface_b_points = _surface_gamma_on_grid(surface_b, quadpoints_phi, quadpoints_theta).reshape((-1, 3))

    surface_a_tree = cKDTree(surface_a_points)
    surface_b_tree = cKDTree(surface_b_points)
    a_to_b = surface_b_tree.query(surface_a_points, k=1)[0]
    b_to_a = surface_a_tree.query(surface_b_points, k=1)[0]
    return {
        "method": "symmetric_nearest_neighbor_distance",
        "surface_a_to_b": _distance_diagnostics(a_to_b),
        "surface_b_to_a": _distance_diagnostics(b_to_a),
        "symmetric_rms_distance": float(np.sqrt(0.5 * (np.mean(a_to_b ** 2) + np.mean(b_to_a ** 2)))),
        "symmetric_max_abs_distance": float(max(np.max(np.abs(a_to_b)), np.max(np.abs(b_to_a)))),
    }


def _boozer_surface_field_data(field, nphi, ntheta, nfp):
    thetas = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    zetas = np.linspace(0.0, 2.0 * np.pi / nfp, nphi, endpoint=False)
    zeta_grid, theta_grid = np.meshgrid(zetas, thetas, indexing="ij")
    points = np.zeros((nphi * ntheta, 3))
    points[:, 0] = 1.0
    points[:, 1] = theta_grid.reshape((-1,))
    points[:, 2] = zeta_grid.reshape((-1,))
    field.set_points(points)

    vmec_modB = field.modB()[:, 0].reshape((nphi, ntheta))
    R = field.R()[:, 0].reshape((nphi, ntheta))
    Z = field.Z()[:, 0].reshape((nphi, ntheta))
    nu = field.nu()[:, 0].reshape((nphi, ntheta))
    phi = zeta_grid - nu

    gamma = np.zeros((nphi, ntheta, 3))
    gamma[:, :, 0] = R * np.cos(phi)
    gamma[:, :, 1] = R * np.sin(phi)
    gamma[:, :, 2] = Z

    return {
        "gamma": gamma,
        "modB": vmec_modB,
        "theta": theta_grid,
        "zeta": zeta_grid,
    }


def _modB_on_gamma(bs, gamma):
    bs.set_points(gamma.reshape((-1, 3)))
    return np.linalg.norm(bs.B().reshape(gamma.shape), axis=2)


def _qi_diagnostics_from_modB_grid(modB_grid, nfp, iota, nphi, nalpha, nBj, nphi_out, phi_shift=None):
    if phi_shift is None:
        field_period = 2.0 * np.pi / nfp
        phi_grid = np.linspace(0.0, field_period, modB_grid.shape[0], endpoint=False)
        phi_shift = float(phi_grid[int(np.argmax(np.max(modB_grid, axis=1)))])

    sampling_surface = SimpleNamespace(nfp=nfp)
    phi_values, alpha_values = _make_qi_sampling_grid(sampling_surface, nphi, nalpha, phi_shift)
    modB_lines = _sample_modB_on_boozer_lines(modB_grid, sampling_surface, phi_values, alpha_values, iota, phi_shift)
    normalized, _, _, _ = _normalize_modB_global(modB_lines)

    bounce_levels = np.linspace(0.0, 1.0, nBj)
    raw_weights = np.zeros((nalpha,))
    bounce_distances = np.zeros((nalpha, nBj))
    branch_locations = np.zeros((nalpha, 2 * nBj - 1))

    for index in range(nalpha):
        _, raw_weights[index], bounce_distances[index, :], branch_locations[index, :] = _build_template_well(
            phi_values,
            normalized[:, index],
            bounce_levels,
        )

    normalized_weights = raw_weights / np.sum(raw_weights)
    mean_bounce_distances = np.sum(bounce_distances * normalized_weights[:, None], axis=0)
    phi_out = np.linspace(phi_values[0], phi_values[-1], nphi_out)
    targets = np.zeros((nalpha, nphi_out))
    original = np.zeros((nalpha, nphi_out))
    residuals = np.zeros((nalpha, nphi_out))

    for index in range(nalpha):
        original_spline = UnivariateSpline(phi_values, normalized[:, index], k=1, s=0)
        targets[index, :] = _make_shuffled_target_values(
            phi_out,
            bounce_distances[index, :],
            bounce_levels,
            branch_locations[index, :],
            mean_bounce_distances,
        )
        original[index, :] = original_spline(phi_out)
        residuals[index, :] = (targets[index, :] - original[index, :]) / np.sqrt(nalpha * nphi_out)

    residual_norms = np.linalg.norm(residuals, axis=1)
    representative_index = int(np.argmax(residual_norms))
    objective = float(np.sum(residuals ** 2))

    return {
        "phi_values": phi_values,
        "alpha_values": alpha_values,
        "phi_out": phi_out,
        "modB_lines": modB_lines,
        "normalized_lines": normalized,
        "targets": targets,
        "original": original,
        "residuals": residuals,
        "representative_index": representative_index,
        "representative_residual_norm": float(residual_norms[representative_index]),
        "mean_bounce_distances": mean_bounce_distances,
        "normalized_weights": normalized_weights,
        "phi_shift": float(phi_shift),
        "residual_norm": float(np.linalg.norm(residuals)),
        "objective": objective,
    }


def _qi_summary(qi):
    return {
        "objective": float(qi["objective"]),
        "residual_norm": float(qi["residual_norm"]),
        "representative_index": int(qi["representative_index"]),
        "representative_residual_norm": float(qi["representative_residual_norm"]),
        "phi_shift": float(qi["phi_shift"]),
    }


def _assess_modB_topology(phi, theta, modB):
    span = float(np.max(modB) - np.min(modB))
    if span <= 1.0e-14:
        return {
            "levels": [],
            "internal_closed_loops": 0,
            "boundary_closed_loops": 0,
            "poloidal_crossings": 0,
            "toroidal_crossings": 0,
            "dominant_wrapping": "degenerate",
            "appears_qi_like": False,
            "verdict": "The sampled |B| is nearly constant on the plotting grid, so contour topology is not informative.",
        }

    levels = np.linspace(np.min(modB) + 0.15 * span, np.max(modB) - 0.15 * span, 8)
    fig, ax = plt.subplots()
    contour = ax.contour(phi, theta, modB.T, levels=levels)
    plt.close(fig)

    tol_phi = 0.02 * (phi[-1] - phi[0] + (phi[1] - phi[0]))
    tol_theta = 0.02 * (theta[-1] - theta[0] + (theta[1] - theta[0]))
    phi_min = float(phi[0])
    phi_max = float(phi[-1])
    theta_min = float(theta[0])
    theta_max = float(theta[-1])

    internal_closed_loops = 0
    boundary_closed_loops = 0
    poloidal_crossings = 0
    toroidal_crossings = 0

    for segments in contour.allsegs:
        for segment in segments:
            if segment.shape[0] < 2:
                continue
            closed = bool(np.linalg.norm(segment[0] - segment[-1]) < max(tol_phi, tol_theta))
            touches_phi_min = bool(np.any(np.isclose(segment[:, 0], phi_min, atol=tol_phi)))
            touches_phi_max = bool(np.any(np.isclose(segment[:, 0], phi_max, atol=tol_phi)))
            touches_theta_min = bool(np.any(np.isclose(segment[:, 1], theta_min, atol=tol_theta)))
            touches_theta_max = bool(np.any(np.isclose(segment[:, 1], theta_max, atol=tol_theta)))
            touches_boundary = touches_phi_min or touches_phi_max or touches_theta_min or touches_theta_max

            if closed and not touches_boundary:
                internal_closed_loops += 1
            elif closed and touches_boundary:
                boundary_closed_loops += 1

            if touches_theta_min and touches_theta_max:
                poloidal_crossings += 1
            if touches_phi_min and touches_phi_max:
                toroidal_crossings += 1

    if poloidal_crossings > toroidal_crossings:
        dominant_wrapping = "poloidal"
    elif toroidal_crossings > poloidal_crossings:
        dominant_wrapping = "toroidal"
    else:
        dominant_wrapping = "mixed"

    appears_qi_like = internal_closed_loops == 0 and dominant_wrapping == "poloidal"
    if internal_closed_loops > 0:
        verdict = "Interior closed |B| contours are present, so the sampled surface is not quasi-isodynamic-like in the strict contour-topology sense."
    elif dominant_wrapping != "poloidal":
        verdict = "The contour set does not predominantly wrap poloidally, so the surface does not yet look convincingly quasi-isodynamic."
    else:
        verdict = "The sampled contours wrap poloidally with no interior closed loops on the inspected levels, which is consistent with quasi-isodynamic-like topology on this surface."

    return {
        "levels": [float(value) for value in levels],
        "internal_closed_loops": int(internal_closed_loops),
        "boundary_closed_loops": int(boundary_closed_loops),
        "poloidal_crossings": int(poloidal_crossings),
        "toroidal_crossings": int(toroidal_crossings),
        "dominant_wrapping": dominant_wrapping,
        "appears_qi_like": bool(appears_qi_like),
        "verdict": verdict,
    }


def _set_equal_3d_limits(ax, xyz_points):
    xyz_points = np.asarray(xyz_points)
    mins = np.min(xyz_points, axis=0)
    maxs = np.max(xyz_points, axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    if radius <= 0.0:
        radius = 1.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def _save_modB_map_plot(reference_modB, vmec_modB, vmec_boozer_coil_modB, output_path, nfp):
    reference_normalized = reference_modB / np.mean(reference_modB)
    vmec_normalized = vmec_modB / np.mean(vmec_modB)
    same_point_normalized = vmec_boozer_coil_modB / np.mean(vmec_boozer_coil_modB)
    delta_same_point = same_point_normalized - vmec_normalized
    delta_reference_vmec = reference_normalized - vmec_normalized

    field_period = 2.0 * np.pi / nfp
    extent = [0.0, field_period, 0.0, 2.0 * np.pi]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5), constrained_layout=True)

    panels = [
        (axes[0, 0], reference_normalized, "Boozer-surface normalized |B|", "viridis", None),
        (axes[0, 1], vmec_normalized, "VMEC/Booz_Xform normalized |B|", "viridis", None),
        (axes[1, 0], delta_reference_vmec, "Reference minus VMEC normalized |B|", "coolwarm", np.max(np.abs(delta_reference_vmec))),
        (axes[1, 1], delta_same_point, "Coil minus VMEC on VMEC Boozer surface", "coolwarm", np.max(np.abs(delta_same_point))),
    ]

    for ax, values, title, cmap, vmax in panels:
        kwargs = {"origin": "lower", "aspect": "auto", "extent": extent, "cmap": cmap}
        if vmax is not None and vmax > 0.0:
            kwargs["vmin"] = -vmax
            kwargs["vmax"] = vmax
        image = ax.imshow(values.T, **kwargs)
        ax.set_title(title)
        ax.set_xlabel("Boozer toroidal angle")
        ax.set_ylabel("Boozer poloidal angle")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_modB_contour_plot(reference_modB, vmec_modB, output_path, nfp):
    field_period = 2.0 * np.pi / nfp
    phi = np.linspace(0.0, field_period, reference_modB.shape[0], endpoint=False)
    theta = np.linspace(0.0, 2.0 * np.pi, reference_modB.shape[1], endpoint=False)
    reference_normalized = reference_modB / np.mean(reference_modB)
    vmec_normalized = vmec_modB / np.mean(vmec_modB)

    combined_min = float(min(np.min(reference_normalized), np.min(vmec_normalized)))
    combined_max = float(max(np.max(reference_normalized), np.max(vmec_normalized)))
    levels = np.linspace(combined_min, combined_max, 12)

    reference_topology = _assess_modB_topology(phi, theta, reference_normalized)
    vmec_topology = _assess_modB_topology(phi, theta, vmec_normalized)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True)
    panels = (
        (axes[0], reference_normalized, "Boozer surface normalized |B| contours", reference_topology),
        (axes[1], vmec_normalized, "VMEC/Booz_Xform normalized |B| contours", vmec_topology),
    )
    for ax, values, title, topology in panels:
        filled = ax.contourf(phi, theta, values.T, levels=levels, cmap="viridis")
        contour = ax.contour(phi, theta, values.T, levels=levels, colors="white", linewidths=0.7, alpha=0.9)
        ax.clabel(contour, inline=True, fontsize=7, fmt="%.2f")
        ax.set_title(title)
        ax.set_xlabel("Boozer toroidal angle")
        ax.set_ylabel("Boozer poloidal angle")
        ax.text(
            0.02,
            0.02,
            f"wrap={topology['dominant_wrapping']}, closed={topology['internal_closed_loops']}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
        fig.colorbar(filled, ax=ax, fraction=0.046, pad=0.04)

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {
        "reference": reference_topology,
        "vmec": vmec_topology,
    }


def _save_qi_heatmap_plot(reference_qi, vmec_qi, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), constrained_layout=True)
    rows = (("Boozer surface", reference_qi), ("VMEC/Booz_Xform", vmec_qi))
    for row_index, (label, qi) in enumerate(rows):
        phi_axis = qi["phi_out"]
        line_axis = np.arange(qi["residuals"].shape[0])
        residual_image = axes[row_index, 0].imshow(
            qi["residuals"],
            origin="lower",
            aspect="auto",
            extent=[phi_axis[0], phi_axis[-1], line_axis[0], line_axis[-1]],
            cmap="coolwarm",
        )
        axes[row_index, 0].set_title(f"{label} QI residuals")
        axes[row_index, 0].set_xlabel("Boozer toroidal angle")
        axes[row_index, 0].set_ylabel("Field-line index")
        fig.colorbar(residual_image, ax=axes[row_index, 0], fraction=0.046, pad=0.04)

        normalized_image = axes[row_index, 1].imshow(
            qi["normalized_lines"].T,
            origin="lower",
            aspect="auto",
            extent=[qi["phi_values"][0], qi["phi_values"][-1], 0, qi["normalized_lines"].shape[1] - 1],
            cmap="viridis",
        )
        axes[row_index, 1].set_title(f"{label} normalized |B| samples")
        axes[row_index, 1].set_xlabel("Boozer toroidal angle")
        axes[row_index, 1].set_ylabel("Field-line index")
        fig.colorbar(normalized_image, ax=axes[row_index, 1], fraction=0.046, pad=0.04)

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_qi_profile_plot(reference_qi, vmec_qi, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), sharey=True, constrained_layout=True)
    for ax, label, qi in ((axes[0], "Boozer surface", reference_qi), (axes[1], "VMEC/Booz_Xform", vmec_qi)):
        index = qi["representative_index"]
        ax.plot(qi["phi_out"], qi["original"][index, :], label="Normalized |B|", color="#4C78A8")
        ax.plot(qi["phi_out"], qi["targets"][index, :], label="Shuffled target", color="#E45756")
        ax.fill_between(qi["phi_out"], qi["original"][index, :], qi["targets"][index, :], color="#72B7B2", alpha=0.25)
        ax.set_title(f"{label} representative line {index}")
        ax.set_xlabel("Boozer toroidal angle")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Normalized |B|")
    axes[1].legend(loc="best")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_surface_comparison_plot(reference_gamma, reference_modB, vmec_gamma, vmec_modB, vmec_boozer_coil_modB, output_path):
    reference_norm = reference_modB / np.mean(reference_modB)
    vmec_norm = vmec_modB / np.mean(vmec_modB)
    diff = vmec_boozer_coil_modB / np.mean(vmec_boozer_coil_modB) - vmec_norm

    reference_cmap = plt.colormaps["viridis"]
    diff_cmap = plt.colormaps["coolwarm"]
    reference_limits = (min(np.min(reference_norm), np.min(vmec_norm)), max(np.max(reference_norm), np.max(vmec_norm)))
    diff_limit = float(np.max(np.abs(diff)))

    fig = plt.figure(figsize=(15.0, 5.0), constrained_layout=True)
    surface_data = [
        (reference_gamma, reference_norm, "Boozer surface |B|", reference_cmap, reference_limits),
        (vmec_gamma, vmec_norm, "VMEC/Booz_Xform |B|", reference_cmap, reference_limits),
        (vmec_gamma, diff, "Coil minus VMEC on VMEC surface", diff_cmap, (-diff_limit, diff_limit)),
    ]

    for index, (gamma, values, title, cmap, limits) in enumerate(surface_data, start=1):
        ax = fig.add_subplot(1, 3, index, projection="3d")
        vmin, vmax = limits
        if vmax <= vmin:
            vmax = vmin + 1.0
        normalized = plt.Normalize(vmin=vmin, vmax=vmax)(values)
        ax.plot_surface(
            gamma[:, :, 0],
            gamma[:, :, 1],
            gamma[:, :, 2],
            facecolors=cmap(normalized),
            linewidth=0,
            antialiased=False,
            shade=False,
        )
        ax.set_title(title)
        ax.set_axis_off()
        _set_equal_3d_limits(ax, gamma.reshape((-1, 3)))
        fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=cmap), ax=ax, fraction=0.046, pad=0.02)

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_vmec_overview_plot(wout_path, output_path):
    with netcdf_file(wout_path, mmap=False) as dataset:
        phi = np.copy(dataset.variables["phi"].data)
        iotaf = np.copy(dataset.variables["iotaf"].data)
        presf = np.copy(dataset.variables["presf"].data)
        DMerc = np.copy(dataset.variables["DMerc"].data)
        ns = int(np.copy(dataset.variables["ns"].data))
        nfp = int(np.copy(dataset.variables["nfp"].data))
        xm = np.copy(dataset.variables["xm"].data)
        xn = np.copy(dataset.variables["xn"].data)
        xm_nyq = np.copy(dataset.variables["xm_nyq"].data)
        xn_nyq = np.copy(dataset.variables["xn_nyq"].data)
        rmnc = np.copy(dataset.variables["rmnc"].data)
        zmns = np.copy(dataset.variables["zmns"].data)
        bmnc = np.copy(dataset.variables["bmnc"].data)
        lasym = int(np.copy(dataset.variables["lasym__logical__"].data))
        if lasym == 1:
            rmns = np.copy(dataset.variables["rmns"].data)
            zmnc = np.copy(dataset.variables["zmnc"].data)
            bmns = np.copy(dataset.variables["bmns"].data)
        else:
            rmns = np.zeros_like(rmnc)
            zmnc = np.zeros_like(rmnc)
            bmns = np.zeros_like(bmnc)

    s = np.linspace(0.0, 1.0, ns)
    theta = np.linspace(0.0, 2.0 * np.pi, 160)
    zeta_slices = np.linspace(0.0, 2.0 * np.pi / nfp, 4, endpoint=False)
    boundary_index = ns - 1
    R = np.zeros((theta.size, zeta_slices.size))
    Z = np.zeros((theta.size, zeta_slices.size))
    for itheta, theta_value in enumerate(theta):
        for izeta, zeta_value in enumerate(zeta_slices):
            angle = xm * theta_value - xn * zeta_value
            R[itheta, izeta] = np.sum(rmnc[boundary_index, :] * np.cos(angle) + rmns[boundary_index, :] * np.sin(angle))
            Z[itheta, izeta] = np.sum(zmns[boundary_index, :] * np.sin(angle) + zmnc[boundary_index, :] * np.cos(angle))

    theta_grid, zeta_grid = np.meshgrid(np.linspace(0.0, 2.0 * np.pi, 80), np.linspace(0.0, 2.0 * np.pi, 120), indexing="ij")
    boundary_modB = np.zeros_like(theta_grid)
    for mode in range(xm_nyq.size):
        angle = xm_nyq[mode] * theta_grid - xn_nyq[mode] * zeta_grid
        boundary_modB += bmnc[boundary_index, mode] * np.cos(angle) + bmns[boundary_index, mode] * np.sin(angle)

    fig, axes = plt.subplots(2, 3, figsize=(13.0, 8.0), constrained_layout=True)
    axes[0, 0].plot(s, iotaf, color="#4C78A8")
    axes[0, 0].set_title("VMEC iota profile")
    axes[0, 0].set_xlabel("s")
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(s, presf, color="#E45756")
    axes[0, 1].set_title("Pressure profile")
    axes[0, 1].set_xlabel("s")
    axes[0, 1].grid(alpha=0.25)

    axes[0, 2].plot(s[:-1], DMerc[:-1], color="#54A24B")
    axes[0, 2].set_title("Mercier stability")
    axes[0, 2].set_xlabel("s")
    axes[0, 2].grid(alpha=0.25)

    for index, zeta_value in enumerate(zeta_slices):
        axes[1, 0].plot(R[:, index], Z[:, index], label=fr"$\zeta={zeta_value:.2f}$")
    axes[1, 0].set_title("Boundary cross-sections")
    axes[1, 0].set_xlabel("R")
    axes[1, 0].set_ylabel("Z")
    axes[1, 0].set_aspect("equal", adjustable="box")
    axes[1, 0].legend(loc="best", fontsize="small")

    image = axes[1, 1].imshow(
        boundary_modB,
        origin="lower",
        aspect="auto",
        extent=[0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi],
        cmap="viridis",
    )
    axes[1, 1].set_title("LCFS |B| from wout")
    axes[1, 1].set_xlabel("Boozer-like toroidal angle")
    axes[1, 1].set_ylabel("Poloidal angle")
    fig.colorbar(image, ax=axes[1, 1], fraction=0.046, pad=0.04)

    axes[1, 2].plot(phi, iotaf, color="#4C78A8", label="iota")
    axes[1, 2].plot(phi, presf / max(np.max(np.abs(presf)), 1.0), color="#E45756", label="pressure / max")
    axes[1, 2].set_title("Flux-coordinate summary")
    axes[1, 2].set_xlabel("phi")
    axes[1, 2].grid(alpha=0.25)
    axes[1, 2].legend(loc="best", fontsize="small")

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_objective_history_plot(history_path, output_path):
    with open(history_path, "r", encoding="utf-8") as stream:
        history = json.load(stream)

    iterations = [entry for entry in history.get("iterations", []) if entry.get("J_nonQIRatio") is not None]
    if not iterations:
        raise RuntimeError(f"No physical iterations were found in {history_path}")

    index = np.arange(len(iterations))
    total_J = np.array([entry["J"] for entry in iterations], dtype=float)
    best_so_far = np.minimum.accumulate(total_J)
    components = {
        "J_nonQIRatio": np.array([entry["J_nonQIRatio"] for entry in iterations], dtype=float),
        "J_iotas": np.array([entry["J_iotas"] for entry in iterations], dtype=float),
        "J_major_radius": np.array([entry["J_major_radius"] for entry in iterations], dtype=float),
        "J_length": np.array([entry["J_length"] for entry in iterations], dtype=float),
    }
    iota_values = np.array([entry["iota"] for entry in iterations], dtype=float)
    major_radius = np.array([entry["major_radius"] for entry in iterations], dtype=float)
    curve_length_total = np.array([entry["curve_length_total"] for entry in iterations], dtype=float)
    grad_norm = np.array([entry["grad_norm"] for entry in iterations], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)

    axes[0, 0].semilogy(index, total_J, marker="o", color="#4C78A8", label="Total J")
    axes[0, 0].semilogy(index, best_so_far, color="#E45756", linewidth=1.5, label="Best so far")
    axes[0, 0].set_title("Total objective over iterations")
    axes[0, 0].set_xlabel("Accepted iteration")
    axes[0, 0].set_ylabel("Objective value")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(loc="best", fontsize="small")

    for name, color in (("J_nonQIRatio", "#4C78A8"), ("J_iotas", "#F58518"), ("J_major_radius", "#54A24B"), ("J_length", "#E45756")):
        axes[0, 1].semilogy(index, np.maximum(components[name], 1.0e-16), marker="o", label=name, color=color)
    axes[0, 1].set_title("Objective components over iterations")
    axes[0, 1].set_xlabel("Accepted iteration")
    axes[0, 1].set_ylabel("Component value")
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend(loc="best", fontsize="small")

    axes[1, 0].plot(index, iota_values, marker="o", color="#4C78A8", label="iota")
    axes[1, 0].plot(index, major_radius, marker="o", color="#F58518", label="major radius")
    axes[1, 0].set_title("Surface parameters over iterations")
    axes[1, 0].set_xlabel("Accepted iteration")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(loc="best", fontsize="small")

    axes[1, 1].plot(index, curve_length_total, marker="o", color="#54A24B", label="total coil length")
    axes[1, 1].semilogy(index, grad_norm, marker="o", color="#E45756", label="grad norm")
    axes[1, 1].set_title("Length and gradient norm")
    axes[1, 1].set_xlabel("Accepted iteration")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(loc="best", fontsize="small")

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {
        "iterations": len(iterations),
        "initial_total_J": float(total_J[0]),
        "final_total_J": float(total_J[-1]),
        "best_total_J": float(np.min(total_J)),
        "initial_qi": float(components["J_nonQIRatio"][0]),
        "final_qi": float(components["J_nonQIRatio"][-1]),
    }


def _apply_vmec_overrides(vmec_input_path, vmec_ns=None, vmec_niter=None, vmec_ftol=None):
    if vmec_ns is None and vmec_niter is None and vmec_ftol is None:
        return {}

    namelist = f90nml.read(vmec_input_path)
    indata = namelist["indata"]
    applied = {}
    if vmec_ns is not None:
        indata["ns_array"] = [int(vmec_ns)]
        applied["ns_array"] = [int(vmec_ns)]
    if vmec_niter is not None:
        indata["niter_array"] = [int(vmec_niter)]
        applied["niter_array"] = [int(vmec_niter)]
    if vmec_ftol is not None:
        indata["ftol_array"] = [float(vmec_ftol)]
        applied["ftol_array"] = [float(vmec_ftol)]
    f90nml.write(namelist, vmec_input_path, force=True)
    return applied


def _validate_wout_file(wout_path, log_path):
    with netcdf_file(wout_path, mmap=False) as dataset:
        has_geometry = "xm" in dataset.variables and "xn" in dataset.variables
        ier_flag = None
        if "ier_flag" in dataset.variables:
            ier_flag = int(dataset.variables["ier_flag"].data)
    if has_geometry:
        return

    failure_hint = "VMEC did not write a complete wout file."
    if Path(log_path).exists():
        with open(log_path, "r", encoding="utf-8") as stream:
            log_text = stream.read()
        for marker in ["ARNORM OR AZNORM EQUAL ZERO IN BCOVAR", "INITIAL JACOBIAN CHANGED SIGN", "EXECUTION TERMINATED NORMALLY"]:
            if marker in log_text:
                failure_hint = marker
                break
    if ier_flag is not None:
        failure_hint = f"{failure_hint} ier_flag={ier_flag}"
    raise RuntimeError(f"{failure_hint} See {log_path}")


def _find_xvmec(explicit_path=None):
    candidates = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path).expanduser())
    env_path = os.environ.get("SIMSOPT_VMEC_XVMEC")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    discovered = shutil.which("xvmec")
    if discovered:
        candidates.append(Path(discovered))

    local_vmec_root = Path.home() / "local" / "vmec2000"
    candidates.extend(local_vmec_root.glob("_skbuild/*/cmake-install/bin/xvmec"))
    candidates.extend(local_vmec_root.glob("_skbuild/*/cmake-build/build/bin/xvmec"))

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    return None


def _run_xvmec_to_wout(vmec_input_path, staging_dir, xvmec_path, vmec_ns=None, vmec_niter=None, vmec_ftol=None):
    staging_dir = Path(staging_dir).expanduser().resolve()
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    staged_input = staging_dir / Path(vmec_input_path).name
    shutil.copy2(vmec_input_path, staged_input)
    overrides = _apply_vmec_overrides(staged_input, vmec_ns=vmec_ns, vmec_niter=vmec_niter, vmec_ftol=vmec_ftol)
    wout_path = staging_dir / _vmec_wout_name(staged_input)
    log_path = staging_dir / "xvmec.log"
    extension = staged_input.name.split("input.", 1)[1] if staged_input.name.startswith("input.") else staged_input.name

    with open(log_path, "w", encoding="utf-8") as stream:
        subprocess.run(
            [str(xvmec_path), extension],
            cwd=staging_dir,
            check=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )

    if not wout_path.exists():
        raise FileNotFoundError(f"Expected VMEC output file was not created: {wout_path}")
    _validate_wout_file(wout_path, log_path)

    return staged_input, wout_path, log_path, overrides


def _load_vmec_equilibrium(vmec_input_path, ntheta, nphi, staging_dir, xvmec_path=None, vmec_ns=None, vmec_niter=None, vmec_ftol=None):
    vmec_input_path = Path(vmec_input_path).expanduser().resolve()
    try:
        equilibrium = Vmec(
            str(vmec_input_path),
            verbose=False,
            ntheta=ntheta,
            nphi=nphi,
            range_surface="half period",
        )
        equilibrium.run()
        return equilibrium, {
            "mode": "direct_input",
            "input": str(vmec_input_path),
        }
    except ValueError as err:
        if "vmec_input__array__" not in str(err):
            raise

    resolved_xvmec = _find_xvmec(xvmec_path)
    if resolved_xvmec is None:
        raise RuntimeError(
            "Vmec(input) failed because of the installed VMEC wrapper array API, and no xvmec executable was found. "
            "Set --xvmec or SIMSOPT_VMEC_XVMEC to a working xvmec binary."
        )

    staged_input, wout_path, log_path, overrides = _run_xvmec_to_wout(
        vmec_input_path,
        staging_dir,
        resolved_xvmec,
        vmec_ns=vmec_ns,
        vmec_niter=vmec_niter,
        vmec_ftol=vmec_ftol,
    )

    equilibrium = Vmec(
        str(wout_path),
        verbose=False,
        ntheta=ntheta,
        nphi=nphi,
        range_surface="half period",
    )
    return equilibrium, {
        "mode": "wout_fallback",
        "input": str(vmec_input_path),
        "xvmec": str(resolved_xvmec),
        "staged_input": str(staged_input),
        "wout_file": str(wout_path),
        "xvmec_log": str(log_path),
        "overrides": overrides,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare BoozerQI coil-field modB against a VMEC+Booz_Xform field map.")
    parser.add_argument("--coils", required=True, help="Path to the optimized Biot-Savart JSON.")
    parser.add_argument("--surface-restart", required=True, help="Path to the exported Boozer surface restart JSON.")
    parser.add_argument("--vmec-input", required=True, help="Path to the exported VMEC input file.")
    parser.add_argument("--output", required=True, help="Path to a JSON report file.")
    parser.add_argument("--nphi", type=int, default=41, help="Number of toroidal grid points for the comparison.")
    parser.add_argument("--ntheta", type=int, default=41, help="Number of poloidal grid points for the comparison.")
    parser.add_argument("--order", type=int, default=3, help="Radial interpolation order for BoozerRadialInterpolant.")
    parser.add_argument("--booz-mpol", type=int, default=24, help="Booz_Xform poloidal resolution.")
    parser.add_argument("--booz-ntor", type=int, default=24, help="Booz_Xform toroidal resolution.")
    parser.add_argument("--xvmec", default=None, help="Optional path to an xvmec executable for fallback VMEC runs.")
    parser.add_argument("--vmec-ns", type=int, default=None, help="Override VMEC ns_array in the staged input.")
    parser.add_argument("--vmec-niter", type=int, default=None, help="Override VMEC niter_array in the staged input.")
    parser.add_argument("--vmec-ftol", type=float, default=None, help="Override VMEC ftol_array in the staged input.")
    parser.add_argument("--plots-dir", default=None, help="Optional directory in which to save comparison plots.")
    parser.add_argument("--history", default=None, help="Optional optimization history JSON for plotting objective progress.")
    parser.add_argument("--qi-nphi", type=int, default=61, help="Number of Boozer toroidal samples for the QI line-shuffling diagnostic.")
    parser.add_argument("--qi-nalpha", type=int, default=11, help="Number of field-line labels for the QI line-shuffling diagnostic.")
    parser.add_argument("--qi-nbj", type=int, default=15, help="Number of bounce levels for the QI line-shuffling diagnostic.")
    parser.add_argument("--qi-nphi-out", type=int, default=241, help="Number of output samples per line for the QI line-shuffling diagnostic.")
    args = parser.parse_args()

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_path.parent / f"{output_path.stem}_vmec_run"
    plots_dir = None if args.plots_dir is None else Path(args.plots_dir).expanduser().resolve()
    if plots_dir is not None:
        plots_dir.mkdir(parents=True, exist_ok=True)
    history_path = None if args.history is None else Path(args.history).expanduser().resolve()

    report = {
        "inputs": {
            "coils": str(Path(args.coils).expanduser().resolve()),
            "surface_restart": str(Path(args.surface_restart).expanduser().resolve()),
            "vmec_input": str(Path(args.vmec_input).expanduser().resolve()),
        },
        "modules": {
            "vmec": vmec is not None,
            "booz_xform": booz_xform is not None,
        },
    }

    bs = load(args.coils)
    restart_payload, restart_surface = _load_surface_restart(args.surface_restart)

    quadpoints_phi = np.linspace(0.0, 1.0 / restart_surface.nfp, args.nphi, endpoint=False)
    quadpoints_theta = np.linspace(0.0, 1.0, args.ntheta, endpoint=False)
    gamma = _surface_gamma_on_grid(restart_surface, quadpoints_phi, quadpoints_theta)
    reference_modB = _modB_on_gamma(bs, gamma)
    report["boozer_qi_reference"] = {
        "iota": float(restart_payload["iota"]),
        "G": None if restart_payload.get("G") is None else float(restart_payload["G"]),
        "stats": _stats(reference_modB),
    }

    if vmec is None or booz_xform is None:
        report["status"] = "blocked_missing_modules"
        report["message"] = (
            "The vmec and booz_xform Python extensions are required to run the VMEC/Booz_Xform comparison. "
            "Install those modules, then rerun this script."
        )
        with open(output_path, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
        raise RuntimeError(report["message"])

    vmec_equilibrium, vmec_execution = _load_vmec_equilibrium(
        args.vmec_input,
        args.ntheta,
        args.nphi,
        staging_dir,
        args.xvmec,
        args.vmec_ns,
        args.vmec_niter,
        args.vmec_ftol,
    )
    vmec_boundary = SurfaceRZFourier.from_wout(
        vmec_execution["wout_file"] if vmec_execution["mode"] == "wout_fallback" else vmec_equilibrium.output_file,
        nphi=args.nphi,
        ntheta=args.ntheta,
        range="half period",
    )
    vmec_boundary_gamma = _surface_gamma_on_grid(vmec_boundary, quadpoints_phi, quadpoints_theta)
    vmec_boundary_coil_modB = _modB_on_gamma(bs, vmec_boundary_gamma)

    field = BoozerRadialInterpolant(
        vmec_equilibrium,
        args.order,
        mpol=args.booz_mpol,
        ntor=args.booz_ntor,
        enforce_vacuum=True,
    )

    boozer_surface_data = _boozer_surface_field_data(field, args.nphi, args.ntheta, vmec_equilibrium.wout.nfp)
    vmec_modB = boozer_surface_data["modB"]
    vmec_boozer_gamma = boozer_surface_data["gamma"]
    vmec_boozer_coil_modB = _modB_on_gamma(bs, vmec_boozer_gamma)
    reference_qi = _qi_diagnostics_from_modB_grid(
        reference_modB,
        restart_surface.nfp,
        float(restart_payload["iota"]),
        args.qi_nphi,
        args.qi_nalpha,
        args.qi_nbj,
        args.qi_nphi_out,
    )
    vmec_qi = _qi_diagnostics_from_modB_grid(
        vmec_modB,
        int(vmec_equilibrium.wout.nfp),
        float(vmec_equilibrium.wout.iotaf[-1]),
        args.qi_nphi,
        args.qi_nalpha,
        args.qi_nbj,
        args.qi_nphi_out,
    )

    report["vmec_boozer"] = {
        "stats": _stats(vmec_modB),
        "nfp": int(vmec_equilibrium.wout.nfp),
        "surface_parameterization": "VMEC Boozer angles mapped back to physical Cartesian points",
        "qi": _qi_summary(vmec_qi),
    }
    report["boozer_qi_reference"]["qi"] = _qi_summary(reference_qi)
    reference_phi = 2.0 * np.pi * np.asarray(quadpoints_phi)
    reference_theta = 2.0 * np.pi * np.asarray(quadpoints_theta)
    vmec_phi = boozer_surface_data["zeta"][:, 0]
    vmec_theta = boozer_surface_data["theta"][0, :]
    report["boozer_qi_reference"]["modB_topology"] = _assess_modB_topology(reference_phi, reference_theta, reference_modB / np.mean(reference_modB))
    report["vmec_boozer"]["modB_topology"] = _assess_modB_topology(vmec_phi, vmec_theta, vmec_modB / np.mean(vmec_modB))
    report["vmec_boundary_geometry"] = {
        "comparison_method": "shape_distance",
        "note": (
            "Direct gamma-to-gamma differences on a shared index grid are not reported here because the restart surface and "
            "the VMEC boundary generally use different poloidal parameterizations."
        ),
        "shape_distance": _surface_shape_diagnostics(
            restart_surface,
            vmec_boundary,
            max(args.nphi, 81),
            max(args.ntheta, 81),
        ),
    }
    report["coil_on_vmec_boundary"] = {
        "stats": _stats(vmec_boundary_coil_modB),
        "comparison_to_restart_surface": _normalized_field_comparison(vmec_boundary_coil_modB, reference_modB),
    }
    report["coil_on_vmec_boozer_surface"] = {
        "stats": _stats(vmec_boozer_coil_modB),
        "comparison_to_vmec_boozer": _normalized_field_comparison(vmec_boozer_coil_modB, vmec_modB),
    }
    report["vmec_execution"] = vmec_execution
    report["legacy_mixed_parameterization_comparison"] = _normalized_field_comparison(vmec_modB, reference_modB)
    report["comparison"] = report["coil_on_vmec_boozer_surface"]["comparison_to_vmec_boozer"]
    if plots_dir is not None:
        modB_maps_path = plots_dir / "modB_boozer_maps.png"
        modB_contours_path = plots_dir / "modB_boozer_contours.png"
        qi_heatmaps_path = plots_dir / "qi_heatmaps.png"
        qi_profiles_path = plots_dir / "qi_line_profiles.png"
        surface_plot_path = plots_dir / "surface_modB_comparison.png"
        vmec_overview_path = plots_dir / "vmec_overview.png"
        objective_history_path = plots_dir / "objective_history.png"
        _save_modB_map_plot(reference_modB, vmec_modB, vmec_boozer_coil_modB, modB_maps_path, int(vmec_equilibrium.wout.nfp))
        contour_topology = _save_modB_contour_plot(reference_modB, vmec_modB, modB_contours_path, int(vmec_equilibrium.wout.nfp))
        _save_qi_heatmap_plot(reference_qi, vmec_qi, qi_heatmaps_path)
        _save_qi_profile_plot(reference_qi, vmec_qi, qi_profiles_path)
        _save_surface_comparison_plot(gamma, reference_modB, vmec_boozer_gamma, vmec_modB, vmec_boozer_coil_modB, surface_plot_path)
        _save_vmec_overview_plot(
            vmec_execution["wout_file"] if vmec_execution["mode"] == "wout_fallback" else vmec_equilibrium.output_file,
            vmec_overview_path,
        )
        history_summary = None
        if history_path is not None:
            history_summary = _save_objective_history_plot(history_path, objective_history_path)
        report["plots"] = {
            "directory": str(plots_dir),
            "modB_maps": str(modB_maps_path),
            "modB_contours": str(modB_contours_path),
            "qi_heatmaps": str(qi_heatmaps_path),
            "qi_line_profiles": str(qi_profiles_path),
            "surface_modB_comparison": str(surface_plot_path),
            "vmec_overview": str(vmec_overview_path),
        }
        if history_summary is not None:
            report["plots"]["objective_history"] = str(objective_history_path)
            report["optimization_history_summary"] = history_summary
        report["contour_topology"] = contour_topology
    report["status"] = "ok"

    with open(output_path, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)

    print(
        f"VMEC/Booz_Xform comparison complete: normalized_rel_l2={report['comparison']['normalized_rel_l2']:.6e}, "
        f"legacy_mixed_parameterization_rel_l2={report['legacy_mixed_parameterization_comparison']['normalized_rel_l2']:.6e}, "
        f"mean_ratio={report['comparison']['mean_ratio']:.6e}"
    )


if __name__ == "__main__":
    main()