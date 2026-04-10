#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import numpy as np

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
from simsopt.mhd import Vmec


def _comparison(coils, vmec, order, booz_mpol, booz_ntor, nphi, ntheta):
    field = BoozerRadialInterpolant(vmec, order, mpol=booz_mpol, ntor=booz_ntor, enforce_vacuum=True)
    thetas = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    zetas = np.linspace(0.0, 2.0 * np.pi / vmec.wout.nfp, nphi, endpoint=False)
    zeta_grid, theta_grid = np.meshgrid(zetas, thetas, indexing="ij")

    points = np.zeros((nphi * ntheta, 3))
    points[:, 0] = 1.0
    points[:, 1] = theta_grid.ravel()
    points[:, 2] = zeta_grid.ravel()
    field.set_points(points)

    vmec_modB = field.modB()[:, 0].reshape((nphi, ntheta))
    R = field.R()[:, 0].reshape((nphi, ntheta))
    Z = field.Z()[:, 0].reshape((nphi, ntheta))
    nu = field.nu()[:, 0].reshape((nphi, ntheta))
    phi = zeta_grid - nu

    xyz = np.zeros((nphi, ntheta, 3))
    xyz[:, :, 0] = R * np.cos(phi)
    xyz[:, :, 1] = R * np.sin(phi)
    xyz[:, :, 2] = Z
    coils.set_points(xyz.reshape((-1, 3)))
    coil_modB = np.linalg.norm(coils.B().reshape((nphi, ntheta, 3)), axis=2)

    normalized_coil = coil_modB / np.mean(coil_modB)
    normalized_vmec = vmec_modB / np.mean(vmec_modB)
    return {
        "order": int(order),
        "booz_mpol": int(booz_mpol),
        "booz_ntor": int(booz_ntor),
        "nphi": int(nphi),
        "ntheta": int(ntheta),
        "absolute_rel_l2": float(np.linalg.norm(coil_modB - vmec_modB) / np.linalg.norm(vmec_modB)),
        "normalized_rel_l2": float(np.linalg.norm(normalized_coil - normalized_vmec) / np.linalg.norm(normalized_vmec)),
        "normalized_max_abs_diff": float(np.max(np.abs(normalized_coil - normalized_vmec))),
        "mean_ratio": float(np.mean(coil_modB) / np.mean(vmec_modB)),
    }


def main():
    parser = argparse.ArgumentParser(description="Sweep postprocessing resolution for a finished BoozerQI VMEC comparison.")
    parser.add_argument("--coils", required=True, help="Path to the optimized Biot-Savart JSON.")
    parser.add_argument("--wout", required=True, help="Path to the finished VMEC wout file.")
    parser.add_argument("--output", required=True, help="Path to the output JSON report.")
    args = parser.parse_args()

    coils = load(args.coils)
    vmec = Vmec(args.wout, verbose=False, ntheta=81, nphi=81, range_surface="half period")
    cases = [
        (3, 24, 24, 41, 41),
        (3, 36, 36, 41, 41),
        (5, 24, 24, 41, 41),
        (5, 36, 36, 41, 41),
        (5, 48, 48, 41, 41),
        (5, 48, 48, 61, 61),
        (5, 48, 48, 81, 81),
    ]
    results = [_comparison(coils, vmec, *case) for case in cases]

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as stream:
        json.dump({"cases": results}, stream, indent=2)

    best = min(results, key=lambda item: item["normalized_rel_l2"])
    print(
        "Resolution sweep complete: "
        f"best normalized_rel_l2={best['normalized_rel_l2']:.6e} "
        f"at order={best['order']}, booz=({best['booz_mpol']},{best['booz_ntor']}), grid=({best['nphi']},{best['ntheta']})"
    )


if __name__ == "__main__":
    main()