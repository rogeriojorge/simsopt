#!/usr/bin/env python3

import json
import os
import sys
from pathlib import Path

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
from scipy.optimize import minimize

from simsopt.geo import (
    BoozerSurface,
    CurveLength,
    Iotas,
    MajorRadius,
    NonQuasiIsodynamicRatio,
    SurfaceXYZTensorFourier,
    Volume,
    boozer_surface_residual,
    curves_to_vtk,
)
from simsopt.objectives import QuadraticPenalty
from simsopt.util import in_github_actions

from boozer_qi_helper import (
    QIConfig,
    QIOptimizationDriver,
    build_minimize_config,
    export_final_artifacts,
    initialize_surface,
    load_coil_seed,
    make_surface,
    maxiter_from_environment,
    qi_resolution_defaults,
    qi_resolution_from_environment,
    resolve_surface_initializer,
    run_taylor_test,
    solve_boozer_surface_with_seed_fallback,
)

r"""
This example optimizes the NCSX coils and currents for QI on a single surface. The objective is

    J = J_QI
        + 0.5*(iota - iota_0)**2
        + 0.5*(major_radius - target_major_radius)**2
        + 0.5*max(sum(CurveLength) - CurveLengthTarget, 0)**2

The top-level flow intentionally mirrors boozerQA.py: build one Boozer surface,
assemble the objective, optionally run checks, then optimize. The extra runtime
controls, restart handling, and VMEC export machinery live in boozer_qi_helper.py.
"""


def print_section(title):
    print(
        "\n" + "#" * 80,
        f"### {title} ".ljust(79, "#"),
        "#" * 80,
        sep="\n",
    )


def print_artifact_summary(artifacts):
    print(f"Wrote optimized Biot-Savart JSON to {artifacts['coils_json']}")
    print(f"Wrote Boozer surface restart to {artifacts['surface_restart']}")
    print(f"Wrote VMEC input to {artifacts['vmec_input']} using phiedge={artifacts['vmec_phiedge']:.6e}")
    print(
        f"Wrote VMEC export diagnostics to {artifacts['vmec_export_diagnostics']} "
        f"with relative_l2={artifacts['vmec_export_relative_l2']:.3e}"
    )
    print(f"Wrote VMEC export cross-section plot to {artifacts['vmec_export_cross_sections_plot']}")
    print(f"Wrote VMEC export 3D plot to {artifacts['vmec_export_surface_plot']}")


def main():
    config = QIConfig.from_environment(REPO_ROOT)
    config.validate()
    config.prepare_output_dir()

    print("Running 2_Intermediate/boozerQI.py")
    print("================================")
    print(f"Optimizer method={config.opt_method}, step_rms_limit={config.step_rms_limit:.2e}")
    print(
        f"Boozer surface mode={config.boozer_type}, skip_taylor={config.skip_taylor}, "
        f"compare_fd={config.compare_fd}, compare_optimizers={config.compare_optimizers}, "
        f"exact_report={config.exact_report}"
    )

    initializer = resolve_surface_initializer(config)
    seed = load_coil_seed(config, initializer)

    mpol_default = 3 if in_github_actions else 6
    ntor_default = 3 if in_github_actions else 6
    mpol = int(os.environ.get("SIMSOPT_BOOZER_QI_MPOL", str(mpol_default)))
    ntor = int(os.environ.get("SIMSOPT_BOOZER_QI_NTOR", str(ntor_default)))
    surface, phis, thetas = make_surface(seed.nfp, mpol, ntor)
    initialize_surface(surface, initializer, seed.ma, seed.nfp, phis, thetas)

    initial_iota = float(initializer.restart_state["iota"]) if initializer.restart_state is not None else -0.406
    volume = Volume(surface)
    vol_target = volume.J()
    boozer_surface = BoozerSurface(
        seed.bs,
        surface,
        volume,
        vol_target,
        constraint_weight=config.exact_ls_weight if config.boozer_type == "ls" else None,
        options={
            "verbose": config.boozer_verbose,
            "exact_ls_fallback": config.exact_ls_fallback,
            "exact_ls_constraint_weight": config.exact_ls_weight,
            "exact_ls_maxiter": config.exact_ls_maxiter,
            "exact_ls_tol": 1e-10,
        },
    )
    initial_res = solve_boozer_surface_with_seed_fallback(
        boozer_surface,
        iota=initial_iota,
        G=(
            float(initializer.restart_state["G"])
            if initializer.restart_state is not None and initializer.restart_state.get("G") is not None
            else seed.G0
        ),
        tol=1e-13,
        maxiter=20,
        verbose=config.boozer_verbose,
    )

    residual = boozer_surface_residual(surface, initial_res["iota"], initial_res["G"], seed.bs, derivatives=0)[0]
    initial_iter = initial_res.get("iter")
    if initial_iter is None and "info" in initial_res:
        initial_iter = getattr(initial_res["info"], "nfev", None)
    initial_solver = "LS" if config.boozer_type == "ls" else ("NEWTON" if "iter" in initial_res else "SEEDED")
    print(
        f"{initial_solver} {initial_res['success']}: iter={initial_iter}, iota={initial_res['iota']:.3f}, "
        f"vol={surface.volume():.3f}, ||residual||={np.linalg.norm(residual):.3e}"
    )

    qi_resolution = qi_resolution_from_environment(qi_resolution_defaults(in_github_actions))
    major_radius = MajorRadius(boozer_surface)
    length_terms = [CurveLength(curve) for curve in seed.base_curves]
    target_iota = float(initial_res["iota"])
    target_major_radius = float(major_radius.J())
    target_curve_length = float(sum(length_terms).J())

    J_major_radius = QuadraticPenalty(major_radius, target_major_radius, "identity")
    J_iotas = QuadraticPenalty(Iotas(boozer_surface), target_iota, "identity")
    J_non_qi_ratio = NonQuasiIsodynamicRatio(
        boozer_surface,
        seed.bs_qi,
        sDIM=qi_resolution["sDIM"],
        nphi=qi_resolution["nphi"],
        nalpha=qi_resolution["nalpha"],
        nBj=qi_resolution["nBj"],
        nphi_out=qi_resolution["nphi_out"],
    )
    J_length = QuadraticPenalty(sum(length_terms), target_curve_length, "max")
    objective = J_non_qi_ratio + J_iotas + J_major_radius + J_length

    if config.write_vtk:
        curves_to_vtk(seed.all_curves, os.path.join(config.out_dir, "curves_init"))
        boozer_surface.surface.to_vtk(os.path.join(config.out_dir, "surf_init"))

    seed.base_currents[0].fix_all()
    driver = QIOptimizationDriver(
        config,
        objective,
        boozer_surface,
        J_non_qi_ratio,
        J_iotas,
        J_major_radius,
        J_length,
        major_radius,
        length_terms,
        seed.bs,
        seed.bs_qi,
        vol_target,
        target_iota,
        target_major_radius,
        qi_resolution,
    )

    if not config.skip_taylor:
        print_section("Perform a Taylor test")
        run_taylor_test(driver.fun, objective.x)

    print_section("Run the optimization")
    initial_dofs = objective.x.copy()
    initial_value, initial_grad = driver.fun(initial_dofs)
    print(f"Initial gradient norm = {np.linalg.norm(initial_grad):.3e}")
    base_snapshot = driver.snapshot()
    maxiter = maxiter_from_environment(in_github_actions)

    if config.compare_fd:
        print_section("Compare against finite differences")
        fd_results = driver.run_directional_fd_comparison(base_snapshot, config.compare_fd_dirs, config.compare_fd_eps)
        driver.history["fd_comparison"] = {"eps": config.compare_fd_eps, "directional": fd_results}
        max_rel = max(result["relative_error"] for result in fd_results) if fd_results else 0.0
        for result in fd_results:
            print(
                f"FD dir {result['direction']}: adjoint={result['adjoint']:.6e}, "
                f"fd={result['finite_difference']:.6e}, rel_err={result['relative_error']:.3e}"
            )
        print(f"Max FD relative error={max_rel:.3e}")
        if config.compare_fd_opt:
            fd_opt = driver.run_fd_subspace_optimization(
                base_snapshot,
                config.compare_fd_subspace,
                config.compare_fd_opt_maxiter,
            )
            driver.history["fd_comparison"]["subspace_optimization"] = fd_opt
            print(
                f"FD reduced-subspace comparison: analytic_fun={fd_opt['analytic_fun']:.6e}, "
                f"fd_fun={fd_opt['fd_fun']:.6e}, step_diff={fd_opt['step_difference_norm']:.3e}, "
                f"fun_diff={fd_opt['fun_difference']:.3e}, subspace={fd_opt['subspace_size']}"
            )

    if config.compare_optimizers:
        print_section("Compare reduced optimizer paths")
        optimizer_comparison = driver.run_reduced_optimizer_comparison(
            base_snapshot,
            config.compare_optimizers_subspace,
            config.compare_optimizers_list,
            config.compare_optimizers_maxiter,
        )
        driver.history["optimizer_comparison"] = optimizer_comparison
        for result in optimizer_comparison["results"]:
            print(
                f"Optimizer {result['method']}: success={result['success']}, nit={result['nit']}, "
                f"fun={result['fun']:.6e}, active_step={result['active_step_norm']:.3e}"
            )

    opt_method, minimize_options, minimize_kwargs = build_minimize_config(
        config.opt_method,
        maxiter,
        initial_dofs.size,
        config.step_rms_limit,
        config.maxls,
        config.maxcor,
    )
    result = minimize(
        driver.fun,
        initial_dofs,
        jac=True,
        method=opt_method,
        options=minimize_options,
        callback=driver.callback,
        tol=1e-15,
        **minimize_kwargs,
    )
    driver.restore_best()

    if config.exact_report and driver.best_good["J"] is not None:
        driver.history["exact_report"] = driver.run_exact_report()
        exact_report = driver.history["exact_report"]
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

    driver.history["result"] = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "nit": int(result.nit),
        "nfev": int(result.nfev),
        "njev": int(result.njev) if getattr(result, "njev", None) is not None else None,
        "initial_J": float(initial_value),
        "final_J": float(result.fun),
        "best_J": float(driver.best_good["J"] if driver.best_good["J"] is not None else result.fun),
    }

    if config.write_vtk:
        curves_to_vtk(seed.all_curves, os.path.join(config.out_dir, "curves_opt"))
        boozer_surface.surface.to_vtk(os.path.join(config.out_dir, "surf_opt"))

    driver.history["artifacts"] = export_final_artifacts(config, seed.bs, boozer_surface)
    history_path = Path(config.history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as stream:
        json.dump(driver.history, stream, indent=2)

    best_J = driver.best_good["J"] if driver.best_good["J"] is not None else result.fun
    print(
        f"Optimization success={result.success}, status={result.status}, nit={result.nit}, "
        f"initial_J={initial_value:.6e}, final_J={result.fun:.6e}, best_J={best_J:.6e}"
    )
    print(f"Wrote optimization history to {history_path}")
    print_artifact_summary(driver.history["artifacts"])
    print("End of 2_Intermediate/boozerQI.py")
    print("================================")


if __name__ == "__main__":
    main()