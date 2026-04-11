import unittest
import json
import numpy as np
import os
import subprocess
import sys
import tempfile
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

from simsopt._core.optimizable import load
from simsopt.field.biotsavart import BiotSavart
from simsopt.geo.surfaceobjectives import ToroidalFlux, QfmResidual, parameter_derivatives, Volume, PrincipalCurvature, MajorRadius, Iotas, NonQuasiSymmetricRatio, NonQuasiIsodynamicRatio, BoozerResidual, _make_shuffled_target_values
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.configs.zoo import get_data
from .surface_test_helpers import get_surface, get_exact_surface, get_boozer_surface


surfacetypes_list = ["SurfaceXYZFourier", "SurfaceRZFourier",
                     "SurfaceXYZTensorFourier"]
stellsym_list = [True, False]


def taylor_test1(f, df, x, epsilons=None, direction=None, atol=1e-9):
    np.random.seed(1)
    f(x)
    if direction is None:
        direction = np.random.rand(*(x.shape))-0.5
    dfx = df(x)@direction
    if epsilons is None:
        epsilons = np.power(2., -np.asarray(range(10, 20)))
    print("###################################################################")
    err_old = 1e9
    for eps in epsilons:
        fpluseps = f(x + eps * direction)
        fminuseps = f(x - eps * direction)
        dfest = (fpluseps-fminuseps)/(2*eps)
        err = np.linalg.norm(dfest - dfx)
        print("taylor test1: ", err, err/err_old)
        np.testing.assert_array_less(err, max(atol, 0.35 * err_old),
                                     err_msg=f"Taylor test failed: err={err:.2e}, threshold={max(atol, 0.35 * err_old):.2e}, "
                                             f"err_old={err_old:.2e}, ratio={err/err_old:.4f}")
        err_old = err
    print("###################################################################")


def taylor_test2(f, df, d2f, x, epsilons=None, direction1=None, direction2=None):
    np.random.seed(1)
    if direction1 is None:
        direction1 = np.random.rand(*(x.shape))-0.5
    if direction2 is None:
        direction2 = np.random.rand(*(x.shape))-0.5

    f(x)
    df0 = df(x) @ direction1
    d2fval = direction2.T @ d2f(x) @ direction1
    if epsilons is None:
        epsilons = np.power(2., -np.asarray(range(7, 20)))
    print("###################################################################")
    err_old = 1e9
    for eps in epsilons:
        fpluseps = df(x + eps * direction2) @ direction1
        d2fest = (fpluseps-df0)/eps
        err = np.abs(d2fest - d2fval)
        print('taylor test2: ', err, err/err_old)
        assert err < 0.6 * err_old
        err_old = err
    print("###################################################################")


class ToroidalFluxTests(unittest.TestCase):
    def test_toroidal_flux_is_constant(self):
        """
        this test ensures that the toroidal flux does not change, regardless
        of the cross section (varphi = constant) across which it is computed
        """
        s = get_exact_surface()
        base_curves, base_currents, ma, nfp, bs= get_data("ncsx")
        bs_tf = BiotSavart(bs.coils)

        gamma = s.gamma()
        num_phi = gamma.shape[0]

        tf_list = np.zeros((num_phi,))
        for idx in range(num_phi):
            tf = ToroidalFlux(s, bs_tf, idx=idx)
            tf_list[idx] = tf.J()
        mean_tf = np.mean(tf_list)

        max_err = np.max(np.abs(mean_tf - tf_list)) / mean_tf
        assert max_err < 1e-2

    def test_toroidal_flux_first_derivative(self):
        """
        Taylor test for partial derivatives of toroidal flux with respect to surface coefficients
        """

        for surfacetype in surfacetypes_list:
            for stellsym in stellsym_list:
                with self.subTest(surfacetype=surfacetype, stellsym=stellsym):
                    self.subtest_toroidal_flux1(surfacetype, stellsym)

    def test_toroidal_flux_second_derivative(self):
        """
        Taylor test for Hessian of toroidal flux with respect to surface coefficients
        """

        for surfacetype in surfacetypes_list:
            for stellsym in stellsym_list:
                with self.subTest(surfacetype=surfacetype, stellsym=stellsym):
                    self.subtest_toroidal_flux2(surfacetype, stellsym)

    def test_toroidal_flux_partial_derivatives_wrt_coils(self):
        """
        Taylor test for partial derivative of toroidal flux with respect to surface coefficients
        """

        for surfacetype in surfacetypes_list:
            for stellsym in stellsym_list:
                with self.subTest(surfacetype=surfacetype, stellsym=stellsym):
                    self.subtest_toroidal_flux3(surfacetype, stellsym)

    def subtest_toroidal_flux1(self, surfacetype, stellsym):
        base_curves, base_currents, ma, nfp, bs = get_data("ncsx")
        bs_tf = BiotSavart(bs.coils)
        s = get_surface(surfacetype, stellsym)

        tf = ToroidalFlux(s, bs_tf)
        coeffs = s.x

        def f(dofs):
            s.x = dofs
            return tf.J()

        def df(dofs):
            s.x = dofs
            return tf.dJ_by_dsurfacecoefficients()
        taylor_test1(f, df, coeffs)

    def subtest_toroidal_flux2(self, surfacetype, stellsym):
        base_curves, base_currents, ma, nfp, bs = get_data("ncsx")
        s = get_surface(surfacetype, stellsym)

        tf = ToroidalFlux(s, bs)
        coeffs = s.x

        def f(dofs):
            s.x = dofs
            return tf.J()

        def df(dofs):
            s.x = dofs
            return tf.dJ_by_dsurfacecoefficients()

        def d2f(dofs):
            s.x = dofs
            return tf.d2J_by_dsurfacecoefficientsdsurfacecoefficients()

        taylor_test2(f, df, d2f, coeffs)

    def subtest_toroidal_flux3(self, surfacetype, stellsym):
        base_curves, base_currents, ma, nfp, bs = get_data("ncsx")
        bs_tf = BiotSavart(bs.coils)
        s = get_surface(surfacetype, stellsym)

        tf = ToroidalFlux(s, bs_tf)
        coeffs = bs_tf.x

        def f(dofs):
            bs_tf.x = dofs
            return tf.J()

        def df(dofs):
            bs_tf.x = dofs
            return tf.dJ_by_dcoils()(bs_tf)
        taylor_test1(f, df, coeffs)


class PrincipalCurvatureTests(unittest.TestCase):
    def test_principal_curvature_first_derivative(self):
        """
        Taylor test for gradient of principal curvature metric.
        """
        for surfacetype in surfacetypes_list:
            for stellsym in stellsym_list:
                with self.subTest(surfacetype=surfacetype, stellsym=stellsym):
                    self.subtest_principal_curvature(surfacetype, stellsym)

    def subtest_principal_curvature(self, surfacetype, stellsym):
        s = get_surface(surfacetype, stellsym)

        pc = PrincipalCurvature(s, kappamax1=1, kappamax2=2.2, weight1=1, weight2=2.)
        coeffs = s.x

        def f(dofs):
            s.x = dofs
            return pc.J()

        def df(dofs):
            s.x = dofs
            return pc.dJ()

        taylor_test1(f, df, coeffs, epsilons=np.power(2., -np.asarray(range(13, 22))))


class ParameterDerivativesTest(unittest.TestCase):
    def test_parameter_derivatives_volume(self):
        """
        Test that parameter derivatives of volume (shape_gradient = 1) match
        parameter derivatives computed from Volume class.
        """
        for surfacetype in surfacetypes_list:
            for stellsym in stellsym_list:
                with self.subTest(surfacetype=surfacetype, stellsym=stellsym):
                    self.subtest_volume(surfacetype, stellsym)

    def subtest_volume(self, surfacetype, stellsym):
        from simsopt.geo import Surface
        s = get_surface(surfacetype, stellsym, mpol=7, ntor=6,
                        ntheta=32, nphi=31, full=True)
        dofs = s.get_dofs()
        vol = Volume(s, range=Surface.RANGE_FIELD_PERIOD)
        dvol_sg = parameter_derivatives(s, np.ones_like(s.gamma()[:, :, 0]))
        dvol = vol.dJ_by_dsurfacecoefficients()
        for i in range(len(dofs)):
            self.assertAlmostEqual(dvol_sg[i], dvol[i], places=10)


class QfmTests(unittest.TestCase):
    def test_qfm_surface_derivative(self):
        """
        Taylor test for derivative of qfm metric wrt surface parameters
        """
        for surfacetype in surfacetypes_list:
            for stellsym in stellsym_list:
                with self.subTest(surfacetype=surfacetype, stellsym=stellsym):
                    self.subtest_qfm1(surfacetype, stellsym)

    def subtest_qfm1(self, surfacetype, stellsym):
        base_curves, base_currents, ma, nfp, bs = get_data("ncsx")
        s = get_surface(surfacetype, stellsym)
        coeffs = s.x
        qfm = QfmResidual(s, bs)

        def f(dofs):
            s.x = dofs
            return qfm.J()

        def df(dofs):
            s.x = dofs
            return qfm.dJ_by_dsurfacecoefficients()
        taylor_test1(f, df, coeffs,
                     epsilons=np.power(2., -np.asarray(range(13, 22))))


class MajorRadiusTests(unittest.TestCase):
    def test_major_radius_derivative(self):
        """
        Taylor test for derivative of surface major radius wrt coil parameters
        """
        for boozer_type in ['exact', 'ls']:
            for label in ["Volume", "ToroidalFlux"]:
                for optimize_G in [True, False]:
                    for weight_inv_modB in [True, False]:
                        with self.subTest(label=label, boozer_type=boozer_type, optimize_G=optimize_G):
                            if boozer_type == 'ls' and label == 'ToroidalFlux':
                                continue
                            if boozer_type == 'exact' and optimize_G is False:
                                continue
                            if boozer_type == 'exact' and weight_inv_modB:
                                continue
                            self.subtest_major_radius_surface_derivative(label, boozer_type, optimize_G, weight_inv_modB)

    def subtest_major_radius_surface_derivative(self, label, boozer_type, optimize_G, weight_inv_modB):
        bs, boozer_surface = get_boozer_surface(label=label, nphi=51, ntheta=51, boozer_type=boozer_type, optimize_G=optimize_G, weight_inv_modB=weight_inv_modB)
        coeffs = bs.x
        mr = MajorRadius(boozer_surface)

        def f(dofs):
            bs.x = dofs
            return mr.J()

        def df(dofs):
            bs.x = dofs
            return mr.dJ()
        taylor_test1(f, df, coeffs,
                     epsilons=np.power(2., -np.asarray(range(13, 18))))


class IotasTests(unittest.TestCase):
    def test_iotas_derivative(self):
        """
        Taylor test for derivative of surface rotational transform wrt coil parameters
        """

        for boozer_type in ['exact', 'ls']:
            for label in ["Volume", "ToroidalFlux"]:
                for optimize_G in [True, False]:
                    for weight_inv_modB in [True, False]:
                        if boozer_type == 'ls' and label == 'ToroidalFlux':
                            continue
                        if boozer_type == 'exact' and optimize_G is False:
                            continue
                        if boozer_type == 'exact' and weight_inv_modB:
                            continue
                        with self.subTest(label=label, boozer_type=boozer_type, optimize_G=optimize_G, weight_inv_modB=weight_inv_modB):
                            self.subtest_iotas_derivative(label, boozer_type, optimize_G, weight_inv_modB)

    def subtest_iotas_derivative(self, label, boozer_type, optimize_G, weight_inv_modB):
        """
        Taylor test for derivative of surface rotational transform wrt coil parameters
        """
        np.random.seed(1)  # Fixed seed for reproducibility across platforms

        bs, boozer_surface = get_boozer_surface(label=label, boozer_type=boozer_type, optimize_G=optimize_G, weight_inv_modB=weight_inv_modB)
        coeffs = bs.x
        io = Iotas(boozer_surface)

        def f(dofs):
            bs.x = dofs
            return io.J()

        def df(dofs):
            bs.x = dofs
            return io.dJ()

        taylor_test1(f, df, coeffs,
                     epsilons=np.power(2., -np.asarray(range(13, 19))), atol=2e-8)


class NonQSRatioTests(unittest.TestCase):
    def test_nonQSratio_derivative(self):
        """
        Taylor test for derivative of surface non QS ratio wrt coil parameters
        """
        for boozer_type in ['exact', 'ls']:
            for label in ["Volume", "ToroidalFlux"]:
                for weight_inv_modB in [True, False]:
                    for optimize_G in [True, False]:
                        for fix_coil_dof in [True, False]:
                            if boozer_type == 'ls' and label == 'ToroidalFlux':
                                continue
                            if boozer_type == 'exact' and optimize_G is False:
                                continue
                            if boozer_type == 'exact' and weight_inv_modB:
                                continue
                            for axis in [False, True]:
                                with self.subTest(label=label, axis=axis, boozer_type=boozer_type, optimize_G=optimize_G, weight_inv_modB=weight_inv_modB, fix_coil_dof=fix_coil_dof):
                                    self.subtest_nonQSratio_derivative(label, axis, boozer_type, optimize_G, weight_inv_modB, fix_coil_dof)

    def subtest_nonQSratio_derivative(self, label, axis, boozer_type, optimize_G, weight_inv_modB, fix_coil_dof):
        bs, boozer_surface = get_boozer_surface(label=label, boozer_type=boozer_type, optimize_G=optimize_G, weight_inv_modB=weight_inv_modB)

        if fix_coil_dof:
            bs.coils[0].curve.fix('xc(0)')

        coeffs = bs.x
        io = NonQuasiSymmetricRatio(boozer_surface, bs, quasi_poloidal=axis)

        def f(dofs):
            bs.x = dofs
            return io.J()

        def df(dofs):
            bs.x = dofs
            return io.dJ()

        taylor_test1(f, df, coeffs,
                     epsilons=np.power(2., -np.asarray(range(13, 19))))


class NonQIRatioSmokeTests(unittest.TestCase):
    def test_nonQI_shuffle_target_handles_duplicate_locations(self):
        phi_values = np.linspace(0.0, 1.0, 9)
        bounce_distances = np.array([0.2, 0.2, 0.2])
        bounce_levels = np.array([0.0, 0.5, 1.0])
        branch_locations = np.array([0.4, 0.4, 0.4, 0.4, 0.4])
        mean_bounce_distances = np.array([0.2, 0.2, 0.2])

        target_values = _make_shuffled_target_values(
            phi_values,
            bounce_distances,
            bounce_levels,
            branch_locations,
            mean_bounce_distances,
        )

        self.assertEqual(target_values.shape, phi_values.shape)
        self.assertTrue(np.all(np.isfinite(target_values)))

    def test_nonQIratio_value_is_finite(self):
        bs, boozer_surface = get_boozer_surface(label="Volume", boozer_type='exact', optimize_G=True, weight_inv_modB=False)
        objective = NonQuasiIsodynamicRatio(boozer_surface, bs, sDIM=10, nphi=31, nalpha=5, nBj=7, nphi_out=41)
        value = objective.J()
        self.assertTrue(np.isfinite(value))
        self.assertGreaterEqual(value, 0.0)

    def test_nonQIratio_derivative_is_finite(self):
        bs, boozer_surface = get_boozer_surface(label="Volume", boozer_type='exact', optimize_G=True, weight_inv_modB=False)
        objective = NonQuasiIsodynamicRatio(boozer_surface, bs, sDIM=6, nphi=21, nalpha=3, nBj=5, nphi_out=21)
        gradient = objective.dJ()
        self.assertEqual(gradient.shape, bs.x.shape)
        self.assertTrue(np.all(np.isfinite(gradient)))

    def test_nonQIratio_derivative_directional_finite_difference(self):
        bs, boozer_surface = get_boozer_surface(label="Volume", boozer_type='ls', optimize_G=True, weight_inv_modB=False)
        objective = NonQuasiIsodynamicRatio(boozer_surface, bs, sDIM=6, nphi=21, nalpha=3, nBj=5, nphi_out=21, phi_shift=0.0)
        coeffs = bs.x.copy()
        np.random.seed(1)
        direction = np.random.rand(*coeffs.shape) - 0.5

        bs.x = coeffs
        objective.recompute_bell()
        gradient = objective.dJ()
        directional_derivative = float(gradient @ direction)

        eps = 2.0 ** -17
        bs.x = coeffs + eps * direction
        objective.recompute_bell()
        plus_value = objective.J()
        bs.x = coeffs - eps * direction
        objective.recompute_bell()
        minus_value = objective.J()

        bs.x = coeffs
        objective.recompute_bell()
        finite_difference = (plus_value - minus_value) / (2 * eps)
        self.assertLess(abs(directional_derivative - finite_difference), 5e-5)

    def test_nonQIratio_value_changes_under_small_perturbation(self):
        bs, boozer_surface = get_boozer_surface(label="Volume", boozer_type='exact', optimize_G=True, weight_inv_modB=False)
        objective = NonQuasiIsodynamicRatio(boozer_surface, bs, sDIM=6, nphi=21, nalpha=3, nBj=5, nphi_out=21)
        x0 = bs.x.copy()
        j0 = objective.J()

        np.random.seed(3)
        perturbation = 1e-3 * (np.random.rand(*x0.shape) - 0.5)
        bs.x = x0 + perturbation
        objective.recompute_bell()
        j1 = objective.J()

        bs.x = x0
        objective.recompute_bell()
        self.assertTrue(np.isfinite(j1))
        self.assertGreater(abs(j1 - j0), 1e-10)


class BoozerQIExampleTests(unittest.TestCase):
    def _run_boozer_qi(self, env):
        repo_root = "/Users/rogerio/local/simsopt_boozer_QI"
        script = os.path.join(repo_root, "examples", "2_Intermediate", "boozerQI.py")
        pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = os.path.join(repo_root, "src") if not pythonpath else os.path.join(repo_root, "src") + os.pathsep + pythonpath
        return subprocess.run(
            [sys.executable, script],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_boozer_qi_example_reduced_runtime(self):
        env = os.environ.copy()
        env.update({
            "SIMSOPT_BOOZER_QI_WRITE_VTK": "0",
            "SIMSOPT_BOOZER_QI_SKIP_TAYLOR": "1",
            "SIMSOPT_BOOZER_QI_MAXITER": "0",
            "SIMSOPT_BOOZER_QI_MPOL": "3",
            "SIMSOPT_BOOZER_QI_NTOR": "3",
            "SIMSOPT_BOOZER_QI_SDIM": "6",
            "SIMSOPT_BOOZER_QI_NPHI": "21",
            "SIMSOPT_BOOZER_QI_NALPHA": "3",
            "SIMSOPT_BOOZER_QI_NBJ": "5",
            "SIMSOPT_BOOZER_QI_NPHI_OUT": "21",
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            env["SIMSOPT_BOOZER_QI_OUT_DIR"] = tmpdir
            result = self._run_boozer_qi(env)

        if result.returncode != 0:
            self.fail(f"boozerQI.py failed with return code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")

        self.assertIn("Running 2_Intermediate/boozerQI.py", result.stdout)
        self.assertIn("Optimization success=", result.stdout)

    def test_boozer_qi_example_fd_comparison(self):
        env = os.environ.copy()
        env.update({
            "SIMSOPT_BOOZER_QI_WRITE_VTK": "0",
            "SIMSOPT_BOOZER_QI_SKIP_TAYLOR": "1",
            "SIMSOPT_BOOZER_QI_COMPARE_FD": "1",
            "SIMSOPT_BOOZER_QI_COMPARE_FD_DIRS": "1",
            "SIMSOPT_BOOZER_QI_COMPARE_FD_EPS": str(2.0 ** -18),
            "SIMSOPT_BOOZER_QI_MAXITER": "0",
            "SIMSOPT_BOOZER_QI_MPOL": "3",
            "SIMSOPT_BOOZER_QI_NTOR": "3",
            "SIMSOPT_BOOZER_QI_SDIM": "6",
            "SIMSOPT_BOOZER_QI_NPHI": "21",
            "SIMSOPT_BOOZER_QI_NALPHA": "3",
            "SIMSOPT_BOOZER_QI_NBJ": "5",
            "SIMSOPT_BOOZER_QI_NPHI_OUT": "21",
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            env["SIMSOPT_BOOZER_QI_OUT_DIR"] = tmpdir
            result = self._run_boozer_qi(env)

        if result.returncode != 0:
            self.fail(f"boozerQI.py FD comparison failed with return code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")

        self.assertIn("Max FD relative error=", result.stdout)
        fd_line = next(line for line in result.stdout.splitlines() if line.startswith("Max FD relative error="))
        fd_error = float(fd_line.split("=", 1)[1])
        self.assertLess(fd_error, 1.0e-4)

    def test_boozer_qi_example_reduced_progress(self):
        env = os.environ.copy()
        env.update({
            "SIMSOPT_BOOZER_QI_WRITE_VTK": "0",
            "SIMSOPT_BOOZER_QI_SKIP_TAYLOR": "1",
            "SIMSOPT_BOOZER_QI_MAXITER": "3",
            "SIMSOPT_BOOZER_QI_MPOL": "3",
            "SIMSOPT_BOOZER_QI_NTOR": "3",
            "SIMSOPT_BOOZER_QI_SDIM": "6",
            "SIMSOPT_BOOZER_QI_NPHI": "21",
            "SIMSOPT_BOOZER_QI_NALPHA": "3",
            "SIMSOPT_BOOZER_QI_NBJ": "5",
            "SIMSOPT_BOOZER_QI_NPHI_OUT": "21",
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            env["SIMSOPT_BOOZER_QI_OUT_DIR"] = tmpdir
            history_path = os.path.join(tmpdir, "history.json")
            env["SIMSOPT_BOOZER_QI_HISTORY_PATH"] = history_path
            result = self._run_boozer_qi(env)
            if result.returncode != 0:
                self.fail(f"boozerQI.py reduced progress run failed with return code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
            with open(history_path, "r", encoding="utf-8") as stream:
                history = json.load(stream)

        self.assertLess(history["result"]["best_J"], history["result"]["initial_J"])

    def test_boozer_qi_example_optimizer_comparison(self):
        env = os.environ.copy()
        env.update({
            "SIMSOPT_BOOZER_QI_WRITE_VTK": "0",
            "SIMSOPT_BOOZER_QI_SKIP_TAYLOR": "1",
            "SIMSOPT_BOOZER_QI_COMPARE_OPTIMIZERS": "1",
            "SIMSOPT_BOOZER_QI_COMPARE_OPTIMIZERS_SUBSPACE": "4",
            "SIMSOPT_BOOZER_QI_COMPARE_OPTIMIZERS_MAXITER": "1",
            "SIMSOPT_BOOZER_QI_MAXITER": "0",
            "SIMSOPT_BOOZER_QI_MPOL": "3",
            "SIMSOPT_BOOZER_QI_NTOR": "3",
            "SIMSOPT_BOOZER_QI_SDIM": "6",
            "SIMSOPT_BOOZER_QI_NPHI": "21",
            "SIMSOPT_BOOZER_QI_NALPHA": "3",
            "SIMSOPT_BOOZER_QI_NBJ": "5",
            "SIMSOPT_BOOZER_QI_NPHI_OUT": "21",
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            env["SIMSOPT_BOOZER_QI_OUT_DIR"] = tmpdir
            history_path = os.path.join(tmpdir, "history.json")
            env["SIMSOPT_BOOZER_QI_HISTORY_PATH"] = history_path
            result = self._run_boozer_qi(env)
            if result.returncode != 0:
                self.fail(f"boozerQI.py optimizer comparison failed with return code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
            with open(history_path, "r", encoding="utf-8") as stream:
                history = json.load(stream)

        self.assertIn("Optimizer L-BFGS-B:", result.stdout)
        self.assertIn("optimizer_comparison", history)
        self.assertEqual(len(history["optimizer_comparison"]["results"]), 3)
        self.assertTrue(all("fun" in entry for entry in history["optimizer_comparison"]["results"]))

    def test_boozer_qi_example_exact_report(self):
        env = os.environ.copy()
        env.update({
            "SIMSOPT_BOOZER_QI_WRITE_VTK": "0",
            "SIMSOPT_BOOZER_QI_SKIP_TAYLOR": "1",
            "SIMSOPT_BOOZER_QI_EXACT_REPORT": "1",
            "SIMSOPT_BOOZER_QI_EXACT_REPORT_MAXITER": "20",
            "SIMSOPT_BOOZER_QI_MAXITER": "0",
            "SIMSOPT_BOOZER_QI_MPOL": "3",
            "SIMSOPT_BOOZER_QI_NTOR": "3",
            "SIMSOPT_BOOZER_QI_SDIM": "6",
            "SIMSOPT_BOOZER_QI_NPHI": "21",
            "SIMSOPT_BOOZER_QI_NALPHA": "3",
            "SIMSOPT_BOOZER_QI_NBJ": "5",
            "SIMSOPT_BOOZER_QI_NPHI_OUT": "21",
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            env["SIMSOPT_BOOZER_QI_OUT_DIR"] = tmpdir
            history_path = os.path.join(tmpdir, "history.json")
            env["SIMSOPT_BOOZER_QI_HISTORY_PATH"] = history_path
            result = self._run_boozer_qi(env)
            if result.returncode != 0:
                self.fail(f"boozerQI.py exact report failed with return code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
            with open(history_path, "r", encoding="utf-8") as stream:
                history = json.load(stream)

        self.assertIn("Exact final report", result.stdout)
        self.assertIn("exact_report", history)
        self.assertTrue(history["exact_report"]["success"])
        self.assertIn("total_J", history["exact_report"])

    def test_boozer_qi_example_exports_vmec_and_coils(self):
        env = os.environ.copy()
        env.update({
            "SIMSOPT_BOOZER_QI_WRITE_VTK": "0",
            "SIMSOPT_BOOZER_QI_SKIP_TAYLOR": "1",
            "SIMSOPT_BOOZER_QI_MAXITER": "0",
            "SIMSOPT_BOOZER_QI_MPOL": "3",
            "SIMSOPT_BOOZER_QI_NTOR": "3",
            "SIMSOPT_BOOZER_QI_SDIM": "6",
            "SIMSOPT_BOOZER_QI_NPHI": "21",
            "SIMSOPT_BOOZER_QI_NALPHA": "3",
            "SIMSOPT_BOOZER_QI_NBJ": "5",
            "SIMSOPT_BOOZER_QI_NPHI_OUT": "21",
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            env["SIMSOPT_BOOZER_QI_OUT_DIR"] = tmpdir
            history_path = os.path.join(tmpdir, "history.json")
            coils_path = os.path.join(tmpdir, "coils_export.json")
            restart_path = os.path.join(tmpdir, "surface_restart.json")
            vmec_input_path = os.path.join(tmpdir, "input.boozer_qi")
            vmec_export_diagnostics_path = os.path.join(tmpdir, "vmec_export_surface_diagnostics.json")
            vmec_export_cross_sections_plot = os.path.join(tmpdir, "vmec_export_cross_sections.png")
            vmec_export_surface_plot = os.path.join(tmpdir, "vmec_export_surface_3d.png")
            env["SIMSOPT_BOOZER_QI_HISTORY_PATH"] = history_path
            env["SIMSOPT_BOOZER_QI_EXPORT_COILS_JSON"] = coils_path
            env["SIMSOPT_BOOZER_QI_EXPORT_SURFACE_RESTART"] = restart_path
            env["SIMSOPT_BOOZER_QI_EXPORT_VMEC_INPUT"] = vmec_input_path
            result = self._run_boozer_qi(env)
            if result.returncode != 0:
                self.fail(f"boozerQI.py export run failed with return code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
            with open(history_path, "r", encoding="utf-8") as stream:
                history = json.load(stream)

            self.assertTrue(os.path.exists(coils_path))
            self.assertTrue(os.path.exists(restart_path))
            self.assertTrue(os.path.exists(vmec_input_path))
            self.assertTrue(os.path.exists(vmec_export_diagnostics_path))
            self.assertTrue(os.path.exists(vmec_export_cross_sections_plot))
            self.assertTrue(os.path.exists(vmec_export_surface_plot))
            self.assertIsInstance(load(coils_path), BiotSavart)
            boundary = SurfaceRZFourier.from_vmec_input(vmec_input_path)
            with open(vmec_export_diagnostics_path, "r", encoding="utf-8") as stream:
                vmec_export_diagnostics = json.load(stream)

        self.assertEqual(Path(history["artifacts"]["coils_json"]).resolve(), Path(coils_path).resolve())
        self.assertEqual(Path(history["artifacts"]["surface_restart"]).resolve(), Path(restart_path).resolve())
        self.assertEqual(Path(history["artifacts"]["vmec_input"]).resolve(), Path(vmec_input_path).resolve())
        self.assertEqual(Path(history["artifacts"]["vmec_export_diagnostics"]).resolve(), Path(vmec_export_diagnostics_path).resolve())
        self.assertEqual(Path(history["artifacts"]["vmec_export_cross_sections_plot"]).resolve(), Path(vmec_export_cross_sections_plot).resolve())
        self.assertEqual(Path(history["artifacts"]["vmec_export_surface_plot"]).resolve(), Path(vmec_export_surface_plot).resolve())
        self.assertEqual(boundary.nfp, 3)
        self.assertLess(vmec_export_diagnostics["selected"]["relative_l2"], 1.0e-2)
        self.assertGreaterEqual(len(vmec_export_diagnostics["candidates"]), 1)
        self.assertIn("Wrote optimized Biot-Savart JSON", result.stdout)
        self.assertIn("Wrote Boozer surface restart", result.stdout)
        self.assertIn("Wrote VMEC input", result.stdout)
        self.assertIn("Wrote VMEC export diagnostics", result.stdout)

    def test_boozer_qi_example_surface_restart_continuation(self):
        env = os.environ.copy()
        env.update({
            "SIMSOPT_BOOZER_QI_WRITE_VTK": "0",
            "SIMSOPT_BOOZER_QI_SKIP_TAYLOR": "1",
            "SIMSOPT_BOOZER_QI_MAXITER": "0",
            "SIMSOPT_BOOZER_QI_MPOL": "3",
            "SIMSOPT_BOOZER_QI_NTOR": "3",
            "SIMSOPT_BOOZER_QI_SDIM": "6",
            "SIMSOPT_BOOZER_QI_NPHI": "21",
            "SIMSOPT_BOOZER_QI_NALPHA": "3",
            "SIMSOPT_BOOZER_QI_NBJ": "5",
            "SIMSOPT_BOOZER_QI_NPHI_OUT": "21",
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            seed_dir = os.path.join(tmpdir, "seed")
            cont_dir = os.path.join(tmpdir, "continuation")
            os.makedirs(seed_dir, exist_ok=True)
            os.makedirs(cont_dir, exist_ok=True)
            coils_path = os.path.join(seed_dir, "coils.json")
            restart_path = os.path.join(seed_dir, "surface_restart.json")
            seed_history_path = os.path.join(seed_dir, "history.json")
            seed_env = env.copy()
            seed_env["SIMSOPT_BOOZER_QI_OUT_DIR"] = seed_dir
            seed_env["SIMSOPT_BOOZER_QI_EXPORT_COILS_JSON"] = coils_path
            seed_env["SIMSOPT_BOOZER_QI_EXPORT_SURFACE_RESTART"] = restart_path
            seed_env["SIMSOPT_BOOZER_QI_HISTORY_PATH"] = seed_history_path
            seed_result = self._run_boozer_qi(seed_env)
            if seed_result.returncode != 0:
                self.fail(f"boozerQI.py seed run failed with return code {seed_result.returncode}\nstdout:\n{seed_result.stdout}\nstderr:\n{seed_result.stderr}")

            cont_env = env.copy()
            cont_env.update({
                "SIMSOPT_BOOZER_QI_OUT_DIR": cont_dir,
                "SIMSOPT_BOOZER_QI_HISTORY_PATH": os.path.join(cont_dir, "history.json"),
                "SIMSOPT_BOOZER_QI_COILS_JSON": coils_path,
                "SIMSOPT_BOOZER_QI_SURFACE_RESTART": restart_path,
                "SIMSOPT_BOOZER_QI_MPOL": "4",
                "SIMSOPT_BOOZER_QI_NTOR": "4",
            })
            cont_result = self._run_boozer_qi(cont_env)
            if cont_result.returncode != 0:
                self.fail(f"boozerQI.py continuation run failed with return code {cont_result.returncode}\nstdout:\n{cont_result.stdout}\nstderr:\n{cont_result.stderr}")

            with open(cont_env["SIMSOPT_BOOZER_QI_HISTORY_PATH"], "r", encoding="utf-8") as stream:
                history = json.load(stream)

        self.assertTrue(np.isfinite(history["result"]["initial_J"]))
        self.assertIn("Using surface initializer from Boozer surface restart", cont_result.stdout)


class BoozerResidualTests(unittest.TestCase):
    def test_boozerresidual_derivative(self):
        """
        Taylor test for derivative of surface non QS ratio wrt coil parameters
        """
        for label in ["Volume"]:
            for optimize_G in [True, False]:
                for weight_inv_modB in [True, False]:
                    with self.subTest(label=label, optimize_G=optimize_G, weight_inv_modB=weight_inv_modB):
                        self.subtest_boozerresidual_derivative(label, optimize_G, weight_inv_modB)

    def subtest_boozerresidual_derivative(self, label, optimize_G, weight_inv_modB):
        bs, boozer_surface = get_boozer_surface(label=label, boozer_type='ls', optimize_G=optimize_G, weight_inv_modB=weight_inv_modB)
        coeffs = bs.x
        br = BoozerResidual(boozer_surface, bs)

        def f(dofs):
            bs.x = dofs
            return br.J()

        def df(dofs):
            bs.x = dofs
            return br.dJ()

        taylor_test1(f, df, coeffs,
                     epsilons=np.power(2., -np.asarray(range(13, 19))))


class LabelTests(unittest.TestCase):
    def test_label_surface_derivative1(self):
        for label in ["Volume", "ToroidalFlux", "Area", "AspectRatio"]:
            for stellsym in stellsym_list:
                for nphi, ntheta in [(13, 14), (None, None), (13, None), (None, 14)]:
                    with self.subTest(label=label, stellsym=stellsym, converge=stellsym):
                        # don't converge the BoozerSurface when stellsym=False because it takes a long time
                        # for a unit test
                        self.subtest_label_derivative1(label, stellsym=stellsym, converge=stellsym, nphi=nphi, ntheta=ntheta)

    def subtest_label_derivative1(self, label, stellsym, converge, nphi, ntheta):
        bs, boozer_surface = get_boozer_surface(label=label, nphi=nphi, ntheta=ntheta, converge=converge, stellsym=stellsym)
        surface = boozer_surface.surface
        label = boozer_surface.label
        coeffs = surface.x

        def f(dofs):
            surface.x = dofs
            return label.J()

        def df(dofs):
            surface.x = dofs
            return label.dJ(partials=True)(surface)

        taylor_test1(f, df, coeffs,
                     epsilons=np.power(2., -np.asarray(range(12, 18))))

    def test_label_surface_derivative2(self):
        for label in ["Volume", "ToroidalFlux", "Area", "AspectRatio"]:
            with self.subTest(label=label):
                self.subtest_label_derivative2(label)

    def subtest_label_derivative2(self, label):
        bs, boozer_surface = get_boozer_surface(label=label)
        surface = boozer_surface.surface
        label = boozer_surface.label
        coeffs = surface.x

        def f(dofs):
            surface.x = dofs
            return label.J()

        def df(dofs):
            surface.x = dofs
            return label.dJ_by_dsurfacecoefficients()

        def d2f(dofs):
            surface.x = dofs
            return label.d2J_by_dsurfacecoefficientsdsurfacecoefficients()

        taylor_test2(f, df, d2f, coeffs,
                     epsilons=np.power(2., -np.asarray(range(13, 19))))
