import unittest
from pathlib import Path
import numpy as np
from simsopt.geo.graph_surface import Surface, SurfaceRZFourier, SurfaceGarabedian

TEST_DIR = Path(__file__).parent / ".." / "test_files"  # ) #.resolve()


class SurfaceTests(unittest.TestCase):
    def test_init(self):
        """
        This test case checks the most common use cases.
        """
        # Try initializing a Surface without or with the optional
        # arguments:
        with self.assertRaises(TypeError):
            s = Surface()
        #self.assertEqual(s.nfp, 1)
        #self.assertTrue(s.stellsym)

        #s = Surface(nfp=3)
        #self.assertEqual(s.nfp, 3)
        #self.assertTrue(s.stellsym, True)

        #s = Surface(stellsym=False)
        #self.assertEqual(s.nfp, 1)
        #self.assertFalse(s.stellsym)

        # Now let's check that we can change nfp and stellsym.
        #s.nfp = 5
        #self.assertEqual(s.nfp, 5)
        #self.assertFalse(s.stellsym)

        #s.stellsym = True
        #self.assertEqual(s.nfp, 5)
        #self.assertTrue(s.stellsym)


class SurfaceRZFourierTests(unittest.TestCase):
    def test_init(self):
        s = SurfaceRZFourier(mpol=3, ntor=2)
        self.assertEqual(s.nfp, 1)
        self.assertTrue(s.stellsym)

        s = SurfaceRZFourier(nfp=3, mpol=3, ntor=2)
        self.assertEqual(s.nfp, 3)
        self.assertTrue(s.stellsym)

        s = SurfaceRZFourier(nfp=2, mpol=3, ntor=2)

        self.assertEqual(s.rc.shape, (4, 5))
        self.assertEqual(s.zs.shape, (4, 5))

        s = SurfaceRZFourier(nfp=10, mpol=1, ntor=3, stellsym=False)
        self.assertEqual(s.rc.shape, (2, 7))
        self.assertEqual(s.zs.shape, (2, 7))
        self.assertEqual(s.rs.shape, (2, 7))
        self.assertEqual(s.zc.shape, (2, 7))

    def test_area_volume(self):
        """
        Test the calculation of area and volume for an axisymmetric surface
        """
        s = SurfaceRZFourier()
        s.set_rc(0, 0, 1.3)
        s.set_rc(1, 0, 0.4)
        s.set_zs(1, 0, 0.2)

        true_area = 15.827322032265993
        true_volume = 2.0528777154265874
        self.assertAlmostEqual(s.area(), true_area, places=4)
        self.assertAlmostEqual(s.volume(), true_volume, places=3)

    def test_get_dofs(self):
        """
        Test that we can convert the degrees of freedom into a 1D vector
        """

        s = SurfaceRZFourier(nfp=2, mpol=3, ntor=2)
        self.assertEqual(s.rc.shape, (4, 5))
        self.assertEqual(s.zs.shape, (4, 5))

        s = SurfaceRZFourier(nfp=10, mpol=1, ntor=3, stellsym=False)
        self.assertEqual(s.rc.shape, (2, 7))
        self.assertEqual(s.zs.shape, (2, 7))
        self.assertEqual(s.rs.shape, (2, 7))
        self.assertEqual(s.zc.shape, (2, 7))

    def test_get_x(self):
        """
        Test that we can convert the degrees of freedom into a 1D vector
        """

        # First try an axisymmetric surface for simplicity:
        s = SurfaceRZFourier()
        s.set_rc(0, 0, 1.3)
        s.set_rc(1, 0, 0.4)
        s.set_zs(1, 0, 0.2)
        x = s.x
        self.assertEqual(x.shape, (3,))
        self.assertAlmostEqual(x[0], 1.3)
        self.assertAlmostEqual(x[1], 0.4)
        self.assertAlmostEqual(x[2], 0.2)

        # Now try a nonaxisymmetric shape:
        s = SurfaceRZFourier(mpol=3, ntor=1)
        s.rc = np.array([[100, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        s.zs = np.array([[101, 102, 13], [14, 15, 16], [17, 18, 19], [20, 21, 22]])
        x = s.x
        self.assertEqual(x.shape, (21,))
        for j in range(21):
            self.assertAlmostEqual(x[j], j + 2)

    def test_set_dofs(self):
        """
        Test that we can set the shape from a 1D vector
        """

        # First try an axisymmetric surface for simplicity:
        s = SurfaceRZFourier()
        s.x = [2.9, -1.1, 0.7]
        self.assertAlmostEqual(s.get_rc(0, 0), 2.9)
        self.assertAlmostEqual(s.get_rc(1, 0), -1.1)
        self.assertAlmostEqual(s.get_zs(1, 0), 0.7)

        # Now try a nonaxisymmetric shape:
        s = SurfaceRZFourier(mpol=3, ntor=1)
        s.x = np.array(list(range(21))) + 1
        self.assertAlmostEqual(s.get_rc(0, -1), 0)
        self.assertAlmostEqual(s.get_rc(0, 0), 1)
        self.assertAlmostEqual(s.get_rc(0, 1), 2)
        self.assertAlmostEqual(s.get_rc(1, -1), 3)
        self.assertAlmostEqual(s.get_rc(1, 0), 4)
        self.assertAlmostEqual(s.get_rc(1, 1), 5)
        self.assertAlmostEqual(s.get_rc(2, -1), 6)
        self.assertAlmostEqual(s.get_rc(2, 0), 7)
        self.assertAlmostEqual(s.get_rc(2, 1), 8)
        self.assertAlmostEqual(s.get_rc(3, -1), 9)
        self.assertAlmostEqual(s.get_rc(3, 0), 10)
        self.assertAlmostEqual(s.get_rc(3, 1), 11)

        self.assertAlmostEqual(s.get_zs(0, -1), 0)
        self.assertAlmostEqual(s.get_zs(0, 0), 0)
        self.assertAlmostEqual(s.get_zs(0, 1), 12)
        self.assertAlmostEqual(s.get_zs(1, -1), 13)
        self.assertAlmostEqual(s.get_zs(1, 0), 14)
        self.assertAlmostEqual(s.get_zs(1, 1), 15)
        self.assertAlmostEqual(s.get_zs(2, -1), 16)
        self.assertAlmostEqual(s.get_zs(2, 0), 17)
        self.assertAlmostEqual(s.get_zs(2, 1), 18)
        self.assertAlmostEqual(s.get_zs(3, -1), 19)
        self.assertAlmostEqual(s.get_zs(3, 0), 20)
        self.assertAlmostEqual(s.get_zs(3, 1), 21)

    def test_from_focus(self):
        """
        Try reading in a focus-format file.
        """
        #filename = os.path.join(TEST_DIR, 'tf_only_half_tesla.plasma')
        filename = TEST_DIR / 'tf_only_half_tesla.plasma'

        s = SurfaceRZFourier.from_focus(filename)

        self.assertEqual(s.nfp, 3)
        self.assertTrue(s.stellsym)
        self.assertEqual(s.rc.shape, (11, 13))
        self.assertEqual(s.zs.shape, (11, 13))
        self.assertAlmostEqual(s.rc[0, 6], 1.408922E+00)
        self.assertAlmostEqual(s.rc[0, 7], 2.794370E-02)
        self.assertAlmostEqual(s.zs[0, 7], -1.909220E-02)
        self.assertAlmostEqual(s.rc[10, 12], -6.047097E-05)
        self.assertAlmostEqual(s.zs[10, 12], 3.663233E-05)

        self.assertAlmostEqual(s.get_rc(0, 0), 1.408922E+00)
        self.assertAlmostEqual(s.get_rc(0, 1), 2.794370E-02)
        self.assertAlmostEqual(s.get_zs(0, 1), -1.909220E-02)
        self.assertAlmostEqual(s.get_rc(10, 6), -6.047097E-05)
        self.assertAlmostEqual(s.get_zs(10, 6), 3.663233E-05)

        true_area = 24.5871075268402
        true_volume = 2.96201898538042
        self.assertAlmostEqual(s.area(), true_area, places=4)
        self.assertAlmostEqual(s.volume(), true_volume, places=3)


class SurfaceGarabedianTests(unittest.TestCase):
    def test_init(self):
        """
        Check that the default surface is what we expect, and that the
        'names' array is correctly aligned.
        """
        #s = optimizable(SurfaceGarabedian(nmin=-1, nmax=2, mmin=-2, mmax=5))
        s = SurfaceGarabedian(nmin=-1, nmax=2, mmin=-2, mmax=5)
        self.assertAlmostEqual(s.Delta[2, 1], 0.1)
        self.assertAlmostEqual(s.Delta[3, 1], 1.0)
        self.assertAlmostEqual(s.get('Delta(0,0)'), 0.1)
        self.assertAlmostEqual(s.get('Delta(1,0)'), 1.0)
        # Verify all other elements are 0:
        d = np.copy(s.Delta)
        d[2, 1] = 0
        d[3, 1] = 0
        np.testing.assert_allclose(d, np.zeros((8, 4)))

        s.set('Delta(-2,-1)', 42)
        self.assertAlmostEqual(s.Delta[0, 0], 42)
        self.assertAlmostEqual(s.get_Delta(-2, -1), 42)

        s.set('Delta(5,-1)', -7)
        self.assertAlmostEqual(s.Delta[7, 0], -7)
        self.assertAlmostEqual(s.get_Delta(5, -1), -7)

        s.set('Delta(-2,2)', 13)
        self.assertAlmostEqual(s.Delta[0, 3], 13)
        self.assertAlmostEqual(s.get_Delta(-2, 2), 13)

        s.set('Delta(5,2)', -5)
        self.assertAlmostEqual(s.Delta[7, 3], -5)
        self.assertAlmostEqual(s.get_Delta(5, 2), -5)

        s.set_Delta(-2, -1, 421)
        self.assertAlmostEqual(s.Delta[0, 0], 421)

        s.set_Delta(5, -1, -71)
        self.assertAlmostEqual(s.Delta[7, 0], -71)

        s.set_Delta(-2, 2, 133)
        self.assertAlmostEqual(s.Delta[0, 3], 133)

        s.set_Delta(5, 2, -50)
        self.assertAlmostEqual(s.Delta[7, 3], -50)

    def test_convert_back(self):
        """
        If we start with a SurfaceRZFourier, convert to Garabedian, and
        convert back to SurfaceFourier, we should get back what we
        started with.
        """
        for mpol in range(1, 4):
            for ntor in range(5):
                for nfp in range(1, 4):
                    sf1 = SurfaceRZFourier(nfp=nfp, mpol=mpol, ntor=ntor)
                    # Set all dofs to random numbers in [-2, 2]:
                    sf1.dofs = ((np.random.rand(sf1.dof_size) - 0.5) * 4)
                    sg = sf1.to_Garabedian()
                    sf2 = sg.to_RZFourier()
                    np.testing.assert_allclose(sf1.rc, sf2.rc)
                    np.testing.assert_allclose(sf1.zs, sf2.zs)


if __name__ == "__main__":
    unittest.main()
