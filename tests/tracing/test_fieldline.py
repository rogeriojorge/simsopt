from simsopt.geo.magneticfieldclasses import ToroidalField
from simsopt.tracing.tracing import compute_fieldlines, particles_to_vtk
import unittest
import numpy as np


def validate_phi_hits(phi_hits, nphis):
    """
    Assert that we are hitting the phi planes in the correct order.
    For the toroidal field, we should always keep increasing in phi.
    """
    for i in range(len(phi_hits)-1):
        this_idx = int(phi_hits[i][1])
        next_idx = int(phi_hits[i+1][1])
        print(this_idx, next_idx)
        if not next_idx == (this_idx + 1) % nphis:
            return False
    return True


class FieldlineTesting(unittest.TestCase):

    def test_poincare(self):
        # Test a toroidal magnetic field with no rotational transform
        R0test = 1.3
        B0test = 0.8
        Bfield = ToroidalField(R0test, B0test)
        r0 = 1.1
        nlines = 10
        nphis = 10
        phis = np.linspace(0, 2*np.pi, nphis, endpoint=False)
        res_tys, res_phi_hits = compute_fieldlines(
            Bfield, r0, nlines, linestep=0.1, tmax=100, phis=phis, stopping_criteria=[])
        for i in range(nlines):
            assert np.allclose(res_tys[i][:, 3], 0.)
            assert np.allclose(np.linalg.norm(res_tys[i][:, 1:3], axis=1), r0+0.1*i)
            assert validate_phi_hits(res_phi_hits[i], nphis)
        particles_to_vtk(res_tys, '/tmp/fieldlines')