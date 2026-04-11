import numpy as np
import jax.numpy as jnp
from jax import value_and_grad
from scipy.interpolate import UnivariateSpline

import simsoptpp as sopp
from .._core.optimizable import Optimizable
from .._core.derivative import Derivative, derivative_dec
from .._core.types import RealArray
from .surface import Surface
from .surfacexyztensorfourier import SurfaceXYZTensorFourier
from ..objectives.utilities import forward_backward

__all__ = ['Area', 'Volume', 'ToroidalFlux', 'PrincipalCurvature',
           'QfmResidual', 'boozer_surface_residual', 'Iotas',
           'MajorRadius', 'NonQuasiSymmetricRatio', 'NonQuasiIsodynamicRatio', 'BoozerResidual',
           'AspectRatio']


class AspectRatio(Optimizable):
    """
    Wrapper class for surface aspect ratio.
    """

    def __init__(self, surface, range=None, nphi=None, ntheta=None):

        if range is not None or nphi is not None or ntheta is not None:
            if range is None:
                if surface.stellsym:
                    range = Surface.RANGE_HALF_PERIOD
                else:
                    range = Surface.RANGE_FIELD_PERIOD
            if nphi is None:
                nphi = len(surface.quadpoints_phi)
            if ntheta is None:
                ntheta = len(surface.quadpoints_theta)
            self.surface = surface.__class__.from_nphi_ntheta(nphi=nphi, ntheta=ntheta, range=range, nfp=surface.nfp, stellsym=surface.stellsym,
                                                              mpol=surface.mpol, ntor=surface.ntor, dofs=surface.dofs)
        else:
            self.surface = surface

        self.range = range
        self.nphi = nphi
        self.ntheta = ntheta

        super().__init__(depends_on=[self.surface])

    def J(self):
        """
        Compute the aspect ratio of a surface.
        """
        return self.surface.aspect_ratio()

    @derivative_dec
    def dJ(self):
        return Derivative({self.surface: self.dJ_by_dsurfacecoefficients()})

    def dJ_by_dsurfacecoefficients(self):
        """
        Calculate the partial derivatives with respect to the surface coefficients.
        """
        return self.surface.daspect_ratio_by_dcoeff()

    def d2J_by_dsurfacecoefficientsdsurfacecoefficients(self):
        """
        Calculate the second partial derivatives with respect to the surface coefficients.
        """
        return self.surface.d2aspect_ratio_by_dcoeff_dcoeff()


class Area(Optimizable):
    """
    Wrapper class for surface area label.
    """

    def __init__(self, surface, range=None, nphi=None, ntheta=None):

        if range is not None or nphi is not None or ntheta is not None:
            if range is None:
                if surface.stellsym:
                    range = Surface.RANGE_HALF_PERIOD
                else:
                    range = Surface.RANGE_FIELD_PERIOD
            if nphi is None:
                nphi = len(surface.quadpoints_phi)
            if ntheta is None:
                ntheta = len(surface.quadpoints_theta)
            self.surface = surface.__class__.from_nphi_ntheta(nphi=nphi, ntheta=ntheta, range=range, nfp=surface.nfp, stellsym=surface.stellsym,
                                                              mpol=surface.mpol, ntor=surface.ntor, dofs=surface.dofs)
        else:
            self.surface = surface

        self.range = range
        self.nphi = nphi
        self.ntheta = ntheta

        super().__init__(depends_on=[self.surface])

    def J(self):
        """
        Compute the area of a surface.
        """
        return self.surface.area()

    @derivative_dec
    def dJ(self):
        return Derivative({self.surface: self.dJ_by_dsurfacecoefficients()})

    def dJ_by_dsurfacecoefficients(self):
        """
        Calculate the partial derivatives with respect to the surface coefficients.
        """
        return self.surface.darea_by_dcoeff()

    def d2J_by_dsurfacecoefficientsdsurfacecoefficients(self):
        """
        Calculate the second partial derivatives with respect to the surface coefficients.
        """
        return self.surface.d2area_by_dcoeffdcoeff()


class Volume(Optimizable):
    """
    Wrapper class for volume label.
    """

    def __init__(self, surface, range=None, nphi=None, ntheta=None):

        if range is not None or nphi is not None or ntheta is not None:
            if range is None:
                if surface.stellsym:
                    range = Surface.RANGE_HALF_PERIOD
                else:
                    range = Surface.RANGE_FIELD_PERIOD
            if nphi is None:
                nphi = len(surface.quadpoints_phi)
            if ntheta is None:
                ntheta = len(surface.quadpoints_theta)
            self.surface = surface.__class__.from_nphi_ntheta(nphi=nphi, ntheta=ntheta, range=range, nfp=surface.nfp, stellsym=surface.stellsym,
                                                              mpol=surface.mpol, ntor=surface.ntor, dofs=surface.dofs)
        else:
            self.surface = surface

        self.range = range
        self.nphi = nphi
        self.ntheta = ntheta

        super().__init__(depends_on=[self.surface])

    def J(self):
        """
        Compute the volume enclosed by the surface.
        """
        return self.surface.volume()

    @derivative_dec
    def dJ(self):
        return Derivative({self.surface: self.dJ_by_dsurfacecoefficients()})

    def dJ_by_dsurfacecoefficients(self):
        """
        Calculate the derivatives with respect to the surface coefficients.
        """
        return self.surface.dvolume_by_dcoeff()

    def d2J_by_dsurfacecoefficientsdsurfacecoefficients(self):
        """
        Calculate the second derivatives with respect to the surface coefficients.
        """
        return self.surface.d2volume_by_dcoeffdcoeff()


class ToroidalFlux(Optimizable):
    r"""
    Given a surface and Biot Savart kernel, this objective calculates

    .. math::
       J &= \int_{S_{\varphi}} \mathbf{B} \cdot \mathbf{n} ~ds, \\
       &= \int_{S_{\varphi}} \text{curl} \mathbf{A} \cdot \mathbf{n} ~ds, \\
       &= \int_{\partial S_{\varphi}} \mathbf{A} \cdot \mathbf{t}~dl,

    where :math:`S_{\varphi}` is a surface of constant :math:`\varphi`, and :math:`\mathbf A`
    is the magnetic vector potential.
    """

    def __init__(self, surface, biotsavart, idx=0, range=None, nphi=None, ntheta=None):

        if range is not None or nphi is not None or ntheta is not None:
            if range is None:
                if surface.stellsym:
                    range = Surface.RANGE_HALF_PERIOD
                else:
                    range = Surface.RANGE_FIELD_PERIOD
            if nphi is None:
                nphi = len(surface.quadpoints_phi)
            if ntheta is None:
                ntheta = len(surface.quadpoints_theta)
            self.surface = surface.__class__.from_nphi_ntheta(nphi=nphi, ntheta=ntheta, range=range, nfp=surface.nfp, stellsym=surface.stellsym,
                                                              mpol=surface.mpol, ntor=surface.ntor, dofs=surface.dofs)
        else:
            self.surface = surface

        self.biotsavart = biotsavart
        self.idx = idx

        self.range = range
        self.nphi = nphi
        self.ntheta = ntheta

        super().__init__(depends_on=[self.surface, biotsavart])

    def recompute_bell(self, parent=None):
        self.invalidate_cache()

    def invalidate_cache(self):
        x = self.surface.gamma()[self.idx]
        self.biotsavart.set_points(x)

    def J(self):
        r"""
        Compute the toroidal flux on the surface where
        :math:`\varphi = \texttt{quadpoints_varphi}[\texttt{idx}]`.
        """
        xtheta = self.surface.gammadash2()[self.idx]
        ntheta = self.surface.gamma().shape[1]
        A = self.biotsavart.A()
        tf = np.sum(A * xtheta)/ntheta
        return tf

    @derivative_dec
    def dJ(self):
        d_s = Derivative({self.surface: self.dJ_by_dsurfacecoefficients()})
        d_c = self.dJ_by_dcoils()
        return d_s + d_c

    def dJ_by_dsurfacecoefficients(self):
        """
        Calculate the partial derivatives with respect to the surface coefficients.
        """
        ntheta = self.surface.gamma().shape[1]
        dA_by_dX = self.biotsavart.dA_by_dX()
        A = self.biotsavart.A()
        dgammadash2 = self.surface.gammadash2()[self.idx, :]
        dgammadash2_by_dc = self.surface.dgammadash2_by_dcoeff()[self.idx, :]

        dx_dc = self.surface.dgamma_by_dcoeff()[self.idx]
        dA_dc = np.sum(dA_by_dX[..., :, None] * dx_dc[..., None, :], axis=1)
        term1 = np.sum(dA_dc * dgammadash2[..., None], axis=(0, 1))
        term2 = np.sum(A[..., None] * dgammadash2_by_dc, axis=(0, 1))

        out = (term1+term2)/ntheta
        return out

    def d2J_by_dsurfacecoefficientsdsurfacecoefficients(self):
        """
        Calculate the second partial derivatives with respect to the surface coefficients.
        """
        ntheta = self.surface.gamma().shape[1]
        dx_dc = self.surface.dgamma_by_dcoeff()[self.idx]
        d2A_by_dXdX = self.biotsavart.d2A_by_dXdX().reshape((ntheta, 3, 3, 3))
        dA_by_dX = self.biotsavart.dA_by_dX()
        dA_dc = np.sum(dA_by_dX[..., :, None] * dx_dc[..., None, :], axis=1)
        d2A_dcdc = np.einsum('jkpl,jpn,jkm->jlmn', d2A_by_dXdX, dx_dc, dx_dc)

        dgammadash2 = self.surface.gammadash2()[self.idx]
        dgammadash2_by_dc = self.surface.dgammadash2_by_dcoeff()[self.idx]

        term1 = np.sum(d2A_dcdc * dgammadash2[..., None, None], axis=-3)
        term2 = np.sum(dA_dc[..., :, None] * dgammadash2_by_dc[..., None, :], axis=-3)
        term3 = np.sum(dA_dc[..., None, :] * dgammadash2_by_dc[..., :, None], axis=-3)

        out = (1/ntheta) * np.sum(term1+term2+term3, axis=0)
        return out

    def dJ_by_dcoils(self):
        """
        Calculate the partial derivatives with respect to the coil coefficients.
        """
        xtheta = self.surface.gammadash2()[self.idx]
        ntheta = self.surface.gamma().shape[1]
        dJ_by_dA = xtheta/ntheta
        dJ_by_dcoils = self.biotsavart.A_vjp(dJ_by_dA)
        return dJ_by_dcoils


class PrincipalCurvature(Optimizable):
    r"""

    Given a Surface, evaluates a metric based on the principal curvatures,
    :math:`\kappa_1` and :math:`\kappa_2`, where :math:`\kappa_1>\kappa_2`.
    This metric is designed to penalize :math:`\kappa_1 > \kappa_{\max,1}` and
    :math:`-\kappa_2 > \kappa_{\max,2}`.

    .. math::
       J &= \int d^2 x \exp \left(- ( \kappa_1 - \kappa_{\max,1})/w_1) \right) \\
         &+ \int d^2 x \exp \left(- (-\kappa_2 - \kappa_{\max,2})/w_2) \right).

    This metric can be used as a regularization within fixed-boundary optimization
    to prevent, for example, surfaces with concave regions
    (large values of :math:`|\kappa_2|`) or surfaces with large elongation
    (large values of :math:`\kappa_1`).

    """

    def __init__(self, surface, kappamax1=1, kappamax2=1, weight1=0.05, weight2=0.05):
        super().__init__(depends_on=[surface])
        self.surface = surface
        self.kappamax1 = kappamax1
        self.kappamax2 = kappamax2
        self.weight1 = weight1
        self.weight2 = weight2

    def J(self):
        curvature = self.surface.surface_curvatures()
        k1 = curvature[:, :, 2]  # larger
        k2 = curvature[:, :, 3]  # smaller
        normal = self.surface.normal()
        norm_normal = np.sqrt(normal[:, :, 0]**2 + normal[:, :, 1]**2 + normal[:, :, 2]**2)
        return np.sum(norm_normal * np.exp(-(k1 - self.kappamax1)/self.weight1)) + \
            np.sum(norm_normal * np.exp(-(-k2 - self.kappamax2)/self.weight2))

    @derivative_dec
    def dJ(self):
        curvature = self.surface.surface_curvatures()
        k1 = curvature[:, :, 2]  # larger
        k2 = curvature[:, :, 3]  # smaller
        normal = self.surface.normal()
        norm_normal = np.sqrt(normal[:, :, 0]**2 + normal[:, :, 1]**2 + normal[:, :, 2]**2)
        dcurvature_dc = self.surface.dsurface_curvatures_by_dcoeff()
        dk1_dc = dcurvature_dc[:, :, 2, :]
        dk2_dc = dcurvature_dc[:, :, 3, :]
        dnormal_dc = self.surface.dnormal_by_dcoeff()
        dnorm_normal_dc = normal[:, :, 0, None]*dnormal_dc[:, :, 0, :]/norm_normal[:, :, None] + \
            normal[:, :, 1, None]*dnormal_dc[:, :, 1, :]/norm_normal[:, :, None] + \
            normal[:, :, 2, None]*dnormal_dc[:, :, 2, :]/norm_normal[:, :, None]
        deriv = np.sum(dnorm_normal_dc * np.exp(-(k1[:, :, None] - self.kappamax1)/self.weight1), axis=(0, 1)) + \
            np.sum(norm_normal[:, :, None] * np.exp(-(k1[:, :, None] - self.kappamax1)/self.weight1) * (- dk1_dc/self.weight1), axis=(0, 1)) + \
            np.sum(dnorm_normal_dc * np.exp(-(-k2[:, :, None] - self.kappamax2)/self.weight2), axis=(0, 1)) + \
            np.sum(norm_normal[:, :, None] * np.exp(-(-k2[:, :, None] - self.kappamax2)/self.weight2) * (dk2_dc/self.weight2), axis=(0, 1))
        return Derivative({self.surface: deriv})


def boozer_surface_residual(surface, iota, G, biotsavart, derivatives=0, weight_inv_modB=False):
    r"""
    For a given surface, this function computes the
    residual

    .. math::
        G\mathbf B_\text{BS}(\mathbf x) - \|\mathbf B_\text{BS}(\mathbf x)\|^2  (\mathbf x_\varphi + \iota  \mathbf x_\theta)

    as well as the derivatives of this residual with respect to surface dofs,
    iota, and G.  In the above, :math:`\mathbf x` are points on the surface, :math:`\iota` is the
    rotational transform on that surface, and :math:`\mathbf B_{\text{BS}}` is the magnetic field
    computed using the Biot-Savart law.

    :math:`G` is known for exact boozer surfaces, so if ``G=None`` is passed, then that
    value is used instead.

    Args:
        surface: The surface to use for the computation
        iota: the surface rotational transform
        G: a constant that is a function of the coil currents in vacuum field
        biotsavart: the Biot-Savart magnetic field
        derivatives: how many spatial derivatives of the residual to compute
        weight_inv_modB: whether or not to weight the residual by :math:`1/\|\mathbf B\|`.  This 
                         is useful to activate so that the residual does not scale with the 
                         coil currents.

    Returns:
        the residual at the surface quadrature points and optionally the spatial derivatives
        of the residual.


    """

    assert derivatives in [0, 1, 2]

    user_provided_G = G is not None
    if not user_provided_G:
        G = 2. * np.pi * np.sum([np.abs(c.current.get_value()) for c in biotsavart.coils]) * (4 * np.pi * 10**(-7) / (2 * np.pi))

    x = surface.gamma()
    xphi = surface.gammadash1()
    xtheta = surface.gammadash2()
    nphi = x.shape[0]
    ntheta = x.shape[1]

    xsemiflat = x.reshape((x.size//3, 3)).copy()

    biotsavart.set_points(xsemiflat)

    biotsavart.compute(derivatives)
    B = biotsavart.B().reshape((nphi, ntheta, 3))

    tang = xphi + iota * xtheta
    B2 = np.sum(B**2, axis=2)
    residual = G*B - B2[..., None] * tang

    if weight_inv_modB:
        modB = np.sqrt(B2)
        w = 1./modB
        rtil = w[:, :, None] * residual
    else:
        rtil = residual.copy()

    rtil_flattened = rtil.reshape((nphi*ntheta*3, ))
    r = rtil_flattened
    if derivatives == 0:
        return r,

    dx_dc = surface.dgamma_by_dcoeff()
    dxphi_dc = surface.dgammadash1_by_dcoeff()
    dxtheta_dc = surface.dgammadash2_by_dcoeff()
    nsurfdofs = dx_dc.shape[-1]

    dB_by_dX = biotsavart.dB_by_dX().reshape((nphi, ntheta, 3, 3))
    dB_dc = np.einsum('ijkl,ijkm->ijlm', dB_by_dX, dx_dc)

    # dresidual_dc = G*dB_dc - 2*np.sum(B[..., None]*dB_dc, axis=2)[:, :, None, :] * tang[..., None] - B2[..., None, None] * (dxphi_dc + iota * dxtheta_dc)
    dresidual_dc = sopp.boozer_dresidual_dc(G, dB_dc, B, tang, B2, dxphi_dc, iota, dxtheta_dc)
    dresidual_diota = -B2[..., None] * xtheta

    if weight_inv_modB:
        dB2_dc = 2*np.einsum('ijk,ijkl->ijl', B, dB_dc, optimize=True)
        dmodB_dc = 0.5*dB2_dc/np.sqrt(B2[:, :, None])
        dw_dc = -dmodB_dc/modB[:, :, None]**2
        drtil_dc = residual[..., None] * dw_dc[:, :, None, :] + w[:, :, None, None] * dresidual_dc
        drtil_diota = w[:, :, None] * dresidual_diota
    else:
        drtil_dc = dresidual_dc.copy()
        drtil_diota = dresidual_diota.copy()

    drtil_dc_flattened = drtil_dc.reshape((nphi*ntheta*3, nsurfdofs))
    drtil_diota_flattened = drtil_diota.reshape((nphi*ntheta*3, 1))

    if user_provided_G:
        dresidual_dG = B

        if weight_inv_modB:
            drtil_dG = w[:, :, None] * dresidual_dG
        else:
            drtil_dG = dresidual_dG.copy()

        drtil_dG_flattened = drtil_dG.reshape((nphi*ntheta*3, 1))
        J = np.concatenate((drtil_dc_flattened, drtil_diota_flattened, drtil_dG_flattened), axis=1)
    else:
        J = np.concatenate((drtil_dc_flattened, drtil_diota_flattened), axis=1)

    if derivatives == 1:
        return r, J

    d2B_by_dXdX = biotsavart.d2B_by_dXdX().reshape((nphi, ntheta, 3, 3, 3))
    d2B_dcdc = np.einsum('ijkpl,ijpn,ijkm->ijlmn', d2B_by_dXdX, dx_dc, dx_dc, optimize=True)
    dB2_dc = 2. * np.einsum('ijl,ijlm->ijm', B, dB_dc, optimize=True)

    term1 = np.einsum('ijlm,ijln->ijmn', dB_dc, dB_dc, optimize=True)
    term2 = np.einsum('ijlmn,ijl->ijmn', d2B_dcdc, B, optimize=True)
    d2B2_dcdc = 2*(term1 + term2)

    term1 = -(dxphi_dc[..., None, :] + iota * dxtheta_dc[..., None, :]) * dB2_dc[..., None, :, None]
    term2 = -(dxphi_dc[..., :, None] + iota * dxtheta_dc[..., :, None]) * dB2_dc[..., None, None, :]
    term3 = -(xphi[..., None, None] + iota * xtheta[..., None, None]) * d2B2_dcdc[..., None, :, :]
    d2residual_by_dcdc = G * d2B_dcdc + term1 + term2 + term3
    d2residual_by_dcdiota = -(dB2_dc[..., None, :] * xtheta[..., :, None] + B2[..., None, None] * dxtheta_dc)
    d2residual_by_diotadiota = np.zeros(dresidual_diota.shape)

    if weight_inv_modB:
        d2B2_dcdc = 2*(np.einsum('ijlm,ijln->ijmn', dB_dc, dB_dc, optimize=True)+np.einsum('ijkpl,ijpn,ijkm,ijl->ijmn', d2B_by_dXdX, dx_dc, dx_dc, B, optimize=True))
        d2modB_dc2 = (2*B2[:, :, None, None] * d2B2_dcdc - dB2_dc[:, :, :, None]*dB2_dc[:, :, None, :])*(1/(4*B2[:, :, None, None]**1.5))
        d2w_dc2 = (2*dmodB_dc[:, :, :, None] * dmodB_dc[:, :, None, :] - modB[:, :, None, None] * d2modB_dc2)/modB[:, :, None, None]**3.

        d2rtil_dcdc = residual[..., None, None] * d2w_dc2[:, :, None, ...] \
            + dw_dc[:, :, None, :, None] * dresidual_dc[:, :, :, None, :] \
            + dw_dc[:, :, None, None, :] * dresidual_dc[:, :, :, :, None] \
            + w[:, :, None, None, None] * d2residual_by_dcdc
        d2rtil_dcdiota = w[:, :, None, None] * d2residual_by_dcdiota + dw_dc[:, :, None, :] * dresidual_diota[..., None]
        d2rtil_diotadiota = np.zeros(dresidual_diota.shape)
    else:
        d2rtil_dcdc = d2residual_by_dcdc.copy()
        d2rtil_dcdiota = d2residual_by_dcdiota.copy()
        d2rtil_diotadiota = d2residual_by_diotadiota.copy()

    d2rtil_dcdc_flattened = d2rtil_dcdc.reshape((nphi*ntheta*3, nsurfdofs, nsurfdofs))
    d2rtil_dcdiota_flattened = d2rtil_dcdiota.reshape((nphi*ntheta*3, nsurfdofs))
    d2rtil_diotadiota_flattened = d2rtil_diotadiota.reshape((nphi*ntheta*3,))

    if user_provided_G:
        d2residual_by_dcdG = dB_dc
        if weight_inv_modB:
            d2rtil_dcdG = dw_dc[:, :, None, :] * dresidual_dG[..., None] + w[:, :, None, None] * d2residual_by_dcdG
        else:
            d2rtil_dcdG = d2residual_by_dcdG.copy()

        d2rtil_dGdG = np.zeros(dresidual_dG.shape)
        d2rtil_dcdG_flattened = d2rtil_dcdG.reshape((nphi*ntheta*3, nsurfdofs))
        d2rtil_diotadG_flattened = np.zeros((nphi*ntheta*3,))
        d2rtil_dGdG_flattened = d2rtil_dGdG.reshape((nphi*ntheta*3,))

        H = np.zeros((nphi*ntheta*3, nsurfdofs + 2, nsurfdofs + 2))
        # noqa turns out linting so that we can align everything neatly
        H[:, :nsurfdofs, :nsurfdofs] = d2rtil_dcdc_flattened        # noqa (0, 0) dcdc
        H[:, :nsurfdofs, nsurfdofs] = d2rtil_dcdiota_flattened     # noqa (0, 1) dcdiota
        H[:, :nsurfdofs, nsurfdofs+1] = d2rtil_dcdG_flattened        # noqa (0, 2) dcdG
        H[:, nsurfdofs, :nsurfdofs] = d2rtil_dcdiota_flattened     # noqa (1, 0) diotadc
        H[:, nsurfdofs, nsurfdofs] = d2rtil_diotadiota_flattened  # noqa (1, 1) diotadiota
        H[:, nsurfdofs, nsurfdofs+1] = d2rtil_diotadiota_flattened  # noqa (1, 2) diotadG
        H[:, nsurfdofs+1, :nsurfdofs] = d2rtil_dcdG_flattened        # noqa (2, 0) dGdc
        H[:, nsurfdofs+1, nsurfdofs] = d2rtil_diotadG_flattened     # noqa (2, 1) dGdiota
        H[:, nsurfdofs+1, nsurfdofs+1] = d2rtil_dGdG_flattened        # noqa (2, 2) dGdG
    else:
        H = np.zeros((nphi*ntheta*3, nsurfdofs + 1, nsurfdofs + 1))
        H[:, :nsurfdofs, :nsurfdofs] = d2rtil_dcdc_flattened        # noqa (0, 0) dcdc
        H[:, :nsurfdofs, nsurfdofs] = d2rtil_dcdiota_flattened     # noqa (0, 1) dcdiota
        H[:, nsurfdofs, :nsurfdofs] = d2rtil_dcdiota_flattened     # noqa (1, 0) diotadc
        H[:, nsurfdofs, nsurfdofs] = d2rtil_diotadiota_flattened  # noqa (1, 1) diotadiota

    return r, J, H


def parameter_derivatives(surface: Surface,
                          shape_gradient: RealArray
                          ) -> RealArray:
    r"""
    Converts the shape gradient of a given figure of merit, :math:`f`,
    to derivatives with respect to parameters defining a surface.  For
    a perturbation to the surface :math:`\delta \vec{x}`, the
    resulting perturbation to the objective function is

    .. math::
      \delta f(\delta \vec{x}) = \int d^2 x \, G \delta \vec{x} \cdot \vec{n}

    where :math:`G` is the shape gradient and :math:`\vec{n}` is the
    unit normal. Given :math:`G`, the parameter derivatives are then
    computed as

    .. math::
      \frac{\partial f}{\partial \Omega} = \int d^2 x \, G \frac{\partial\vec{x}}{\partial \Omega} \cdot \vec{n},

    where :math:`\Omega` is any parameter of the surface.

    Args:
        surface: The surface to use for the computation
        shape_gradient: 2d array of size (numquadpoints_phi,numquadpoints_theta)

    Returns:
        1d array of size (ndofs)
    """
    N = surface.normal()
    dx_by_dc = surface.dgamma_by_dcoeff()
    N_dot_dx_by_dc = np.einsum('ijk,ijkl->ijl', N, dx_by_dc)
    nphi = surface.gamma().shape[0]
    ntheta = surface.gamma().shape[1]
    return np.einsum('ijk,ij->k', N_dot_dx_by_dc, shape_gradient) / (ntheta * nphi)


class QfmResidual(Optimizable):
    r"""
    For a given surface :math:`S`, this class computes the residual

    .. math::
        f(S) = \frac{\int_{S} d^2 x \, (\textbf{B} \cdot \hat{\textbf{n}})^2}{\int_{S} d^2 x \, B^2}

    where :math:`\textbf{B}` is the magnetic field from :mod:`biotsavart`,
    :math:`\hat{\textbf{n}}` is the unit normal on a given surface, and the
    integration is performed over the surface. Derivatives are computed wrt the
    surface dofs.
    """

    def __init__(self, surface, biotsavart):
        self.surface = surface
        self.biotsavart = biotsavart
        self.biotsavart.append_parent(self.surface)
        super().__init__(depends_on=[surface, biotsavart])

    def recompute_bell(self, parent=None):
        self.invalidate_cache()

    def invalidate_cache(self):
        x = self.surface.gamma()
        xsemiflat = x.reshape((-1, 3))
        self.biotsavart.set_points(xsemiflat)

    def J(self):
        N = self.surface.normal()
        norm_N = np.linalg.norm(N, axis=2)
        n = N/norm_N[:, :, None]
        x = self.surface.gamma()
        nphi = x.shape[0]
        ntheta = x.shape[1]
        B = self.biotsavart.B().reshape((nphi, ntheta, 3))
        B_n = np.sum(B * n, axis=2)
        norm_B = np.linalg.norm(B, axis=2)
        return np.sum(B_n**2 * norm_N)/np.sum(norm_B**2 * norm_N)

    def dJ_by_dsurfacecoefficients(self):
        """
        Calculate the derivatives with respect to the surface coefficients
        """

        # we write the objective as J = J1/J2, then we compute the partial derivatives
        # dJ1_by_dgamma, dJ1_by_dN, dJ2_by_dgamma, dJ2_by_dN and then use the vjp functions
        # to get the derivatives wrt to the surface dofs
        x = self.surface.gamma()
        nphi = x.shape[0]
        ntheta = x.shape[1]
        dB_by_dX = self.biotsavart.dB_by_dX().reshape((nphi, ntheta, 3, 3))
        B = self.biotsavart.B().reshape((nphi, ntheta, 3))
        N = self.surface.normal()
        norm_N = np.linalg.norm(N, axis=2)

        B_N = np.sum(B * N, axis=2)
        dJ1dx = (2*B_N/norm_N)[:, :, None] * (np.sum(dB_by_dX*N[:, :, None, :], axis=3))
        dJ1dN = (2*B_N/norm_N)[:, :, None] * B - (B_N**2/norm_N**3)[:, :, None] * N

        dJ2dx = 2 * np.sum(dB_by_dX*B[:, :, None, :], axis=3) * norm_N[:, :, None]
        dJ2dN = (np.sum(B*B, axis=2)/norm_N)[:, :, None] * N

        J1 = np.sum(B_N**2 / norm_N)  # same as np.sum(B_n**2 * norm_N)
        J2 = np.sum(B**2 * norm_N[:, :, None])

        # d_J1 = self.surface.dnormal_by_dcoeff_vjp(dJ1dN) + self.surface.dgamma_by_dcoeff_vjp(dJ1dx)
        # d_J2 = self.surface.dnormal_by_dcoeff_vjp(dJ2dN) + self.surface.dgamma_by_dcoeff_vjp(dJ2dx)
        # deriv = d_J1/J2 - d_J2*J1/(J2*J2)

        deriv = self.surface.dnormal_by_dcoeff_vjp(dJ1dN/J2 - dJ2dN*J1/(J2*J2)) \
            + self.surface.dgamma_by_dcoeff_vjp(dJ1dx/J2 - dJ2dx*J1/(J2*J2))
        return deriv


class MajorRadius(Optimizable):
    r"""
    This wrapper objective computes the major radius of a toroidal Boozer surface and supplies
    its derivative with respect to coils

    Args:
        boozer_surface: The surface to use for the computation
    """

    def __init__(self, boozer_surface):
        super().__init__(depends_on=[boozer_surface])
        self.boozer_surface = boozer_surface
        self.surface = boozer_surface.surface
        self.recompute_bell()

    def J(self):
        if self._J is None:
            self.compute()
        return self._J

    @derivative_dec
    def dJ(self):
        if self._dJ is None:
            self.compute()
        return self._dJ

    def recompute_bell(self, parent=None):
        self._J = None
        self._dJ = None

    def compute(self):
        if self.boozer_surface.need_to_run_code:
            res = self.boozer_surface.res
            res = self.boozer_surface.run_code(res['iota'], G=res['G'])

        surface = self.surface
        self._J = surface.major_radius()

        booz_surf = self.boozer_surface
        iota, G, P, L, U, dconstraint_dcoils_vjp = _get_boozer_linearization(booz_surf)

        # tack on dJ_diota = dJ_dG = 0 to the end of dJ_ds
        dJ_ds = np.zeros(L.shape[0])
        dj_ds = surface.dmajor_radius_by_dcoeff()
        dJ_ds[:dj_ds.size] = dj_ds
        adj = forward_backward(P, L, U, dJ_ds)

        adj_times_dg_dcoil = dconstraint_dcoils_vjp(adj, booz_surf, iota, G)
        self._dJ = -1 * adj_times_dg_dcoil


class NonQuasiSymmetricRatio(Optimizable):
    r"""
    This objective decomposes the field magnitude :math:`B(\varphi,\theta)` into quasisymmetric and
    non-quasisymmetric components.  For quasi-axisymmetry, we compute

    .. math::
        B_{\text{QS}} &= \frac{\int_0^1 B \|\mathbf n\| ~d\varphi}{\int_0^1 \|\mathbf n\| ~d\varphi} \\
        B_{\text{non-QS}} &= B - B_{\text{QS}}

    where :math:`B = \| \mathbf B(\varphi,\theta) \|_2`.  
    For quasi-poloidal symmetry, an analagous formula is used, but the integral is computed in the :math:`\theta` direction.
    The objective computed by this penalty is

    .. math::
        J &= \frac{\int_{\Gamma_{s}} B_{\text{non-QS}}^2~dS}{\int_{\Gamma_{s}} B_{\text{QS}}^2~dS} \\

    When :math:`J` is zero, then there is perfect QS on the given boozer surface. The ratio of the QS and non-QS components
    of the field is returned to avoid dependence on the magnitude of the field strength.  Note that this penalty is computed
    on an auxilliary surface with quadrature points that are different from those on the input Boozer surface.  This is to allow
    for a spectrally accurate evaluation of the above integrals. Note that if boozer_surface.surface.stellsym == True, 
    computing this term on the half-period with shifted quadrature points is ~not~ equivalent to computing on the full-period 
    with unshifted points.  This is why we compute on an auxilliary surface with quadrature points on the full period.

    Args:
        boozer_surface: input boozer surface on which the penalty term is evaluated,
        biotsavart: biotsavart object (not necessarily the same as the one used on the Boozer surface). 
        sDIM: integer that determines the resolution of the quadrature points placed on the auxilliary surface.  
        quasi_poloidal: `False` for quasiaxisymmetry and `True` for quasipoloidal symmetry
    """

    def __init__(self, boozer_surface, bs, sDIM=20, quasi_poloidal=False):
        # only SurfaceXYZTensorFourier for now
        assert type(boozer_surface.surface) is SurfaceXYZTensorFourier

        Optimizable.__init__(self, depends_on=[boozer_surface])
        in_surface = boozer_surface.surface
        self.boozer_surface = boozer_surface

        surface = in_surface
        phis = np.linspace(0, 1/in_surface.nfp, 2*sDIM, endpoint=False)
        thetas = np.linspace(0, 1., 2*sDIM, endpoint=False)
        surface = SurfaceXYZTensorFourier(mpol=in_surface.mpol, ntor=in_surface.ntor, stellsym=in_surface.stellsym, nfp=in_surface.nfp, quadpoints_phi=phis, quadpoints_theta=thetas, dofs=in_surface.dofs)

        self.axis = 1 if quasi_poloidal else 0
        self.in_surface = in_surface
        self.surface = surface
        self.biotsavart = bs
        self.recompute_bell()

    def recompute_bell(self, parent=None):
        self._J = None
        self._dJ = None

    def J(self):
        if self._J is None:
            self.compute()
        return self._J

    @derivative_dec
    def dJ(self):
        if self._dJ is None:
            self.compute()
        return self._dJ

    def compute(self):
        if self.boozer_surface.need_to_run_code:
            res = self.boozer_surface.res
            res = self.boozer_surface.run_code(res['iota'], G=res['G'])

        self.biotsavart.set_points(self.surface.gamma().reshape((-1, 3)))
        axis = self.axis

        # compute J
        surface = self.surface
        nphi = surface.quadpoints_phi.size
        ntheta = surface.quadpoints_theta.size

        B = self.biotsavart.B()
        B = B.reshape((nphi, ntheta, 3))
        modB = np.sqrt(B[:, :, 0]**2 + B[:, :, 1]**2 + B[:, :, 2]**2)

        nor = surface.normal()
        dS = np.sqrt(nor[:, :, 0]**2 + nor[:, :, 1]**2 + nor[:, :, 2]**2)

        B_QS = np.mean(modB * dS, axis=axis) / np.mean(dS, axis=axis)

        if axis == 0:
            B_QS = B_QS[None, :]
        else:
            B_QS = B_QS[:, None]

        B_nonQS = modB - B_QS
        self._J = np.mean(dS * B_nonQS**2) / np.mean(dS * B_QS**2)

        booz_surf = self.boozer_surface
        iota, G, P, L, U, dconstraint_dcoils_vjp = _get_boozer_linearization(booz_surf)

        dJ_by_dB = self.dJ_by_dB().reshape((-1, 3))
        dJ_by_dcoils = self.biotsavart.B_vjp(dJ_by_dB)

        # tack on dJ_diota = dJ_dG = 0 to the end of dJ_ds
        dJ_ds = np.zeros(L.shape[0])
        dj_ds = self.dJ_by_dsurfacecoefficients()
        dJ_ds[:dj_ds.size] = dj_ds
        adj = forward_backward(P, L, U, dJ_ds)

        adj_times_dg_dcoil = dconstraint_dcoils_vjp(adj, booz_surf, iota, G)
        self._dJ = dJ_by_dcoils-adj_times_dg_dcoil

    def dJ_by_dB(self):
        """
        Return the partial derivative of the objective with respect to the magnetic field
        """
        surface = self.surface
        nphi = surface.quadpoints_phi.size
        ntheta = surface.quadpoints_theta.size
        axis = self.axis

        B = self.biotsavart.B()
        B = B.reshape((nphi, ntheta, 3))

        modB = np.sqrt(B[:, :, 0]**2 + B[:, :, 1]**2 + B[:, :, 2]**2)
        nor = surface.normal()
        dS = np.sqrt(nor[:, :, 0]**2 + nor[:, :, 1]**2 + nor[:, :, 2]**2)

        denom = np.mean(dS, axis=axis)
        B_QS = np.mean(modB * dS, axis=axis) / denom

        if axis == 0:
            B_QS = B_QS[None, :]
        else:
            B_QS = B_QS[:, None]

        B_nonQS = modB - B_QS

        dmodB_dB = B / modB[..., None]
        dnum_by_dB = B_nonQS[..., None] * dmodB_dB * dS[:, :, None] / (nphi * ntheta)  # d J_nonQS / dB_ijk
        ddenom_by_dB = B_QS[..., None] * dmodB_dB * dS[:, :, None] / (nphi * ntheta)  # dJ_QS/dB_ijk
        num = 0.5*np.mean(dS * B_nonQS**2)
        denom = 0.5*np.mean(dS * B_QS**2)
        return (denom * dnum_by_dB - num * ddenom_by_dB) / denom**2

    def dJ_by_dsurfacecoefficients(self):
        """
        Return the partial derivative of the objective with respect to the surface coefficients
        """
        surface = self.surface
        nphi = surface.quadpoints_phi.size
        ntheta = surface.quadpoints_theta.size
        axis = self.axis

        B = self.biotsavart.B()
        B = B.reshape((nphi, ntheta, 3))
        modB = np.sqrt(B[:, :, 0]**2 + B[:, :, 1]**2 + B[:, :, 2]**2)

        nor = surface.normal()
        dnor_dc = surface.dnormal_by_dcoeff()
        dS = np.sqrt(nor[:, :, 0]**2 + nor[:, :, 1]**2 + nor[:, :, 2]**2)
        dS_dc = (nor[:, :, 0, None]*dnor_dc[:, :, 0, :] + nor[:, :, 1, None]*dnor_dc[:, :, 1, :] + nor[:, :, 2, None]*dnor_dc[:, :, 2, :])/dS[:, :, None]

        B_QS = np.mean(modB * dS, axis=axis) / np.mean(dS, axis=axis)

        if axis == 0:
            B_QS = B_QS[None, :]
        else:
            B_QS = B_QS[:, None]

        B_nonQS = modB - B_QS

        dB_by_dX = self.biotsavart.dB_by_dX().reshape((nphi, ntheta, 3, 3))
        dx_dc = surface.dgamma_by_dcoeff()
        dB_dc = np.einsum('ijkl,ijkm->ijlm', dB_by_dX, dx_dc, optimize=True)

        modB = np.sqrt(B[:, :, 0]**2 + B[:, :, 1]**2 + B[:, :, 2]**2)
        dmodB_dc = (B[:, :, 0, None] * dB_dc[:, :, 0, :] + B[:, :, 1, None] * dB_dc[:, :, 1, :] + B[:, :, 2, None] * dB_dc[:, :, 2, :])/modB[:, :, None]

        num = np.mean(modB * dS, axis=axis)
        denom = np.mean(dS, axis=axis)
        dnum_dc = np.mean(dmodB_dc * dS[..., None] + modB[..., None] * dS_dc, axis=axis)
        ddenom_dc = np.mean(dS_dc, axis=axis)
        B_QS_dc = (dnum_dc * denom[:, None] - ddenom_dc * num[:, None])/denom[:, None]**2

        if axis == 0:
            B_QS_dc = B_QS_dc[None, :, :]
        else:
            B_QS_dc = B_QS_dc[:, None, :]

        B_nonQS_dc = dmodB_dc - B_QS_dc

        num = 0.5*np.mean(dS * B_nonQS**2)
        denom = 0.5*np.mean(dS * B_QS**2)
        dnum_by_dc = np.mean(0.5*dS_dc * B_nonQS[..., None]**2 + dS[..., None] * B_nonQS[..., None] * B_nonQS_dc, axis=(0, 1))
        ddenom_by_dc = np.mean(0.5*dS_dc * B_QS[..., None]**2 + dS[..., None] * B_QS[..., None] * B_QS_dc, axis=(0, 1))
        dJ_by_dc = (denom * dnum_by_dc - num * ddenom_by_dc) / denom**2
        return dJ_by_dc


def _make_qi_aux_surface(in_surface, sDIM):
    phis = np.linspace(0, 1/in_surface.nfp, 2*sDIM, endpoint=False)
    thetas = np.linspace(0, 1., 2*sDIM, endpoint=False)
    return SurfaceXYZTensorFourier(mpol=in_surface.mpol, ntor=in_surface.ntor, stellsym=in_surface.stellsym,
                                   nfp=in_surface.nfp, quadpoints_phi=phis, quadpoints_theta=thetas,
                                   dofs=in_surface.dofs)


def _normalize_modB_global(values, epsilon=1e-15):
    values = np.asarray(values, dtype=float)
    minimum = np.min(values)
    maximum = np.max(values)
    scale = max(maximum - minimum, epsilon)
    return (values - minimum) / scale, minimum, maximum, scale


def _squash_left_branch(branch):
    branch = np.asarray(branch, dtype=float).copy()
    index_max = np.argmax(branch)
    branch[:index_max] = branch[index_max]
    for index in range(len(branch) - 1):
        if branch[index] <= branch[index + 1]:
            index_final = len(branch) - 1
            for follower in range(index + 1, len(branch)):
                if branch[follower] < branch[index]:
                    index_final = follower
                    break
            branch[index:index_final] = branch[index]
    return branch


def _squash_right_branch(branch):
    branch = np.asarray(branch, dtype=float).copy()
    index_max = np.argmax(branch)
    branch[index_max:] = branch[index_max]
    for index in range(len(branch) - 1, 1, -1):
        if branch[index - 1] >= branch[index]:
            index_final = 0
            for follower in range(index - 1, 1, -1):
                if branch[follower] < branch[index]:
                    index_final = follower
                    break
            branch[index_final + 1:index] = branch[index]
    return branch


def _stretch_left_branch(phi_values, branch, pmax=50, pmin=15):
    phi_values = np.asarray(phi_values, dtype=float)
    branch = np.asarray(branch, dtype=float)
    if phi_values.size < 2 or phi_values[-1] == phi_values[0]:
        return np.zeros_like(branch)
    x_values = (phi_values - phi_values[0]) / (phi_values[-1] - phi_values[0])
    left_half = x_values < 0.5
    r1 = 1 - branch[0]
    r2 = -branch[-1]
    cosine_term = ((np.cos(2 * np.pi * x_values) + 1) / 2)
    return left_half * r1 * cosine_term**pmax + (~left_half) * r2 * cosine_term**pmin


def _stretch_right_branch(phi_values, branch, pmax=50, pmin=15):
    phi_values = np.asarray(phi_values, dtype=float)
    branch = np.asarray(branch, dtype=float)
    if phi_values.size < 2 or phi_values[-1] == phi_values[0]:
        return np.zeros_like(branch)
    x_values = (phi_values - phi_values[0]) / (phi_values[-1] - phi_values[0])
    left_half = x_values < 0.5
    r1 = 1 - branch[-1]
    r2 = -branch[0]
    cosine_term = ((np.cos(2 * np.pi * x_values) + 1) / 2)
    return left_half * r2 * cosine_term**pmin + (~left_half) * r1 * cosine_term**pmax


def _get_branches(phi_values, branch, level, maximum, minimum):
    phi_values = np.asarray(phi_values, dtype=float)
    branch = np.asarray(branch, dtype=float)

    if phi_values.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    if phi_values.size == 1 or branch.size == 1:
        phi_single = float(phi_values[0])
        return phi_single, phi_single, 0.0, 0.0

    differences = branch - level
    sign_products = differences[:-1] * differences[1:]
    indices = np.where(sign_products < 0)[0]
    indices = np.sort(indices)

    if level == minimum or level < minimum:
        index_minimum = int(np.argmin(branch))
        phi_minimum = phi_values[index_minimum]
        return phi_minimum, phi_minimum, float(index_minimum), float(index_minimum)
    if level == maximum or level > maximum:
        return phi_values[0], phi_values[-1], 0.0, float(len(branch) - 1)

    if len(indices) < 2:
        indices = np.where(sign_products <= 0)[0]
        for index in range(1, len(indices)):
            if indices[index] != indices[index - 1] + 1:
                indices = [indices[index - 1], indices[-1]]
                break

    if len(indices) < 2:
        if len(indices) == 1:
            index = int(indices[0])
            delta_y = branch[index] - branch[index + 1]
            delta_x = phi_values[index] - phi_values[index + 1]
            slope = delta_y / delta_x if delta_x != 0 else 0.0
            if slope != 0:
                intercept = branch[index] - slope * phi_values[index]
                phi_hit = float((level - intercept) / slope)
            else:
                phi_hit = float(phi_values[index])
            return phi_hit, phi_hit, 0.0, 0.0

        equal_hits = np.where(np.isclose(branch, level, rtol=0.0, atol=1e-14))[0]
        if equal_hits.size >= 2:
            first = int(equal_hits[0])
            last = int(equal_hits[-1])
            return phi_values[first], phi_values[last], 0.0, 0.0
        if equal_hits.size == 1:
            hit = int(equal_hits[0])
            phi_hit = float(phi_values[hit])
            return phi_hit, phi_hit, 0.0, 0.0

        index_minimum = int(np.argmin(np.abs(differences)))
        phi_hit = float(phi_values[index_minimum])
        return phi_hit, phi_hit, 0.0, 0.0

    if len(indices) > 2:
        indices = [indices[0], indices[-1]]

    index_1 = int(indices[0])
    index_2 = int(indices[1])

    delta_y_1 = branch[index_1] - branch[index_1 + 1]
    delta_x_1 = phi_values[index_1] - phi_values[index_1 + 1]
    slope_1 = delta_y_1 / delta_x_1
    intercept_1 = branch[index_1] - slope_1 * phi_values[index_1]
    phi_1 = (level - intercept_1) / slope_1 if slope_1 != 0 else phi_values[index_1]

    delta_y_2 = branch[index_2] - branch[index_2 + 1]
    delta_x_2 = phi_values[index_2] - phi_values[index_2 + 1]
    slope_2 = delta_y_2 / delta_x_2
    intercept_2 = branch[index_2] - slope_2 * phi_values[index_2]
    phi_2 = (level - intercept_2) / slope_2 if slope_2 != 0 else phi_values[index_2 + 1]

    return phi_1, phi_2, slope_1, slope_2


def _repair_shifted_qi_branches(left_branch, right_branch):
    left_branch = np.asarray(left_branch, dtype=float).copy()
    right_branch = np.asarray(right_branch, dtype=float).copy()
    count = len(left_branch)
    for index in range(count - 1):
        if left_branch[index + 1] - left_branch[index] < 0:
            right_branch[-index - 2] = right_branch[-index - 2] + (left_branch[index] - left_branch[index + 1] + 1e-12)
            left_branch[index + 1] = left_branch[index] + 1e-12
        if right_branch[-index - 1] - right_branch[-index - 2] < 0:
            left_branch[index + 1] = left_branch[index + 1] + (right_branch[-index - 1] - right_branch[-index - 2] - 1e-12)
            right_branch[-index - 2] = right_branch[-index - 1] - 1e-12
    return left_branch, right_branch


def _enforce_strictly_increasing(values, min_spacing=None):
    values = np.asarray(values, dtype=float).copy()
    if values.size <= 1:
        return values

    finite_values = values[np.isfinite(values)]
    scale = max(1.0, float(np.max(np.abs(finite_values)))) if finite_values.size else 1.0
    spacing = 1e-12 * scale if min_spacing is None else float(min_spacing)

    if not np.isfinite(values[0]):
        values[0] = 0.0
    for index in range(1, values.size):
        if not np.isfinite(values[index]) or values[index] <= values[index - 1]:
            values[index] = values[index - 1] + spacing
    return values


def _squash_left_branch_sources(branch):
    values = np.asarray(branch, dtype=float).copy()
    sources = np.arange(values.size, dtype=int)
    if values.size <= 1:
        return sources

    index_max = int(np.argmax(values))
    values[:index_max] = values[index_max]
    sources[:index_max] = sources[index_max]
    for index in range(len(values) - 1):
        if values[index] <= values[index + 1]:
            index_final = len(values) - 1
            for follower in range(index + 1, len(values)):
                if values[follower] < values[index]:
                    index_final = follower
                    break
            values[index:index_final] = values[index]
            sources[index:index_final] = sources[index]
    return sources


def _squash_right_branch_sources(branch):
    values = np.asarray(branch, dtype=float).copy()
    sources = np.arange(values.size, dtype=int)
    if values.size <= 1:
        return sources

    index_max = int(np.argmax(values))
    values[index_max:] = values[index_max]
    sources[index_max:] = sources[index_max]
    for index in range(len(values) - 1, 1, -1):
        if values[index - 1] >= values[index]:
            index_final = 0
            for follower in range(index - 1, 1, -1):
                if values[follower] < values[index]:
                    index_final = follower
                    break
            values[index_final + 1:index] = values[index]
            sources[index_final + 1:index] = sources[index]
    return sources


def _build_branch_crossing_descriptor(phi_values, branch, level, maximum, minimum):
    phi_values = np.asarray(phi_values, dtype=float)
    branch = np.asarray(branch, dtype=float)
    descriptor = {"level": float(level)}

    if phi_values.size == 0:
        descriptor.update({"mode": "constant", "phi1": 0.0, "phi2": 0.0})
        return descriptor
    if phi_values.size == 1 or branch.size == 1:
        phi_single = float(phi_values[0])
        descriptor.update({"mode": "constant", "phi1": phi_single, "phi2": phi_single})
        return descriptor

    differences = branch - level
    sign_products = differences[:-1] * differences[1:]
    indices = np.where(sign_products < 0)[0]
    indices = np.sort(indices)

    if level == minimum or level < minimum:
        phi_minimum = float(phi_values[int(np.argmin(branch))])
        descriptor.update({"mode": "constant", "phi1": phi_minimum, "phi2": phi_minimum})
        return descriptor
    if level == maximum or level > maximum:
        descriptor.update({"mode": "constant", "phi1": float(phi_values[0]), "phi2": float(phi_values[-1])})
        return descriptor

    if len(indices) < 2:
        indices = np.where(sign_products <= 0)[0]
        for index in range(1, len(indices)):
            if indices[index] != indices[index - 1] + 1:
                indices = np.asarray([indices[index - 1], indices[-1]])
                break

    if len(indices) < 2:
        if len(indices) == 1:
            descriptor.update({"mode": "single-segment", "index": int(indices[0])})
            return descriptor

        equal_hits = np.where(np.isclose(branch, level, rtol=0.0, atol=1e-14))[0]
        if equal_hits.size >= 2:
            descriptor.update({
                "mode": "constant",
                "phi1": float(phi_values[int(equal_hits[0])]),
                "phi2": float(phi_values[int(equal_hits[-1])]),
            })
            return descriptor
        if equal_hits.size == 1:
            phi_hit = float(phi_values[int(equal_hits[0])])
            descriptor.update({"mode": "constant", "phi1": phi_hit, "phi2": phi_hit})
            return descriptor

        phi_hit = float(phi_values[int(np.argmin(np.abs(differences)))])
        descriptor.update({"mode": "constant", "phi1": phi_hit, "phi2": phi_hit})
        return descriptor

    if len(indices) > 2:
        indices = np.asarray([indices[0], indices[-1]])

    descriptor.update({"mode": "pair", "index_1": int(indices[0]), "index_2": int(indices[1])})
    return descriptor


def _build_shift_repair_linearization(left_branch, right_branch):
    left = np.asarray(left_branch, dtype=float).copy()
    right = np.asarray(right_branch, dtype=float).copy()
    left_fixes = []
    right_fixes = []
    count = len(left)
    for index in range(count - 1):
        fix_left = left[index + 1] - left[index] < 0
        left_fixes.append(bool(fix_left))
        if fix_left:
            right[-index - 2] = right[-index - 2] + (left[index] - left[index + 1] + 1e-12)
            left[index + 1] = left[index] + 1e-12

        fix_right = right[-index - 1] - right[-index - 2] < 0
        right_fixes.append(bool(fix_right))
        if fix_right:
            left[index + 1] = left[index + 1] + (right[-index - 1] - right[-index - 2] - 1e-12)
            right[-index - 2] = right[-index - 1] - 1e-12

    return {"left_fixes": left_fixes, "right_fixes": right_fixes}


def _build_strictly_increasing_linearization(values, min_spacing=None):
    values = np.asarray(values, dtype=float).copy()
    finite_values = values[np.isfinite(values)]
    scale = max(1.0, float(np.max(np.abs(finite_values)))) if finite_values.size else 1.0
    spacing = 1e-12 * scale if min_spacing is None else float(min_spacing)
    adjusted = values.copy()
    reset_first = not np.isfinite(adjusted[0]) if adjusted.size else False
    if reset_first:
        adjusted[0] = 0.0
    fix_flags = []
    for index in range(1, adjusted.size):
        fix = (not np.isfinite(adjusted[index])) or adjusted[index] <= adjusted[index - 1]
        fix_flags.append(bool(fix))
        if fix:
            adjusted[index] = adjusted[index - 1] + spacing
    return {"reset_first": reset_first, "fix_flags": fix_flags, "spacing": spacing}


def _build_piecewise_linear_interval_indices(x_nodes, x_query):
    x_nodes = np.asarray(x_nodes, dtype=float)
    x_query = np.asarray(x_query, dtype=float)
    if x_nodes.size <= 1:
        return np.zeros_like(x_query, dtype=int)
    return np.clip(np.searchsorted(x_nodes, x_query, side='right') - 1, 0, x_nodes.size - 2)


def _trapz_1d(values, x_values):
    values = np.asarray(values, dtype=float)
    x_values = np.asarray(x_values, dtype=float)
    if values.size <= 1:
        return 0.0
    return float(np.sum(0.5 * (values[1:] + values[:-1]) * (x_values[1:] - x_values[:-1])))


def _stretch_left_branch_jax(phi_values, branch, pmax=50, pmin=15):
    if branch.shape[0] < 2:
        return jnp.zeros_like(branch)
    span = phi_values[-1] - phi_values[0]
    x_values = (phi_values - phi_values[0]) / span
    left_half = x_values < 0.5
    r1 = 1.0 - branch[0]
    r2 = -branch[-1]
    cosine_term = (jnp.cos(2.0 * jnp.pi * x_values) + 1.0) / 2.0
    return left_half * r1 * cosine_term**pmax + (~left_half) * r2 * cosine_term**pmin


def _stretch_right_branch_jax(phi_values, branch, pmax=50, pmin=15):
    if branch.shape[0] < 2:
        return jnp.zeros_like(branch)
    span = phi_values[-1] - phi_values[0]
    x_values = (phi_values - phi_values[0]) / span
    left_half = x_values < 0.5
    r1 = 1.0 - branch[-1]
    r2 = -branch[0]
    cosine_term = (jnp.cos(2.0 * jnp.pi * x_values) + 1.0) / 2.0
    return left_half * r2 * cosine_term**pmin + (~left_half) * r1 * cosine_term**pmax


def _evaluate_branch_crossing_fixed(branch, phi_values, descriptor):
    level = descriptor["level"]
    mode = descriptor["mode"]
    if mode == "constant":
        return jnp.asarray(descriptor["phi1"]), jnp.asarray(descriptor["phi2"])

    def _phi_from_segment(index):
        delta_y = branch[index] - branch[index + 1]
        delta_x = phi_values[index] - phi_values[index + 1]
        slope = delta_y / delta_x
        intercept = branch[index] - slope * phi_values[index]
        return jnp.where(slope != 0.0, (level - intercept) / slope, phi_values[index])

    if mode == "single-segment":
        phi_hit = _phi_from_segment(descriptor["index"])
        return phi_hit, phi_hit

    phi_1 = _phi_from_segment(descriptor["index_1"])
    phi_2 = _phi_from_segment(descriptor["index_2"])
    return phi_1, phi_2


def _repair_shifted_qi_branches_fixed(left_branch, right_branch, data):
    left = left_branch
    right = right_branch
    count = left.shape[0]
    for index in range(count - 1):
        if data["left_fixes"][index]:
            right = right.at[-index - 2].set(right[-index - 2] + (left[index] - left[index + 1] + 1e-12))
            left = left.at[index + 1].set(left[index] + 1e-12)
        if data["right_fixes"][index]:
            left = left.at[index + 1].set(left[index + 1] + (right[-index - 1] - right[-index - 2] - 1e-12))
            right = right.at[-index - 2].set(right[-index - 1] - 1e-12)
    return left, right


def _enforce_strictly_increasing_fixed(values, data):
    adjusted = values
    if adjusted.shape[0] == 0:
        return adjusted
    if data["reset_first"]:
        adjusted = adjusted.at[0].set(0.0)
    spacing = data["spacing"]
    for index, fix in enumerate(data["fix_flags"], start=1):
        if fix:
            adjusted = adjusted.at[index].set(adjusted[index - 1] + spacing)
    return adjusted


def _evaluate_piecewise_linear_fixed(x_nodes, y_nodes, x_query, interval_indices):
    if x_nodes.shape[0] <= 1:
        return jnp.full(x_query.shape, y_nodes[0] if y_nodes.shape[0] else 0.0)
    indices = jnp.asarray(interval_indices)
    x0 = x_nodes[indices]
    x1 = x_nodes[indices + 1]
    y0 = y_nodes[indices]
    y1 = y_nodes[indices + 1]
    denom = x1 - x0
    t = jnp.where(denom != 0.0, (x_query - x0) / denom, 0.0)
    return y0 + t * (y1 - y0)


def _normalize_modB_global_fixed(values, data):
    flat = values.reshape((-1,))
    minimum = flat[data["minimum_index"]]
    scale = jnp.maximum(flat[data["maximum_index"]] - minimum, data["epsilon"])
    return (values - minimum) / scale


def _build_template_well_fixed(branch_values, data, phi_values):
    index_minimum = data["index_minimum"]
    left_branch = branch_values[:index_minimum + 1]
    right_branch = branch_values[index_minimum:]
    left_sources = jnp.asarray(data["left_sources"])
    right_sources = jnp.asarray(data["right_sources"])

    left_squashed = left_branch[left_sources]
    right_squashed = right_branch[right_sources]
    left_phi = phi_values[:index_minimum + 1]
    right_phi = phi_values[index_minimum:]

    left_values = left_squashed + _stretch_left_branch_jax(left_phi, left_squashed)
    right_values = right_squashed + _stretch_right_branch_jax(right_phi, right_squashed)
    template_values = jnp.concatenate((left_values[:-1], right_values))

    error_sq = (branch_values - template_values) ** 2
    integral = jnp.sum(0.5 * (error_sq[1:] + error_sq[:-1]) * (phi_values[1:] - phi_values[:-1]))
    weight_raw = (phi_values[-1] - phi_values[0]) / jnp.maximum(integral, 1e-15)

    bounce_distances = []
    branch_locations = []
    for descriptor in data["crossings"]:
        phi_1, phi_2 = _evaluate_branch_crossing_fixed(template_values, phi_values, descriptor)
        bounce_distances.append(phi_2 - phi_1)
        branch_locations.append((phi_1, phi_2))

    bounce_distances = jnp.stack(bounce_distances)
    branch_locations = jnp.asarray(branch_locations)
    left_locations = branch_locations[:, 0][::-1]
    right_locations = branch_locations[:, 1]
    return template_values, weight_raw, bounce_distances, jnp.concatenate((left_locations, right_locations[1:]))


def _build_qi_objective_linearization(modB_lines, phi_values, nBj, nphi_out):
    phi_values = np.asarray(phi_values, dtype=float)
    modB_lines = np.asarray(modB_lines, dtype=float)
    bounce_levels = np.linspace(0.0, 1.0, nBj)
    phi_out = np.linspace(phi_values[0], phi_values[-1], nphi_out)
    normalized, _, _, _ = _normalize_modB_global(modB_lines)
    flat = modB_lines.reshape((-1,))

    data = {
        "bounce_levels": bounce_levels,
        "phi_values": phi_values,
        "phi_out": phi_out,
        "nalpha": normalized.shape[1],
        "nphi_out": nphi_out,
        "epsilon": 1e-15,
        "minimum_index": int(np.argmin(flat)),
        "maximum_index": int(np.argmax(flat)),
        "original_interp_indices": _build_piecewise_linear_interval_indices(phi_values, phi_out),
        "alpha_data": [],
    }

    raw_weights = []
    bounce_distances = []
    branch_locations = []
    for index in range(normalized.shape[1]):
        branch = normalized[:, index]
        index_minimum = int(np.argmin(branch))
        template_values, weight_raw, current_bounce, current_locations = _build_template_well(phi_values, branch, bounce_levels)
        alpha_data = {
            "index_minimum": index_minimum,
            "left_sources": _squash_left_branch_sources(branch[:index_minimum + 1]),
            "right_sources": _squash_right_branch_sources(branch[index_minimum:]),
            "crossings": [
                _build_branch_crossing_descriptor(phi_values, template_values, level, 1.0, 0.0)
                for level in bounce_levels
            ],
        }
        data["alpha_data"].append(alpha_data)
        raw_weights.append(weight_raw)
        bounce_distances.append(current_bounce)
        branch_locations.append(current_locations)

    raw_weights = np.asarray(raw_weights)
    bounce_distances = np.asarray(bounce_distances)
    branch_locations = np.asarray(branch_locations)
    normalized_weights = raw_weights / np.sum(raw_weights)
    mean_bounce = np.sum(bounce_distances * normalized_weights[:, None], axis=0)
    target_levels = np.concatenate((np.flip(bounce_levels), bounce_levels[1:]))

    for index, alpha_data in enumerate(data["alpha_data"]):
        bounce_delta = (bounce_distances[index, :] - mean_bounce) / 2.0
        left_branch = branch_locations[index, :bounce_levels.size] + np.flip(bounce_delta)
        right_branch = branch_locations[index, bounce_levels.size - 1:] - bounce_delta
        repair_data = _build_shift_repair_linearization(left_branch, right_branch)
        left_branch, right_branch = _repair_shifted_qi_branches(left_branch, right_branch)
        shifted_locations = np.concatenate((left_branch, right_branch[1:]))
        increasing_data = _build_strictly_increasing_linearization(shifted_locations)
        shifted_locations = _enforce_strictly_increasing(shifted_locations)
        alpha_data["repair_data"] = repair_data
        alpha_data["increasing_data"] = increasing_data
        alpha_data["target_interval_indices"] = _build_piecewise_linear_interval_indices(shifted_locations, phi_out)
        alpha_data["target_levels"] = target_levels

    return data


def _qi_objective_from_raw_lines_fixed(raw_lines, data):
    phi_values = jnp.asarray(data["phi_values"])
    phi_out = jnp.asarray(data["phi_out"])
    bounce_levels = jnp.asarray(data["bounce_levels"])
    normalized = _normalize_modB_global_fixed(raw_lines, data)

    raw_weights = []
    bounce_distances = []
    branch_locations = []
    for index, alpha_data in enumerate(data["alpha_data"]):
        branch = normalized[:, index]
        _, weight_raw, current_bounce, current_locations = _build_template_well_fixed(branch, alpha_data, phi_values)
        raw_weights.append(weight_raw)
        bounce_distances.append(current_bounce)
        branch_locations.append(current_locations)

    raw_weights = jnp.stack(raw_weights)
    bounce_distances = jnp.stack(bounce_distances)
    branch_locations = jnp.stack(branch_locations)
    normalized_weights = raw_weights / jnp.sum(raw_weights)
    mean_bounce = jnp.sum(bounce_distances * normalized_weights[:, None], axis=0)
    target_levels = jnp.asarray(data["alpha_data"][0]["target_levels"])

    residuals = []
    normalization = jnp.sqrt(data["nalpha"] * data["nphi_out"])
    for index, alpha_data in enumerate(data["alpha_data"]):
        branch = normalized[:, index]
        bounce_delta = (bounce_distances[index, :] - mean_bounce) / 2.0
        left_branch = branch_locations[index, :bounce_levels.shape[0]] + jnp.flip(bounce_delta)
        right_branch = branch_locations[index, bounce_levels.shape[0] - 1:] - bounce_delta
        left_branch, right_branch = _repair_shifted_qi_branches_fixed(left_branch, right_branch, alpha_data["repair_data"])
        shifted_locations = jnp.concatenate((left_branch, right_branch[1:]))
        shifted_locations = _enforce_strictly_increasing_fixed(shifted_locations, alpha_data["increasing_data"])
        target_values = _evaluate_piecewise_linear_fixed(
            shifted_locations,
            target_levels,
            phi_out,
            alpha_data["target_interval_indices"],
        )
        original_values = _evaluate_piecewise_linear_fixed(
            phi_values,
            branch,
            phi_out,
            data["original_interp_indices"],
        )
        residuals.append((target_values - original_values) / normalization)

    residual = jnp.concatenate(residuals)
    return jnp.dot(residual, residual)


def _qi_objective_and_gradient_from_raw_lines(modB_lines, phi_values, nBj, nphi_out):
    data = _build_qi_objective_linearization(modB_lines, phi_values, nBj, nphi_out)
    objective = lambda values: _qi_objective_from_raw_lines_fixed(values, data)
    value, gradient = value_and_grad(objective)(jnp.asarray(modB_lines, dtype=jnp.float64))
    return float(value), np.asarray(gradient, dtype=float)


def _build_boozer_line_sampling_linearization(surface, phi_values, alpha_values, iota, phi_shift):
    field_period_normalized = 1.0 / surface.nfp
    phi_normalized = np.mod(phi_values / (2 * np.pi), field_period_normalized)
    nphi_grid = surface.quadpoints_phi.size
    ntheta_grid = surface.quadpoints_theta.size
    dphi = field_period_normalized / nphi_grid
    dtheta = 1.0 / ntheta_grid

    theta_physical = alpha_values[None, :] + iota * (phi_values[:, None] - phi_shift)
    theta_normalized = np.mod(theta_physical / (2 * np.pi), 1.0)

    phi_index = np.broadcast_to(
        np.mod(phi_normalized[:, None] / dphi, nphi_grid),
        theta_normalized.shape,
    )
    theta_index = np.mod(theta_normalized / dtheta, ntheta_grid)

    phi_index_0 = np.floor(phi_index).astype(int)
    theta_index_0 = np.floor(theta_index).astype(int)
    phi_index_1 = (phi_index_0 + 1) % nphi_grid
    theta_index_1 = (theta_index_0 + 1) % ntheta_grid
    phi_weight = phi_index - phi_index_0
    theta_weight = theta_index - theta_index_0

    return {
        "phi_index_0": phi_index_0,
        "phi_index_1": phi_index_1,
        "theta_index_0": theta_index_0,
        "theta_index_1": theta_index_1,
        "phi_weight": phi_weight,
        "theta_weight": theta_weight,
        "dtheta": dtheta,
        "dtheta_diota": np.broadcast_to(
            (phi_values[:, None] - phi_shift) / (2 * np.pi),
            theta_normalized.shape,
        ),
    }


def _pullback_boozer_line_sampling(sample_gradient, modB_grid, sampling_data):
    grad_grid = np.zeros_like(modB_grid)
    grad_iota = 0.0

    phi_index_0 = sampling_data["phi_index_0"]
    phi_index_1 = sampling_data["phi_index_1"]
    theta_index_0 = sampling_data["theta_index_0"]
    theta_index_1 = sampling_data["theta_index_1"]
    phi_weight = sampling_data["phi_weight"]
    theta_weight = sampling_data["theta_weight"]

    for row in range(sample_gradient.shape[0]):
        for column in range(sample_gradient.shape[1]):
            gradient = sample_gradient[row, column]
            i0 = phi_index_0[row, column]
            i1 = phi_index_1[row, column]
            j0 = theta_index_0[row, column]
            j1 = theta_index_1[row, column]
            wp = phi_weight[row, column]
            wt = theta_weight[row, column]

            w00 = (1 - wp) * (1 - wt)
            w01 = (1 - wp) * wt
            w10 = wp * (1 - wt)
            w11 = wp * wt

            grad_grid[i0, j0] += gradient * w00
            grad_grid[i0, j1] += gradient * w01
            grad_grid[i1, j0] += gradient * w10
            grad_grid[i1, j1] += gradient * w11

            val00 = modB_grid[i0, j0]
            val01 = modB_grid[i0, j1]
            val10 = modB_grid[i1, j0]
            val11 = modB_grid[i1, j1]
            dsample_dtheta = ((1 - wp) * (val01 - val00) + wp * (val11 - val10)) / sampling_data["dtheta"]
            grad_iota += gradient * dsample_dtheta * sampling_data["dtheta_diota"][row, column]

    return grad_grid, float(grad_iota)


def _qi_objective_single_surface(target_values, source_values, nalpha, nphi_out):
    target_values = np.asarray(target_values, dtype=float)
    source_values = np.asarray(source_values, dtype=float)
    residual = (target_values - source_values) / np.sqrt(nalpha * nphi_out)
    return float(np.dot(residual.reshape((-1,)), residual.reshape((-1,))))


def _periodic_bilinear_interpolate(values, period_phi, phi_query, theta_query):
    values = np.asarray(values, dtype=float)
    nphi, ntheta = values.shape
    dphi = period_phi / nphi
    dtheta = 1.0 / ntheta

    phi_index = np.mod(phi_query / dphi, nphi)
    theta_index = np.mod(theta_query / dtheta, ntheta)

    phi_index_0 = np.floor(phi_index).astype(int)
    theta_index_0 = np.floor(theta_index).astype(int)
    phi_index_1 = (phi_index_0 + 1) % nphi
    theta_index_1 = (theta_index_0 + 1) % ntheta

    phi_weight = phi_index - phi_index_0
    theta_weight = theta_index - theta_index_0

    val00 = values[phi_index_0, theta_index_0]
    val01 = values[phi_index_0, theta_index_1]
    val10 = values[phi_index_1, theta_index_0]
    val11 = values[phi_index_1, theta_index_1]

    return ((1 - phi_weight) * (1 - theta_weight) * val00
            + (1 - phi_weight) * theta_weight * val01
            + phi_weight * (1 - theta_weight) * val10
            + phi_weight * theta_weight * val11)


def _make_qi_sampling_grid(surface, nphi, nalpha, phi_shift):
    field_period = 2 * np.pi / surface.nfp
    if phi_shift is None:
        phi_shift = 0.0
    phi_values = np.linspace(phi_shift, phi_shift + field_period, nphi)
    alpha_values = np.linspace(0.0, 2 * np.pi, nalpha, endpoint=False)
    return phi_values, alpha_values


def _sample_modB_on_boozer_lines(modB_grid, surface, phi_values, alpha_values, iota, phi_shift):
    field_period_normalized = 1.0 / surface.nfp
    phi_normalized = np.mod(phi_values / (2 * np.pi), field_period_normalized)
    sampled = np.zeros((phi_values.size, alpha_values.size))
    for index, alpha in enumerate(alpha_values):
        theta_physical = alpha + iota * (phi_values - phi_shift)
        theta_normalized = np.mod(theta_physical / (2 * np.pi), 1.0)
        sampled[:, index] = _periodic_bilinear_interpolate(modB_grid, field_period_normalized, phi_normalized, theta_normalized)
    return sampled


def _build_template_well(phi_values, branch_values, bounce_levels):
    index_minimum = int(np.argmin(branch_values))
    left_branch = branch_values[:index_minimum + 1].copy()
    right_branch = branch_values[index_minimum:].copy()
    left_phi = phi_values[:index_minimum + 1].copy()
    right_phi = phi_values[index_minimum:].copy()

    left_branch = _squash_left_branch(left_branch)
    right_branch = _squash_right_branch(right_branch)

    left_branch = left_branch + _stretch_left_branch(left_phi, left_branch)
    right_branch = right_branch + _stretch_right_branch(right_phi, right_branch)
    left_branch = left_branch[:-1]
    template_values = np.concatenate((left_branch, right_branch))

    weight_spline = UnivariateSpline(phi_values, np.abs(branch_values - template_values)**2, k=1, s=0)
    weight_raw = (phi_values[-1] - phi_values[0]) / weight_spline.integral(phi_values[0], phi_values[-1])

    bounce_distances = np.zeros((bounce_levels.size,))
    branch_locations = np.zeros((2 * bounce_levels.size - 1,))
    for index, level in enumerate(bounce_levels):
        phi_1, phi_2, _, _ = _get_branches(phi_values, template_values, level, 1.0, 0.0)
        bounce_distances[index] = phi_2 - phi_1
        branch_locations[bounce_levels.size - index - 1] = phi_1
        branch_locations[bounce_levels.size + index - 1] = phi_2

    return template_values, weight_raw, bounce_distances, branch_locations


def _make_shuffled_target_values(phi_values, branch_values, bounce_levels, branch_locations, mean_bounce_distances):
    bounce_delta = (branch_values - mean_bounce_distances) / 2
    left_branch = branch_locations[:bounce_levels.size].copy()
    right_branch = branch_locations[bounce_levels.size - 1:].copy()
    left_branch = left_branch + np.flip(bounce_delta)
    right_branch = right_branch - bounce_delta
    left_branch, right_branch = _repair_shifted_qi_branches(left_branch, right_branch)

    shifted_locations = np.concatenate((left_branch, right_branch[1:]))
    target_levels = np.concatenate((np.flip(bounce_levels), bounce_levels[1:]))
    shifted_locations = _enforce_strictly_increasing(shifted_locations)
    spline = UnivariateSpline(shifted_locations, target_levels, k=1, s=0)
    return spline(phi_values)


def _get_boozer_linearization(booz_surf):
    res = booz_surf.res
    if not res.get('success', True):
        raise RuntimeError("Boozer surface solve failed during objective evaluation.")
    if 'PLU' not in res or 'vjp' not in res:
        raise RuntimeError("Boozer surface solve did not provide derivative metadata.")
    iota = res['iota']
    G = res['G']
    P, L, U = res['PLU']
    return iota, G, P, L, U, res['vjp']


def _qi_residual_vector_single_surface(modB_lines, phi_values, nBj, nphi_out):
    normalized, _, _, _ = _normalize_modB_global(modB_lines)
    nphi, nalpha = normalized.shape
    bounce_levels = np.linspace(0.0, 1.0, nBj)
    template_values = np.zeros_like(normalized)
    raw_weights = np.zeros((nalpha,))
    bounce_distances = np.zeros((nalpha, nBj))
    branch_locations = np.zeros((nalpha, 2 * nBj - 1))

    for index in range(nalpha):
        template_values[:, index], raw_weights[index], bounce_distances[index, :], branch_locations[index, :] = _build_template_well(phi_values, normalized[:, index], bounce_levels)

    normalized_weights = raw_weights / np.sum(raw_weights)
    mean_bounce_distances = np.sum(bounce_distances * normalized_weights[:, None], axis=0)
    phi_out = np.linspace(phi_values[0], phi_values[-1], nphi_out)
    residuals = np.zeros((nalpha, nphi_out))

    for index in range(nalpha):
        original_spline = UnivariateSpline(phi_values, normalized[:, index], k=1, s=0)
        target_values = _make_shuffled_target_values(phi_out, bounce_distances[index, :], bounce_levels,
                                                     branch_locations[index, :], mean_bounce_distances)
        residuals[index, :] = (target_values - original_spline(phi_out)) / np.sqrt(nalpha * nphi_out)

    return residuals.reshape((-1,))


class NonQuasiIsodynamicRatio(Optimizable):
    r"""
    Milestone-1 quasi-isodynamic objective on a Boozer surface.

    The current implementation ports the legacy single-surface well-shuffling residual
    to a BoozerSurface auxiliary grid and defines the scalar objective as the squared norm
    of the normalized residual vector. Derivatives are propagated through the Boozer-surface
    solve using the linearized constraint system together with Biot-Savart vector-Jacobian
    products, so coil gradients are computed without a finite-difference outer loop.
    """

    def __init__(self, boozer_surface, bs, sDIM=20, nphi=151, nalpha=31, nBj=51, nphi_out=2000, phi_shift=None, smoothing=None):
        assert type(boozer_surface.surface) is SurfaceXYZTensorFourier

        Optimizable.__init__(self, depends_on=[boozer_surface, bs])
        self.boozer_surface = boozer_surface
        self.in_surface = boozer_surface.surface
        self.surface = _make_qi_aux_surface(self.in_surface, sDIM)
        self.biotsavart = bs
        self.nphi = nphi
        self.nalpha = nalpha
        self.nBj = nBj
        self.nphi_out = nphi_out
        self.phi_shift = phi_shift
        self.smoothing = smoothing
        self.recompute_bell()

    def recompute_bell(self, parent=None):
        self._J = None
        self._dJ = None

    def J(self):
        if self._J is None:
            self.compute()
        return self._J

    def residuals(self):
        return self._residual_vector_from_current_state()

    @derivative_dec
    def dJ(self):
        if self._dJ is None:
            self.compute()
        return self._dJ

    def _ensure_current_boozer_surface(self):
        if self.boozer_surface.need_to_run_code:
            res = self.boozer_surface.res
            try:
                self.boozer_surface.run_code(res['iota'], G=res['G'])
            except np.linalg.LinAlgError:
                self.boozer_surface.res['success'] = False
                return False
        return self.boozer_surface.res.get('success', True)

    def _objective_value_from_current_state(self):
        try:
            residual = self._residual_vector_from_current_state()
        except RuntimeError:
            return 1e6
        return float(np.dot(residual, residual))

    def _resolve_phi_shift(self, modB):
        if self.phi_shift is not None:
            return float(self.phi_shift)
        return float(2 * np.pi * self.surface.quadpoints_phi[np.argmax(np.max(modB, axis=1))])

    def _residual_vector_from_current_state(self):
        success = self._ensure_current_boozer_surface()
        if not success:
            raise RuntimeError("Boozer surface solve failed while evaluating the QI residual.")

        self.surface.set_dofs(self.in_surface.get_dofs())
        self.biotsavart.set_points(self.surface.gamma().reshape((-1, 3)))
        nphi_aux = self.surface.quadpoints_phi.size
        ntheta_aux = self.surface.quadpoints_theta.size
        B = self.biotsavart.B().reshape((nphi_aux, ntheta_aux, 3))
        modB = np.sqrt(np.sum(B**2, axis=2))

        phi_shift = self.phi_shift
        if phi_shift is None:
            phi_shift = 2 * np.pi * self.surface.quadpoints_phi[np.argmax(np.max(modB, axis=1))]

        phi_values, alpha_values = _make_qi_sampling_grid(self.surface, self.nphi, self.nalpha, phi_shift)
        modB_lines = _sample_modB_on_boozer_lines(modB, self.surface, phi_values, alpha_values,
                                                  self.boozer_surface.res['iota'], phi_shift)
        return _qi_residual_vector_single_surface(modB_lines, phi_values, self.nBj, self.nphi_out)

    def compute(self):
        success = self._ensure_current_boozer_surface()
        if not success:
            self._J = 1e6
            self._dJ = Derivative({})
            return

        self.surface.set_dofs(self.in_surface.get_dofs())
        surface_points = self.surface.gamma().reshape((-1, 3))
        self.biotsavart.set_points(surface_points)
        self.biotsavart.compute(1)

        nphi_aux = self.surface.quadpoints_phi.size
        ntheta_aux = self.surface.quadpoints_theta.size
        B = self.biotsavart.B().reshape((nphi_aux, ntheta_aux, 3))
        modB = np.sqrt(np.sum(B**2, axis=2))
        modB_safe = np.maximum(modB, 1e-15)

        phi_shift = self._resolve_phi_shift(modB)
        phi_values, alpha_values = _make_qi_sampling_grid(self.surface, self.nphi, self.nalpha, phi_shift)
        iota = self.boozer_surface.res['iota']
        G = self.boozer_surface.res['G']
        modB_lines = _sample_modB_on_boozer_lines(modB, self.surface, phi_values, alpha_values, iota, phi_shift)

        self._J, dJ_dmodB_lines = _qi_objective_and_gradient_from_raw_lines(modB_lines, phi_values, self.nBj, self.nphi_out)

        sampling_data = _build_boozer_line_sampling_linearization(self.surface, phi_values, alpha_values, iota, phi_shift)
        dJ_dmodB_grid, dJ_diota = _pullback_boozer_line_sampling(dJ_dmodB_lines, modB, sampling_data)

        dJ_dB = dJ_dmodB_grid[:, :, None] * B / modB_safe[:, :, None]
        dJ_by_dcoils = self.biotsavart.B_vjp(dJ_dB.reshape((-1, 3)))

        dB_by_dX = self.biotsavart.dB_by_dX().reshape((nphi_aux, ntheta_aux, 3, 3))
        dJ_dX = np.einsum('ijb,ijbc->ijc', dJ_dB, dB_by_dX, optimize=True)
        dJ_dsurface = self.surface.dgamma_by_dcoeff_vjp(dJ_dX)

        booz_surf = self.boozer_surface
        iota, G, P, L, U, dconstraint_dcoils_vjp = _get_boozer_linearization(booz_surf)
        dJ_ds = np.zeros(L.shape[0])
        dJ_dsurface = np.asarray(dJ_dsurface, dtype=float)
        dJ_ds[:dJ_dsurface.size] = dJ_dsurface
        if G is not None:
            dJ_ds[-2] = dJ_diota
        else:
            dJ_ds[-1] = dJ_diota

        adj = forward_backward(P, L, U, dJ_ds)
        adj_times_dg_dcoil = dconstraint_dcoils_vjp(adj, booz_surf, iota, G)
        self._dJ = dJ_by_dcoils - adj_times_dg_dcoil


class Iotas(Optimizable):
    """
    This term returns the rotational transform on a boozer surface as well as its derivative
    with respect to the coil degrees of freedom.
    """

    def __init__(self, boozer_surface):
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[boozer_surface])
        self.boozer_surface = boozer_surface
        self.biotsavart = boozer_surface.biotsavart
        self.recompute_bell()

    def J(self):
        if self._J is None:
            self.compute()
        return self._J

    @derivative_dec
    def dJ(self):
        if self._dJ is None:
            self.compute()
        return self._dJ

    def recompute_bell(self, parent=None):
        self._J = None
        self._dJ = None
        self._dJ_by_dcoefficients = None
        self._dJ_by_dcoilcurrents = None

    def compute(self):
        if self.boozer_surface.need_to_run_code:
            res = self.boozer_surface.res
            res = self.boozer_surface.run_code(res['iota'], G=res['G'])

        self._J = self.boozer_surface.res['iota']

        booz_surf = self.boozer_surface
        iota, G, P, L, U, dconstraint_dcoils_vjp = _get_boozer_linearization(booz_surf)

        dJ_ds = np.zeros(L.shape[0])
        if G is not None:
            # tack on dJ_diota = 1, and  dJ_dG = 0 to the end of dJ_ds
            dJ_ds[-2] = 1.
        else:
            # tack on dJ_diota = 1 to the end of dJ_ds
            dJ_ds[-1] = 1.

        adj = forward_backward(P, L, U, dJ_ds)

        adj_times_dg_dcoil = dconstraint_dcoils_vjp(adj, booz_surf, iota, G)
        self._dJ = -1.*adj_times_dg_dcoil


class BoozerResidual(Optimizable):
    r"""
    This term returns the Boozer residual penalty term

    .. math::
       J = \int_0^{1/n_{\text{fp}}} \int_0^1 \| \mathbf r \|^2 ~d\theta ~d\varphi + w (\text{label.J()-boozer_surface.constraint_weight})^2.

    where

    .. math::
        \mathbf r = \frac{1}{\|\mathbf B\|}[G\mathbf B_\text{BS}(\mathbf x) - ||\mathbf B_\text{BS}(\mathbf x)||^2  (\mathbf x_\varphi + \iota  \mathbf x_\theta)]

    """

    def __init__(self, boozer_surface, bs):
        Optimizable.__init__(self, depends_on=[boozer_surface])
        in_surface = boozer_surface.surface
        self.boozer_surface = boozer_surface

        # same number of points as on the solved surface
        phis = in_surface.quadpoints_phi
        thetas = in_surface.quadpoints_theta

        s = SurfaceXYZTensorFourier(mpol=in_surface.mpol, ntor=in_surface.ntor, stellsym=in_surface.stellsym, nfp=in_surface.nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        s.set_dofs(in_surface.get_dofs())

        self.constraint_weight = boozer_surface.constraint_weight
        self.in_surface = in_surface
        self.surface = s
        self.biotsavart = bs
        self.recompute_bell()

    def J(self):
        """
        Return the value of the penalty function.
        """

        if self._J is None:
            self.compute()
        return self._J

    @derivative_dec
    def dJ(self):
        """
        Return the derivative of the penalty function with respect to the coil degrees of freedom.
        """

        if self._dJ is None:
            self.compute()
        return self._dJ

    def recompute_bell(self, parent=None):
        self._J = None
        self._dJ = None

    def compute(self):
        if self.boozer_surface.need_to_run_code:
            res = self.boozer_surface.res
            res = self.boozer_surface.run_code(res['iota'], G=res['G'])

        self.surface.set_dofs(self.in_surface.get_dofs())
        self.biotsavart.set_points(self.surface.gamma().reshape((-1, 3)))

        nphi = self.surface.quadpoints_phi.size
        ntheta = self.surface.quadpoints_theta.size
        num_points = 3 * nphi * ntheta

        # compute J
        surface = self.surface
        iota = self.boozer_surface.res['iota']
        G = self.boozer_surface.res['G']
        r, J = boozer_surface_residual(surface, iota, G, self.biotsavart, derivatives=1, weight_inv_modB=self.boozer_surface.res['weight_inv_modB'])
        rtil = np.concatenate((r/np.sqrt(num_points), [np.sqrt(self.constraint_weight)*(self.boozer_surface.label.J()-self.boozer_surface.targetlabel)]))
        self._J = 0.5*np.sum(rtil**2)

        booz_surf = self.boozer_surface
        iota, G, P, L, U, dconstraint_dcoils_vjp = _get_boozer_linearization(booz_surf)

        dJ_by_dB = self.dJ_by_dB()
        dJ_by_dcoils = self.biotsavart.B_vjp(dJ_by_dB)

        # dJ_diota, dJ_dG  to the end of dJ_ds are on the end
        dl = np.zeros((J.shape[1],))
        dlabel_dsurface = self.boozer_surface.label.dJ_by_dsurfacecoefficients()
        dl[:dlabel_dsurface.size] = dlabel_dsurface
        Jtil = np.concatenate((J/np.sqrt(num_points), np.sqrt(self.constraint_weight) * dl[None, :]), axis=0)
        dJ_ds = Jtil.T@rtil

        adj = forward_backward(P, L, U, dJ_ds)

        adj_times_dg_dcoil = dconstraint_dcoils_vjp(adj, booz_surf, iota, G)
        self._dJ = dJ_by_dcoils - adj_times_dg_dcoil

    def dJ_by_dB(self):
        """
        Return the partial derivative of the objective with respect to the magnetic field
        """

        surface = self.surface
        res = self.boozer_surface.res
        nphi = self.surface.quadpoints_phi.size
        ntheta = self.surface.quadpoints_theta.size
        num_points = 3 * nphi * ntheta
        r, r_dB = boozer_surface_residual_dB(surface, self.boozer_surface.res['iota'], self.boozer_surface.res['G'], self.biotsavart, derivatives=0, weight_inv_modB=res['weight_inv_modB'])

        r /= np.sqrt(num_points)
        r_dB /= np.sqrt(num_points)

        dJ_by_dB = r[:, None]*r_dB
        dJ_by_dB = np.sum(dJ_by_dB.reshape((-1, 3, 3)), axis=1)
        return dJ_by_dB


def boozer_surface_dexactresidual_dcoils_dcurrents_vjp(lm, booz_surf, iota, G):
    r"""
    For a given surface with points :math:`x` on it, this function computes the
    vector-Jacobian product of:

    .. math::
        \lambda^T \frac{d\mathbf{r}}{d\text{coils   }} &= [G\lambda - 2\lambda\|\mathbf B(\mathbf x)\| (\mathbf{x}_\varphi + \iota \mathbf{x}_\theta) ]^T \frac{d\mathbf B}{d\text{coils}} \\ 
        \lambda^T \frac{d\mathbf{r}}{d\text{currents}} &= [G\lambda - 2\lambda\|\mathbf B(\mathbf x)\| (\mathbf{x}_\varphi + \iota \mathbf{x}_\theta) ]^T \frac{d\mathbf B}{d\text{currents}}

    where :math:`\mathbf{r}` is the Boozer residual.

    Args:
        lm: adjoint variable,
        booz_surf: boozer surface,
        iota: rotational transform on the boozer surface,
        G: constant on boozer surface,
    """
    surface = booz_surf.surface
    biotsavart = booz_surf.biotsavart

    # G must be provided here
    assert G is not None

    res, dres_dB = boozer_surface_residual_dB(surface, iota, G, biotsavart)
    dres_dB = dres_dB.reshape((-1, 3, 3))

    lm_label = lm[-1]
    lmask = np.zeros(booz_surf.res["mask"].shape)
    lmask[booz_surf.res["mask"]] = lm[:-1]
    lm_cons = lmask.reshape((-1, 3))

    lm_times_dres_dB = np.sum(lm_cons[:, :, None] * dres_dB, axis=1).reshape((-1, 3))
    lm_times_dres_dcoils = biotsavart.B_vjp(lm_times_dres_dB)
    lm_times_dlabel_dcoils = lm_label*booz_surf.label.dJ(partials=True)(biotsavart, as_derivative=True)

    return lm_times_dres_dcoils+lm_times_dlabel_dcoils


def boozer_surface_dlsqgrad_dcoils_vjp(lm, booz_surf, iota, G, weight_inv_modB=True):
    r"""
    For a given surface with points x on it, this function computes the
    vector-Jacobian product of \lm^T * dlsqgrad_dcoils, \lm^T * dlsqgrad_dcurrents:
    lm^T dresidual_dcoils    = lm^T [dr_dsurface]^T[dr_dcoils]    + sum r_i lm^T d2ri_dsdc
    lm^T dresidual_dcurrents = lm^T [dr_dsurface]^T[dr_dcurrents] + sum r_i lm^T d2ri_dsdcurrents

    G is known for exact boozer surfaces, so if G=None is passed, then that
    value is used instead.
    """

    surface = booz_surf.surface
    biotsavart = booz_surf.biotsavart
    nphi = surface.quadpoints_phi.size
    ntheta = surface.quadpoints_theta.size
    num_points = 3 * nphi * ntheta
    # r, dr_dB, J, d2residual_dsurfacedB, d2residual_dsurfacedgradB
    boozer = boozer_surface_residual_dB(surface, iota, G, biotsavart, derivatives=1, weight_inv_modB=weight_inv_modB)
    r = boozer[0]/np.sqrt(num_points)
    dr_dB = boozer[1].reshape((-1, 3, 3))/np.sqrt(num_points)
    dr_ds = boozer[2]/np.sqrt(num_points)
    d2r_dsdB = boozer[3]/np.sqrt(num_points)
    d2r_dsdgradB = boozer[4]/np.sqrt(num_points)

    v1 = np.sum(np.sum(lm[:, None]*dr_ds.T, axis=0).reshape((-1, 3, 1)) * dr_dB, axis=1)
    v2 = np.sum(r.reshape((-1, 3, 1))*np.sum(lm[None, None, :]*d2r_dsdB, axis=-1).reshape((-1, 3, 3)), axis=1)
    v3 = np.sum(r.reshape((-1, 3, 1, 1))*np.sum(lm[None, None, None, :]*d2r_dsdgradB, axis=-1).reshape((-1, 3, 3, 3)), axis=1)
    dres_dcoils = biotsavart.B_and_dB_vjp(v1+v2, v3)
    return dres_dcoils[0]+dres_dcoils[1]


def boozer_surface_residual_dB(surface, iota, G, biotsavart, derivatives=0, weight_inv_modB=False):
    """
    For a given surface with points x on it, this function computes the
    differentiated residual
       d/dB[ G*B_BS(x) - ||B_BS(x)||^2 * (x_phi + iota * x_theta) ]
    as well as the derivatives of this residual with respect to surface dofs,
    iota, and G.
    G is known for exact boozer surfaces, so if G=None is passed, then that
    value is used instead.
    """

    user_provided_G = G is not None
    if not user_provided_G:
        G = 2. * np.pi * np.sum(np.abs([c.current.get_value() for c in biotsavart.coils])) * (4 * np.pi * 10**(-7) / (2 * np.pi))

    x = surface.gamma()
    xphi = surface.gammadash1()
    xtheta = surface.gammadash2()
    nphi = x.shape[0]
    ntheta = x.shape[1]

    xsemiflat = x.reshape((x.size//3, 3)).copy()

    biotsavart.set_points(xsemiflat)

    B = biotsavart.B().reshape((nphi, ntheta, 3))

    tang = xphi + iota * xtheta
    residual = G*B - np.sum(B**2, axis=2)[..., None] * tang

    GI = np.eye(3, 3) * G
    dresidual_dB = GI[None, None, :, :] - 2. * tang[:, :, :, None] * B[:, :, None, :]

    if weight_inv_modB:
        B2 = np.sum(B**2, axis=2)
        modB = np.sqrt(B2)
        w = 1./modB
        dw_dB = -B/B2[:, :, None]**1.5
        rtil = w[:, :, None] * residual
        drtil_dB = residual[:, :, :, None] * dw_dB[:, :, None, :] + dresidual_dB * w[:, :, None, None]
    else:
        rtil = residual.copy()
        drtil_dB = dresidual_dB.copy()

    rtil_flattened = rtil.reshape((nphi*ntheta*3, ))
    drtil_dB_flattened = drtil_dB.reshape((nphi*ntheta*3, 3))

    if derivatives == 0:
        return rtil_flattened, drtil_dB_flattened

    dx_dc = surface.dgamma_by_dcoeff()
    dxphi_dc = surface.dgammadash1_by_dcoeff()
    dxtheta_dc = surface.dgammadash2_by_dcoeff()
    nsurfdofs = dx_dc.shape[-1]

    dB_by_dX = biotsavart.dB_by_dX().reshape((nphi, ntheta, 3, 3))
    dB_dc = np.einsum('ijkl,ijkm->ijlm', dB_by_dX, dx_dc, optimize=True)
    dtang_dc = dxphi_dc + iota * dxtheta_dc
    dresidual_dc = G*dB_dc \
        - 2*np.sum(B[..., None]*dB_dc, axis=2)[:, :, None, :] * tang[..., None] \
        - np.sum(B**2, axis=2)[..., None, None] * dtang_dc
    dresidual_diota = -np.sum(B**2, axis=2)[..., None] * xtheta

    d2residual_dcdB = -2*dB_dc[:, :, None, :, :] * tang[:, :, :, None, None] - 2*B[:, :, None, :, None] * dtang_dc[:, :, :, None, :]
    d2residual_diotadB = -2.*B[:, :, None, :] * xtheta[:, :, :, None]
    d2residual_dcdgradB = -2.*B[:, :, None, None, :, None]*dx_dc[:, :, None, :, None, :]*tang[:, :, :, None, None, None]
    idx = np.arange(3)
    d2residual_dcdgradB[:, :, idx, :, idx, :] += dx_dc * G

    if weight_inv_modB:
        dB2_dc = 2*np.einsum('ijk,ijkl->ijl', B, dB_dc, optimize=True)
        dmodB_dc = 0.5*dB2_dc/modB[:, :, None]
        dw_dc = -dmodB_dc/B2[:, :, None]

        d2w_dcdB = -(dB_dc * B2[:, :, None, None]**1.5 - 1.5*dB2_dc[:, :, None, :]*modB[:, :, None, None]*B[:, :, :, None])/B2[:, :, None, None]**3
        d2w_dcdgradB = dw_dB[:, :, None, :, None] * dx_dc[:, :, :, None, :]

        drtil_dc = dresidual_dc * w[:, :, None, None] + dw_dc[:, :, None, :] * residual[..., None]
        drtil_diota = w[:, :, None] * dresidual_diota
        d2rtil_dcdB = dresidual_dc[:, :, :, None, :]*dw_dB[:, :, None, :, None]  \
            + dresidual_dB[:, :, :, :, None]*dw_dc[:, :, None, None, :] \
            + d2residual_dcdB*w[:, :, None, None, None] \
            + residual[:, :, :, None, None]*d2w_dcdB[:, :, None, :, :]
        d2rtil_diotadB = dw_dB[:, :, None, :]*dresidual_diota[:, :, :, None] + w[:, :, None, None]*d2residual_diotadB
        d2rtil_dcdgradB = d2w_dcdgradB[:, :, None, :, :, :]*residual[:, :, :, None, None, None] + d2residual_dcdgradB*w[:, :, None, None, None, None]
    else:
        drtil_dc = dresidual_dc.copy()
        drtil_diota = dresidual_diota.copy()
        d2rtil_dcdB = d2residual_dcdB.copy()
        d2rtil_diotadB = d2residual_diotadB.copy()
        d2rtil_dcdgradB = d2residual_dcdgradB.copy()

    drtil_dc_flattened = drtil_dc.reshape((nphi*ntheta*3, nsurfdofs))
    drtil_diota_flattened = drtil_diota.reshape((nphi*ntheta*3, 1))
    d2rtil_dcdB_flattened = d2rtil_dcdB.reshape((nphi*ntheta*3, 3, nsurfdofs))
    d2rtil_diotadB_flattened = d2rtil_diotadB.reshape((nphi*ntheta*3, 3, 1))
    d2rtil_dcdgradB_flattened = d2rtil_dcdgradB.reshape((nphi*ntheta*3, 3, 3, nsurfdofs))
    d2rtil_diotadgradB_flattened = np.zeros((nphi*ntheta*3, 3, 3, 1))

    if user_provided_G:
        dresidual_dG = B
        d2residual_dGdB = np.ones((nphi*ntheta, 3, 3))
        d2residual_dGdB[:, :, :] = np.eye(3)[None, :, :]
        d2residual_dGdB = d2residual_dGdB.reshape((nphi, ntheta, 3, 3))
        d2residual_dGdgradB = np.zeros((nphi, ntheta, 3, 3, 3))

        if weight_inv_modB:
            drtil_dG = dresidual_dG * w[:, :, None]
            d2rtil_dGdB = d2residual_dGdB * w[:, :, None, None] + dw_dB[:, :, None, :]*dresidual_dG[:, :, :, None]
            d2rtil_dGdgradB = d2residual_dGdgradB.copy()
        else:
            drtil_dG = dresidual_dG.copy()
            d2rtil_dGdB = d2residual_dGdB.copy()
            d2rtil_dGdgradB = d2residual_dGdgradB.copy()

        drtil_dG_flattened = drtil_dG.reshape((nphi*ntheta*3, 1))
        d2rtil_dGdB_flattened = d2rtil_dGdB.reshape((nphi*ntheta*3, 3, 1))
        d2rtil_dGdgradB_flattened = d2rtil_dGdgradB.reshape((nphi*ntheta*3, 3, 3, 1))

        J = np.concatenate((drtil_dc_flattened, drtil_diota_flattened, drtil_dG_flattened), axis=1)
        d2rtil_dsurfacedB = np.concatenate((d2rtil_dcdB_flattened,
                                            d2rtil_diotadB_flattened,
                                            d2rtil_dGdB_flattened), axis=-1)
        d2rtil_dsurfacedgradB = np.concatenate((d2rtil_dcdgradB_flattened,
                                                d2rtil_diotadgradB_flattened,
                                                d2rtil_dGdgradB_flattened), axis=-1)
    else:
        J = np.concatenate((drtil_dc_flattened, drtil_diota_flattened), axis=1)
        d2rtil_dsurfacedB = np.concatenate((d2rtil_dcdB_flattened, d2rtil_diotadB_flattened), axis=-1)
        d2rtil_dsurfacedgradB = np.concatenate((d2rtil_dcdgradB_flattened, d2rtil_diotadgradB_flattened), axis=-1)

    if derivatives == 1:
        return rtil_flattened, drtil_dB_flattened, J, d2rtil_dsurfacedB, d2rtil_dsurfacedgradB
