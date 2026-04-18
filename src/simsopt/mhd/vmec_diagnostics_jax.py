# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the MIT License

"""JAX-backed VMEC diagnostics.

This module keeps the SIMSOPT-facing API surface for VMEC-JAX diagnostics, but
the actual quasisymmetry implementation now lives only in :mod:`vmec_jax`.
"""

from __future__ import annotations

try:
    from vmec_jax.quasisymmetry import (
        quasisymmetry_ratio_residual_from_state as _qs_from_state,
        quasisymmetry_ratio_residual_from_wout as _qs_from_wout,
    )
except Exception as exc:  # pragma: no cover - optional dependency
    _import_error = exc
    _qs_from_state = None
    _qs_from_wout = None
else:
    _import_error = None


__all__ = [
    "QuasisymmetryRatioResidualJax",
    "quasisymmetry_ratio_residual_from_wout_jax",
    "quasisymmetry_ratio_residual_from_state_jax",
]


def _require_vmec_jax_quasisymmetry():
    if _qs_from_state is None or _qs_from_wout is None:
        raise ImportError(
            "vmec_diagnostics_jax requires vmec_jax with the quasisymmetry module available."
        ) from _import_error


def quasisymmetry_ratio_residual_from_wout_jax(
    wout,
    *,
    surfaces,
    helicity_m: int = 1,
    helicity_n: int = 0,
    weights=None,
    ntheta: int = 63,
    nphi: int = 64,
):
    """Evaluate the VMEC-only quasisymmetry residual from a wout-like object."""
    _require_vmec_jax_quasisymmetry()
    return _qs_from_wout(
        wout,
        surfaces=surfaces,
        helicity_m=helicity_m,
        helicity_n=helicity_n,
        weights=weights,
        ntheta=ntheta,
        nphi=nphi,
    )


def quasisymmetry_ratio_residual_from_state_jax(
    vmec,
    state,
    *,
    surfaces,
    helicity_m: int = 1,
    helicity_n: int = 0,
    weights=None,
    ntheta: int = 63,
    nphi: int = 64,
):
    """Evaluate the VMEC-only QS residual directly from a solved ``VmecJax`` state."""
    _require_vmec_jax_quasisymmetry()
    vmec._ensure_context()
    context = vmec.get_context()
    return _qs_from_state(
        state=state,
        static=vmec.get_static(),
        indata=vmec._indata_raw,
        signgs=int(vmec._signgs),
        flux_local=context.flux,
        prof_local={"pressure": context.pressure},
        pressure_local=context.pressure,
        surfaces=surfaces,
        helicity_m=helicity_m,
        helicity_n=helicity_n,
        weights=weights,
        ntheta=ntheta,
        nphi=nphi,
    )


class QuasisymmetryRatioResidualJax:
    """JAX-backed VMEC-only quasisymmetry residual for :class:`VmecJax`."""

    def __init__(
        self,
        vmec,
        surfaces,
        helicity_m: int = 1,
        helicity_n: int = 0,
        weights=None,
        ntheta: int = 63,
        nphi: int = 64,
    ) -> None:
        self.vmec = vmec
        self.surfaces = list(surfaces)
        self.helicity_m = int(helicity_m)
        self.helicity_n = int(helicity_n)
        self.weights = None if weights is None else list(weights)
        self.ntheta = int(ntheta)
        self.nphi = int(nphi)

    def compute_from_wout(self, wout):
        return quasisymmetry_ratio_residual_from_wout_jax(
            wout,
            surfaces=self.surfaces,
            helicity_m=self.helicity_m,
            helicity_n=self.helicity_n,
            weights=self.weights,
            ntheta=self.ntheta,
            nphi=self.nphi,
        )

    def compute_from_state(self, state):
        return quasisymmetry_ratio_residual_from_state_jax(
            self.vmec,
            state,
            surfaces=self.surfaces,
            helicity_m=self.helicity_m,
            helicity_n=self.helicity_n,
            weights=self.weights,
            ntheta=self.ntheta,
            nphi=self.nphi,
        )

    def compute(self, x_free):
        return self.compute_from_wout(self.vmec.get_wout(x_free))

    def residuals_from_wout(self, wout):
        return self.compute_from_wout(wout)["residuals1d"]

    def residuals_from_state(self, state):
        return self.compute_from_state(state)["residuals1d"]

    def residuals(self, x_free):
        return self.compute(x_free)["residuals1d"]

    def profile_from_wout(self, wout):
        return self.compute_from_wout(wout)["profile"]

    def profile_from_state(self, state):
        return self.compute_from_state(state)["profile"]

    def profile(self, x_free):
        return self.compute(x_free)["profile"]

    def total_from_wout(self, wout):
        return self.compute_from_wout(wout)["total"]

    def total_from_state(self, state):
        return self.compute_from_state(state)["total"]

    def total(self, x_free):
        return self.compute(x_free)["total"]
