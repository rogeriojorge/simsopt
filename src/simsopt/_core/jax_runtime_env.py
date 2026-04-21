"""Helpers for setting JAX/XLA runtime environment defaults early."""

from __future__ import annotations

import os


def _ensure_min_numeric_env(name: str, minimum: int) -> None:
    value = os.environ.get(name)
    if value is None:
        os.environ[name] = str(minimum)
        return
    try:
        parsed = int(str(value).strip())
    except Exception:
        os.environ[name] = str(minimum)
        return
    if parsed < minimum:
        os.environ[name] = str(minimum)


def configure_jax_runtime_env() -> None:
    """Set conservative JAX/XLA defaults before importing JAX.

    This raises weak log-filter settings to a warning-suppression floor and
    disables vmec_jax's persistent compilation cache by default for SIMSOPT
    workflows, since current JAX releases can flood stderr with repeated
    ``pjrt_executable.cc`` warnings on cache hits. Users can opt back in by
    setting either cache environment variable explicitly before import.
    """

    _ensure_min_numeric_env("TF_CPP_MIN_LOG_LEVEL", 2)
    _ensure_min_numeric_env("ABSL_MIN_LOG_LEVEL", 2)
    _ensure_min_numeric_env("GLOG_minloglevel", 2)

    if "VMEC_JAX_COMPILATION_CACHE_DIR" not in os.environ and "JAX_COMPILATION_CACHE_DIR" not in os.environ:
        os.environ["VMEC_JAX_COMPILATION_CACHE_DIR"] = "disabled"

