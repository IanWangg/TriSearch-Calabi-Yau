from __future__ import annotations

"""Centralized CYTools configuration."""

import os
from pathlib import Path

_DEFAULT_MOSEK_LICENSE_PATH = "/home/yiranwang/mosek/mosek.lic"
MOSEK_LICENSE_PATH = os.environ.get("MOSEKLM_LICENSE_FILE", _DEFAULT_MOSEK_LICENSE_PATH)
REGULARITY_BACKEND = os.environ.get("CYTOOLS_REGULARITY_BACKEND", "mosek")

_configured = False


def configure_cytools() -> None:
    """Configure CYTools once, using environment variables when available."""
    global _configured
    if _configured:
        return
    _configured = True

    if MOSEK_LICENSE_PATH and Path(MOSEK_LICENSE_PATH).exists():
        os.environ.setdefault("MOSEKLM_LICENSE_FILE", MOSEK_LICENSE_PATH)

    try:
        from cytools import config

        if MOSEK_LICENSE_PATH and Path(MOSEK_LICENSE_PATH).exists():
            config.set_mosek_path(MOSEK_LICENSE_PATH)
        enable = getattr(config, "enable_experimental_features", None)
        if callable(enable):
            enable()
    except ModuleNotFoundError:
        pass
