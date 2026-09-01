"""Public configuration API.

Configuration data models live in :mod:`models`; file loading lives in
:mod:`loader`. Importing from ``text_renderer.config`` remains the supported
public interface.
"""

from text_renderer.config.loader import get_app_cfg, get_cfg, import_module_from_file
from text_renderer.config.models import (
    AppCfg,
    FixedPerspectiveTransformCfg,
    FixedTextColorCfg,
    GeneratorCfg,
    GrayStyleCfg,
    NormPerspectiveTransformCfg,
    PerspectiveTransformCfg,
    RenderCfg,
    RuntimeCfg,
    SafeTextColorCfg,
    StorageCfg,
    TextColorCfg,
    UniformPerspectiveTransformCfg,
)

__all__ = [
    "AppCfg",
    "FixedPerspectiveTransformCfg",
    "FixedTextColorCfg",
    "GeneratorCfg",
    "GrayStyleCfg",
    "NormPerspectiveTransformCfg",
    "PerspectiveTransformCfg",
    "RenderCfg",
    "RuntimeCfg",
    "SafeTextColorCfg",
    "StorageCfg",
    "TextColorCfg",
    "UniformPerspectiveTransformCfg",
    "get_app_cfg",
    "get_cfg",
    "import_module_from_file",
]
