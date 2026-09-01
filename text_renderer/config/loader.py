import os
import traceback
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import List

from text_renderer.config.models import AppCfg, GeneratorCfg


def get_cfg(config_file: str) -> List[GeneratorCfg]:
    """Load a config file and return its generator configurations."""
    return get_app_cfg(config_file).generators


def get_app_cfg(config_file: str, selected_languages=None) -> AppCfg:
    """Load either a YAML application config or a legacy Python config."""
    suffix = Path(config_file).suffix.lower()
    if suffix in {".yaml", ".yml"}:
        from text_renderer.config.yaml_config import load_yaml_app_config

        return load_yaml_app_config(config_file, selected_languages)

    if selected_languages:
        raise ValueError("--language is only supported by version 2 YAML configs")

    print(os.path.join(os.getcwd(), config_file))
    module = import_module_from_file(config_file)
    if module is None:
        raise RuntimeError(f"Load config module failed: {config_file}")
    cfgs = getattr(module, "configs", None)
    if cfgs is None:
        raise RuntimeError(f"Load configs failed: {config_file}")

    assert all(
        isinstance(cfg, GeneratorCfg) for cfg in cfgs
    ), "Please make sure all items in configs is GeneratorCfg"

    return AppCfg(generators=cfgs)


def import_module_from_file(full_path_to_module):
    """Import a Python configuration module from a file path."""
    try:
        module_path = Path(full_path_to_module)
        spec = spec_from_file_location(module_path.stem, module_path)
        if spec is None or spec.loader is None:
            return None
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:
        traceback.print_exc()
        return None
