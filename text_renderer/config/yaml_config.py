import math
import os
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import yaml

from text_renderer.config import (
    AppCfg,
    FixedPerspectiveTransformCfg,
    FixedTextColorCfg,
    GeneratorCfg,
    GrayStyleCfg,
    NormPerspectiveTransformCfg,
    RenderCfg,
    RuntimeCfg,
    SafeTextColorCfg,
    StorageCfg,
    UniformPerspectiveTransformCfg,
)
from text_renderer.corpus import EnumCorpus, EnumCorpusCfg
from text_renderer.effect import (
    DropoutHorizontal,
    DropoutRand,
    DropoutVertical,
    Effects,
    Emboss,
    MotionBlur,
    Padding,
)


def _mapping(value: Any, section: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"YAML section '{section}' must be a mapping")
    return dict(value)


def _validate_keys(
    data: Mapping[str, Any], section: str, allowed: Iterable[str], required=()
) -> None:
    allowed = set(allowed)
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(
            f"Unknown keys in YAML section '{section}': {sorted(unknown)}"
        )
    missing = set(required) - set(data)
    if missing:
        raise ValueError(
            f"Missing keys in YAML section '{section}': {sorted(missing)}"
        )


def _path(value: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(value))))


def _pair(value: Sequence, name: str, cast=float) -> Tuple:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"'{name}' must contain exactly two values")
    result = cast(value[0]), cast(value[1])
    if result[0] > result[1]:
        raise ValueError(f"'{name}' minimum must not exceed its maximum")
    return result


def _clean_line(
    line: str,
    charset,
    strip: bool,
    normalize_spaces: bool,
    remove_punctuation: bool,
    allowed_punctuation: str,
) -> str:
    if strip:
        line = line.strip()
    if charset is not None:
        line = "".join(char for char in line if char in charset)
    if remove_punctuation:
        line = "".join(
            char
            for char in line
            if not unicodedata.category(char).startswith("P")
            or char in allowed_punctuation
        )
    if normalize_spaces:
        line = " ".join(line.split())
    return line.strip() if strip else line


def _apply_length_policy(
    text: str, min_length: int, max_length: int, long_text: str
) -> List[str]:
    if max_length <= 0 or len(text) <= max_length:
        candidates = [text]
    elif long_text == "split":
        candidates = [
            text[start : start + max_length].strip()
            for start in range(0, len(text), max_length)
        ]
    elif long_text == "truncate":
        candidates = [text[:max_length].strip()]
    elif long_text == "drop":
        candidates = []
    else:
        raise ValueError("corpus.long_text must be split, truncate, or drop")
    return [item for item in candidates if len(item) >= min_length]


def _load_corpus_items(data: Mapping[str, Any], num_images: int) -> List[str]:
    path = _path(data["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Corpus file does not exist: {path}")

    charset = None
    charset_path = data.get("charset_path")
    if charset_path:
        charset_file = _path(charset_path)
        if not charset_file.is_file():
            raise FileNotFoundError(f"Charset file does not exist: {charset_file}")
        charset = set(charset_file.read_text(encoding="utf-8-sig").splitlines())
        charset.add(" ")

    min_length = int(data.get("min_length", 1))
    max_length = int(data.get("max_length", -1))
    long_text = str(data.get("long_text", "split"))
    if min_length < 0:
        raise ValueError("corpus.min_length must be non-negative")
    if max_length != -1 and max_length < min_length:
        raise ValueError("corpus.max_length must be -1 or >= corpus.min_length")
    if long_text not in {"split", "truncate", "drop"}:
        raise ValueError("corpus.long_text must be split, truncate, or drop")

    items = []
    with path.open("r", encoding="utf-8-sig") as corpus_file:
        for raw_line in corpus_file:
            text = _clean_line(
                raw_line.rstrip("\r\n"),
                charset,
                bool(data.get("strip", True)),
                bool(data.get("normalize_spaces", True)),
                bool(data.get("remove_punctuation", True)),
                str(data.get("allowed_punctuation", "！？")),
            )
            if not text:
                continue
            items.extend(
                _apply_length_policy(
                    text,
                    min_length,
                    max_length,
                    long_text,
                )
            )

    if not items:
        raise ValueError(f"Corpus is empty after filtering: {path}")

    capacity_multiplier = float(data.get("capacity_multiplier", 1.0))
    if capacity_multiplier <= 0:
        raise ValueError("corpus.capacity_multiplier must be > 0")
    target_count = max(num_images, int(math.ceil(num_images * capacity_multiplier)))
    if len(items) < target_count:
        items *= int(math.ceil(target_count / len(items)))
    return items


def _text_color_cfg(data: Mapping[str, Any]):
    strategy = str(data.get("text_color_strategy", "safe")).lower()
    if strategy == "safe":
        cfg = SafeTextColorCfg()
    elif strategy == "fixed":
        cfg = FixedTextColorCfg()
    elif strategy == "gray":
        cfg = GrayStyleCfg()
    else:
        raise ValueError("render.text_color_strategy must be safe, fixed, or gray")

    rgb = data.get("fixed_rgb", [0, 0, 0])
    if not isinstance(rgb, (list, tuple)) or len(rgb) != 3:
        raise ValueError("render.fixed_rgb must contain exactly three values")
    cfg.r, cfg.g, cfg.b = (int(value) for value in rgb)
    cfg.alpha = _pair(data.get("text_alpha_range", [245, 255]), "text_alpha_range", int)
    return cfg


def _perspective_cfg(data: Mapping[str, Any]):
    if not data or not bool(data.get("enabled", False)):
        return None
    kind = str(data.get("type", "uniform")).lower()
    values = {
        "x": float(data.get("x", 10)),
        "y": float(data.get("y", 5)),
        "z": float(data.get("z", 1.5)),
        "scale": int(data.get("scale", 1)),
        "fovy": int(data.get("fovy", 50)),
    }
    classes = {
        "fixed": FixedPerspectiveTransformCfg,
        "uniform": UniformPerspectiveTransformCfg,
        "normal": NormPerspectiveTransformCfg,
    }
    if kind not in classes:
        raise ValueError("perspective.type must be fixed, uniform, or normal")
    return classes[kind](**values)


def _effects_cfg(items: Any):
    if not items:
        return None
    if not isinstance(items, list):
        raise ValueError("render.corpus_effects must be a list")
    registry = {
        "padding": Padding,
        "dropout_rand": DropoutRand,
        "dropout_horizontal": DropoutHorizontal,
        "dropout_vertical": DropoutVertical,
        "emboss": Emboss,
        "motion_blur": MotionBlur,
    }
    effects = []
    for index, raw_effect in enumerate(items):
        effect = _mapping(raw_effect, f"render.corpus_effects[{index}]")
        kind = effect.pop("type", None)
        if kind not in registry:
            raise ValueError(
                f"Unsupported corpus effect '{kind}'. Supported: {sorted(registry)}"
            )
        for key, value in list(effect.items()):
            if isinstance(value, list):
                effect[key] = tuple(value)
        effects.append(registry[kind](**effect))
    return Effects(effects)


def _resolve_output_dir(value: str) -> Path:
    timestamp = datetime.now().strftime("%m%d%H_%M_%S")
    return _path(str(value).replace("{timestamp}", timestamp))


def load_yaml_app_config(config_file: str) -> AppCfg:
    config_path = Path(config_file)
    with config_path.open("r", encoding="utf-8-sig") as stream:
        root = yaml.safe_load(stream) or {}
    root = _mapping(root, "root")
    _validate_keys(
        root,
        "root",
        {"version", "runtime", "storage", "generator", "corpus", "font", "background", "render"},
        {"generator", "corpus", "font", "background"},
    )
    if int(root.get("version", 1)) != 1:
        raise ValueError("Only YAML config version 1 is supported")

    runtime_data = _mapping(root.get("runtime"), "runtime")
    _validate_keys(runtime_data, "runtime", {"num_processes", "log_period"})
    runtime = RuntimeCfg(
        num_processes=int(runtime_data.get("num_processes", 2)),
        log_period=float(runtime_data.get("log_period", 10)),
    )
    if runtime.num_processes < 0:
        raise ValueError("runtime.num_processes must be >= 0")
    if runtime.log_period <= 0:
        raise ValueError("runtime.log_period must be > 0")

    storage_data = _mapping(root.get("storage"), "storage")
    _validate_keys(
        storage_data,
        "storage",
        {"dataset", "jpg_quality", "lmdb_map_size_mb", "lmdb_cache_size", "lmdb_map_increment_mb"},
    )
    storage = StorageCfg(
        dataset=str(storage_data.get("dataset", "img")),
        jpg_quality=int(storage_data.get("jpg_quality", 95)),
        lmdb_map_size_mb=int(storage_data.get("lmdb_map_size_mb", 20480)),
        lmdb_cache_size=int(storage_data.get("lmdb_cache_size", 500)),
        lmdb_map_increment_mb=int(storage_data.get("lmdb_map_increment_mb", 500)),
    )
    if storage.dataset not in {"lmdb", "img"}:
        raise ValueError("storage.dataset must be lmdb or img")
    if not 1 <= storage.jpg_quality <= 100:
        raise ValueError("storage.jpg_quality must be between 1 and 100")
    if min(storage.lmdb_map_size_mb, storage.lmdb_cache_size, storage.lmdb_map_increment_mb) <= 0:
        raise ValueError("LMDB storage values must be positive")

    generator_data = _mapping(root["generator"], "generator")
    _validate_keys(generator_data, "generator", {"num_images", "output_dir"}, {"num_images", "output_dir"})
    num_images = int(generator_data["num_images"])
    if num_images <= 0:
        raise ValueError("generator.num_images must be > 0")

    corpus_data = _mapping(root["corpus"], "corpus")
    _validate_keys(
        corpus_data,
        "corpus",
        {
            "path", "charset_path", "strip", "normalize_spaces", "remove_punctuation",
            "allowed_punctuation", "min_length", "max_length", "long_text",
            "capacity_multiplier", "clip_length", "char_spacing", "join_str",
            "orientation", "rotate_characters",
        },
        {"path"},
    )
    items = _load_corpus_items(corpus_data, num_images)

    font_data = _mapping(root["font"], "font")
    _validate_keys(font_data, "font", {"dir", "size"}, {"dir", "size"})
    font_dir = _path(font_data["dir"])
    if not font_dir.is_dir():
        raise FileNotFoundError(f"Font directory does not exist: {font_dir}")
    font_size = _pair(font_data["size"], "font.size", int)
    if font_size[0] >= font_size[1]:
        raise ValueError("font.size requires min < max")

    background_data = _mapping(root["background"], "background")
    _validate_keys(background_data, "background", {"dir", "preload"}, {"dir"})
    background_dir = _path(background_data["dir"])
    if not background_dir.is_dir():
        raise FileNotFoundError(f"Background directory does not exist: {background_dir}")

    render_data = _mapping(root.get("render"), "render")
    _validate_keys(
        render_data,
        "render",
        {
            "write_mode", "text_color_mode", "text_color_strategy", "fixed_rgb",
            "gray_text_range", "text_alpha_range", "perspective", "corpus_effects",
            "return_bg_and_mask",
        },
    )
    write_mode = str(render_data.get("write_mode", "oneline"))
    if write_mode not in {"oneline", "multiline"}:
        raise ValueError("render.write_mode must be oneline or multiline")
    text_color_mode = str(render_data.get("text_color_mode", "gray"))
    if text_color_mode not in {"gray", "color"}:
        raise ValueError("render.text_color_mode must be gray or color")

    orientation = str(corpus_data.get("orientation", "horizontal"))
    if orientation not in {"horizontal", "vertical"}:
        raise ValueError("corpus.orientation must be horizontal or vertical")
    clip_length = int(corpus_data.get("clip_length", -1))
    if clip_length == 0 or clip_length < -1:
        raise ValueError("corpus.clip_length must be -1 or > 0")
    char_spacing = corpus_data.get("char_spacing", -1)
    if isinstance(char_spacing, list):
        char_spacing = _pair(char_spacing, "corpus.char_spacing", float)

    corpus = EnumCorpus(
        EnumCorpusCfg(
            items=items,
            font_dir=font_dir,
            font_size=font_size,
            clip_length=clip_length,
            char_spacing=char_spacing,
            horizontal=orientation == "horizontal",
            rotate_chars=bool(corpus_data.get("rotate_characters", True)),
            remove_punctuation=bool(corpus_data.get("remove_punctuation", True)),
            allowed_punctuation=str(corpus_data.get("allowed_punctuation", "！？")),
            join_str=str(corpus_data.get("join_str", "")),
            text_color_cfg=_text_color_cfg(render_data),
        )
    )

    perspective_data = _mapping(render_data.get("perspective"), "render.perspective")
    _validate_keys(
        perspective_data,
        "render.perspective",
        {"enabled", "type", "x", "y", "z", "scale", "fovy"},
    )
    render_cfg = RenderCfg(
        corpus=corpus,
        corpus_effects=_effects_cfg(render_data.get("corpus_effects")),
        bg_dir=background_dir,
        pre_load_bg_img=bool(background_data.get("preload", True)),
        perspective_transform=_perspective_cfg(perspective_data),
        return_bg_and_mask=bool(render_data.get("return_bg_and_mask", False)),
        write_mode=write_mode,
        text_color_mode=text_color_mode,
        gray_text_range=_pair(render_data.get("gray_text_range", [0, 12]), "gray_text_range", int),
        text_alpha_range=_pair(render_data.get("text_alpha_range", [245, 255]), "text_alpha_range", int),
    )
    generator = GeneratorCfg(
        num_image=num_images,
        save_dir=_resolve_output_dir(generator_data["output_dir"]),
        render_cfg=render_cfg,
    )
    return AppCfg(generators=[generator], runtime=runtime, storage=storage)
