"""
Synthetic multi-line call-number recognition LMDB generator (streamlined).

Each sample renders one call-number sticker (1-3 lines, one box covering all
lines) directly on a small canvas, expands the text-block edges, and writes
the resulting recognition piece straight into an LMDB. The padded piece and
its conventional four-point box are then rotated 90 degrees counterclockwise.
The expanded area is sampled from the continuous sticker/book background, not
filled with a synthetic border. No intermediate files are created.
Pieces whose tight text box has a middle height/width ratio can be rotated
90 degrees counterclockwise at a configurable probability; very flat pieces
stay upright and very tall pieces rotate deterministically.

Output:
    <out>/   # LMDB, record format identical to lv_tools.lmdbs.workflows.ocr:
             # id-000000001 ... with transcription / points / image / scores / file

Note: third-line ``v`` and ``.v`` prefixes are rendered verbatim and are also
included verbatim in the transcription label.

Usage:
    python -m scripts.generate.callnumber_rec_multiline_lmdb --out F:/dataset/OCR/3.text_rec/synthesis/rec_pieces/multi_language_090116_34_25/call_number_rec_multiline_lmdb --num 1000 --output-scale 0.8
    python -m scripts.generate.callnumber_rec_multiline_lmdb --patterns '[{"lengths":[2,2,2],"weight":1},{"lengths":[2,2,0],"weight":2},{"lengths":[0,0,3],"weight":1}]'
    python -m scripts.generate.callnumber_rec_multiline_lmdb --generator-strategy learned --num 10000
"""

from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
import random
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from scripts.generate import callnumber_det as call_number_renderer
from text_renderer.corpus.call_number_generator import (
    PatternConfig,
    build_call_number_generator,
)
from lv_tools.dataset_io.lmdb_backend import lmdb_2_min_size
from lv_tools.lmdbs.workflows.ocr import (
    LmdbDataset,
)

# Natural context on every side, relative to the rotated text-box height.
_DEFAULT_CONTEXT_EXPANSION_RATIO = 1.0
_DEFAULT_ROTATION_ASPECT_RATIO_THRESHOLD_L = 0.7
_DEFAULT_ROTATION_ASPECT_RATIO_THRESHOLD_R = 1.5
_DEFAULT_PATTERNS = [
    {"lengths": [2, 2, 2], "weight": 1.5},
    {"lengths": [3, 3, 3], "weight": 1.5},
    {"lengths": [2, 1, 1], "weight": 1},
    # {"lengths": [4, 4, 0], "weight": 1},
    # {"lengths": [5, 5, 0], "weight": 1},
    # {"lengths": [6, 6, 0], "weight": 1},
    # {"lengths": [7, 7, 0], "weight": 1},

    {"lengths": [2, 1, 0], "weight": 2.5},
    {"lengths": [2, 0, 2], "weight": 2.5},
    {"lengths": [3, 0, 3], "weight": 2.5}
    ]

# 同一种字体复用的样本数。字体文件加载是最大开销，
# 每样本换字体（pick_font_for_image 预载全部字号）会慢一个数量级。
_SAMPLES_PER_FONT = 20

_font_cache = {}        # {(path, size): font}，仅当前字体路径的字号，切换路径时清空
_current_font_path = None


def _scale_pixels(v):
    """Scale a pixel value by the current image scale (call_number_renderer._IMAGE_SCALE_FACTOR)."""
    return call_number_renderer._scale_pixels(v)


def _select_font_for_batch():
    """每 _SAMPLES_PER_FONT 个样本换一种字体，字体对象按 (路径, 字号) 持久缓存。

    不调用 call_number_renderer.pick_font_for_image（其按当前 scale 预载 13~24 个字号且每次清空
    缓存），而是让 call_number_renderer.get_font 按需加载到本脚本的持久缓存里。
    """
    global _current_font_path
    path = random.choice(call_number_renderer._FONT_PATHS) if call_number_renderer._FONT_PATHS else None
    if path != _current_font_path:
        _current_font_path = path
        _font_cache.clear()
        call_number_renderer._CURRENT_FONT_PATH = path
        call_number_renderer._FONT_CACHE = _font_cache


def _paper_fill():
    """Return a neutral, slightly tinted paper color instead of pure white."""
    base = random.randint(202, 246)
    tint = random.choice(
        [
            (5, 2, -3),   # warm/aged paper
            (-3, 1, 5),   # cool paper
            (2, 2, 2),    # neutral gray paper
        ]
    )
    return tuple(max(0, min(255, base + value)) for value in tint) + (255,)


def _add_surface_texture(img):
    """Add subtle illumination and sensor/paper variation to flat fills."""
    arr = np.asarray(img, dtype=np.int16)
    height, width = arr.shape[:2]
    x_gradient = np.linspace(0, random.uniform(-7, 7), width, dtype=np.float32)
    y_gradient = np.linspace(0, random.uniform(-7, 7), height, dtype=np.float32)
    illumination = y_gradient[:, None] + x_gradient[None, :]
    grain = np.random.normal(0, random.uniform(0.8, 2.4), (height, width, 1))
    arr = np.clip(arr + illumination[:, :, None] + grain, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _compute_crop_and_text_polygon(text_box, image_size, side_ratio):
    """Expand into the rendered background and return the relative text box."""
    x0, y0, x1, y1 = text_box
    width, height = image_size
    text_x0 = max(0, int(np.floor(x0)))
    text_y0 = max(0, int(np.floor(y0)))
    text_x1 = min(width, int(np.ceil(x1)))
    text_y1 = min(height, int(np.ceil(y1)))
    text_w = text_x1 - text_x0

    # After CCW rotation the text-box height equals its current width.
    side = max(0, int(round(text_w * side_ratio)))
    crop_x0 = max(0, text_x0 - side)
    crop_y0 = max(0, text_y0 - side)
    crop_x1 = min(width, text_x1 + side)
    crop_y1 = min(height, text_y1 + side)

    # Conventional order before rotation: LT -> RT -> RB -> LB.
    points = [
        [text_x0 - crop_x0, text_y0 - crop_y0],
        [text_x1 - crop_x0, text_y0 - crop_y0],
        [text_x1 - crop_x0, text_y1 - crop_y0],
        [text_x0 - crop_x0, text_y1 - crop_y0],
    ]
    return (crop_x0, crop_y0, crop_x1, crop_y1), points


def _rotate_90_ccw_with_points(image, points):
    """Rotate an image and its boundary-coordinate polygon 90 degrees CCW."""
    source_width = image.shape[1]
    rotated_points = [[y, source_width - x] for x, y in points]
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), rotated_points


def _should_rotate_for_aspect_ratio(
    text_box_aspect_ratio,
    threshold_l,
    threshold_r,
    rotation_probability,
):
    """Apply the flat/random/tall rotation policy to h/w."""
    if text_box_aspect_ratio <= threshold_l:
        return False
    if text_box_aspect_ratio >= threshold_r:
        return True
    return random.random() < rotation_probability


def _label_from_lines(lines):
    """Build the transcription from exactly the text that was rendered."""
    return " ".join(lines)


def _current_font_file_name():
    """Return the filename of the font selected for the current batch."""
    return None if _current_font_path is None else Path(_current_font_path).name


def generate_and_write_sample(
    writer,
    sample_index,
    call_number_generator,
    scale_range=(1.8, 3.4),
    side_ratio=_DEFAULT_CONTEXT_EXPANSION_RATIO,
    rotation_aspect_ratio_threshold_l=_DEFAULT_ROTATION_ASPECT_RATIO_THRESHOLD_L,
    rotation_aspect_ratio_threshold_r=_DEFAULT_ROTATION_ASPECT_RATIO_THRESHOLD_R,
    rotation_probability=0.30,
    output_scale=0.5,
    noise=False,
):
    """Render one sticker and write the multi-line recognition piece to the LMDB."""
    if sample_index % _SAMPLES_PER_FONT == 0:
        _select_font_for_batch()
    # 随机画布尺度：模拟不同分辨率来源的检测切片
    call_number_renderer._IMAGE_SCALE_FACTOR = random.uniform(*scale_range)

    lines = call_number_generator.generate()

    # 贴纸尺寸：与检测数据生成时一致
    longest_line_length = max(map(len, lines))
    minimum_text_width = _scale_pixels(longest_line_length * 12 + 6)
    label_w = max(
        random.randint(_scale_pixels(30), _scale_pixels(62)),
        minimum_text_width,
    )
    label_h = random.randint(_scale_pixels(48), _scale_pixels(78))
    margin_x = random.randint(_scale_pixels(24), _scale_pixels(56))
    margin_y = random.randint(_scale_pixels(24), _scale_pixels(56))
    canvas_width = label_w + 2 * margin_x
    canvas_height = label_h + 2 * margin_y

    # 贴纸四周的书脊色背景
    color = random.choice(call_number_renderer.BOOK_COLORS)
    shade = tuple(max(0, min(255, int(c * random.uniform(0.86, 1.08)))) for c in color)
    img = Image.new("RGBA", (canvas_width, canvas_height), shade + (255,))
    draw = ImageDraw.Draw(img, "RGBA")

    label_left, label_top = margin_x, margin_y
    label_right, label_bottom = label_left + label_w, label_top + label_h

    # Keep the text region on gray/warm paper instead of a flat pure-white
    # sticker. Border variation remains useful when the source crop is enlarged.
    fill = _paper_fill()
    border = random.choice(
        [
            (155, 61, 38, 255),
            (178, 75, 42, 255),
            (40, 90, 150, 255),
            (25, 75, 140, 255),
            (92, 105, 116, 255),
        ]
    )
    shadow_offset = random.randint(1, max(1, _scale_pixels(2)))
    draw.rounded_rectangle(
        [label_left + shadow_offset, label_top + shadow_offset, label_right + shadow_offset, label_bottom + shadow_offset],
        radius=random.randint(3, 7),
        fill=(0, 0, 0, random.randint(25, 60)),
    )
    draw.rounded_rectangle([label_left, label_top, label_right, label_bottom], radius=random.randint(3, 7), fill=fill,
                           outline=border, width=random.randint(1, 2))
    if random.random() < 0.55:
        draw.line([label_left + 2, label_top + 6, label_right - 2, label_top + 6], fill=border, width=1)
        draw.line([label_left + 2, label_bottom - 6, label_right - 2, label_bottom - 6], fill=border, width=1)

    # 多行文字，返回逐行的紧凑字形框
    line_bboxes = call_number_renderer.draw_fitted_multiline_text(
        img,
        (label_left + 2, label_top + 2, label_right - 2, label_bottom - 2),
        lines,
        fill=(12, 12, 12, 235),
        add_version_dot=False,
    )

    # Tight union box for all lines. It remains the annotation inside the
    # padded upright piece before image and box are rotated together.
    x0 = min(p[0][0] for p in line_bboxes)
    y0 = min(p[0][1] for p in line_bboxes)
    x1 = max(p[1][0] for p in line_bboxes)
    y1 = max(p[2][1] for p in line_bboxes)
    text_box_aspect_ratio = (y1 - y0) / (x1 - x0)
    (crop_left, crop_top, crop_right, crop_bottom), points = _compute_crop_and_text_polygon(
        (x0, y0, x1, y1),
        (canvas_width, canvas_height),
        side_ratio,
    )

    img = _add_surface_texture(img.convert("RGB"))
    if noise:
        img = call_number_renderer.add_noise_and_blur(img)
    text_piece = cv2.cvtColor(
        np.asarray(img)[crop_top:crop_bottom, crop_left:crop_right],
        cv2.COLOR_RGB2BGR,
    )
    should_rotate = _should_rotate_for_aspect_ratio(
        text_box_aspect_ratio,
        rotation_aspect_ratio_threshold_l,
        rotation_aspect_ratio_threshold_r,
        rotation_probability,
    )
    if should_rotate:
        piece, points = _rotate_90_ccw_with_points(text_piece, points)
        rotation_degrees = 90
    else:
        piece = text_piece
        rotation_degrees = 0

    source_height, source_width = piece.shape[:2]
    target_width = max(1, int(round(source_width * output_scale)))
    target_height = max(1, int(round(source_height * output_scale)))
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    piece = cv2.resize(
        piece,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )
    points = [
        [int(round(x * scale_x)), int(round(y * scale_y))]
        for x, y in points
    ]

    record_id = writer.read_count() + 1
    label = _label_from_lines(lines)
    file_name = "image-{:09d}_0_{}.jpg".format(
        record_id,
        rotation_degrees,
    )
    writer.write(
        "id-{:09d}".format(record_id),
        piece,
        label,
        bbox=points,
        font_base=_current_font_file_name(),
        scores=1,
        file=file_name,
    )


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=f"./call_number_rec_multiline_{timestamp}_lmdb")
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--generator-strategy",
        choices=("grammar", "learned"),
        default="learned",
        help=(
            "call-number generation strategy; learned uses the embedded "
            "distribution model"
        ),
    )
    parser.add_argument(
        "--patterns",
        default=_DEFAULT_PATTERNS,
        help=(
            "JSON pattern configs with non-negative, unbounded visible line lengths "
            "and a sampling weight, "
            "and a zero length omits that line, "
            "used for grammar generation or the embedded distribution-model structure; "
            "when omitted, both strategies use the built-in patterns, "
            "e.g. [{\"lengths\":[2,2,2],\"weight\":1},"
            "{\"lengths\":[2,2,0],\"weight\":2},"
            "{\"lengths\":[3,3,3],\"weight\":1}]"
        ),
    )
    parser.add_argument(
        "--line3-prefix-prob",
        type=float,
        default=0.20,
        help="probability that a line 3 of length 2+ uses a valid v/.v start",
    )
    parser.add_argument(
        "--line3-dot-v-prob",
        type=float,
        default=0.50,
        help="conditional probability of .v rather than v when both fit",
    )
    parser.add_argument(
        "--line1-letter-mode",
        choices=("uniform", "learned"),
        default="uniform",
        help="first-line letter sampling: uniform A-Z or learned frequency",
    )
    parser.add_argument(
        "--rotation-aspect-ratio-threshold-l",
        type=float,
        default=_DEFAULT_ROTATION_ASPECT_RATIO_THRESHOLD_L,
        help="h/w at or below this value never rotates",
    )
    parser.add_argument(
        "--rotation-aspect-ratio-threshold-r",
        type=float,
        default=_DEFAULT_ROTATION_ASPECT_RATIO_THRESHOLD_R,
        help="h/w at or above this value always rotates",
    )
    parser.add_argument(
        "--rotation-prob",
        type=float,
        default=0.50,
        help="rotation probability when h/w is between the two thresholds",
    )
    parser.add_argument(
        "--output-scale",
        type=float,
        default=0.8,
        help="final image width/height multiplier; default halves both dimensions",
    )
    parser.add_argument("--scale-min", type=float, default=1.6, help="min canvas scale (text size)")
    parser.add_argument("--scale-max", type=float, default=2.2, help="max canvas scale (text size)")
    parser.add_argument("--side-ratio", type=float, default=_DEFAULT_CONTEXT_EXPANSION_RATIO,
                    help="natural-background expansion relative to rotated text-box height")
    parser.add_argument("--noise", action="store_true", help="enable noise/blur augmentation")
    args = parser.parse_args()

    if args.scale_min <= 0 or args.scale_min > args.scale_max:
        parser.error("require 0 < --scale-min <= --scale-max")
    if args.side_ratio < 0:
        parser.error("require --side-ratio >= 0")
    if args.rotation_aspect_ratio_threshold_l <= 0:
        parser.error("require --rotation-aspect-ratio-threshold-l > 0")
    if args.rotation_aspect_ratio_threshold_r <= args.rotation_aspect_ratio_threshold_l:
        parser.error(
            "require --rotation-aspect-ratio-threshold-l < "
            "--rotation-aspect-ratio-threshold-r"
        )
    if not 0 <= args.rotation_prob <= 1:
        parser.error("require 0 <= --rotation-prob <= 1")
    if not 0 < args.output_scale <= 1:
        parser.error("require 0 < --output-scale <= 1")

    random.seed(args.seed)
    np.random.seed(args.seed)

    try:
        raw_patterns = args.patterns
        if isinstance(raw_patterns, str):
            raw_patterns = json.loads(raw_patterns)
        if not isinstance(raw_patterns, list) or not raw_patterns:
            raise ValueError("--patterns must be a non-empty JSON list")

        patterns = []
        for index, item in enumerate(raw_patterns):
            if not isinstance(item, dict):
                raise ValueError(
                    "--patterns[{}] must be an object with lengths and optional weight".format(
                        index
                    )
                )
            if "lengths" not in item:
                raise ValueError("--patterns[{}] is missing 'lengths'".format(index))

            unknown_keys = set(item) - {"lengths", "weight"}
            if unknown_keys:
                raise ValueError(
                    "--patterns[{}] has unknown keys: {}".format(
                        index,
                        ", ".join(sorted(unknown_keys)),
                    )
                )

            lengths = item["lengths"]
            if not isinstance(lengths, (list, tuple)):
                raise ValueError(
                    "--patterns[{}].lengths must be a JSON array".format(index)
                )

            patterns.append(
                PatternConfig(
                    lengths=tuple(lengths),
                    weight=item.get("weight", 1.0),
                )
            )

        call_number_generator = build_call_number_generator(
            args.generator_strategy,
            patterns=patterns,
            line3_prefix_probability=args.line3_prefix_prob,
            line3_dot_v_probability=args.line3_dot_v_prob,
            line1_letter_mode=args.line1_letter_mode,
            rng=random.Random(args.seed),
        )
    except (json.JSONDecodeError, TypeError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print("Call-number generator: {}".format(call_number_generator.summary()))

    output_path = Path(args.out)
    if output_path.exists():
        shutil.rmtree(output_path)

    writer = LmdbDataset(str(output_path), map_size=200 * 1024 ** 2)
    for idx in tqdm(range(args.num)):
        generate_and_write_sample(
            writer,
            idx,
            call_number_generator,
            scale_range=(args.scale_min, args.scale_max),
            side_ratio=args.side_ratio,
            rotation_aspect_ratio_threshold_l=args.rotation_aspect_ratio_threshold_l,
            rotation_aspect_ratio_threshold_r=args.rotation_aspect_ratio_threshold_r,
            rotation_probability=args.rotation_prob,
            output_scale=args.output_scale,
            noise=args.noise,
        )
    writer.close()  # 写入 num-samples 元数据

    lmdb_2_min_size(output_path)
    print(f"Done: {output_path} ({writer.read_count()} records)")


if __name__ == "__main__":
    main()
