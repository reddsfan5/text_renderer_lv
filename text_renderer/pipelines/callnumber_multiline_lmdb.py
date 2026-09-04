"""
Synthetic multi-line call-number recognition LMDB generator.

Each sample renders one call-number sticker (row or column writing), expands
the text-block edges from the continuous background, and writes the piece to
LMDB. Layout structure is controlled by direction-specific default patterns so
final h/w stays within a usable recognition range without rotation-as-correction.

Usage:
    python -m text_renderer.cli callnumber-multiline --write-direction row --num 1000
    python -m text_renderer.cli callnumber-multiline --write-direction column --num 1000
    python -m text_renderer.cli callnumber-multiline --patterns '[{"lengths":[3,3,0],"weight":2}]'
"""

from datetime import datetime
from pathlib import Path

import argparse
import json
import random
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from text_renderer.postprocess import visualize_lmdb
from text_renderer.renderers import callnumber as call_number_renderer
from text_renderer.corpus.call_number_generator import (
    PatternConfig,
    build_call_number_generator,
)
from lv_tools.dataset_io.lmdb_backend import lmdb_2_min_size
from lv_tools.lmdbs.workflows.ocr import (
    LmdbDataset,
)

_DEFAULT_CONTEXT_EXPANSION_RATIO = 1.0
_DEFAULT_MAX_ASPECT_RATIO = 1.7
_MAX_LAYOUT_RETRIES = 50

# Edit these tables to change default layout distributions.
# Manually keep estimated h/w around <= 1.2 for the matching write_direction:
#   row:    n_rows / max_line_len
#   column: max_col_len / n_cols
_DEFAULT_PATTERNS_ROW = [
    {"lengths": [5, 5, 0], "weight": 1.0},
    {"lengths": [4, 4, 0], "weight": 1.0},
    {"lengths": [3, 3, 0], "weight": 1.5},
    {"lengths": [3, 3, 3], "weight": 1.5},
    {"lengths": [3, 0, 3], "weight": 1.5},
    {"lengths": [2, 1, 0], "weight": 1.5},
    {"lengths": [2, 0, 2], "weight": 1.5},
]

_DEFAULT_PATTERNS_COLUMN = [
    {"lengths": [2, 2, 2], "weight": 2},
    {"lengths": [2, 2, 0], "weight": 1.5},
    {"lengths": [1, 1, 1], "weight": 1.0},
    {"lengths": [2, 1, 0], "weight": 1},
    {"lengths": [3, 3, 3], "weight": 2},
]

_SAMPLES_PER_FONT = 20

_font_cache = {}
_current_font_path = None


def _scale_pixels(v):
    """Scale a pixel value by the current image scale."""
    return call_number_renderer._scale_pixels(v)


def _select_font_for_batch():
    """Reuse one font path for _SAMPLES_PER_FONT samples."""
    global _current_font_path
    path = (
        random.choice(call_number_renderer._FONT_PATHS)
        if call_number_renderer._FONT_PATHS
        else None
    )
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
            (5, 2, -3),
            (-3, 1, 5),
            (2, 2, 2),
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


def _nonempty_lengths(lengths):
    return [int(length) for length in lengths if int(length) > 0]


def layout_aspect_from_lengths(lengths, write_direction):
    """Estimate text-box h/w from pattern lengths under row/column writing."""
    values = _nonempty_lengths(lengths)
    if not values:
        raise ValueError("lengths must contain at least one positive value")
    if write_direction == "row":
        return len(values) / max(values)
    if write_direction == "column":
        return max(values) / len(values)
    raise ValueError("write_direction must be row or column")


def layout_aspect_from_lines(lines, write_direction):
    """Estimate text-box h/w from generated line strings."""
    nonempty = [line for line in lines if line]
    if not nonempty:
        raise ValueError("lines must contain at least one non-empty string")
    lengths = [len(line) for line in nonempty]
    return layout_aspect_from_lengths(lengths, write_direction)


def default_patterns_for_direction(write_direction):
    if write_direction == "row":
        return [dict(item) for item in _DEFAULT_PATTERNS_ROW]
    if write_direction == "column":
        return [dict(item) for item in _DEFAULT_PATTERNS_COLUMN]
    raise ValueError("write_direction must be row or column")


def filter_patterns_by_aspect(patterns, max_aspect_ratio, write_direction):
    """Keep PatternConfig items whose estimated h/w <= max_aspect_ratio."""
    kept = []
    dropped = []
    for pattern in patterns:
        aspect = layout_aspect_from_lengths(pattern.lengths, write_direction)
        if aspect <= max_aspect_ratio:
            kept.append(pattern)
        else:
            dropped.append((pattern, aspect))
    return kept, dropped


def _compute_crop_and_text_polygon(text_box, image_size, side_ratio):
    """Expand into the rendered background and return the relative text box."""
    x0, y0, x1, y1 = text_box
    width, height = image_size
    text_x0 = max(0, int(np.floor(x0)))
    text_y0 = max(0, int(np.floor(y0)))
    text_x1 = min(width, int(np.ceil(x1)))
    text_y1 = min(height, int(np.ceil(y1)))
    text_w = max(1, text_x1 - text_x0)

    side = max(0, int(round(text_w * side_ratio)))
    crop_x0 = max(0, text_x0 - side)
    crop_y0 = max(0, text_y0 - side)
    crop_x1 = min(width, text_x1 + side)
    crop_y1 = min(height, text_y1 + side)

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


def _label_from_lines(lines):
    """Build the transcription from exactly the text that was rendered."""
    return " ".join(lines)


def _current_font_file_name():
    """Return the filename of the font selected for the current batch."""
    return None if _current_font_path is None else Path(_current_font_path).name


def _sample_lines(call_number_generator, write_direction, max_aspect_ratio):
    """Generate lines that pass the layout aspect estimate."""
    last_error = None
    for _ in range(_MAX_LAYOUT_RETRIES):
        lines = call_number_generator.generate()
        try:
            aspect = layout_aspect_from_lines(lines, write_direction)
        except ValueError as error:
            last_error = error
            continue
        if aspect <= max_aspect_ratio:
            return lines
    raise RuntimeError(
        "could not sample a layout with h/w <= {}; last error: {}".format(
            max_aspect_ratio,
            last_error,
        )
    )


def generate_and_write_sample(
    writer,
    sample_index,
    call_number_generator,
    write_direction="row",
    max_aspect_ratio=_DEFAULT_MAX_ASPECT_RATIO,
    scale_range=(1.8, 3.4),
    side_ratio=_DEFAULT_CONTEXT_EXPANSION_RATIO,
    rotation_probability=0.0,
    output_scale=0.5,
    noise=False,
):
    """Render one sticker and write it when the measured aspect is valid.

    Returns:
        True if a record was written, False if layout/aspect was rejected.
    """
    if sample_index % _SAMPLES_PER_FONT == 0:
        _select_font_for_batch()
    call_number_renderer._IMAGE_SCALE_FACTOR = random.uniform(*scale_range)

    try:
        lines = _sample_lines(
            call_number_generator, write_direction, max_aspect_ratio
        )
    except RuntimeError:
        return False

    if write_direction == "row":
        width_units = max(map(len, lines))
        height_units = len(lines)
    else:
        width_units = len(lines)
        height_units = max(map(len, lines))

    minimum_text_width = _scale_pixels(width_units * 12 + 6)
    minimum_text_height = _scale_pixels(height_units * 14 + 6)
    label_w = max(
        random.randint(_scale_pixels(30), _scale_pixels(62)),
        minimum_text_width,
    )
    label_h = max(
        random.randint(_scale_pixels(48), _scale_pixels(78)),
        minimum_text_height,
    )
    margin_x = random.randint(_scale_pixels(24), _scale_pixels(56))
    margin_y = random.randint(_scale_pixels(24), _scale_pixels(56))
    canvas_width = label_w + 2 * margin_x
    canvas_height = label_h + 2 * margin_y

    color = random.choice(call_number_renderer.BOOK_COLORS)
    shade = tuple(max(0, min(255, int(c * random.uniform(0.86, 1.08)))) for c in color)
    img = Image.new("RGBA", (canvas_width, canvas_height), shade + (255,))
    draw = ImageDraw.Draw(img, "RGBA")

    label_left, label_top = margin_x, margin_y
    label_right, label_bottom = label_left + label_w, label_top + label_h

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
        [
            label_left + shadow_offset,
            label_top + shadow_offset,
            label_right + shadow_offset,
            label_bottom + shadow_offset,
        ],
        radius=random.randint(3, 7),
        fill=(0, 0, 0, random.randint(25, 60)),
    )
    draw.rounded_rectangle(
        [label_left, label_top, label_right, label_bottom],
        radius=random.randint(3, 7),
        fill=fill,
        outline=border,
        width=random.randint(1, 2),
    )
    if random.random() < 0.55:
        draw.line(
            [label_left + 2, label_top + 6, label_right - 2, label_top + 6],
            fill=border,
            width=1,
        )
        draw.line(
            [label_left + 2, label_bottom - 6, label_right - 2, label_bottom - 6],
            fill=border,
            width=1,
        )

    text_box = (
        label_left + 2,
        label_top + 2,
        label_right - 2,
        label_bottom - 2,
    )
    if write_direction == "column":
        unit_bboxes = call_number_renderer.draw_fitted_multicolumn_text(
            img,
            text_box,
            lines,
            fill=(12, 12, 12, 235),
        )
    else:
        unit_bboxes = call_number_renderer.draw_fitted_multiline_text(
            img,
            text_box,
            lines,
            fill=(12, 12, 12, 235),
            add_version_dot=False,
        )
    if not unit_bboxes:
        return False

    x0 = min(points[0][0] for points in unit_bboxes)
    y0 = min(points[0][1] for points in unit_bboxes)
    x1 = max(points[1][0] for points in unit_bboxes)
    y1 = max(points[2][1] for points in unit_bboxes)
    text_w = max(1e-6, x1 - x0)
    text_h = max(1e-6, y1 - y0)
    text_box_aspect_ratio = text_h / text_w
    if text_box_aspect_ratio > max_aspect_ratio:
        return False

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

    rotation_degrees = 0
    piece = text_piece
    if rotation_probability > 0 and random.random() < rotation_probability:
        # After 90° CCW, new h/w = old w/h.
        if (1.0 / text_box_aspect_ratio) <= max_aspect_ratio:
            piece, points = _rotate_90_ccw_with_points(text_piece, points)
            rotation_degrees = 90

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
    file_name = "image-{:09d}_0_{}.jpg".format(record_id, rotation_degrees)
    writer.write(
        "id-{:09d}".format(record_id),
        piece,
        label,
        bbox=points,
        font_base=_current_font_file_name(),
        scores=1,
        file=file_name,
    )
    return True


def _parse_patterns(raw_patterns):
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
    return patterns


def main(arg_list=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(prog="text-renderer callnumber-multiline")
    parser.add_argument("--out", default=fr"F:\dataset\OCR\3.text_rec\synthesis\rec_pieces/call_number_rec_multiline_{timestamp}_lmdb")
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--write-direction",
        choices=("row", "column"),
        default="row",
        help="row=stacked horizontal lines; column=side-by-side vertical columns",
    )
    parser.add_argument(
        "--max-aspect-ratio",
        type=float,
        default=_DEFAULT_MAX_ASPECT_RATIO,
        help="reject layouts whose estimated/measured h/w exceeds this value",
    )
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
        default=None,
        help=(
            "JSON pattern list; when omitted, uses direction-specific defaults "
            "(_DEFAULT_PATTERNS_ROW / _DEFAULT_PATTERNS_COLUMN)"
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
        "--rotation-prob",
        type=float,
        default=0.0,
        help="optional augmentation: rotate only if post-rotation h/w still fits",
    )
    parser.add_argument(
        "--output-scale",
        type=float,
        default=0.8,
        help="final image width/height multiplier",
    )
    parser.add_argument("--scale-min", type=float, default=1.6, help="min canvas scale")
    parser.add_argument("--scale-max", type=float, default=2.2, help="max canvas scale")
    parser.add_argument(
        "--side-ratio",
        type=float,
        default=_DEFAULT_CONTEXT_EXPANSION_RATIO,
        help="natural-background expansion relative to text-box width",
    )
    parser.add_argument("--noise", action="store_true", help="enable noise/blur augmentation")
    parser.add_argument(
        "--sample-count",
        type=int,
        default=50,
        help="generate this many visual samples after generation; 0 disables it",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output directory",
    )
    args = parser.parse_args(arg_list)

    if args.scale_min <= 0 or args.scale_min > args.scale_max:
        parser.error("require 0 < --scale-min <= --scale-max")
    if args.side_ratio < 0:
        parser.error("require --side-ratio >= 0")
    if args.max_aspect_ratio <= 0:
        parser.error("require --max-aspect-ratio > 0")
    if not 0 <= args.rotation_prob <= 1:
        parser.error("require 0 <= --rotation-prob <= 1")
    if not 0 < args.output_scale <= 1:
        parser.error("require 0 < --output-scale <= 1")
    if args.num <= 0:
        parser.error("require --num > 0")
    if args.sample_count < 0:
        parser.error("require --sample-count >= 0")

    random.seed(args.seed)
    np.random.seed(args.seed)

    try:
        user_provided_patterns = args.patterns is not None
        raw_patterns = (
            args.patterns
            if user_provided_patterns
            else default_patterns_for_direction(args.write_direction)
        )
        patterns = _parse_patterns(raw_patterns)
        if user_provided_patterns:
            patterns, dropped = filter_patterns_by_aspect(
                patterns, args.max_aspect_ratio, args.write_direction
            )
            for pattern, aspect in dropped:
                print(
                    "drop pattern lengths={} estimated_h/w={:.3f} > {}".format(
                        list(pattern.lengths),
                        aspect,
                        args.max_aspect_ratio,
                    )
                )
            if not patterns:
                raise ValueError(
                    "no patterns remain under --max-aspect-ratio {} for "
                    "write-direction {}; adjust patterns or threshold".format(
                        args.max_aspect_ratio,
                        args.write_direction,
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
    print(
        "Call-number generator: {} | write_direction={} | max_aspect_ratio={}".format(
            call_number_generator.summary(),
            args.write_direction,
            args.max_aspect_ratio,
        )
    )

    output_path = Path(args.out)
    if output_path.exists():
        if not args.overwrite:
            parser.error(
                "output already exists: {}; pass --overwrite to replace it".format(
                    output_path
                )
            )
        shutil.rmtree(output_path)

    writer = LmdbDataset(str(output_path), map_size=200 * 1024 ** 2)
    written = 0
    attempts = 0
    max_attempts = args.num * _MAX_LAYOUT_RETRIES
    progress = tqdm(total=args.num)
    while written < args.num and attempts < max_attempts:
        attempts += 1
        ok = generate_and_write_sample(
            writer,
            written,
            call_number_generator,
            write_direction=args.write_direction,
            max_aspect_ratio=args.max_aspect_ratio,
            scale_range=(args.scale_min, args.scale_max),
            side_ratio=args.side_ratio,
            rotation_probability=args.rotation_prob,
            output_scale=args.output_scale,
            noise=args.noise,
        )
        if ok:
            written += 1
            progress.update(1)
    progress.close()
    if written < args.num:
        writer.close()
        raise RuntimeError(
            "only wrote {}/{} records after {} attempts; "
            "relax --max-aspect-ratio or change patterns".format(
                written, args.num, attempts
            )
        )
    writer.close()

    lmdb_2_min_size(output_path)
    print(f"Done: {output_path} ({writer.read_count()} records)")
    if args.sample_count:
        try:
            visualize_lmdb(output_path, args.sample_count)
            print("Samples: {}".format(output_path / "samples"))
        except Exception as error:
            print("Warning: could not generate samples: {}".format(error))

    return output_path


if __name__ == "__main__":
    main()
