"""
Synthetic call-number label detection dataset generator (LabelMe format).

Output for each sample:
    image.jpg
    image.json   # LabelMe annotation

The annotation label is:
    "call_number_label"

"""

from pathlib import Path

import cv2
import math
import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import base64, json, random, re
import numpy as np
import pandas as pd
# ---- Call-number line pools (loaded from Excel in main) ----
_CN_POOL_L1 = []            # line1: filtered len ≤ 3, punctuation stripped
_CN_L2_BY_LEN = {}          # line2: {length: [items]}, punctuation stripped
_CN_L3_BY_LEN = {}          # line3: {length: [items]}, punctuation stripped
_CN_COMMON_LENGTHS = []     # lengths present in both L2 and L3
_WEIGHTED_LENGTHS = []      # weighted: 70% len=2, 15% each len=1/3

BOOK_COLORS = [
    (190, 35, 28), (214, 218, 191), (86, 160, 139), (233, 233, 225),
    (25, 86, 135), (55, 76, 142), (112, 81, 170), (20, 130, 130),
    (232, 180, 90), (50, 50, 55), (245, 245, 245), (130, 44, 42),
]
TITLE_POOL = [
    "统计学", "应用统计学", "数据分析", "现代基础统计学", "简单统计学",
    "统计学习方法", "管理学", "国是千万家", "创新统计思维", "西文注释详解",
    "女士品茶", "数据如何误导我们", "数学", "原理与结果"
]
FONT_DIR = Path(r"D:\lxd_code\OCR\OCR_SOURCE\font\font_set\简体-简体-低风险\常规类_已更正\等宽")
_FONT_PATHS = sorted(FONT_DIR.glob("*.ttf")) if FONT_DIR.exists() else []
_CURRENT_FONT_PATH = None   # one font per image
_FONT_CACHE = {}            # {(path, size): ImageFont}, preloaded on switch
_REFERENCE_IMAGE_HEIGHT = 640               # reference image height for scaling
_IMAGE_SCALE_FACTOR = 1.0                # H / _BASE_H, set by the pipeline

DEFAULT_XLSX = r"D:\lxd_code\OCR\OCR_SOURCE\corpus\call_number\2026-06-26-13-32-44.xlsx"


def _scale_pixels(v):
    """Scale a pixel value relative to base image height 640."""
    return max(0, int(v * _IMAGE_SCALE_FACTOR))


def _remove_punctuation(text):
    """Remove punctuation: keep only alphanumeric characters."""
    return re.sub(r'[^a-zA-Z0-9]', '', text)


def load_callnumber_pools(xlsx_path=DEFAULT_XLSX):
    """Load call-number lines from Excel, strip punctuation, group by length.

    - All text: punctuation stripped (not rendered at all).
    - L1: keep only items with length ≤ 3.
    - L2, L3: grouped by length; empty-after-strip items dropped.
    - Length selection: 70% len=2, 15% each len=1/3.
    """
    global _CN_POOL_L1, _CN_L2_BY_LEN, _CN_L3_BY_LEN, _CN_COMMON_LENGTHS, _WEIGHTED_LENGTHS

    df = pd.read_excel(xlsx_path)

    # L1: strip punctuation, filter raw length == 2
    l1_raw = df['callnumber1'].dropna().astype(str).apply(_remove_punctuation)
    _CN_POOL_L1 = l1_raw[l1_raw.str.len() == 2].tolist()

    # L2, L3: strip punctuation, drop empty, group by length
    l2_raw = df['callnumber2'].dropna().astype(str).apply(_remove_punctuation)
    l3_raw = df['callnumber3'].dropna().astype(str).apply(_remove_punctuation)

    l2_raw = l2_raw[l2_raw.str.len() > 0]
    l3_raw = l3_raw[l3_raw.str.len() > 0]

    _CN_L2_BY_LEN = {k: g.tolist() for k, g in l2_raw.groupby(l2_raw.str.len())}
    _CN_L3_BY_LEN = {k: g.tolist() for k, g in l3_raw.groupby(l3_raw.str.len())}

    _CN_COMMON_LENGTHS = sorted(
        set(_CN_L2_BY_LEN) & set(_CN_L3_BY_LEN) & {1, 2}
    )

    _WEIGHTED_LENGTHS = [1] * 50 + [2] * 50

    total_l2 = sum(len(v) for v in _CN_L2_BY_LEN.values())
    total_l3 = sum(len(v) for v in _CN_L3_BY_LEN.values())
    print(f"Call-number pools loaded: L1={len(_CN_POOL_L1)} (len≤3), "
          f"L2={total_l2}, L3={total_l3}, "
          f"common_lengths={_CN_COMMON_LENGTHS}, "
          f"weighted_ratio={{1:15, 2:70, 3:15}}")


def pick_font_for_image():
    """Preload one font (all sizes) for the current image."""
    global _CURRENT_FONT_PATH, _FONT_CACHE
    _CURRENT_FONT_PATH = random.choice(_FONT_PATHS) if _FONT_PATHS else None
    _FONT_CACHE.clear()
    if _CURRENT_FONT_PATH:
        for size in range(_scale_pixels(12), _scale_pixels(19)):
            _FONT_CACHE[(_CURRENT_FONT_PATH, size)] = ImageFont.truetype(str(_CURRENT_FONT_PATH), size=size)


def get_font(size=None):
    if size is None:
        size = _scale_pixels(20)
    if _CURRENT_FONT_PATH:
        key = (_CURRENT_FONT_PATH, size)
        if key not in _FONT_CACHE:
            _FONT_CACHE[key] = ImageFont.truetype(str(_CURRENT_FONT_PATH), size=size)
        return _FONT_CACHE[key]
    return ImageFont.load_default()


def generate_random_call_number_lines():
    """Generate 2–3 line call number. Punctuation already stripped from pools.

    - L2 and L3 have the same length (balanced layout).
    - Length distribution: ~70% len=2, ~15% each len=1/3.
    - 2-line vs 3-line: ~50/50.
    """
    if random.random() < 0.5:
        return [random.choice(_CN_POOL_L1),
                random.choice(_CN_L2_BY_LEN[random.choice(_WEIGHTED_LENGTHS)])]
    else:
        length = random.choice(_WEIGHTED_LENGTHS)
        return [random.choice(_CN_POOL_L1),
                random.choice(_CN_L2_BY_LEN[length]),
                random.choice(_CN_L3_BY_LEN[length])]

def _get_line_bbox(line, font):
    """Get tight glyph bbox via FreeType font metrics.

    font.getbbox(line, anchor='lt') returns (left, top, right, bottom)
    relative to the left-top anchor point. These are exact glyph bounds.
    """
    left, top, right, bottom = font.getbbox(line, anchor='lt')
    return left, top, right, bottom


def fit_font_for_box(lines, max_w, max_h, min_size=None, max_size=None):
    if min_size is None:
        min_size = _scale_pixels(13)
    if max_size is None:
        max_size = _scale_pixels(17) + random.randint(-1, 1)
    for fs in range(max_size, min_size - 1, -1):
        font = get_font(fs)
        gap = max(_scale_pixels(1), int(fs * random.uniform(.29, .3)))
        ok = True
        for line in lines:
            _, _, r, b = _get_line_bbox(line, font)
            if r > max_w or b > max_h:
                ok = False
                break
        if not ok:
            continue
        total_h = 0
        max_lw = 0
        for line in lines:
            _, _, r, b = _get_line_bbox(line, font)
            total_h += b
            max_lw = max(max_lw, r)
        total_h += gap * max(0, len(lines) - 1)
        if max_lw <= max_w and total_h <= max_h:
            return font, gap
    return get_font(min_size), max(_scale_pixels(1), int(min_size * 0.12))


def draw_fitted_multiline_text(
    img_rgba,
    box,
    lines,
    fill=(12, 12, 12, 235),
    add_version_dot=True,
):
    """Draw multi-line text centered, return per-line tight 4-point bboxes.

    Uses font.getbbox(anchor='lt') for accurate glyph-level bounds,
    and draw.text(anchor='lt') for consistent positioning.

    Returns:
        list of [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], one 4-point bbox per line
    """
    x1, y1, x2, y2 = box
    pad_x = max(_scale_pixels(2), int((x2 - x1) * 0.08))
    pad_y = max(_scale_pixels(2), int((y2 - y1) * 0.07))
    inner_x1 = x1 + pad_x
    inner_y1 = y1 + pad_y
    inner_x2 = x2 - pad_x
    inner_y2 = y2 - pad_y
    inner_w = inner_x2 - inner_x1
    inner_h = inner_y2 - inner_y1

    max_w = max(_scale_pixels(6), int(0.90 * inner_w))
    max_h = max(_scale_pixels(6), int(0.90 * inner_h))

    font, gap = fit_font_for_box(lines, max_w, max_h)

    # Measure each line's glyph bbox
    line_rects = [_get_line_bbox(line, font) for line in lines]
    heights = [r[3] for r in line_rects]  # bottom
    block_h = sum(heights) + gap * max(0, len(lines) - 1)

    # Center the text block vertically in the inner box
    cur_y = inner_y1 + (inner_h - block_h) / 2.0

    # Measure dot separately for tight placement
    _, _, dot_w, dot_h = _get_line_bbox(".", font)

    draw = ImageDraw.Draw(img_rgba, "RGBA")
    line_bboxes = []
    for i, line in enumerate(lines):
        _, _, lw, lh = line_rects[i]
        # Horizontal center (based on text width, dot will be placed left of it)
        tx = inner_x1 + (inner_w - lw) / 2.0
        draw.text((tx, cur_y), line, font=font, fill=fill, anchor='lt')

        if add_version_dot and i == len(lines) - 1 and line.startswith("v"):
            # Dot: bottom-left of text, touching
            dot_x = tx - dot_w + 1
            dot_y = cur_y + lh - dot_h
            draw.text((dot_x, dot_y), ".", font=font, fill=fill, anchor='lt')
            # Merge dot + text into one bbox
            line_bboxes.append([
                [dot_x, cur_y],
                [tx + lw, cur_y],
                [tx + lw, cur_y + lh],
                [dot_x, cur_y + lh],
            ])
        else:
            line_bboxes.append([
                [tx, cur_y],
                [tx + lw, cur_y],
                [tx + lw, cur_y + lh],
                [tx, cur_y + lh],
            ])
        cur_y += lh + gap

    return line_bboxes


def _measure_horizontal_text(text, font):
    """Return (width, height) of horizontal text via FreeType bbox."""
    _, _, right, bottom = _get_line_bbox(text, font)
    return max(1, int(math.ceil(right))), max(1, int(math.ceil(bottom)))


def _render_column_strip(text, font, fill):
    """Render one unit as horizontal text, then rotate 90° counterclockwise.

    Sideways glyphs read top-to-bottom after CCW rotation of the horizontal strip.
    """
    text_w, text_h = _measure_horizontal_text(text, font)
    pad = 1
    strip = Image.new("RGBA", (text_w + 2 * pad, text_h + 2 * pad), (0, 0, 0, 0))
    draw = ImageDraw.Draw(strip, "RGBA")
    draw.text((pad, pad), text, font=font, fill=fill, anchor="lt")
    return strip.rotate(90, expand=True, resample=Image.BICUBIC)


def fit_font_for_columns(columns, max_w, max_h, min_size=None, max_size=None):
    """Fit a font so rotated column strips fit inside max_w x max_h."""
    if min_size is None:
        min_size = _scale_pixels(13)
    if max_size is None:
        max_size = _scale_pixels(17) + random.randint(-1, 1)
    for fs in range(max_size, min_size - 1, -1):
        font = get_font(fs)
        col_gap = max(_scale_pixels(1), int(fs * random.uniform(0.35, 0.45)))
        col_widths = []
        col_heights = []
        ok = True
        for column in columns:
            if not column:
                continue
            # After 90° CCW: horizontal (w, h) becomes column (h, w).
            text_w, text_h = _measure_horizontal_text(column, font)
            width, height = text_h, text_w
            if width > max_w or height > max_h:
                ok = False
                break
            col_widths.append(width)
            col_heights.append(height)
        if not ok or not col_widths:
            continue
        total_w = sum(col_widths) + col_gap * max(0, len(col_widths) - 1)
        total_h = max(col_heights)
        if total_w <= max_w and total_h <= max_h:
            return font, col_gap
    return (
        get_font(min_size),
        max(_scale_pixels(1), int(min_size * 0.2)),
    )


def draw_fitted_multicolumn_text(
    img_rgba,
    box,
    lines,
    fill=(12, 12, 12, 235),
):
    """Draw each line as a 90°-rotated column strip, left to right.

    Each ``line`` is rendered horizontally then rotated 90° counterclockwise so
    glyphs appear sideways and form a vertical column.

    Returns:
        list of [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], one 4-point bbox per column
    """
    x1, y1, x2, y2 = box
    pad_x = max(_scale_pixels(2), int((x2 - x1) * 0.08))
    pad_y = max(_scale_pixels(2), int((y2 - y1) * 0.07))
    inner_x1 = x1 + pad_x
    inner_y1 = y1 + pad_y
    inner_x2 = x2 - pad_x
    inner_y2 = y2 - pad_y
    inner_w = inner_x2 - inner_x1
    inner_h = inner_y2 - inner_y1

    max_w = max(_scale_pixels(6), int(0.90 * inner_w))
    max_h = max(_scale_pixels(6), int(0.90 * inner_h))

    columns = [line for line in lines if line]
    if not columns:
        return []

    font, col_gap = fit_font_for_columns(columns, max_w, max_h)

    strips = [_render_column_strip(column, font, fill) for column in columns]
    block_w = sum(strip.size[0] for strip in strips) + col_gap * max(0, len(strips) - 1)
    block_h = max(strip.size[1] for strip in strips)
    cur_x = inner_x1 + (inner_w - block_w) / 2.0
    block_top = inner_y1 + (inner_h - block_h) / 2.0

    column_bboxes = []
    for strip in strips:
        sw, sh = strip.size
        paste_y = int(round(block_top + (block_h - sh) / 2.0))
        paste_x = int(round(cur_x))
        img_rgba.paste(strip, (paste_x, paste_y), strip)
        column_bboxes.append(
            [
                [paste_x, paste_y],
                [paste_x + sw, paste_y],
                [paste_x + sw, paste_y + sh],
                [paste_x, paste_y + sh],
            ]
        )
        cur_x += sw + col_gap

    return column_bboxes


def draw_vertical_title(draw, x, top, bottom, w, color):
    title = random.choice(TITLE_POOL)
    if random.random() < 0.5:
        font = get_font(random.randint(16, 22))
        yy = top + random.randint(20, 70)
        title_color = (25, 25, 25, 180) if sum(color) > 360 else (245, 245, 245, 210)
        max_chars = max(2, min(len(title), int((bottom - top - 50) / 25)))
        for ch in title[:max_chars]:
            draw.text((x + w / 2 - 9, yy), ch, font=font, fill=title_color)
            yy += random.randint(20, 28)
            if yy > bottom - 25:
                break

def add_noise_and_blur(img):
    arr = np.asarray(img).astype(np.int16)
    noise_sigma = random.uniform(2, 8)
    noise = np.random.normal(0, noise_sigma, arr.shape).astype(np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    out = Image.fromarray(arr)
    if random.random() < 0.60:
        out = out.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.12, 0.75)))
    if random.random() < 0.45:
        out = ImageEnhance.Contrast(out).enhance(random.uniform(0.90, 1.18))
    if random.random() < 0.45:
        out = ImageEnhance.Brightness(out).enhance(random.uniform(0.90, 1.12))
    return out

def draw_book(draw, img_rgba, x, bottom, w, h, color):
    top = bottom - h
    x2 = x + w
    shade = tuple(max(0, min(255, int(c * random.uniform(0.86, 1.08)))) for c in color)
    draw.rounded_rectangle([x, top, x2, bottom], radius=random.randint(0, 4), fill=shade + (255,))
    draw.line([x, top, x, bottom], fill=(0, 0, 0, 95), width=random.randint(1, 3))
    draw.line([x2 - 1, top, x2 - 1, bottom], fill=(255, 255, 255, 45), width=1)
    # if random.random() < 0.60 and w >= 34:
    #     draw_vertical_title(draw, x, top, bottom, w, color)
    return top

def draw_label(draw, img_rgba, x, bottom, w, top, with_label=True):
    """Draw a call-number sticker on the book spine.

    Returns:
        (list_of_line_bboxes, joined_text)  or  (None, None)
        list_of_line_bboxes: [[x1,y1,x2,y2], ...] — one tight bbox per line of text
    """
    if not with_label or w < _scale_pixels(26):
        return None, None

    max_label_w = max(_scale_pixels(22), int(w * 0.88))
    min_label_w = max(_scale_pixels(18), int(w * 0.64))
    label_w = min_label_w if min_label_w >= max_label_w else random.randint(min_label_w, max_label_w)

    available_h = bottom - top - _scale_pixels(24)
    label_h = min(random.randint(_scale_pixels(48), _scale_pixels(78)), max(_scale_pixels(36), available_h))
    if label_h < _scale_pixels(30):
        return None, None

    lx = x + (w - label_w) / 2 + random.uniform(-1.2, 1.2)
    ly = bottom - label_h - random.randint(_scale_pixels(45), _scale_pixels(86))
    ly = max(top + _scale_pixels(10), min(ly, bottom - label_h - _scale_pixels(10)))
    rx, by = lx + label_w, ly + label_h

    fill = (244 + random.randint(-4, 4), 246 + random.randint(-4, 4), 247 + random.randint(-4, 4), 255)
    border = random.choice([(40, 90, 150, 255), (25, 75, 140, 255), (80, 120, 160, 255)])
    draw.rounded_rectangle([lx, ly, rx, by], radius=random.randint(3, 7), fill=fill, outline=border, width=random.randint(1, 2))
    if random.random() < 0.55:
        draw.line([lx + 2, ly + 6, rx - 2, ly + 6], fill=border, width=1)
        draw.line([lx + 2, by - 6, rx - 2, by - 6], fill=border, width=1)

    lines = generate_random_call_number_lines()
    line_bboxes = draw_fitted_multiline_text(img_rgba, (lx + 2, ly + 2, rx - 2, by - 2), lines, fill=(12, 12, 12, 235))
    return line_bboxes, "\n".join(lines)

def make_labelme_json(image_name, width, height, shapes, include_imageData=False, image_full_path=None):
    data = None
    if include_imageData and image_full_path is not None:
        with open(image_full_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
    return {
        "version": "5.5.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": data,
        "imageHeight": height,
        "imageWidth": width,
    }

def generate_one(out_dir, idx, W=1600, H=640, include_imageData=False):
    pick_font_for_image()  # one font per image

    arr = np.zeros((H, W, 4), dtype=np.uint8)
    for y in range(H):
        v = int(5 + 20 * (y / H))
        arr[y, :, :] = [v, v, v + 1, 255]
    img = Image.fromarray(arr, "RGBA")
    draw = ImageDraw.Draw(img, "RGBA")

    shelf_top = int(H * random.uniform(0.9, 1))
    draw.rectangle([0, shelf_top, W, H], fill=(8, 8, 8, 255))
    draw.rectangle([0, shelf_top - _scale_pixels(5), W, shelf_top + _scale_pixels(8)], fill=(22, 22, 22, 255))


    x = random.randint(_scale_pixels(20), _scale_pixels(60))
    bottom = shelf_top - random.randint(0, _scale_pixels(8))
    shapes = []

    while x < W - _scale_pixels(45):
        if random.random() < 0.06 and x > W * 0.55:
            x += random.randint(_scale_pixels(35), _scale_pixels(120))
            continue

        w = random.randint(_scale_pixels(35), _scale_pixels(70))
        h = random.randint(int(H * 0.82), int(H * 0.96))
        if random.random() < 0.10:
            h = random.randint(int(H * 0.35), int(H * 0.62))
        top = draw_book(draw, img, x, bottom, w, h, random.choice(BOOK_COLORS))
        bbox_list, call_text = draw_label(draw, img, x, bottom, w, top, with_label=(random.random() < 1))
        if bbox_list is not None:
            for line_pts in bbox_list:
                # line_pts is 4-point polygon: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                shapes.append({
                    "label": "Bar",
                    "points": [[round(p[0], 2), round(p[1], 2)] for p in line_pts],
                    "group_id": None,
                    "description": call_text,
                    "shape_type": "polygon",
                    "flags": {}
                })

        draw.rectangle([x + w - 1, top, x + w + 1, bottom], fill=(0, 0, 0, 90))
        x += w + random.randint(0, _scale_pixels(5))

    # if random.random() < 0.35:
    #     bx = random.randint(int(W * 0.70), int(W * 0.88))
    #     bh = random.randint(int(H * 0.40), int(H * 0.72))
    #     by = bottom - bh
    #     draw.rounded_rectangle([bx, by, bx + 18, bottom + 10], radius=4, fill=(170, 174, 176, 190))
    #     draw.line([bx + 8, by, bx + 8, bottom + 10], fill=(230, 230, 230, 160), width=2)
    #     draw.rectangle([bx - 20, bottom - 12, bx + 70, bottom + 7], fill=(125, 128, 130, 150))

    # img = add_noise_and_blur(img.convert("RGB"))
    img = img.convert('RGB')
    image_name = f"{os.path.basename(out_dir)}_{idx:06d}.jpg"
    json_name = f"{os.path.basename(out_dir)}_{idx:06d}.json"
    image_path = out_dir / image_name
    json_path = out_dir / json_name
    img.save(image_path, quality=98)

    ann = make_labelme_json(
        image_name=image_name,
        width=W,
        height=H,
        shapes=shapes,
        include_imageData=include_imageData,
        image_full_path=image_path,
    )
    json_path.write_text(json.dumps(ann, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(shapes)

