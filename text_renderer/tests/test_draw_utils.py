from pathlib import Path

from PIL import ImageFont

from text_renderer.utils.draw_utils import draw_text_on_bg_hv
from text_renderer.utils.font_text import FontText, font_getsize


FONT_PATH = Path(__file__).resolve().parents[2] / "lmdbs/lmdbs/simhei.ttf"


def test_horizontal_padding_uses_one_character_width():
    font = ImageFont.truetype(str(FONT_PATH), 30)
    text = "1234567890"
    font_text = FontText(font, text, str(FONT_PATH), horizontal=True)

    mask, bbox, _ = draw_text_on_bg_hv(font_text, char_spacing=0)

    sizes = [font_getsize(font, char) for char in text]
    single_char_width = max(width for width, _ in sizes)
    assert mask.width == sum(width for width, _ in sizes) + 2 * single_char_width
    assert mask.height == max(height for _, height in sizes) + 4 * single_char_width

    alpha_bbox = mask.getchannel("A").getbbox()
    left, top, right, bottom = alpha_bbox
    assert bbox == [
        [left, top],
        [right - 1, top],
        [right - 1, bottom - 1],
        [left, bottom - 1],
    ]
