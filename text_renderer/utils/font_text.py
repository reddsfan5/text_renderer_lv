from dataclasses import dataclass

from PIL.ImageFont import FreeTypeFont
from fontTools.ttLib import ttFont


def font_getsize(font: FreeTypeFont, text: str):
    """Return the legacy Pillow ``getsize`` result on old and new versions."""
    if hasattr(font, "getsize"):
        return font.getsize(text)
    _, _, right, bottom = font.getbbox(text)
    return right, bottom


def font_getoffset(font: FreeTypeFont, text: str):
    """Return the legacy Pillow ``getoffset`` result on old and new versions."""
    if hasattr(font, "getoffset"):
        return font.getoffset(text)
    left, top, _, _ = font.getbbox(text)
    return left, top


@dataclass
class FontText:
    font: FreeTypeFont
    text: str
    font_path: str
    horizontal: bool = True
    rotate_chars: bool = True

    @property
    def xy(self):
        offset = font_getoffset(self.font, self.text)
        left, top, right, bottom = self.font.getmask(self.text).getbbox()
        return 0 - offset[0] - left, 0 - offset[1]

    @property
    def offset(self):
        return font_getoffset(self.font, self.text)

    @property
    def size(self) -> [int, int]:
        """
        Get text size without offset

        Returns:
            width, height
        """
        if self.horizontal:
            offset = font_getoffset(self.font, self.text)
            size = font_getsize(self.font, self.text)
            width = size[0] - offset[0]
            height = size[1] - offset[1]
            left, top, right, bottom = self.font.getmask(self.text).getbbox()
            return right - left, height
        else:
            widths = [
                font_getsize(self.font, c)[0] - font_getoffset(self.font, c)[0]
                for c in self.text
            ]
            width = max(widths)
            height = sum(font_getsize(self.font, c)[1] for c in self.text)
            height -= font_getoffset(self.font, self.text[0])[1]
            return height, width
if __name__ == '__main__':

    from PIL import ImageFont

    font = FreeTypeFont(r'D:\lxd_code\OCR\OCR_SOURCE\font\font_set\藏文\ctrc-betsu.ttf', 100)
    fonttext = FontText(font,'སྲུང་སྐྱོང་།ལུང་ཧྲེང་ཧྭ་འགྱིག་སོགས་ཁེ་ལས་ལ་བཅར་འདྲི',r'D:\lxd_code\OCR\OCR_SOURCE\font\font_set\藏文\ctrc-betsu.ttf')




    print(fonttext.size)
    print(fonttext.xy)
    print(fonttext.offset)
