from dataclasses import dataclass

from PIL.ImageFont import FreeTypeFont
from fontTools.ttLib import ttFont


@dataclass
class FontText:
    font: FreeTypeFont
    text: str
    font_path: str
    horizontal: bool = True

    @property
    def xy(self):
        offset = self.font.getoffset(self.text)
        left, top, right, bottom = self.font.getmask(self.text).getbbox()
        return 0 - offset[0] - left, 0 - offset[1]

    @property
    def offset(self):
        return self.font.getoffset(self.text)

    # @property
    # def size(self) -> [int, int]:
    #     """
    #     Get text size without offset

    #     Returns:
    #         width, height
    #     """
    #     if self.horizontal:
    #         offset = self.font.getoffset(self.text)
    #         size = self.font.getsize(self.text)
    #         width = size[0] - offset[0]
    #         height = size[1] - offset[1]
    #         left, top, right, bottom = self.font.getmask(self.text).getbbox()
    #         return right - left, height
    #     else:
    #         widths = [self.font.getsize(c)[0] - self.font.getoffset(c)[0] for c in self.text]
    #         width = max(widths)
    #         height = sum([self.font.getsize(c)[1] for c in self.text]) - self.font.getoffset(self.text[0])[1]
    #         return height, width
        
    @property
    def size(self) -> [int, int]:
        if self.horizontal:
            # 使用 getbbox() 替代 getsize() 和 getoffset()
            bbox = self.font.getbbox(self.text)
            width = bbox[2] - bbox[0]  # right - left
            height = bbox[3] - bbox[1] # bottom - top
            return width, height
        else:
            # 垂直文本处理
            char_widths = []
            char_heights = []
            
            for char in self.text:
                char_bbox = self.font.getbbox(char)
                char_width = char_bbox[2] - char_bbox[0]
                char_height = char_bbox[3] - char_bbox[1]
                char_widths.append(char_width)
                char_heights.append(char_height)
                
            total_height = sum(char_heights)
            max_char_width = max(char_widths) if char_widths else 0
            return total_height, max_char_width

if __name__ == '__main__':

    from PIL import ImageFont

    font = FreeTypeFont(r'D:\lxd_code\OCR\OCR_SOURCE\font\font_set\藏文\ctrc-betsu.ttf', 100)
    fonttext = FontText(font,'སྲུང་སྐྱོང་།ལུང་ཧྲེང་ཧྭ་འགྱིག་སོགས་ཁེ་ལས་ལ་བཅར་འདྲི',r'D:\lxd_code\OCR\OCR_SOURCE\font\font_set\藏文\ctrc-betsu.ttf')




    print(fonttext.size)
    print(fonttext.xy)
    print(fonttext.offset)