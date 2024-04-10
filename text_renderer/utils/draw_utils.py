import copy
import os
import random
from os import path as osp
from typing import Tuple, Union
import sys
import numpy as np
from PIL import ImageDraw, Image
from PIL.Image import Image as PILImage

from costum_utils.text_segmentation import limit_text_and_add_space
from text_renderer.utils.font_text import FontText
if (lv_tools:=r'D:\lxd_code\lv_tools') not in sys.path:
    sys.path.append(lv_tools)
from task_ocr_text_render.digit_str_gen import number_to_text,number_to_text_with_parenthesis
CLOSE_APOSTROPHE = {'【', '】', '（', '）', '《', '》', '“', '”', '〔', '〕', '〈', '〉', '「', '」', '『', '』', '〖',
                    '〗'}  # ord大于256的闭合标点， '{', '}'不分全角半角，其ord小于256。


class Imgerror(RuntimeError):
    def __init__(self, arg=None):
        self.args = arg


def need_rotate(char):
    # 1. 数字，英文，闭合标点 不需要旋转。 数字切出来，基本都是非旋转的。
    # 2. 非闭合标点 旋转后最好居中（这个可先不管）
    # 3.中文需要旋转
    # return False
    if ord(char) < 256 or char in CLOSE_APOSTROPHE:
        return False
    else:
        return True


def transparent_img(size: Tuple[int, int]) -> PILImage:
    """

    Args:
        size: (width, height)

    Returns:

    """
    return Image.new("RGBA", (size[0], size[1]), (255, 255, 255, 0))


def draw_text_on_bg_hv(
        font_text: FontText,
        text_color: Tuple[int, int, int, int] = (0, 0, 0, 255),
        char_spacing: Union[float, Tuple[float, float]] = -1,
        save_dir: str = ''
) -> PILImage:
    """

    Parameters
    ----------
    font_text : FontText
    text_color : RGBA
        Default is black
    char_spacing : Union[float, Tuple[float, float]]
        Draw character with spacing. If tuple, random choice between [min, max)
        Set -1 to disable

    Returns
    -------
        PILImage:
            RGBA Pillow image with text on a transparent image
    -------

    """
    if char_spacing == -1:
        if font_text.horizontal:
            return _draw_text_on_bg(font_text, text_color)
        else:
            char_spacing = 0

    chars_size = []
    widths = []
    heights = []

    for c in font_text.text:
        size = font_text.font.getsize(c)
        if need_rotate(c):
            chars_size.append(size)
            widths.append(size[0])
            heights.append(size[1])
        else:
            chars_size.append((size[1], size[0]))
            widths.append(size[1])
            heights.append(size[0])

    if font_text.horizontal:
        width = sum(widths)
        height = max(heights)
    else:
        width = max(widths)
        height = sum(heights)

    char_spacings = []

    cs_height = font_text.size[1]
    for i in range(len(font_text.text)):
        if isinstance(char_spacing, list) or isinstance(char_spacing, tuple):
            s = np.random.uniform(*char_spacing)
            char_spacings.append(int(s * cs_height))
        else:
            char_spacings.append(int(char_spacing * cs_height))

    if font_text.horizontal:
        width += sum(char_spacings[:-1])
    else:
        height += sum(char_spacings[:-1])

    # 长宽估算，生成掩码
    # text_mask = transparent_img((width, height))
    # x方向两头填充字符数
    x_pad_chars_num = 4
    text_mask = transparent_img((3 * width, x_pad_chars_num * width + height))  # 四周的padding 平均一个height。
    pre_img = copy.deepcopy(text_mask)
    draw = ImageDraw.Draw(text_mask)

    x_start = c_x = width
    y_start = c_y = (x_pad_chars_num // 2) * width
    horizontal_content = []
    if font_text.horizontal:
        y_offset = font_text.offset[1]
        for i, c in enumerate(font_text.text):
            draw.text((c_x, c_y - y_offset), c, fill=text_color, font=font_text.font)

            c_x += chars_size[i][0] + char_spacings[i]
    else:
        x_offset = font_text.offset[0]
        # 纵横书写，预留位置。实现中英文 书脊名字的排列形式。
        vertical_location = []
        vertical_text = []
        for i, c in enumerate(font_text.text):
            if need_rotate(c):
                # 卧倒中文书写
                draw.text((c_x - x_offset, c_y), c, fill=text_color, font=font_text.font)
                if c != ' ' and (np.array(text_mask) == pre_img).all():

                    print(f'{osp.basename(font_text.font_path)}-出现字体残缺不齐全')
                    raise Imgerror()
                else:
                    pre_img = np.array(text_mask)
            else:
                vertical_location.append((c_y, text_mask.width - (c_x - x_offset + widths[i])))
                vertical_text.append(c)

            c_y += chars_size[i][1] + char_spacings[i]
        text_mask = text_mask.rotate(90, expand=True)
        draw2 = ImageDraw.Draw(text_mask)
        pre_img = np.array(draw2)
        for vt, loc in zip(vertical_text, vertical_location):
            draw2.text(loc, vt, fill=text_color, font=font_text.font)
            if vt != ' ' and (np.array(text_mask) == pre_img).all():
                print(f'{osp.basename(font_text.font_path)}-出现字体残缺不齐全')
                raise Imgerror('字体残缺不齐全')
            else:
                pre_img = np.array(text_mask)

    pre_img = np.array(text_mask)[..., :3]
    points = np.argwhere(pre_img < 255)
    xmin = np.min(points[:, 1])
    ymin = np.min(points[:, 0])
    xmax = np.max(points[:, 1])
    ymax = np.max(points[:, 0])
    box_n = np.asarray([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
    if save_dir:
        if not osp.exists(save_dir):
            os.mkdir(save_dir)
        # s = base64.b64encode(os.urandom(3)).decode("utf8")
        # s = s.replace("\\", "").replace("/", "").replace("=","").replace("+","")
        # box det

        # 存储用于字体展示
        text_mask.save(osp.join(save_dir, osp.basename(font_text.font_path) + '.png'))

    # bbox = [[x_start, y_start], [x_start + sum(heights), y_start], [x_start + sum(heights), y_start + max(widths)],
    #         [x_start, y_start + max(widths)]]
    bbox = box_n.tolist()
    font_base = osp.basename(font_text.font_path)

    return text_mask, bbox, font_base



def draw_text_on_bg_with_digit_in_one_line(
        font_text: FontText,
        text_color: Tuple[int, int, int, int] = (0, 0, 0, 255),
        char_spacing: Union[float, Tuple[float, float]] = -1,
        save_dir: str = ''
) -> PILImage:
    """

    Parameters
    ----------
    font_text : FontText
    text_color : RGBA
        Default is black
    char_spacing : Union[float, Tuple[float, float]]
        Draw character with spacing. If tuple, random choice between [min, max)
        Set -1 to disable

    Returns
    -------
        PILImage:
            RGBA Pillow image with text on a transparent image
    -------

    """

    char_spacings = []

    ori_text = font_text.text
    text_with_space = limit_text_and_add_space(ori_text)





    cs_height = font_text.size[1]
    for i in range(len(font_text.text)):
        if isinstance(char_spacing, list) or isinstance(char_spacing, tuple):
            s = np.random.uniform(*char_spacing)
            char_spacings.append(int(s * cs_height))
        else:
            char_spacings.append(int(char_spacing * cs_height))


    # 长宽估算，生成掩码
    # text_mask = transparent_img((width, height))
    # x方向两头填充字符数
    x_pad_chars_num = 4


    text_mask = transparent_img((3 * width, x_pad_chars_num * width + height))  # 四周的padding 平均一个height。
    pre_img = copy.deepcopy(text_mask)
    draw = ImageDraw.Draw(text_mask)

    x_start = c_x = width
    y_start = c_y = (x_pad_chars_num // 2) * width
    horizontal_content = []
    if font_text.horizontal:
        y_offset = font_text.offset[1]
        for i, c in enumerate(font_text.text):
            draw.text((c_x, c_y - y_offset), c, fill=text_color, font=font_text.font)

            c_x += chars_size[i][0] + char_spacings[i]
    else:
        x_offset = font_text.offset[0]
        # 纵横书写，预留位置。实现中英文 书脊名字的排列形式。
        vertical_location = []
        vertical_text = []
        for i, c in enumerate(font_text.text):
            if need_rotate(c):
                # 卧倒中文书写
                draw.text((c_x - x_offset, c_y), c, fill=text_color, font=font_text.font)
                if c != ' ' and (np.array(text_mask) == pre_img).all():

                    print(f'{osp.basename(font_text.font_path)}-出现字体残缺不齐全')
                    raise Imgerror()
                else:
                    pre_img = np.array(text_mask)
            else:
                vertical_location.append((c_y, text_mask.width - (c_x - x_offset + widths[i])))
                vertical_text.append(c)

            c_y += chars_size[i][1] + char_spacings[i]
        text_mask = text_mask.rotate(90, expand=True)
        draw2 = ImageDraw.Draw(text_mask)
        pre_img = np.array(draw2)
        for vt, loc in zip(vertical_text, vertical_location):
            draw2.text(loc, vt, fill=text_color, font=font_text.font)
            if vt != ' ' and (np.array(text_mask) == pre_img).all():
                print(f'{osp.basename(font_text.font_path)}-出现字体残缺不齐全')
                raise Imgerror('字体残缺不齐全')
            else:
                pre_img = np.array(text_mask)

    pre_img = np.array(text_mask)[..., :3]
    points = np.argwhere(pre_img < 255)
    xmin = np.min(points[:, 1])
    ymin = np.min(points[:, 0])
    xmax = np.max(points[:, 1])
    ymax = np.max(points[:, 0])
    box_n = np.asarray([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
    if save_dir:
        if not osp.exists(save_dir):
            os.mkdir(save_dir)
        # s = base64.b64encode(os.urandom(3)).decode("utf8")
        # s = s.replace("\\", "").replace("/", "").replace("=","").replace("+","")
        # box det

        # 存储用于字体展示
        text_mask.save(osp.join(save_dir, osp.basename(font_text.font_path) + '.png'))

    # bbox = [[x_start, y_start], [x_start + sum(heights), y_start], [x_start + sum(heights), y_start + max(widths)],
    #         [x_start, y_start + max(widths)]]
    bbox = box_n.tolist()
    font_base = osp.basename(font_text.font_path)

    return text_mask, bbox, font_base




def draw_text_on_bg(
        font_text: FontText,
        text_color: Tuple[int, int, int, int] = (0, 0, 0, 255),
        char_spacing: Union[float, Tuple[float, float]] = -1,
        save_dir: str = ''
) -> PILImage:
    """

    Parameters
    ----------
    font_text : FontText
    text_color : RGBA
        Default is black
    char_spacing : Union[float, Tuple[float, float]]
        Draw character with spacing. If tuple, random choice between [min, max)
        Set -1 to disable

    Returns
    -------
        PILImage:
            RGBA Pillow image with text on a transparent image
    -------

    """
    if char_spacing == -1:
        if font_text.horizontal:
            return _draw_text_on_bg(font_text, text_color)
        else:
            char_spacing = 0

    chars_size = []
    widths = []
    heights = []

    for c in font_text.text:
        size = font_text.font.getsize(c)
        if need_rotate(c):
            chars_size.append(size)
            widths.append(size[0])
            heights.append(size[1])
        else:
            chars_size.append((size[1], size[0]))
            widths.append(size[1])
            heights.append(size[0])

    if font_text.horizontal:
        width = sum(widths)
        height = max(heights)
    else:
        width = max(widths)
        height = sum(heights)

    char_spacings = []

    cs_height = font_text.size[1]
    for i in range(len(font_text.text)):
        if isinstance(char_spacing, list) or isinstance(char_spacing, tuple):
            s = np.random.uniform(*char_spacing)
            char_spacings.append(int(s * cs_height))
        else:
            char_spacings.append(int(char_spacing * cs_height))

    if font_text.horizontal:
        width += sum(char_spacings[:-1])
    else:
        height += sum(char_spacings[:-1])

    # 长宽估算，生成掩码
    # text_mask = transparent_img((width, height))
    text_mask = transparent_img((3 * width, 2 * width + height))  # 四周的padding 平均一个height。
    draw = ImageDraw.Draw(text_mask)

    c_x = random.randint(0, 2 * width)
    c_y = random.randint(0, 2 * width)
    horizontal_content = []
    if font_text.horizontal:
        y_offset = font_text.offset[1]
        for i, c in enumerate(font_text.text):
            draw.text((c_x, c_y - y_offset), c, fill=text_color, font=font_text.font)
            c_x += chars_size[i][0] + char_spacings[i]
    else:
        x_offset = font_text.offset[0]
        # 纵横书写，预留位置。实现中英文 书脊名字的排列形式。
        vertical_location = []
        vertical_text = []
        for i, c in enumerate(font_text.text):
            if need_rotate(c):
                draw.text((c_x - x_offset, c_y), c, fill=text_color, font=font_text.font)
            else:
                vertical_location.append((c_y, text_mask.width - (c_x - x_offset + widths[i])))
                vertical_text.append(c)

            c_y += chars_size[i][1] + char_spacings[i]
        text_mask = text_mask.rotate(90, expand=True)
        draw2 = ImageDraw.Draw(text_mask)
        for vt, loc in zip(vertical_text, vertical_location):
            draw2.text(loc, vt, fill=text_color, font=font_text.font)
        if save_dir:
            if not osp.exists(save_dir):
                os.mkdir(save_dir)
            # s = base64.b64encode(os.urandom(3)).decode("utf8")
            # s = s.replace("\\", "").replace("/", "").replace("=","").replace("+","")
            text_mask.save(osp.join(save_dir, osp.splitext(osp.basename(font_text.font_path))[0] + '.png'))

    return text_mask


def draw_text_on_bg_backup(
        font_text: FontText,
        text_color: Tuple[int, int, int, int] = (0, 0, 0, 255),
        char_spacing: Union[float, Tuple[float, float]] = -1,
) -> PILImage:
    """

    Parameters
    ----------
    font_text : FontText
    text_color : RGBA
        Default is black
    char_spacing : Union[float, Tuple[float, float]]
        Draw character with spacing. If tuple, random choice between [min, max)
        Set -1 to disable

    Returns
    -------
        PILImage:
            RGBA Pillow image with text on a transparent image
    -------

    """
    if char_spacing == -1:
        if font_text.horizontal:
            return _draw_text_on_bg(font_text, text_color)
        else:
            char_spacing = 0

    chars_size = []
    widths = []
    heights = []

    for c in font_text.text:
        size = font_text.font.getsize(c)
        chars_size.append(size)
        widths.append(size[0])
        heights.append(size[1])

    if font_text.horizontal:
        width = sum(widths)
        height = max(heights)
    else:
        width = max(widths)
        height = sum(heights)

    char_spacings = []

    cs_height = font_text.size[1]
    for i in range(len(font_text.text)):
        if isinstance(char_spacing, list) or isinstance(char_spacing, tuple):
            s = np.random.uniform(*char_spacing)
            char_spacings.append(int(s * cs_height))
        else:
            char_spacings.append(int(char_spacing * cs_height))

    if font_text.horizontal:
        width += sum(char_spacings[:-1])
    else:
        height += sum(char_spacings[:-1])

    # 长宽估算，生成掩码
    text_mask = transparent_img((width, height))
    draw = ImageDraw.Draw(text_mask)

    c_x = 0
    c_y = 0

    if font_text.horizontal:
        y_offset = font_text.offset[1]
        for i, c in enumerate(font_text.text):
            draw.text((c_x, c_y - y_offset), c, fill=text_color, font=font_text.font)
            c_x += chars_size[i][0] + char_spacings[i]
    else:
        x_offset = font_text.offset[0]
        for i, c in enumerate(font_text.text):
            draw.text((c_x - x_offset, c_y), c, fill=text_color, font=font_text.font)
            c_y += chars_size[i][1] + char_spacings[i]
        text_mask = text_mask.rotate(90, expand=True)

    return text_mask


def _draw_text_on_bg(
        font_text: FontText,
        text_color: Tuple[int, int, int, int] = (0, 0, 0, 255),
) -> PILImage:
    """
    Draw text

    Parameters
    ----------
    font_text : FontText
    text_color : RGBA
        Default is black

    Returns
    -------
        PILImage:
            RGBA Pillow image with text on a transparent image
    """
    text_width, text_height = font_text.size
    text_mask = transparent_img((text_width, text_height))
    draw = ImageDraw.Draw(text_mask)

    xy = font_text.xy

    # TODO: figure out anchor
    draw.text(
        xy,
        font_text.text,
        font=font_text.font,
        fill=text_color,
        anchor=None,
    )

    return text_mask


from PIL import Image, ImageDraw, ImageFont

def draw_multiline_text_centered(image_path, text, font_path, font_size):
    # 打开底图
    image = Image.new("RGB", (500, 500), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    # 设置字体
    font = ImageFont.truetype(font_path, font_size)

    # 分割文本为多行
    lines = text.split('\n')
    max_width = 0
    line_heights = []

    # 计算每行的宽度和高度
    for line in lines:
        width, height = draw.textsize(line, font=font)
        max_width = max(max_width, width)
        line_heights.append(height)

    # 计算总高度
    total_height = sum(line_heights)

    # 计算绘制起始位置
    y = (image.height - total_height) // 2

    # 逐行绘制文本
    for line, height in zip(lines, line_heights):
        width, _ = draw.textsize(line, font=font)
        x = (image.width - width) // 2
        draw.text((x, y), line, font=font, fill="black")
        y += height

    # 保存或展示图像
    image.show()

# 示例使用




if __name__ == '__main__':

    '''
    1. 多行书写的方式可较为容易的实现数字横着
    2. 潜在隐患是因为同时书写，如果写空字不容易检查到。（当然，如果字体map文件过滤的好，这个问题基本不用担心）
    3. 需接入到文本替换逻辑和标注框逻辑里。
    4. 需lmdb逻辑。
    
    当然，找到恰当的位置插入，应该省事不少。
    
    '''

    print(number_to_text_with_parenthesis(5))


    # from PIL import Image, ImageDraw, ImageFont
    #
    # # create an image
    # out = Image.new("RGB", (500, 500), (255, 255, 255))
    #
    # # get a font
    # fnt = ImageFont.truetype(r"D:\lxd_code\OCR\OCR_SOURCE\font\font_set - 副本\简体-简体-低风险\粗体\字魂4456号-悠然飘扬体.ttf", 40)
    # # get a drawing context
    # d = ImageDraw.Draw(out)
    #
    # # draw multiline text
    # bbox = d.multiline_textbbox((100, 100), "你\n的\n名\n\n\n字\n26563", font=fnt)
    # d.rectangle(bbox, outline=(0, 0, 255, 255))
    # print(bbox)
    # d.multiline_text((100, 100), "你\n的\n名\n\n\n字\n26563", font=fnt, fill=(0, 0, 0))
    # out.show()

    # draw_multiline_text_centered("path_to_your_image.jpg", "你\n的\n名\n\n\n字\n26563", r"D:\lxd_code\OCR\OCR_SOURCE\font\font_set - 副本\简体-简体-低风险\粗体\字魂4456号-悠然飘扬体.ttf", 20)