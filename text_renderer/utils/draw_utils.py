import copy
import random
from typing import Tuple, Union
from os import path as osp
import PIL.Image
from PIL import ImageDraw, Image
from PIL.Image import Image as PILImage
import numpy as np
import os
import base64
from text_renderer.utils.font_text import FontText

CLOSE_APOSTROPHE = {'【', '】', '（', '）', '《', '》', '“', '”', '〔', '〕', '〈', '〉', '「', '」', '『', '』', '〖',
                    '〗'}  # ord大于256的闭合标点， '{', '}'不分全角半角，其ord小于256。

class Imgerror(RuntimeError):
    def __init__(self, arg=None):
        self.args = arg


def need_rotate(char):
    # 1. 数字，英文，闭合标点 不需要旋转。 数字切出来，基本都是非旋转的。
    # 2. 非闭合标点 旋转后最好居中（这个可先不管）
    # 3.中文需要旋转
    if ord(char)<256 or char in CLOSE_APOSTROPHE:
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
    text_mask = transparent_img((3 * width, 10 * width + height))  # 四周的padding 平均一个height。
    pre_img = copy.deepcopy(text_mask)
    draw = ImageDraw.Draw(text_mask)

    # c_x = random.randint(0,2*width)
    # c_y = random.randint(0,2*width)
    x_start = c_x = width
    y_start = c_y = 5*width
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
                if (np.array(text_mask) == pre_img).all():

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
            if (np.array(text_mask) == pre_img).all():
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


        text_mask.save(osp.join(save_dir, osp.basename(font_text.font_path) + '.png'))

    # bbox = [[x_start, y_start], [x_start + sum(heights), y_start], [x_start + sum(heights), y_start + max(widths)],
    #         [x_start, y_start + max(widths)]]
    bbox = box_n.tolist()
    font_base = osp.basename(font_text.font_path)


    return text_mask, bbox,font_base


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
