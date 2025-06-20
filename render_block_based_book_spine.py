import copy
import faulthandler
import itertools
import os
import random
import traceback
from abc import ABC
from pathlib import Path
from typing import Literal, Dict, Tuple, Union
from os import path as osp
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from PIL import Image as PILImage
from lv_tools.cores.poly_ops import rectangle2points
from lv_tools.cores.str_utils import rand_str, is_need_rotate
from lv_tools.data_parsing.alva_json_ops import get_book_spines_from_jd
from lv_tools.data_parsing.labelme_json_constructor import construct_labelme_jd
from lv_tools.dataset_io.data_loader import ImgArrayJDLoader, ImgJsonLoader
from lv_tools.dataset_io.data_saver import LmdbJsonLSaver, LmdbSaver, JsonLSaver
from lv_tools.dataset_io.namer import NameIt
from lv_tools.font.safe_font import FontChoice
from lv_tools.img_tools.bg_gen import BgGener, Block_BG
from lv_tools.img_tools.img_paste import points_shift
from text_renderer.config.block_based_config import create_splitter, FONT_COLOR_CONFIG
from text_renderer.utils import FontText
from text_renderer.utils.draw_utils import transparent_img, Imgerror

faulthandler.enable()

'''
书脊与文本的交互，书脊长宽限制文本总字数。进行文本选择。
'''

'''
职责划分：
文本切分类。（把长文本切分为基本等长的短文本）-》包括依据单词为单位切分的英文，字符为单位的中文，语义为单位的作者等。
    -》 每个单位统一成列表元素。
        * ['This','is','a','sample']
        * ['这','是','一','个','例','子']
        * ['Sophia Turner','David Beckham','Olivia Rodriguez','Sarah Jessica Parker']

文本处理类。包括英文的首字母大写，全字母大写，全字母小写等。英语字母内添加空格来控制字间距等。

'''

'''
在给定画布大小上，书写特定文本，自动适配行数，以充分利用画布空间。

1. 先初始化行数n为1.
2. 给定画布，将画布大小按照行高平均分为n行。
3. 从给定文本列表中，随机选择一个文本，文本按照要书写的行数进行划分（切分点只能在空格处，不能把单词切开。）大致分配均匀即可。
4. 文本切分后用最长的那个文本在单行高度上进行书写测试，文字由1号子逐步放大，直到高度达到阈值，或者宽度达到阈值进行文字大小调整。
文字大小调整后，如果w方向先达到阈值而h方向远小于高度百分比或者某绝对像素，说明文字太多，一行写不下，就将画布在高度方向平均分为n+1行，文本分为n+1块。
重复2过程，知道宽度达到阈值前，可满足高度要求。
--》
5.如果行数增加到行高太小，就放弃对应字数的条目。
6.重选字数小于当前文本字数的某条文本进行1-5过程。

##长宽比是个相对概念，文本串的长宽比 过高于 画布长宽比，就需要分行。以宽度为基准，假设背景和文本的宽都是1，则长宽比的比较，就简化为长的比较。（变化中寻求不变量）

'''

'''
写不下就截取，截取到不可再截的单位元素还超界的话，就放弃相关item。

每一个类别的信息的最长者做代表，进行区域预估。  权重调节各个类别的预计文字大小。
比如：（书名：）
'''


class TextSize:
    def __init__(self,oritation:Literal['h','v','auto']):
        self.oritation = oritation





def get_text_size_hv(
        font_text: FontText,
        char_spacing: Union[float, Tuple[float, float]] = -1,
) -> tuple[int,int]:

    chars_size = []
    widths = []
    heights = []

    for c in font_text.text:
        size = font_text.font.getsize(c)
        if is_need_rotate(c):
            chars_size.append(size)
            widths.append(size[0])
            heights.append(size[1])
        else:
            chars_size.append((size[1], size[0]))
            widths.append(size[1])
            heights.append(size[0])

    width = sum(widths)
    height = max(heights)

    char_spacings = []

    cs_height = font_text.size[1]

    char_spacings.extend([int(char_spacing * cs_height)]*len(font_text.text))

    width += sum(char_spacings[:-1])


    return width, height








def get_text_size(text: str, font: ImageFont) -> tuple[int, int]:
    """给定文本和字体，获取文本的宽度和高度"""
    bbox = font.getbbox(text)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    return w, h


def find_max_allowed_font_size(text: str, max_width: int, max_height: int, font_path: str, step: int = 3):
    """找到适合给定空间的最大字体大小,在实际书写时，未必按照最大的来写，而是在一定范围内随机选择一个字体大小"""
    font_size = 9
    last_valid = None
    if not text:
        raise ValueError("Text cannot be empty")
    # text为空会导致死循环。
    while True:
        font = ImageFont.truetype(font_path, font_size)
        text_width, text_height = get_text_size(text, font)
        #
        if text_width > max_width or text_height > max_height:
            break

        last_valid = font_size
        font_size += step

    return last_valid if last_valid else 1


class EntireImgRender:
    '''

    1.
    2.记录切片的投影和反投影矩阵
    todo  3.下一阶段：书脊两侧进行阴影追加（以模拟一些书脊边缘）->当前不需要，我们目前不用大模型做书脊检测
    '''

    '''
    将整书图投影书脊的逻辑与书脊写字的逻辑串起来，可以防止之前出现过的过度缩放问题。多大的空间写多大的字，而不是像之前的方案那样，
    写到一个通用书脊上再具体贴回到高矮胖瘦随机的书脊上。
    '''

    def __init__(self, data_loader: ImgArrayJDLoader):
        self.data_loader = data_loader

    def __call__(self):
        img_arr, jd = self.data_loader.choice()
        img_path = jd['image_path']
        book_spines = get_book_spines_from_jd(jd)
        for book_spine in book_spines:
            '''
            自适应扩充投影，可能会导致返回时，书脊之间的覆盖。
            所以，如果我们不关注图书检测的话，应该基于原始标注框进行投影，然后将书脊返回到等大的自选的背景图上。
            这样可以保证书脊之间的不重叠。
            
            '''
            pass


# class BlockRender:
#     '''
#     1.给定文本和切片，自动选择合适的行数进行渲染 (切片的方向由外部控制)
#     2. 写不下时，需要一些文本裁切手段
#     3. RenderOneLine 是其成员。
#     4. block内的所有文字应该使用同一种字体，只是字号随机向下抖动，大小写随机变化，颜色随机变化。  字体事关
#     '''
#     '''
#     将文本按照分隔符切分开后，均匀分配到 min(n,len(words))行。
#     分隔符：
#         标题中的【：】【空格】
#         作者中的【,】
#         索书号中的【/】
#     '''
#     '''
#     块的渲染应该知道自己属于哪个种类的文本信息。然后针对相应种类的信息做出对应的行为。
#     比如：
#     大小控制，分隔符控制，颜色控制，字体控制，文本间距控制等。
#     '''
#
#     def __init__(self, safe_color: SafeTextColorCfg = SafeTextColorCfg(),
#                  info_class: Literal['0主标题', '1副标题', '2分辑号', '3版本', '4丛书项', '5作者', '6出版社', '7索书号',
#                  '8杂项'] = None):
#         self.safe_color = safe_color
#         self.info_class = info_class
#         self.delimiter = self._get_delimiter()
#
#     def _get_delimiter(self) -> str:
#         '''
#         1. 标题中的【：】【空格】
#         2. 作者中的【,】
#         3. 索书号中的【/】
#         '''
#         if self.info_class in ['0主标题', '1副标题']:
#             return ' '
#         elif self.info_class == '5作者':
#             return ','
#         elif self.info_class == '7索书号':
#             return '/'
#         else:
#             return ' '
#
#     def split_text(self, text: str, n: int) -> list:
#         """将文本按照分割符号至多分割成n行，尽量均匀并不切断单词"""
#         '''
#         原则上，text切分到最小时，如果还无法满足需求，那再切就没有意义了。即：逻辑上不应该出现 【n > 分隔符数量】 的情况。
#         '''
#         words = self._break_into_pieces(text)
#         if len(words) <= n:
#             return words
#
#         avg = len(words) // n
#         remainder = len(words) % n
#         result = []
#         current = 0
#         for i in range(n):
#             length = avg + 1 if i < remainder else avg
#             joinner = self.delimiter
#             result.append(joinner.join(words[current:current + length]))
#             current += length
#         return result
#
#     def get_max_divisions(self, text: str) -> int:
#         words = self._break_into_pieces(text)
#         return len(words)
#
#     def _break_into_pieces(self, text: str) -> list:
#         words = text.split(self.delimiter)
#         return words
#
#     def layout_text(self, img_arr: np.ndarray, text: str, min_line_height: int = 15, height_threshold: float = 0.39,
#                     font_path: str = None):
#         """自动排版文本到画布
#         """
#         '''
#         1.是否要切分，是否可用的两个依据：
#             * 能不能更好（切分后是否会在更大的字体或占比下，恰当显示文本）
#             * 能不能使用（如果不可再切（切到分割最大），文字大小是否可用）
#         '''
#
#         canvas_height, canvas_width = img_arr.shape[:2]
#         # todo 索书号背景测试
#         img_arr = cv2.imread(r'F:\dataset\OCR\callnumber_gen\callnumber_bg_normal\IMG_20230727_134503_tingting0727_v1.jpg')
#         img_arr = cv2.rotate(img_arr, cv2.ROTATE_90_COUNTERCLOCKWISE)
#         img_arr = cv2.resize(img_arr,dsize=(canvas_width,canvas_height))
#
#         n = 1
#         while True:
#             line_height = canvas_height / n
#             # 字体太小就没有再分的必要了
#             if line_height < min_line_height:
#                 return img_arr, []
#
#             lines = self.split_text(text, n)
#
#             longest_line = max(lines, key=lambda x: len(x))
#
#             font_size = find_max_allowed_font_size(
#                 longest_line,
#                 int(canvas_width * random.uniform(.7, .85)),
#                 int(line_height * random.uniform(.75, .9)),
#                 font_path,
#                 step=3
#             )
#
#             # 验证字体适配情况
#             font = ImageFont.truetype(font_path, font_size)
#             bbox = font.getbbox(longest_line, anchor='lt')
#             text_height = bbox[3] - bbox[1]
#
#             height_ratio = text_height / line_height
#
#             # 判断是否需要增加行数
#             if height_ratio < height_threshold and n < self.get_max_divisions(text):
#                 n += 1
#                 continue
#
#             shapes = []
#             # 行高分配
#             line_height = int(text_height + min(line_height - text_height, text_height * random.uniform(0.15, 0.5)))
#             block_core_height = int(line_height * n)
#
#             h_start = int(canvas_height / 2 - block_core_height / 2)
#
#             # 根据文本高度，合理安排上下间隔，再次分配书写位置。
#             positions = [(0, h_start + int(i * line_height), canvas_width, h_start + int((i + 1) * line_height))
#                          for i in range(n)]
#             result = {
#                 'text': lines,
#                 'font_size': font_size,
#                 'positions': positions,
#                 'n': n
#             }
#
#             img = Image.fromarray(img_arr)
#             safe_color = self.safe_color.get_color(img)
#             draw = ImageDraw.Draw(img)
#
#             for line, (x1, y1, x2, y2) in zip(result['text'], result['positions']):
#                 if not line:
#                     continue
#                 shape = {}
#                 font_size = find_max_allowed_font_size(
#                     line,
#                     int(canvas_width * random.uniform(.9, .92)),
#                     int(line_height * random.uniform(.88, .92)),
#                     font_path,
#                     step=3
#                 )
#
#                 font = ImageFont.truetype(font_path, font_size)
#
#                 # 计算文本位置（垂直居中）
#                 bbox = font.getbbox(line, anchor='lt')
#
#                 text_width = bbox[2] - bbox[0]
#                 text_height = bbox[3] - bbox[1]
#                 x = x1 + (x2 - x1 - text_width) / 2 - bbox[0]  # 起笔位置跟文本边界并不一定一样。
#                 y = y1 + (y2 - y1 - text_height) / 2 - bbox[1]
#
#                 bbox = draw.textbbox((x, y), line, font=font, anchor='lt')  # bbox 必须通过计算得到，不是简单的起始点和宽高可以直接计算。
#                 draw.text((x, y), line, font=font, fill=safe_color, stroke_width=0, anchor='lt')
#                 # print('书写完成')
#                 shape['label'] = line
#                 shape['points'] = rectangle2points([bbox[:2], bbox[2:]])
#                 shape['font_name'] = os.path.basename(font_path)
#                 shape['font_size'] = font_size
#                 shape['group_id'] = self.info_class
#                 shapes.append(shape)
#             # for shape in shapes:
#             #     img = cv2.polylines(np.array(img), [np.array(shape['points'], np.int32)], True, (255, 0, 0), 2)
#
#             img = np.array(img)
#
#             # img = cv2.polylines(img,[np.array(topleft_wh2points((0,0,canvas_width,canvas_height)),np.int32)],True,(255,255,0),2,lineType=2)
#
#             img = Image.fromarray(img)
#
#             return img, shapes

class BlockRender:
    '''
    1.给定文本和切片，自动选择合适的行数进行渲染 (切片的方向由外部控制)
    2. 写不下时，需要一些文本裁切手段
    3. RenderOneLine 是其成员。
    4. block内的所有文字应该使用同一种字体，只是字号随机向下抖动，大小写随机变化，颜色随机变化。  字体事关
    '''
    '''
    将文本按照分隔符切分开后，均匀分配到 min(n,len(words))行。
    分隔符：
        标题中的【：】【空格】
        作者中的【,】
        索书号中的【/】
    '''
    '''
    块的渲染应该知道自己属于哪个种类的文本信息。然后针对相应种类的信息做出对应的行为。
    比如：
    大小控制，分隔符控制，颜色控制，字体控制，文本间距控制等。
    '''
    callnumber_bg_root = r'F:\dataset\OCR\callnumber_gen\callnumber_bg_and_wh'
    img_json_loader = ImgJsonLoader(callnumber_bg_root)
    callnumber_bg_gener = Block_BG(img_json_loader)

    def __init__(self,
                 info_class: Literal['0主标题', '1副标题', '2分辑号', '3版本', '4丛书项', '5作者', '6出版社', '7索书号',
                 '8杂项'] = None):
        self.safe_color = FONT_COLOR_CONFIG[info_class]
        self.info_class = info_class

    def layout_text(self, img_arr: np.ndarray, text: str, min_line_height: int = 15, height_threshold: float = 0.39,
                    font_path: str = None):
        """自动排版文本到画布
        """
        '''
        1.是否要切分，是否可用的两个依据：
            * 能不能更好（切分后是否会在更大的字体或占比下，恰当显示文本）
            * 能不能使用（如果不可再切（切到分割最大），文字大小是否可用）
        '''
        splitter = create_splitter(self.info_class, text)
        canvas_height, canvas_width = img_arr.shape[:2]
        cor_h, cor_w = canvas_height, canvas_width
        # todo 是否换背，换背来源 :索书号背景测试
        # img_arr = cv2.imread(
        #     r'F:\dataset\OCR\callnumber_gen\callnumber_bg_normal\IMG_20230727_134503_tingting0727_v1.jpg')
        # todo 这只是临时解决方案，目前没想到怎么抽象出这个问题。
        if self.info_class == '7索书号':
            # callnumber_bg_root = r'F:\dataset\OCR\callnumber_gen\callnumber_bg_and_wh'
            # img_json_loader = ImgJsonLoader(callnumber_bg_root)
            # callnumber_bg_gener = Block_BG(img_json_loader)
            # s = time.time()
            img_arr, (w, h) = self.callnumber_bg_gener.get_bg_and_wh()
            img_h, img_w = img_arr.shape[:2]
            img_arr = cv2.rotate(img_arr, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img_arr = cv2.resize(img_arr, dsize=(canvas_width, canvas_height))
            cor_w = w * canvas_width / img_w
            cor_h = h * canvas_height / img_h
            # print(time.time()-s)

        n = 1
        while True:
            # line_height = canvas_height / n
            line_height = cor_h / n
            # 字体太小就没有再分的必要了
            if line_height < min_line_height:
                return img_arr, []

            lines = splitter.split_text(n)

            longest_line = max(lines, key=lambda x: len(x))

            # todo 给出最大宽高
            font_size = find_max_allowed_font_size(
                longest_line,
                int(cor_w),
                int(line_height * random.uniform(.8, .95)),
                font_path,
                step=3
            )

            # 验证字体适配情况
            font = ImageFont.truetype(font_path, font_size)
            bbox = font.getbbox(longest_line, anchor='lt')
            text_height = bbox[3] - bbox[1]

            height_ratio = text_height / line_height

            # 判断是否需要增加行数
            if height_ratio < height_threshold and n < splitter.get_max_divisions():
                n += 1
                continue

            shapes = []
            # 行高分配
            line_height = int(text_height + min(line_height - text_height, text_height * random.uniform(0.15, 0.5)))
            block_core_height = int(line_height * n)

            h_start = int(canvas_height / 2 - block_core_height / 2)

            # 根据文本高度，合理安排上下间隔，再次分配书写位置。
            positions = [(0, h_start + int(i * line_height), canvas_width, h_start + int((i + 1) * line_height))
                         for i in range(n)]
            result = {
                'text': lines,
                'font_size': font_size,
                'positions': positions,
                'n': n
            }

            img = Image.fromarray(img_arr)
            safe_color = self.safe_color.get_color(img)
            draw = ImageDraw.Draw(img)

            for line, (x1, y1, x2, y2) in zip(result['text'], result['positions']):
                if not line:
                    continue
                shape = {}
                font_size = find_max_allowed_font_size(
                    line,
                    int(canvas_width * random.uniform(.9, .92)),
                    int(line_height * random.uniform(.88, .92)),
                    font_path,
                    step=3
                )

                font = ImageFont.truetype(font_path, font_size)

                # 计算文本位置（垂直居中）
                bbox = font.getbbox(line, anchor='lt')

                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                if text_height < min_line_height:
                    continue

                x = x1 + (x2 - x1 - text_width) / 2 - bbox[0]  # 起笔位置跟文本边界并不一定一样。
                y = y1 + (y2 - y1 - text_height) / 2 - bbox[1]

                bbox = draw.textbbox((x, y), line, font=font, anchor='lt')  # bbox 必须通过计算得到，不是简单的起始点和宽高可以直接计算。
                draw.text((x, y), line, font=font, fill=safe_color, stroke_width=0, anchor='lt')
                # print('书写完成')
                shape['label'] = line
                shape['points'] = rectangle2points([bbox[:2], bbox[2:]])
                shape['font_name'] = os.path.basename(font_path)
                shape['font_size'] = font_size
                shape['group_id'] = self.info_class
                shapes.append(shape)
            # for shape in shapes:
            #     img = cv2.polylines(np.array(img), [np.array(shape['points'], np.int32)], True, (255, 0, 0), 2)

            img = np.array(img)

            # img = cv2.polylines(img,[np.array(topleft_wh2points((0,0,canvas_width,canvas_height)),np.int32)],True,(255,255,0),2,lineType=2)

            img = Image.fromarray(img)

            return img, shapes


class BlockRenderFactory:
    @staticmethod
    def get_block_render(
            info_class: Literal['0主标题', '1副标题', '2分辑号', '3版本', '4丛书项', '5作者', '6出版社', '7索书号',
            '8杂项'] = None) -> BlockRender:
        return BlockRender(info_class=info_class)


class DynamicBookSpineRender:
    '''

    将书脊划分，空间比较合理的分配给各类文本。对对应的每个块进行渲染。

    获取block，结合文本的类别等信息，决定切片是否旋转（这个决定了文字方向）。

    记录旋转与否


    '''
    '''
    暂时实现一个简单的，按照硬编码的比例分配的方案。

    '''

    def __init__(self, book_corbus: dict, book_spine_img: np.ndarray, font_choicer: FontChoice,
                 block_render_factory: BlockRenderFactory = BlockRenderFactory()):
        self.book_corpus = book_corbus
        self.book_spine_img = book_spine_img
        self.book_spine_h, self.book_spine_w = book_spine_img.shape[:2]
        self.font_choicer = font_choicer
        self.block_render_factory = block_render_factory

    def _get_split_info(self):
        '''todo 书脊划分方案
        1.真正的划分应该是根据文本信息的长度，书脊块的大小动态的划分。暂时先按照硬编码的比例分配。
        2. 真正的划分应该基于文本具体包含的类别来。
        -------
        '''
        series_portion = .1
        author_portion = .3

        if self.book_corpus.get('2分辑号'):
            use_series_portion = random.choice([series_portion, 0])
        else:
            use_series_portion = 0

        if self.book_corpus.get('5作者'):
            use_author_portion = random.choice([author_portion, 0])
        else:
            use_author_portion = 0

        v1 = [('0主标题', .45 + (series_portion - use_series_portion) + (author_portion - use_author_portion)),
              ('2分辑号', use_series_portion), ('5作者', use_author_portion), ('7索书号', .15)]
        v2 = [('5作者', use_author_portion),
              ('0主标题', .45 + (series_portion - use_series_portion) + (author_portion - use_author_portion)),
              ('2分辑号', use_series_portion), ('7索书号', .15)]

        return random.choice([v1, v2])

    def render(self):
        book_blocks = self._get_split_info()

        start = 0
        end = 0
        location_info = []
        draws_with_shapes = []
        for info_class, end_portion in book_blocks:
            if not end_portion:
                continue

            flag = 0
            end += int(self.book_spine_w * end_portion)
            location_info.append((info_class, (start, end)))

            block = self.book_spine_img[:, start:end, :]

            if info_class in ['2分辑号', '7索书号'] and random.choice((0, 1)):
                block = np.rot90(block, 3)
                flag = 1

            font_path = FONT_LOADER[info_class].choice_safe_font(self.book_corpus[info_class])
            # wrapper.print_stats()
            # font_path = self.font_choice.choice_safe_font(self.book_corpus[info_class])
            if not font_path:
                continue

            block_render = self.block_render_factory.get_block_render(info_class)

            drawed_img, shapes = block_render.layout_text(block,
                                                          text=self.book_corpus[info_class], font_path=font_path)
            if flag == 1:
                w, h = drawed_img.size
                src = [[0, 0], [w, 0], [w, h], [0, h]]
                dst = [[[0, 0], [h, 0], [h, w], [0, w]][i % 4] for i in range(3, 3 + 4)]

                M = cv2.getPerspectiveTransform(np.array([src], np.float32), np.array([dst], np.float32))
                for shape in shapes:
                    shape['points'] = cv2.perspectiveTransform(np.array(shape['points'], np.float32).reshape(1, -1, 2),
                                                               M).reshape(-1, 2)
                    # 点序校准
                    shape['points'] = [shape['points'][i % 4] for i in range(1, 1 + 4)]

                drawed_img = drawed_img.transpose(Image.ROTATE_90)

            # for shape in shapes:
            #     drawed_img = cv2.polylines(np.array(drawed_img), [np.array(shape['points'], np.int32)], True, (255, 0, 0), 2)
            #     start_points = np.array(shape['points'][0],np.int32).tolist()
            #     drawed_img = cv2.circle(drawed_img, start_points, 2,
            #                                (255, 0, 0), 2)
            start = end
            draws_with_shapes.append((drawed_img, shapes))

        shift_value = 0
        book_spine_shapes = []
        for img, shapes in draws_with_shapes:

            for shape in shapes:
                shape['points'] = points_shift(shape['points'], shift_value, 0)
                book_spine_shapes.append(shape)
            # new_draws_with_shapes.append((img,shapes))
            shift_value += img.size[0]

        ret_img_arr = np.concatenate([img_pair[0] for img_pair in draws_with_shapes], axis=1)

        # for shape in book_spine_shapes:
        #     drawed_img = cv2.polylines(np.array(ret_img_arr), [np.array(shape['points'], np.int32)], True, (255, 0, 0), 2)
        #     start_points = np.array(shape['points'][0],np.int32).tolist()
        #     ret_img_arr = cv2.circle(drawed_img, start_points, 2,
        #                                (255, 0, 0), 2)

        return ret_img_arr, book_spine_shapes


class OneLineRender(ABC):
    '''
    常规或多变的形式，渲染一行高度的区域。
    1. 需给出整行文本的bbox.
    2. 给出渲染后的文本的大小写情况。
    '''
    pass


class RenderOneLineMixStyle:
    '''
    给定一行文本进行多色，多字体，介词可能略小的多形态渲染
    '''
    pass


class RenderOneLineConsistantStyle:
    pass


class PatternPaster:
    '''
    只关乎真实模拟，不关乎检测识别，最后阶段再管。
    1.书脊留白处增加图案。
    '''
    pass


def render_a_book_spine(item_dict: dict, bg_img_arr: np.ndarray,
                        font_choicer: FontChoice):
    book_corpus = {
        '0主标题': str(item_dict['0主标题']),
        '5作者': str(item_dict['5作者']),
        '2分辑号': random.choice(['volume ', 'Vol. ', '']) + str(random.randint(1, 300)),
        '7索书号': str(item_dict['7索书号'])}

    book_spine_render = DynamicBookSpineRender(book_corpus,
                                               bg_img_arr,
                                               block_render_factory=BlockRenderFactory(),
                                               font_choicer=font_choicer)
    ret_img_arr, book_spine_shapes = book_spine_render.render()
    h, w = ret_img_arr.shape[:2]
    src = [[0, 0], [w, 0], [w, h], [0, h]]
    dst = [[[0, 0], [h, 0], [h, w], [0, w]][i % 4] for i in range(1, 1 + 4)]

    M = cv2.getPerspectiveTransform(np.array([src], np.float32), np.array([dst], np.float32))
    for shape in book_spine_shapes:
        shape['points'] = cv2.perspectiveTransform(np.array(shape['points'], np.float32).reshape(1, -1, 2),
                                                   M).reshape(-1, 2).tolist()
        # 点序校准
        shape['points'] = [shape['points'][i % 4] for i in range(3, 3 + 4)]

    drawed_img = Image.fromarray(ret_img_arr).transpose(Image.ROTATE_270)

    w, h = drawed_img.size

    randstr = rand_str(5)

    img_path_name = randstr + '.jpg'
    labelme_jd = construct_labelme_jd(shapes=book_spine_shapes, imagePath=img_path_name, imageHeight=h, imageWidth=w)

    return np.array(drawed_img), labelme_jd


if __name__ == "__main__":
    # bg_root = r'D:\lxd_code\bar_dm\dm_bar_base\indoorCVPR_09\Images'
    bg_root = r'D:\lxd_code\OCR\OCR_SOURCE\bg\bg_ori'
    eng_font_root: str = r'D:\lxd_code\OCR\OCR_SOURCE\font\font_set\english\jinke_miaomu_done\english_miaomu'

    callnumber_font_root = r'D:\lxd_code\OCR\OCR_SOURCE\font\font_set\索书号可用字体'
    en_font_choicer = FontChoice(eng_font_root)
    callnumber_font_choicer = FontChoice(callnumber_font_root)
    FONT_LOADER: Dict[Literal['0主标题', '1副标题', '2分辑号', '3版本', '4丛书项', '5作者', '6出版社', '7索书号',
    '8杂项'], FontChoice] = {
        "0主标题": en_font_choicer,
        "1副标题": en_font_choicer,
        "2分辑号": en_font_choicer,
        "3版本": en_font_choicer,
        "4丛书项": en_font_choicer,
        "5作者": en_font_choicer,
        "6出版社": en_font_choicer,
        "7索书号": callnumber_font_choicer,
        "8杂项": en_font_choicer
    }

    img_paths = list(Path(bg_root).glob('*.jpg'))
    data_num = 1 * 10 ** 6
    count = 0
    stem = NameIt(task_name='book_spine_info', data_source='syn', feature='en_book_openlib', data_num=data_num)
    lmdb_save_path = rf'F:\dataset\OCR\3-2.book_info_classes\syn_book_spine_pieces_with_callnumber\{stem}'
    #

    mimic_spine = BgGener(bg_root)
    lmdb_saver = LmdbSaver(lmdb_save_path)
    jsonl_saver = JsonLSaver(os.path.join(lmdb_save_path, f'{os.path.basename(lmdb_save_path)}.jsonl'),
                             cache_capacity=500)
    data_saver = LmdbJsonLSaver(lmdb_saver, jsonl_saver)
    # df = pd.read_feather(r'F:\dataset\OCR\foreign_book\english_book\abdallahwagih-7K\openlibrary_9780-with_callnumber.feather')
    df = pd.read_feather(
        r'F:\dataset\OCR\foreign_book\english_book\openlibrary\v5-openlibrary_9780-9781-with_callnumber_norm_author-1230W.feather')

    font_choicer = FontChoice(eng_font_root)

    with data_saver as saver:

        for index, row in tqdm(itertools.cycle(df.iterrows())):
            try:
                if count > data_num:
                    break

                row = row.dropna()
                h, w = (random.randint(70, 150), 1200)
                # h, w = (random.randint(30, 50), 1200)
                spine_arr = mimic_spine((h, w), show_poly='')

                img_arr, jd = render_a_book_spine(item_dict=row, bg_img_arr=spine_arr,
                                                  font_choicer=font_choicer)

                saver.put(img_arr, jd)
                count += 1

            except:
                traceback.print_exc()
                continue
