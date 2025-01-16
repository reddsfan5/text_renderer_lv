import base64
import os.path
import random
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Tuple, Literal

import PIL.Image
import cv2
import numpy as np
from PIL.Image import Image as PILImage
from PIL.ImageFont import FreeTypeFont
from loguru import logger

from lv_tools.cores.json_io import load_json_to_dict, save_json
from lv_tools.cores.poly_ops import order_rectify, point_distance, is_clockwise
from lv_tools.cores.typing_custom import BoundingBox, Points
from lv_tools.img_tools.img_aug import img_aug
from lv_tools.json_tools.labelme_json_constructor import construct_one_shape, construct_labelme_jd
from lv_tools.task_book_info_gen.bg_gen import MimicSpineBg
from lv_tools.task_book_info_gen.book_info_content import BookSpineContent
from lv_tools.task_book_info_gen.book_info_template import BookInfoTemplate
from lv_tools.task_ocr_text_render.text_preprocess import remove_continue_space
from text_renderer.config import RenderCfg
from text_renderer.render import Render
from text_renderer.utils.bbox import BBox
from text_renderer.utils.draw_utils import Imgerror, draw_text_on_bg_hv_no_pad, draw_text_on_bg_backup
from text_renderer.utils.draw_utils import draw_text_on_bg_multi_line
from text_renderer.utils.errors import PanicError
from text_renderer.utils.math_utils import PerspectiveTransform
from text_renderer.utils.types import FontColor, is_list

class BookSpineRender:
    '''

    此类职责过多，需要解耦
    * 存储器需要解耦出来
        * 怎么存，存成什么格式应该有传入的存储器决定。
        * 存储器不干预文件的细节，其应该由生成器决定。

    {'0主标题': ['马克思恩格斯书信集'],
     '1副标题': [],
     '2分辑号': [],
     '3版本': [],
     '4丛书项': ['马列主义经典著作典藏文库'],
     '5作者': ['(德)卡尔·马克思', '(德)弗里德里希·恩格斯著'],
     '6出版社': ['中央编译出版社'],
     '第二责任者': []})


    '''

    def __init__(self, cfg: RenderCfg,
                 book_spine_content: BookSpineContent,
                 book_info_template: BookInfoTemplate,
                 bg_generator: MimicSpineBg,
                 dst_dir: str):
        # 初始化开销不大
        self.cfg = cfg  # cfg 的问题在于如果没有一个持续维护的文档，时间长了，就不知道这个黑盒子到底携带多少参数了。好处是写代码自由。
        self.corpus = cfg.corpus[0] if isinstance(cfg.corpus, list) and len(cfg.corpus) == 1 else cfg.corpus
        self._corpus_check()
        self.dst_dir = dst_dir
        self.book_spine_content = book_spine_content
        self.book_info_template = book_info_template

        self.bg_gener = bg_generator
        self.text_orientation = 1  # 0为横向，1为纵向

    def _corpus_check(self):
        if is_list(self.corpus) and is_list(self.cfg.corpus_effects):
            if len(self.corpus) != len(self.cfg.corpus_effects):
                raise PanicError(
                    f"corpus length({self.corpus}) is not equal to corpus_effects length({self.cfg.corpus_effects})"
                )

        if is_list(self.corpus) and (
                self.cfg.corpus_effects and not is_list(self.cfg.corpus_effects)
        ):
            raise PanicError("corpus is list, corpus_effects is not list")

        if not is_list(self.corpus) and is_list(self.cfg.corpus_effects):
            raise PanicError("corpus_effects is list, corpus is not list")

    @staticmethod
    def _len_item_title(item: dict) -> int:
        title_text_num = sum([len(title_piece) for title_piece in item['0主标题']])
        return title_text_num


    def adeptive_adjust_bbox_size(self, src_points: BoundingBox, dst_patch_size: tuple[float, float]):
        w, h = dst_patch_size

        (cx, cy), (b_w, b_h), angle = cv2.minAreaRect(np.array(src_points, np.float32))

        ratio = w / h  # 长短边比例。
        rand_ratio = random.uniform(1., 2.)

        if b_w <= b_h:
            dst_bw = b_w
            dst_bh = int(min(b_w * ratio * rand_ratio, b_h))
        else:
            dst_bh = b_h
            dst_bw = int(min(b_h * ratio * rand_ratio, b_w))

        src_points = cv2.boxPoints(((cx, cy), (dst_bw, dst_bh), angle))


        return src_points

    def limit_hw(self, h: int, w: int, target_max_h=1200) -> Tuple[int, int]:
        h_new = min(target_max_h, h)
        w_new = max(60, int(h_new * w / h))
        return h_new, w_new

    def points_resize(self, points: Points, wh_ratio: Tuple) -> Points:
        return [[int(point[0] * wh_ratio[0]), int(point[1] * wh_ratio[1])] for point in points]

    # @retry
    def __call__(self, *args, **kwargs):
        '''
        遍历渲染书脊
        1. 检测框信息包括坐标，文字和文字类别
        2. 书脊信息包括参考模板名字，便于检测
        3.

        '''
        try:
            self.corpus.normalized_corpus = self.book_spine_content.get_one_item_with_title()
            len_book_title = self._len_item_title(self.corpus.normalized_corpus)
            self.book_info_template_path = self.book_info_template.get_template_json_path_by_book_name_length(len_book_title)

            # 核心逻辑

            template_jd = load_json_to_dict(self.book_info_template_path)
            h, w = template_jd['imageHeight'], template_jd['imageWidth']

            # ⭐对目标书籍的大小进行限制，并修改对应json相关信息,掠过索书号，给下一阶段去贴
            bg_h, bg_w = self.limit_hw(h, w)
            h_ratio, w_ratio = bg_h / h, bg_w / w
            new_shapes = []
            for info_index in range(len(template_jd['shapes'])):
                shape = template_jd['shapes'][info_index]
                shape['points'] = self.points_resize(shape['points'], (w_ratio, h_ratio))
                if isinstance(shape.get('group_id'), str) and shape.get('group_id') == '7索书号':
                    continue
                new_shapes.append(shape)
            template_jd['shapes'] = new_shapes
            template_jd['imageHeight'], template_jd['imageWidth'] = bg_h, bg_w

            bg = self.bg_gener((bg_h, bg_w))
            text_len_limit = 1.2
            dst_shapes = []

            for info_index in range(len(template_jd['shapes'])):
                info_class = template_jd['shapes'][info_index]['group_id']


                shape = template_jd['shapes'][info_index]
                src_points = shape['points']

                # 解决异常点数的问题
                if len(src_points) != 4 and len(src_points) > 2:
                    pprint(src_points)
                    min_area_rect = cv2.minAreaRect(np.array(src_points, np.float32))
                    src_points = cv2.boxPoints(min_area_rect)

                # 顺时针校正
                if not is_clockwise(src_points):
                    # src_points = src_points[:1]+src_points[-1:0:-1] # 这种写法不太符合直觉
                    src_points = src_points[:1] + src_points[1:][::-1]

                src_points = order_rectify(src_points)
                # 过滤掉横向文本
                # 根据原始框的长宽比，估计原文的文字方向（除非进行了方向标注，或者某种先验佐证，否则就需要这个估算，无法确知）
                src_w, src_h = point_distance(src_points[0], src_points[1]), point_distance(src_points[1],
                                                                                            src_points[2])
                if src_w > src_h:
                    continue

                # dataframe中取文本
                # todo 根据剩余内容长度动态延长
                if not self.corpus.normalized_corpus.get(info_class):
                    continue
                else:
                    text = ''
                    while self.corpus.normalized_corpus[info_class] and len(text) < text_len_limit:
                        if text:
                            text += ' '
                        text += self.corpus.normalized_corpus[info_class].pop()

                    template_label_len = len(template_jd['shapes'][info_index]['label'])

                    # 标注框应该有自动缩回去的能力

                    # if len(text) < math.sqrt(template_label_len):
                    #     continue
                    # else:
                    cut_len_min = int(template_label_len)
                    cut_len_max = int(template_label_len * text_len_limit)
                    text = text[:random.randint(cut_len_min, cut_len_max)]

                # todo lv 单行，混排文本切换开关：oneline，multiline
                # todo 文本颜色切换
                cropped_bg, text, M_anti, transformed_text_mask, bbox, font_base = self.gen_one_corpus(
                    template_jd, info_index, text, bg, write_mode='oneline', cmap='color')
                img = cropped_bg

                img = img.convert("RGB")
                np_img = np.array(img)
                # np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
                piece_with_black_pad = cv2.warpPerspective(np_img, M_anti, (bg_w, bg_h))
                _, mask_fg = cv2.threshold(cv2.cvtColor(piece_with_black_pad, cv2.COLOR_BGR2GRAY), 1, 255,
                                           cv2.THRESH_BINARY)
                kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask_fg = cv2.erode(mask_fg, kernal, iterations=3)
                _, mask_fg = cv2.threshold(mask_fg, 1, 255, cv2.THRESH_BINARY)
                piece_with_black_pad = cv2.bitwise_and(piece_with_black_pad, piece_with_black_pad, mask=mask_fg)

                mask_bg = ~mask_fg

                bg_with_black_piece = cv2.bitwise_and(bg, bg, mask=mask_bg)
                bg = cv2.add(piece_with_black_pad, bg_with_black_piece)
                # np_img = self.norm(np_img)
                bbox_to_bg = cv2.perspectiveTransform(np.array([bbox], np.float32), M_anti)
                dst_shape = construct_one_shape(label=remove_continue_space(text),
                                                points=np.squeeze(bbox_to_bg).tolist(), group_id=info_class)
                dst_shapes.append(dst_shape)
            time_str = datetime.now().strftime("%Y%m%d-%H%M%S%f")

            template_img_path = self.book_info_template_path.replace('.json', '.jpg')
            # template_img = cv2.imread(template_img_path)

            # merged_show_ret = np.hstack([bg, template_img])

            s = base64.b64encode(os.urandom(5)).decode("utf8")
            s = s.replace("\\", "").replace("/", "").replace("=", "").replace("+", "")
            img_base = f'{s}_{Path(template_img_path).stem}_{time_str}.jpg'

            if not os.path.exists(self.dst_dir):
                os.mkdir(self.dst_dir)
            cv2.imwrite(rf'{self.dst_dir}\{img_base}',
                        bg)

            dst_jd = construct_labelme_jd(dst_shapes, img_base, bg_h, bg_w)
            save_json(os.path.join(self.dst_dir, img_base[:-3] + 'json'), dst_jd)
            # return np_img, text, bbox

        except Exception as e:
            raise Imgerror(e)
            # logger.exception(e)
            # raise e

    def gen_one_corpus(self, jd: dict, info_index: int, text: str, bg: np.ndarray,
                       write_mode: Literal['oneline', 'multiline'] = 'oneline', cmap='color') -> Tuple[
        PILImage, str, PILImage, PILImage]:

        font_text = self.corpus.get_font_text(text)
        # 这个bg即是透视模拟出来的书脊，所有文字都投影到这个书脊上
        pil_bg = PIL.Image.fromarray(bg)
        if cmap == 'color':
            if self.cfg.text_color_cfg is not None:
                text_color = self.cfg.text_color_cfg.get_color(pil_bg)

            # corpus text_color has higher priority than RenderCfg.text_color_cfg
            if self.corpus.cfg.text_color_cfg is not None:
                text_color = self.corpus.cfg.text_color_cfg.get_color(pil_bg)
        else:
            gray_value = random.randint(5, 35)  # 颜色不可以为0，这样影响图像融合的贴图逻辑。
            opac_value = random.randint(245, 255)
            text_color = (gray_value, gray_value, gray_value, opac_value)
        # 书写文本接口,写在透明背景上
        if write_mode == 'oneline':
            if self.text_orientation == 1:
                text_mask, bbox, font_base = draw_text_on_bg_hv_no_pad(
                    font_text, text_color, char_spacing=self.corpus.cfg.char_spacing,
                    save_dir=r'D:\lxd_code\OCR\OCR_SOURCE\font\font_show'
                )
            else:
                font_text.horizontal = True
                text_mask, bbox, font_base = draw_text_on_bg_backup(
                    font_text, text_color, char_spacing=self.corpus.cfg.char_spacing,
                )
        elif write_mode == 'multiline':
            text_mask, bbox, font_base = draw_text_on_bg_multi_line(
                font_text, text_color, char_spacing=self.corpus.cfg.char_spacing,
                save_dir=r'D:\lxd_code\OCR\OCR_SOURCE\font\font_show'
            )

        # from matplotlib import pyplot as plt
        # import cv2
        # text_mask_arr = np.array(text_mask)[...,:3].astype(np.uint8)
        # ret = cv2.polylines(text_mask_arr, np.array([bbox], np.int32), True, (0, 255, 0), 16)
        # plt.imshow(ret)
        # plt.show()

        if self.cfg.corpus_effects is not None:
            text_mask, _ = self.cfg.corpus_effects.apply_effects(
                text_mask, BBox.from_size(text_mask.size)
            )

        if self.cfg.perspective_transform is not None:
            transformer = PerspectiveTransform(self.cfg.perspective_transform)
            # TODO: refactor this, now we must call get_transformed_size to call gen_warp_matrix
            _ = transformer.get_transformed_size(text_mask.size)

            try:
                (
                    transformed_text_mask,
                    transformed_text_pnts,
                ) = transformer.do_warp_perspective(text_mask)
            except Exception as e:
                logger.exception(e)
                logger.error(font_text.font_path, "text", font_text.text)
                raise e
        else:
            transformed_text_mask = text_mask
        # 白背景的字融合到目标背景上
        pers_bg, M_anti = self.paste_text_mask_on_bg_with_M_anti(pil_bg, transformed_text_mask, jd, info_index)

        return pers_bg, font_text.text, M_anti, transformed_text_mask, bbox, font_base

    # todo 裁切修改为背景投影逻辑
    def paste_text_mask_on_bg(
            self, bg: PILImage, transformed_text_mask: PILImage
    ) -> Tuple[PILImage, PILImage]:
        """

        Args:
            bg:
            transformed_text_mask:

        Returns:

        """
        # x_offset, y_offset = utils.random_xy_offset(transformed_text_mask.size, bg.size)
        # 为了控制背景裁切区域，牺牲背景多样性
        x_offset, y_offset = 0, 0
        bg = self.bg_manager.guard_bg_size(bg, transformed_text_mask.size)
        bg = bg.crop(
            (
                x_offset,
                y_offset,
                x_offset + transformed_text_mask.width,
                y_offset + transformed_text_mask.height,
            )
        )
        if self.cfg.return_bg_and_mask:
            _bg = bg.copy()
        else:
            _bg = bg
        bg.paste(transformed_text_mask, (0, 0), mask=transformed_text_mask)
        return bg, _bg

    def paste_text_mask_on_bg_with_M_anti(
            self, bg: PILImage, transformed_text_mask: PILImage, jd: dict, info_index: int
    ) -> Tuple[PILImage, PILImage]:
        """

        Args:
            bg:
            transformed_text_mask:

        Returns:

        """

        bg = np.array(bg)
        # 这里假定所有文字都是纵向书写的(取消这个假定)
        piece_w, piece_h = transformed_text_mask.size

        dst_points = [[0, 0], [piece_w, 0], [piece_w, piece_h], [0, piece_h]]

        shape = jd['shapes'][info_index]
        src_points = shape['points']

        # 解决异常点数的问题
        if len(src_points) != 4:
            min_area_rect = cv2.minAreaRect(np.array(src_points, np.float32))
            src_points = cv2.boxPoints(min_area_rect)

        src_points = self.adeptive_adjust_bbox_size(src_points, (piece_w, piece_h))

        if not is_clockwise(src_points):
            # src_points = src_points[:1]+src_points[-1:0:-1] # 这种写法不太符合直觉
            src_points = src_points[:1] + src_points[1:][::-1]

        src_points = order_rectify(src_points)

        # 根据原始框的长宽比，估计原文的文字方向（除非进行了方向标注，或者某种先验佐证，否则就需要这个估算，无法确知）
        # src_w, src_h = point_distance(src_points[0], src_points[1]), point_distance(src_points[1], src_points[2])
        if self.text_orientation == 1:
            dst_points = [dst_points[(i + 3) % 4] for i in range(4)]
        '''
        应该传递文字方向信息，以确定文字书写方向。所以，还应该增加横向文本书写逻辑。
        '''

        M = cv2.getPerspectiveTransform(np.array(src_points, dtype=np.float32), np.array(dst_points, dtype=np.float32))
        M_anti = cv2.getPerspectiveTransform(np.array(dst_points, dtype=np.float32),
                                             np.array(src_points, dtype=np.float32))
        dst_img = cv2.warpPerspective(bg, M, (piece_w, piece_h))
        bg = PIL.Image.fromarray(dst_img)
        bg.paste(transformed_text_mask, (0, 0), mask=transformed_text_mask)

        return bg, M_anti

    def get_text_color(self, bg: PILImage, text: str, font: FreeTypeFont) -> FontColor:
        # TODO: better get text color
        # text_mask = self.draw_text_on_transparent_bg(text, font)
        np_img = np.array(bg)
        # mean = np.mean(np_img, axis=2)
        mean = np.mean(np_img)

        alpha = np.random.randint(110, 255)
        r = np.random.randint(0, int(mean * 0.7))
        g = np.random.randint(0, int(mean * 0.7))
        b = np.random.randint(0, int(mean * 0.7))
        fg_text_color = (r, g, b, alpha)

        return fg_text_color

    def _should_apply_layout(self) -> bool:
        return isinstance(self.corpus, list) and len(self.corpus) > 1

    def norm(self, image: np.ndarray) -> np.ndarray:
        if self.cfg.gray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.cfg.height != -1 and self.cfg.height != image.shape[0]:
            height, width = image.shape[:2]
            width = int(width // (height / self.cfg.height))
            image = cv2.resize(
                image, (width, self.cfg.height), interpolation=cv2.INTER_CUBIC
            )

        return image


class CallnumberRender(BookSpineRender):

    def _init_corpus(self, book_spine_content: BookSpineContent, book_info_template: BookInfoTemplate):
        '''


        '''

        while True:
            self.corpus.normalized_corpus = book_spine_content.choice()
            callnumber_list = self.corpus.normalized_corpus['7索书号']
            callnumber_num = len(callnumber_list)
            self.book_info_template_path = book_info_template.get_template_json_path_by_callnumber_part_num(callnumber_num)
            break

    def __call__(self, *args, **kwargs) -> Tuple[np.ndarray, str]:
        '''
        遍历渲染书脊
        1. 检测框信息包括坐标，文字和文字类别
        2. 书脊信息包括参考模板名字，便于检测
        3.

        '''
        try:

            # 核心逻辑

            template_jd = load_json_to_dict(self.book_info_template_path)
            bg_h, bg_w = template_jd['imageHeight'], template_jd['imageWidth']
            # bg = self.bg_gener.gen_mimic_spine((bg_h, bg_w))
            bg = cv2.imread(self.book_info_template_path.replace('.json', '.jpg'))
            bg = img_aug(bg)

            # text_len_limit = 1.2
            dst_shapes = []

            for info_index in range(len(template_jd['shapes'])):
                info_class = template_jd['shapes'][info_index]['group_id']

                shape = template_jd['shapes'][info_index]
                src_points = shape['points']

                # 解决异常点数的问题
                if len(src_points) != 4 and len(src_points) > 2:
                    pprint(src_points)
                    min_area_rect = cv2.minAreaRect(np.array(src_points, np.float32))
                    src_points = cv2.boxPoints(min_area_rect)

                # 顺时针校正
                if not is_clockwise(src_points):
                    # src_points = src_points[:1]+src_points[-1:0:-1] # 这种写法不太符合直觉
                    src_points = src_points[:1] + src_points[1:][::-1]

                src_points = order_rectify(src_points)

                # 根据原始框的长宽比，估计原文的文字方向（除非进行了方向标注，或者某种先验佐证，否则就需要这个估算，无法确知）
                src_w, src_h = point_distance(src_points[0], src_points[1]), point_distance(src_points[1],
                                                                                            src_points[2])
                self.text_orientation = 0 if src_w > src_h * 1.5 else 1

                # dataframe中取文本
                if not self.corpus.normalized_corpus.get(info_class):
                    continue
                else:
                    text = self.corpus.normalized_corpus[info_class].pop()

                # todo lv 单行，混排文本切换开关：oneline，multiline
                cropped_bg, text, M_anti, transformed_text_mask, bbox, font_base = self.gen_one_corpus(
                    template_jd, info_index, text, bg, write_mode='oneline')
                img = cropped_bg

                img = img.convert("RGB")
                np_img = np.array(img)
                # np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
                piece_with_black_pad = cv2.warpPerspective(np_img, M_anti, (bg_w, bg_h))
                _, mask_fg = cv2.threshold(cv2.cvtColor(piece_with_black_pad, cv2.COLOR_BGR2GRAY), 1, 255,
                                           cv2.THRESH_BINARY)
                kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask_fg = cv2.erode(mask_fg, kernal, iterations=2)
                _, mask_fg = cv2.threshold(mask_fg, 1, 255, cv2.THRESH_BINARY)
                piece_with_black_pad = cv2.bitwise_and(piece_with_black_pad, piece_with_black_pad, mask=mask_fg)

                mask_bg = ~mask_fg

                bg_with_black_piece = cv2.bitwise_and(bg, bg, mask=mask_bg)
                bg = cv2.add(piece_with_black_pad, bg_with_black_piece)
                # np_img = self.norm(np_img)
                bbox_to_bg = cv2.perspectiveTransform(np.array([bbox], np.float32), M_anti)
                dst_shape = construct_one_shape(label=text, points=np.squeeze(bbox_to_bg).tolist(), group_id=info_class,
                                                font_base=font_base)
                dst_shapes.append(dst_shape)
            time_str = datetime.now().strftime("%Y%m%d-%H%M%S%f")

            template_img_path = self.book_info_template_path.replace('.json', '.jpg')
            template_img = cv2.imread(template_img_path)

            # merged_show_ret = np.hstack([bg, template_img])

            s = base64.b64encode(os.urandom(5)).decode("utf8")
            s = s.replace("\\", "").replace("/", "").replace("=", "").replace("+", "")
            img_base = f'{s}_{Path(template_img_path).stem}_{time_str}.jpg'

            if not os.path.exists(self.dst_dir):
                os.mkdir(self.dst_dir)
            cv2.imwrite(rf'{self.dst_dir}\{img_base}',
                        bg)

            dst_jd = construct_labelme_jd(dst_shapes, img_base, bg_h, bg_w)
            save_json(os.path.join(self.dst_dir, img_base[:-3] + 'json'), dst_jd)
            return np_img, text, bbox

        except Exception as e:
            raise Imgerror(e)
            # logger.exception(e)
            # raise e


if __name__ == '__main__':
    r'''
    见：D:\lxd_code\text_renderer_lv\render_a_spine.py
     '''
    # print(random.randint(2,2))
