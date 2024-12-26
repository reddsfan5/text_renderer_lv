import base64
import os
import random
import time
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from shapely import Polygon

from lv_tools.cores.file_ops import make_dir
from lv_tools.cores.json_io import save_json
from lv_tools.cores.poly_ops import rectangle_2_bbox
from lv_tools.json_tools.labelme_json_constructor import construct_one_shape, construct_labelme_jd
from lv_tools.task_book_info_gen.bg_gen import MimicSpineBg
from lv_tools.task_ocr.text_data_4m import TextData4mWordKey
from text_renderer.config import get_cfg
from text_renderer.render import RenderOne

render: RenderOne


class SentenceRenderer:
    def __init__(self, img_gen: TextData4mWordKey, textrender: RenderOne):
        '''
        渲染器配置
        '''
        self.img_gen = img_gen
        self.textrender = textrender

    def __call__(self, sentence: str, safe_h: int = 30) -> tuple[np.ndarray, str]:

        img_list = []
        img_hs = []
        for word in sentence.split(' '):

            try:
                img, _ = self.img_gen(word)
            except:
                img, _ = self.textrender(word)
            img_list.append(img)
            img_hs.append(img.shape[0])

        ave_h = np.array(img_hs).mean()
        img_h = int(max(safe_h, ave_h))
        space = int(img_h * random.uniform(.3, 1))

        normed_img_list = []

        space_img = np.ones((img_h, space, 3), dtype=np.uint8) * 255

        for img in img_list[:-1]:
            img = norm_img_size(img, img_h)
            normed_img_list.append(img)
            normed_img_list.append(space_img)

        normed_img_list.append(norm_img_size(img_list[-1], img_h))

        ret = np.concatenate(normed_img_list, axis=1)

        return ret, sentence


class TextRenderAdapter(RenderOne):

    def text_board_expand(self, xs: int, ys: int, xe: int, ye: int, board: int = 5):

        xs, ys, xe, ye = max(0, xs - board), max(0, ys - board), xe + board, ye + board

        return xs, ys, xe, ye

    def __call__(self, *args, **kwargs):
        data = super().__call__(*args)
        if data is not None:
            bbox = data[2]
            (xs, ys), (xe, ye) = bbox[0], bbox[2]

            xs, ys, xe, ye = self.text_board_expand(xs, ys, xe, ye)

            img_arr = data[0][ys:ye, xs:xe, :]

            if show := kwargs.get('show'):
                s = base64.b64encode(os.urandom(3)).decode("utf8")
                s = s.replace("\\", "").replace("/", "").replace("=", "").replace("+", "")
                if not os.path.exists(show_dir := f'./{show}'):
                    os.mkdir(show_dir)
                cv2.imwrite(os.path.join(show_dir, s + 'ori_text.jpg'), data[0])
                cv2.imwrite(os.path.join(show_dir, s + 'cuted_text.jpg'), img_arr)

            return img_arr, data[1]


def norm_img_size(img: np.ndarray, dst_h: int):
    h, w = img.shape[:2]
    ratio = dst_h / h
    img = cv2.resize(img, (int(w * ratio), dst_h))
    return img


class TextCorpusGen:
    def __init__(self, txt_file_path: Union[str, Path]):
        self.txt_file_path = txt_file_path

    def generate_corpus(self):
        with open(self.txt_file_path, 'r', encoding='utf8') as f:
            # return (line for line in f if line.strip())  # ValueError: I/O operation on closed file.
            lines = [line.strip() for line in f if line.strip()]
            random.shuffle(lines)
            for line in lines:
                yield line


class PasteText:

    def __init__(self, bg_img: np.ndarray, text_img: np.ndarray, text: str, jd: dict = None):

        self.bg_img = bg_img
        self.bg_h, self.bg_w = bg_img.shape[:2]
        self.text_img = text_img
        self.text_h, self.text_w = text_img.shape[:2]
        self.jd = jd if jd is not None else construct_labelme_jd(shapes=[], imagePath='', imageHeight=self.bg_h,
                                                                 imageWidth=self.bg_w)
        self.text = text

        self.occupied_positions = self.get_occupied_positions()

    def get_occupied_positions(self):

        occupied_positions = []
        if self.jd and self.jd.get('shapes'):
            for shape in self.jd.get('shapes'):
                points = shape.get('points')
                occupied_positions.append(points)
        return occupied_positions

    def paste_text(self, try_times: int = 100):

        for _ in range(try_times):

            if self.bg_w <= self.text_w or self.bg_h <= self.text_h:
                return self.bg_img, self.jd

            # 尝试100次
            x0 = random.randint(0, self.bg_w - self.text_w)
            y0 = random.randint(0, self.bg_h - self.text_h)
            x1 = x0 + self.text_w
            y1 = y0 + self.text_h
            text_bbox = rectangle_2_bbox([[x0, y0], [x1, y1]])

            # 如果该位置没有重叠，则选择该位置
            if not any(Polygon(text_bbox).intersects(Polygon(points)) for points in self.occupied_positions):
                new_shape = construct_one_shape(self.text, text_bbox)

                self.jd['shapes'].append(new_shape)

                break
        else:
            return self.bg_img, self.jd
        # 将切片放置到背景图上

        self.bg_img[y0:y1, x0:x1, :] = self.text_img
        return self.bg_img, self.jd


def mimic_en_book_spine(root_4m: str = r'D:\lxd_code\OCR\OCR_SOURCE\text_data\Union14M-L\full_images',
                        ann_word_path: str = r'D:\lxd_code\OCR\OCR_SOURCE\text_data\Union14M-L\entire\word_pathlist_horizontal.json',
                        bg_dir: str = r'D:\lxd_code\bar_dm\dm_bar_base\indoorCVPR_09\Images',
                        dst_dir: str = r'F:\dataset\OCR\3-2.book_info_classes\syn_entire_img_rec_book_info\syn_en_book_spine_expand_5_1217',
                        cfg_path: str = r'./example_data/effect_layout_example.py',
                        corpus_file_path: str = r'F:\dataset\OCR\图书目录\english_book\Open_Library_ol_dump_works_2023-02-28_title.txt',
                        book_spine_h: int = 1200,
                        book_spine_w: int = 120):
    make_dir(dst_dir)

    mimic_spine = MimicSpineBg(bg_dir)
    generator_cfg = get_cfg(cfg_path)[0]
    text_gener = TextCorpusGen(corpus_file_path).generate_corpus()
    img_getter = TextData4mWordKey(root_4m, ann_word_path)
    render = TextRenderAdapter(generator_cfg.render_cfg)
    sentence_render = SentenceRenderer(img_getter, render)
    # 文本裁切测试。
    # for sentence in text_gener:
    #     sentence = sentence.strip()
    #     render(sentence, show='expand_board')

    for _ in range(10 ** 6):
        bg = mimic_spine.gen_mimic_spine_bg((book_spine_h, book_spine_w))
        jd = None

        for i in range(3):
            sentence = next(text_gener).strip()

            text_img, text = sentence_render(sentence)

            text_img = cv2.rotate(text_img, cv2.ROTATE_90_CLOCKWISE)

            book_spine_paster = PasteText(bg, text_img, text, jd)
            bg, jd = book_spine_paster.paste_text()
        s = base64.b64encode(os.urandom(8)).decode("utf8")
        s = s.replace("\\", "").replace("/", "").replace("=", "").replace("+", "")
        img_path = s + '.jpg'
        cv2.imwrite(os.path.join(dst_dir, img_path), bg)
        jd['imagePath'] = img_path
        save_json(os.path.join(dst_dir, s + '.json'), jd)






if __name__ == "__main__":
    r'''    
    --config .\example_data/effect_layout_example.py --dataset lmdb --num_processes 1 --log_period 2
    
    --config .\example_data/example.py --dataset lmdb --num_processes 1 --log_period 2

    font show: E:\lxd\OCR_project\OCR_SOURCE\font\font_show
    font not suport: E:\lxd\OCR_project\OCR_SOURCE\font
    '''
    # mimic_en_book_spine()
