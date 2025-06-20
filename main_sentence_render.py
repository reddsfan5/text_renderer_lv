import base64
import itertools
import os
import random

import cv2
import numpy as np
from PIL import Image
from shapely import Polygon

from lv_tools.cores.file_ops import make_dir
from lv_tools.cores.json_io import save_json
from lv_tools.cores.poly_ops import rectangle2points
from lv_tools.data_parsing.labelme_json_constructor import construct_one_shape, construct_labelme_jd
from lv_tools.img_tools.bg_gen import BgGener
from lv_tools.img_tools.img_resize import ImgResizeWithPadding
from lv_tools.task_ocr.text_data_4m import TextData4mWordKey
from main_single_process_for_debug import DBWriterProcess
from text_renderer.config import get_cfg, SafeTextColorCfg
from text_renderer.corpus.book_spine_corpus import TextCorpusGen
from text_renderer.dataset import LmdbDataset
from text_renderer.render import RenderOne
from text_renderer.render_by_gt_img import WordRender, TextRenderAdapter

render: RenderOne


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
            text_bbox = rectangle2points([[x0, y0], [x1, y1]])

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
                        dst_dir: str = r'F:\dataset\OCR\3-2.book_info_classes\syn_entire_img_rec_book_info\syn_en_book_spine_expand_5_250103',
                        cfg_path: str = r'./example_data/effect_layout_example.py',
                        corpus_file_path: str = r'F:\dataset\OCR\图书目录\english_book\Open_Library_ol_dump_works_2023-02-28_title.txt',
                        book_spine_h: int = 1200,
                        book_spine_w: int = 120,
                        spliter: str = ' ',
                        is_add_space: bool = True):
    make_dir(dst_dir)

    mimic_spine = BgGener(bg_dir)
    generator_cfg = get_cfg(cfg_path)[0]
    text_gener = TextCorpusGen(corpus_file_path).corpus_gener()
    img_getter = TextData4mWordKey(root_4m, ann_word_path)
    render = TextRenderAdapter(generator_cfg.render_cfg)
    # sentence_render = SentenceRender(img_getter, render,spliter=spliter,is_add_space=is_add_space)
    sentence_render = WordRender(img_getter, render, spliter=spliter, is_add_space=is_add_space)
    # 文本裁切测试。
    # for sentence in text_gener:
    #     sentence = sentence.strip()
    #     render(sentence, show='expand_board')

    for _ in range(10 ** 2):
        bg = mimic_spine((book_spine_h, book_spine_w))
        jd = None

        for i in range(3):
            sentence = next(text_gener).strip()

            text_img, text = sentence_render(sentence)

            text_img = cv2.rotate(text_img, cv2.ROTATE_90_CLOCKWISE)

            book_spine_paster = PasteText(bg, text_img, text, jd)  # 寻找合适的留白区域书写文字
            bg, jd = book_spine_paster.paste_text()
        s = base64.b64encode(os.urandom(8)).decode("utf8")
        s = s.replace("\\", "").replace("/", "").replace("=", "").replace("+", "")
        img_path = s + '.jpg'
        cv2.imwrite(os.path.join(dst_dir, img_path), bg)
        jd['imagePath'] = img_path
        save_json(os.path.join(dst_dir, s + '.json'), jd)


class Config:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir


def gen_word_rec_piece(root_char_img_dir: str = r'D:\lxd_code\OCR\OCR_SOURCE\text_data\Union14M-L\full_images',
                       ann_word_path: str = r'D:\lxd_code\OCR\OCR_SOURCE\text_data\Union14M-L\entire\word_pathlist_horizontal.json',
                       cfg_path: str = r'./example_data/effect_layout_example.py',
                       corpus_file_path: str = r'F:\dataset\OCR\图书目录\english_book\Open_Library_ol_dump_works_2023-02-28_title.txt',
                       bg_dir: str = r'D:\lxd_code\bar_dm\dm_bar_base\indoorCVPR_09\pure',
                       spliter: str = ' ',
                       is_add_space: bool = True,
                       num=10 ** 6):
    # config = Config(dst_dir)
    # generator_cfg = config

    generator_cfg = get_cfg(cfg_path)[0]
    db_writer_process = DBWriterProcess(
        LmdbDataset, generator_cfg, 2
    )
    text_gener = TextCorpusGen(corpus_file_path).corpus_gener()
    text_gener = itertools.cycle(text_gener)
    img_getter = TextData4mWordKey(root_char_img_dir, ann_word_path)
    render = TextRenderAdapter(generator_cfg.render_cfg)
    mimic_spine = BgGener(bg_dir)
    color_gener = SafeTextColorCfg()
    word_render = WordRender(img_getter, render, spliter=spliter, is_add_space=is_add_space)
    # 文本裁切测试。

    count = 0

    while count < num:
        sentence = next(text_gener).strip()
        try:

            text_img, text = word_render(sentence)
        except:
            continue
        count += 1
        text_img_h, text_img_w = text_img.shape[:2]

        dst_h = int(text_img_h + 2 * text_img_h)
        dst_w = int(text_img_w + 5 * text_img_h)

        text_img, bbox = ImgResizeWithPadding(.5, .5).resize(text_img,dst_h,dst_w )

        alpha_anti = cv2.cvtColor(text_img, cv2.COLOR_BGR2GRAY)

        # alpha = 255 - alpha_anti
        alpha = 255 - (alpha_anti // 2)

        bg_img = mimic_spine((dst_h, dst_w), is_blur=False)
        # bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)

        text_color = color_gener.get_color(bg_img)
        bg_img_pil = Image.fromarray(bg_img)

        font_img_pil = Image.new('RGB', (dst_w, dst_h), color=text_color)

        callnumber_img_pil = Image.fromarray(text_img)

        font_img_pil.paste(callnumber_img_pil, (0, 0), Image.fromarray(alpha_anti))

        bg_img_pil.paste(font_img_pil, (0, 0), mask=Image.fromarray(alpha))

        text_img = np.array(bg_img_pil)
        # 反相
        text_img = text_img if random.randint(0, 2) else 255 - text_img

        ret = {
            'image': text_img,
            'label': text,
            'bbox': bbox,
            'font': ''
        }
        db_writer_process.gen_data(ret)


if __name__ == "__main__":
    r'''    
    --config .\example_data/effect_layout_example.py --dataset lmdb --num_processes 1 --log_period 2
    
    --config .\example_data/example.py --dataset lmdb --num_processes 1 --log_period 2

    font show: E:\lxd\OCR_project\OCR_SOURCE\font\font_show
    font not suport: E:\lxd\OCR_project\OCR_SOURCE\font
    '''

    # mimic_en_book_spine(root_4m=r'D:\lxd_code\OCR\OCR_SOURCE\text_data\gnt\png-label\gnt-images',
    #                     ann_word_path=r'D:\lxd_code\OCR\OCR_SOURCE\text_data\gnt\word_pathlist_mini.json',
    #                     dst_dir = r'F:\dataset\OCR\3-2.book_info_classes\syn_entire_img_rec_book_info\syn_en_book_spine_expand_5_250103-17',
    #                     corpus_file_path=r'D:\lxd_code\OCR\OCR_SOURCE\corpus\anhuidaxue_call_number\anhuidaxue-callnumber.txt',
    #                     spliter='',
    #                     is_add_space=False)

    # ⭐基于中科院自动化所数据的手写字符识别⭐
    book_name_author_corpus_path = r'D:\lxd_code\OCR\OCR_SOURCE\corpus\author_bookname\bookname_author_0418_sample_200w.txt'
    callnumber_corpus_path = r'D:\lxd_code\OCR\OCR_SOURCE\corpus\anhuidaxue_call_number\anhuidaxue-callnumber.txt'
    gen_word_rec_piece(root_char_img_dir=r'D:\lxd_code\OCR\OCR_SOURCE\text_data\gnt\png-label\gnt-images',
                       ann_word_path=r'D:\lxd_code\OCR\OCR_SOURCE\text_data\gnt\word_pathlist.json',
                       corpus_file_path=book_name_author_corpus_path,
                       spliter='',
                       bg_dir=r'D:\lxd_code\OCR\OCR_SOURCE\bg',
                       is_add_space=False,
                       num=1 * 10 ** 2)

    # img_path = r'D:\lxd_code\OCR\OCR_SOURCE\text_data\gnt\png-label\gnt-images\001-f\97.png'
    # char_in = '‘'
    # img_arr = cv2.imread(img_path)
    #
    # img = img_height_normalize(img_arr,10,char_in)
    #
    # from matplotlib import pyplot as plt
    # plt.imshow(img)
    # plt.show()
    # root_4m: str = r'D:\lxd_code\OCR\OCR_SOURCE\text_data\Union14M-L\full_images'
    # ann_word_path: str = r'D:\lxd_code\OCR\OCR_SOURCE\text_data\Union14M-L\entire\word_pathlist_horizontal.json'
    # bg_dir: str = r'D:\lxd_code\bar_dm\dm_bar_base\indoorCVPR_09\Images'
    # dst_dir: str = r'F:\dataset\OCR\3-2.book_info_classes\syn_entire_img_rec_book_info\syn_en_book_spine_expand_5_1217'
    # cfg_path: str = r'./example_data/effect_layout_example.py'
    # corpus_file_path: str = r'F:\dataset\OCR\图书目录\english_book\Open_Library_ol_dump_works_2023-02-28_title.txt'
    # book_spine_h: int = 1200
    # book_spine_w: int = 120
    #
    #
    # make_dir(dst_dir)
    #
    # mimic_spine = BgGener(bg_dir)
    # generator_cfg = get_cfg(cfg_path)[0]
    # text_gener = TextCorpusGen(corpus_file_path).corpus_gener()
    # img_getter = TextData4mWordKey(root_4m, ann_word_path)
    # render = TextRenderAdapter(generator_cfg.render_cfg)
    # sentence_render = SentenceRender(img_getter, render)
    # # sentence_render = SentenceRender(img_getter, '')
    # for text in text_gener:
    #     ret = sentence_render(text)
    #
    #
    #
    #
    #     from matplotlib import pyplot as plt
    #     plt.imshow(ret)
    #     plt.show()
