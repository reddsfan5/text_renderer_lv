import base64
import os
import random

import cv2
import numpy as np

from lv_tools.img_tools.img_resize import norm_img_size, img_height_normalize
from lv_tools.task_ocr.text_data_4m import TextData4mWordKey
from text_renderer.render import RenderOne
from lv_tools.cores.str_utils import is_need_rotate


class SentenceRender:
    def __init__(self, img_gen: TextData4mWordKey, textrender: RenderOne=None, spliter=' ', is_add_space: bool = True):
        '''
        渲染器配置
        '''
        self.img_gen = img_gen
        self.textrender = textrender
        self.spliter = spliter
        self.is_add_space = is_add_space

    def __call__(self, sentence: str, safe_h: int = 30) -> tuple[np.ndarray, str]:

        img_list = []
        img_heights = []
        sentence_eles = self.split_sentence(sentence)
        for word in sentence_eles:

            try:
                img, _ = self.img_gen(word)
            except:
                img, _ = self.textrender(word)
            img_list.append(img)
            img_heights.append(img.shape[0])

        ave_h = np.array(img_heights).mean()
        img_h = int(max(safe_h, ave_h))

        space = int(img_h * random.uniform(.3, 1))
        space_img = np.ones((img_h, space, 3), dtype=np.uint8) * 255

        normed_img_list = []

        for img in img_list[:-1]:
            img = norm_img_size(img, img_h)
            normed_img_list.append(img)
            if self.is_add_space:
                normed_img_list.append(space_img)

        normed_img_list.append(norm_img_size(img_list[-1], img_h))

        ret = np.concatenate(normed_img_list, axis=1)

        return ret, sentence

    def split_sentence(self, sentence: str):
        if self.spliter:
            sentence = sentence.split(self.spliter)
        return sentence


class WordRender(SentenceRender):

    # def is_support_all_chr(self,word:str):

    def __call__(self, word: str, safe_h: int = 30) -> tuple[np.ndarray, str]:
        img_list = []
        img_heights = []
        word_units = self.split_sentence(word)

        for word in word_units:

            try:
                img, _ = self.img_gen(word, is_case_sensitivity=True)
            except:
                # img, _ = self.textrender(word)
                raise ValueError('not support all chr')

            if is_need_rotate(word):
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img_list.append(img)
            img_heights.append(img.shape[0])

        ave_h = np.array(img_heights).mean()
        img_h = int(max(safe_h, ave_h))

        space = int(img_h * random.uniform(.3, 1))
        space_img = np.ones((img_h, space, 3), dtype=np.uint8) * 255

        normed_img_list = []

        for index, img in enumerate(img_list[:-1]):
            img, _ = img_height_normalize(img, img_h, word_units[index])

            normed_img_list.append(img)
            if self.is_add_space:
                normed_img_list.append(space_img)

        normed_img_list.append(img_height_normalize(img_list[-1], img_h, word_units[-1])[0])

        ret = np.concatenate(normed_img_list, axis=1)

        return ret, word_units


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
