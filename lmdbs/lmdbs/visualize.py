# # coding:utf-8
# import base64
# import json
# import os
# from io import BytesIO
# from typing import Literal
# import PIL
# import cv2
# import numpy as np
# from PIL import Image, ImageDraw, ImageFont
#
# from lmdb_loader import LMDBLoader
#
#
# class Visualize(object):
#     font_dict = {'ch':'simhei.ttf',
#                  'kr':'Gumi Romance.ttf',
#                  'jp':'sarasa-mono-slab-k-semibold.ttf'}
#
#     def __init__(self, dir='e:/temp/',lan:Literal['ch','en','kr','jp']='ch'):
#         self.dir = dir
#         self.font  = ImageFont.truetype('/'.join(__file__.split('\\')[:-1]) + f"/{self.font_dict[lan]}", size=15)
#
#
#     def decode_wordBB(self, word_boxes):
#         if word_boxes is not None:
#             word_boxes = json.loads(word_boxes)  # word_boxes.decode('utf-8'))
#             word_box = []
#             i = 0
#             for j in range(4):
#                 word_box.append([word_boxes[2 * j][i], word_boxes[2 * j + 1][i]])
#             word_boxes = word_box
#         return word_boxes
#
#     def draw_Pts(self, img, pnts, color):
#         """
#                     :param img: gray image, will be convert to BGR image
#                     :param pnts: left-top, right-top, right-bottom, left-bottom
#                     :param color:
#                     :return:
#                     """
#         if isinstance(pnts, np.ndarray):
#             pnts = pnts.astype(np.int32)
#
#         if len(img.shape) > 2:
#             dst = img
#         else:
#             img = np.clip(img, 0., 255.).astype(np.uint8)
#             dst = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#
#         thickness = 1
#         linetype = cv2.LINE_AA
#         for i in range(len(pnts)):
#             pt0 = pnts[i]
#             pt1 = pnts[(i + 1) % len(pnts)]
#             cv2.line(dst, (int(pt0[0]), int(pt0[1])), (int(pt1[0]), int(pt1[1])), color=color, thickness=thickness,
#                      lineType=linetype)
#         return dst
#
#     def draw_bbox(self, img, bbox, color):
#         if bbox is not None:
#             bbox = self.decode_wordBB(bbox)
#             pnts = np.array(bbox)
#             img = self.draw_Pts(img, pnts.astype(int), color)
#         return img
#
#     def paste_label_to_img(self, img, label):
#         txt_img = np.zeros((20, img.shape[1], img.shape[2]), dtype=np.uint8)
#         txt_img = Image.fromarray(txt_img)  # np.uint8(txt_img))
#         draw = ImageDraw.Draw(txt_img)
#         # font = ImageFont.truetype('/'.join(__file__.split('/')[:-1]) + "/simhei.ttf", size=15)  # simkai.ttf
#         # font = ImageFont.truetype('/'.join(__file__.split('\\')[:-1]) + "/Gumi Romance.ttf", size=15)  # simkai.ttf,WantedSans-Medium.ttf
#         draw.text((0, 0), label, fill=(0, 200, 200), font=self.font)
#         txt_img = np.array(txt_img)
#         img = np.concatenate([img, txt_img], axis=0)  # in h
#         return img
#
#     def draw_text(self, img, text, pos=(0, 0)):
#         txt_img = Image.fromarray(img)  # np.uint8(txt_img))
#         draw = ImageDraw.Draw(txt_img)
#         font = ImageFont.truetype('/'.join(__file__.split('/')[:-1]) + "/simhei.ttf", size=15)  # simkai.ttf
#         draw.text(pos, text, fill=(0, 200, 200), font=font)
#         txt_img = np.array(txt_img)
#         return txt_img
#
#     def get_file_name(self, rand_idx=-1, ext='', file_name=None):
#         if rand_idx < 0:
#             rand_idx = np.random.randint(0, 100)
#         if file_name is None:
#             default_path = './temp/'
#             file_name = default_path + str(rand_idx) + ext + '.jpg'
#         else:
#             default_path, fname = os.path.split(file_name)
#         if not os.path.exists(default_path):
#             os.mkdir(default_path)
#
#         return file_name, rand_idx
#
#     def convert_image_to_display(self, image):
#         if isinstance(image, bytes):
#             assert type(image) is bytes and len(image) > 0, "invalid input 'img' in DecodeImage"
#             image = np.frombuffer(image, dtype='uint8')
#             image = cv2.imdecode(image, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
#             # else:
#             #     image = cv2.imdecode(image, 1)
#         if image.max() < 100:
#             image = (image - image.min()) / (image.max() - image.min()) * 255
#         if not (1 == image.shape[2] or 3 == image.shape[2]):
#             image = image.transpose((1, 2, 0))
#         if image.shape[1] < 350:
#             temp = image
#             image = np.zeros((temp.shape[0], 350, temp.shape[2]), dtype=np.uint8)
#             image[:, 0:temp.shape[1], :] = temp
#         return image
#
#     def get_rect(self, rc):
#         if 'width' in rc:
#             rc = [rc['x'], rc['y'], rc['x'] + rc['width'], rc['y'] + rc['height']]
#         elif 'w' in rc:
#             rc = [rc['x'], rc['y'], rc['x'] + rc['w'], rc['y'] + rc['h']]
#         return rc
#
#     def draw_rect(self, image, rc, cr=(255, 0, 0)):
#         rc = self.get_rect(rc)
#         rc = np.array(rc).astype(int)
#         image = self.draw_Pts(image, [rc[:2], [rc[2], rc[1]], rc[2:], [rc[0], rc[3]]], color=cr)
#         return image
#
#     def display_type_key(self, image, k, v):
#         if 'image' == k:
#             return image  # continue
#         if 'rect' == k:
#             image = self.draw_rect(image, v, color=(255, 0, 0))
#         elif 'points' == k:
#             image = self.draw_Pts(image, v, color=(255, 0, 0))
#         else:
#             if not isinstance(v, str):
#                 v = str(v)
#             image = self.paste_label_to_img(image, k + ': ' + v)
#         return image
#
#     def display_infor(self, data, rand_idx=-1, ext='', file_name=None):
#         # if not DISPLAY_DBG_INFO and file_name is None:
#         #     return 0
#         image = data['image'] if isinstance(data, dict) else data
#         image = self.convert_image_to_display(image).copy()
#         if isinstance(data, dict):
#             for k, v in data.items():
#                 if 'points' == k:
#                     image = self.display_type_key(image, k, v)
#                 elif isinstance(v, list):
#                     for elem in v:
#                         for k2, v2 in elem.items():
#                             image = self.display_type_key(image, k2, v2)
#                 else:
#                     image = self.display_type_key(image, k, v)
#         file_name, rand_idx = self.get_file_name(rand_idx, ext, file_name)
#         # cv2.imwrite(file_name, image)
#
#         cv2.imencode('.jpg', image)[1].tofile(file_name)
#         return rand_idx
#
#     def draw_elements(self, image, elements, cr=(255, 0, 0)):
#         for elem in elements:
#             if 'oriRect' in elem:
#                 elem['rect'] = elem['oriRect']
#             if 'rect' not in elem:
#                 elem['rect'] = [elem['x1'], elem['y1'], elem['x2'], elem['y2']] if 'x1' in elem else [0, 0, 1, 1]
#             mes = ''
#             for k, v in elem.items():
#                 if 'rect' == k:
#                     image = self.draw_rect(image, v, cr=cr)
#                 elif 'image' != k:
#                     if not isinstance(v, str):
#                         v = str(v)
#                     mes = mes + ' ' + k + ':' + str(v)
#
#             pos = self.get_rect(elem['rect'])[:2] if 'rect' in elem else (0, 0)
#             image = self.draw_text(image, mes, pos)
#         return image
#
#     def display_elements(self, image, elements, labels=None, rand_idx=-1, ext='', file_name=None):
#         image = self.convert_image_to_display(image)
#         image = self.draw_elements(image, elements)
#         if labels is not None:
#             image = self.draw_elements(image, labels, (0, 255, 0))
#
#         file_name, rand_idx = self.get_file_name(rand_idx, ext, file_name)
#         cv2.imwrite(file_name, image)
#         return rand_idx
#
#
# def visualize_some_sample(
#         lmdb_path=r'D:\dataset\bar_code\a_bar\a_rec\barcode_comp\cylinder\bar_rec_len_5_12p2_0804_cylinder_v1',
#
#         num=100,
#         lan='ch'):
#     lmdb_loader = LMDBLoader(lmdb_path, rand=True)  # , gen_labels=True)
#     print(f'样本量：{lmdb_loader.num_samples}')
#     dst_path = lmdb_path + '/samples/'
#     vis = Visualize(lan=lan)
#     if not os.path.exists(dst_path):
#         os.mkdir(dst_path)
#     for j in range(num):
#         i = j  # np.random.randint(0, lmdb_loader.num_samples)
#         data = lmdb_loader[i]
#         if data is None or data['image'] is None:
#             continue
#         vis.display_infor(data, file_name=dst_path + str(i) + '_infor.jpg')
#
#
# def show_sp_data(lmdb_path: str, id=86064):
#     lmdb_loader = LMDBLoader(lmdb_path, rand=False)  # , gen_labels=True)
#     val = lmdb_loader.txn.get(('id-' + str(id).zfill(9)).encode()).decode('utf8')
#     print(val)
#     img_b64 = json.loads(val)['image']
#     with BytesIO() as bio:
#         bio.write(base64.b64decode(img_b64))
#         pil_img = PIL.Image.open(bio)
#         from matplotlib import pyplot as plt
#         plt.imshow(np.array(pil_img))
#         plt.show()
from lv_tools.lmdbs.visualize import visualization_some_sample


if __name__ == '__main__':
    visualization_some_sample(
        r'D:\dataset\OCR\lmdb_datatest_062010_44_30\effect_layout_image\effect_ensembleauthor')

    # cvt_data()
    # lmdb_info(r'\\192.168.1.11\dataset\bar_code\a_bar\a_det\synthesis\bar_spine_1226_6num_remainder_v4lmdb')
    # li = [str(i).zfill(6) for i in range(10**6)]
    # print(li[:10])
    # lmdb_path = r'F:\dataset\OCR\2.text_det\syn_digit_series_with_parenthesis_lmdb_num_5000'
    # with lmdb.open(lmdb_path) as env:
    #     with env.begin(write=False) as txn:
    #         for k,v in txn.cursor():
    #             if not k.decode('utf8').startswith('id-'):
    #                 print(f"{k.decode('utf8')}---{v.decode('unicode_escape')}")
