# -*- coding:utf-8 -*-
# @Author  : __author__ = ''
# @time    : 2021/12/21 16:52
# @File    : QRProject.py
# @Software:PyCharm
import base64
import pickle
import sys
from collections import defaultdict
from pathlib import Path
import paddle.device
from matplotlib import pyplot as plt
import json
import os
import shutil
from typing import List
from pprint import pprint
from tqdm import tqdm
import cv2
import loguru
import numpy as np
from shutil import copyfile
from os import path as osp
import random


def get_specific_file_paths(root, suffixs=None):
    '''
    递归查找目录下特定后缀的文件，返回Path对象列表
    '''
    img_dir_path = Path(root)
    if suffixs:
        file_paths = [file_path for file_path in img_dir_path.glob('**/*') if str(file_path).endswith(suffixs)]
    else:
        file_paths = [file_path for file_path in img_dir_path.glob('**/*')]
    return file_paths


LABEL_TUPLE = (
    ('指天抬头男', 'A50'),
    ('招手男', 'A43'),
    ('双耳灰陶', 'A1'),
    ('双复系盘口壶', 'A5'),
    ('双耳鸡头', 'A11'),
    ('南瓜碎口壶', 'A38'),
    ('多嘴刺猬壶', 'A18'),
    ('盘蛇瓶', 'A39'),
    ('葫芦瓶', 'A10'),
    ('广口陶罐', 'A15'),
    ('青釉痰盂', 'A21'),
    ('龙泉净瓶', 'A49'),
    ('尖嘴柱罐', 'A2'),
    ('双曲空心罐', 'A26'),
    ('平顶花团坛', 'A27'),
    ('蒙古帽众星拱月罐', 'A28'),
    ('六角星圆柱罐', 'A30'),
    ('蒙古帽梵高花卉罐', 'A41'),
    ('陶瓷酒杯', 'A24'),
    ('围棋搭口坛', 'A25'),
    ('青花山水碗', 'A29'),
    ('绿釉盘口壶', 'A32'),
    ('火焰种子', 'A33'),
    ('黑番茄', 'A37'),
    ('塑料感大盘', 'A47'),
    ('雕饰砚', 'A3'),
    ('扁平砚', 'A36'),
    ('玉箫吹破石章', 'A4'),
    ('镀金配饰', 'A6'),
    ('对月梳', 'A7'),
    ('琵琶梳', 'A8'),
    ('有子花盘', 'A9'),
    ('无子花盘', 'A16'),
    ('蓝菊小碟', 'A17'),
    ('八开花卉盘', 'A19'),
    ('牛角罗汉', 'A12'),
    ('立像观音', 'A13'),
    ('方口三头龙鼎', 'A14'),
    ('菱花铜镜', 'A20'),
    ('洗手绿金盆', 'A22'),
    ('长柄铜漏斗', 'A23'),
    ('贝刮削器', 'A31'),
    ('虎形枕', 'A34'),
    ('铜如来佛', 'A35'),
    ('风花雪月瓷枕', 'A40'),
    ('五体投地', 'A42'),
    ('人首蛇身俑', 'A44'),
    ('石雕走龙', 'A45'),
    ('夜游赤壁', 'A46'),
    ('抱子观音', 'A48'),
)
ch2en = {label[0]: label[1] for label in LABEL_TUPLE}
en2ch = {label[1]: label[0] for label in LABEL_TUPLE}

# class MyEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         elif isinstance(obj, bytes):
#             return str(obj, encoding='utf-8')
#         elif isinstance(obj, np.integer):
#             return int(obj)
#         elif isinstance(obj, np.floating):
#             return float(obj)
#         return json.JSONEncoder.default(self, obj)


'''
需要重构的部分：
1. 关键点所在的 shapes。
2. 图像尺寸
3. 图像位置（图像与对应的json文件所在的位置，形成一种惯例。那就是放一起，这样相对位置好整。（当然这部分开放接口参数留有自由度））
4. 图像数据置为空。
'''


def getJsonDict(json_path):
    '''
    :param json_path: json_file 路径。
    :return:json_dict
    '''
    with open(json_path, encoding='utf8', mode='r') as f:
        s = f.read()
        # s = s.replace('\\','\\\\') # 防止莫名转义错误
        json_dict = json.loads(s)
    return json_dict


def getLabel(json_dict):
    label_list = [block['label'] for block in json_dict['shapes']]  # 获得某张图的json中的polygon的label.
    return label_list


def get_boxes(json_path):
    loguru.logger.add('logs/multi_point_imgs.logs', level='INFO')

    json_dict = getJsonDict(json_path)
    point_list = [block['points'] for block in json_dict['shapes']]
    return point_list


def getHomogPoint(json_dict):
    point_list_array = np.array([point for block in json_dict['shapes'] for point in block['points']])
    one = np.ones((point_list_array.shape[0], 1))
    homog = np.concatenate((point_list_array, one), axis=-1)
    # 为了避免多点（>4点）标注的干扰，暂时策略，直接忽略包含这些标注的图（这些图都是马马虎虎的选手标注的，标注框质量不好，不太贴合。）
    if len(point_list_array) % 4 == 0:
        return homog
    else:
        return [0]


def orderCalibration(point_list, img_str):
    '''
    按照四点标注比对点的顺序。
    :param point_list:
    :return:
    '''
    for i in range(0, len(point_list), 4):
        tem = point_list[i:i + 4]
        x_list = []
        y_list = []
        for j in tem:
            x_list.append(j[0])
            y_list.append(j[1])
        x_list.sort()
        y_list.sort()
        # 点顺序的检测，基于假设：0，1 点的纵坐标是最小。0点横坐标，小于1点横坐标。
        assert (tem[0][1] in y_list[:2] and tem[1][1] in y_list[:2] and tem[0][0] < tem[1][0]), \
            f'图片：【{img_str}】，索引号为【{i // 4}】的框的点顺序异常,请核实\n{np.array(tem[..., :2]).astype(np.int32)}'


def cstPoly4(points_list: List[list]) -> List[List[list]]:
    '''
    类型转换是为了符合dump的数据要求。
    :param points_list:传入非嵌套坐标点列表即可，
    :return:返回嵌套的polygon列表。
    '''
    polygon_list = []
    bbox = []
    for i in range(len(points_list)):
        if i % 4 == 0:
            if bbox:
                polygon_list.append(bbox)
                bbox = []  # bbox.clear() # 出错.可变数据类型.
        bbox.append([float(points_list[i][0]), float(points_list[i][1])])
    polygon_list.append(bbox)
    return polygon_list


def modifyJson(json_dict, polygon_list, h, w, img_basename, label='Book', rlt_path=''):
    '''
    根据提供的信息，对json字典完成更新，不需要返回值。
    惯例json与对应的图在一个文件夹内。 否则提供相对路径：rel_path
    :param json_dict:
    :param polygon_list:
    :param h:
    :param w:
    :param img_path:
    :param rlt_path:
    :return:
    '''
    shapes = json_dict['shapes']
    # assert len(shapes) == len(polygon_list), f'传入标注框数目{len(polygon_list)}与json_dict[shapes]数目{len(shapes)}不匹配'
    # 标注框更新
    for i in range(len(shapes)):
        json_dict['shapes'][i]['points'] = polygon_list[i]
        if label:
            json_dict['shapes'][i]['label'] = label

    json_dict['imageData'] = None
    json_dict['imageHeight'] = h
    json_dict['imageWidth'] = w
    json_dict['imagePath'] = os.path.join(rlt_path, img_basename)  # 相对于json的路径


def outlier_count(json_dir=r'D:\image_distortion_lv\train_data\Lib_simple\annotations'):
    loguru.logger.add('logs/multi_point_imgs.logs', level='INFO')
    json_files = os.listdir(json_dir)
    for json_basename in json_files:
        json_dict = getJsonDict(os.path.join(json_dir, json_basename))
        point_list_array = [block['points'] for block in json_dict['shapes'] if len(block['points']) > 4]
        if (ret := len(point_list_array)):
            # 将标注不规范的json都删除了
            # os.remove(os.path.join(json_dir,json_basename))
            loguru.logger.info(f'{json_basename} --- {ret}')


def remove_isolate_img(rootdir, jsondir='annotations', imgdir='images'):
    '''
    删除无标签的图片数据。
    - rootdir
        - jsondir
            - json1
            - json2
        - imgdir
            - img1
            - img2
    :param rootdir:
    :param jsondir:
    :param imgdir:
    :return:
    '''
    basenames = os.listdir(os.path.join(rootdir, jsondir))
    stems = [os.path.splitext(basename)[0] for basename in basenames if basename.endswith('json')]
    img_basename = os.listdir(os.path.join(rootdir, imgdir))
    for img in img_basename:
        if os.path.splitext(img)[0] not in stems:
            os.remove(os.path.join(rootdir, imgdir, img))


def coco2labelme(coco_path, labelme_path, dst_path):
    jd_coco = getJsonDict(coco_path)
    jd_lm = getJsonDict(labelme_path)
    coco_kp = jd_coco['annotations'][0]['keypoints']
    kp_lm = [coco_kp[0:2], coco_kp[3:5], coco_kp[6:8], coco_kp[9:11]]
    jd_lm['shapes'][0]['points'] = kp_lm
    img_path, img_h, img_w = jd_coco['images'][0]['file_name'], jd_coco['images'][0]['height'], jd_coco['images'][0][
        'width']
    img = cv2.imread(coco_path.replace('.json', '.jpg'))
    # img_h,img_w = img.shape[0],img.shape[1]
    jd_lm['imagePath'] = img_path
    jd_lm['imageHeight'] = img_h
    jd_lm['imageWidth'] = img_w
    with open(dst_path, mode='w', encoding='utf8') as f:
        data = json.dumps(jd_lm)
        f.write(data)


def coco2labelme_batch(coco_dir, labelme_path, dst_dir):
    # 列出coco_dir 之下的所有json，准备转码
    coco_jsons = [json_file for json_file in os.listdir(coco_dir) if json_file.endswith('.json')]
    for jc in tqdm(coco_jsons):
        if jc.startswith('dmRectPour'):
            continue
        coco_path = os.path.join(coco_dir, jc)
        img_path = os.path.join(coco_dir, jc.replace('.json', '.jpg'))
        json_dst_path = os.path.join(dst_dir, jc)
        img_dst_path = os.path.join(dst_dir, jc.replace('.json', '.jpg'))
        coco2labelme(coco_path, labelme_path, json_dst_path)
        copyfile(img_path, img_dst_path)


def only4left(json_path, dst_path):
    jd = getJsonDict(json_path)
    for box in jd['shapes']:
        if len(box['points']) != 4:
            return
    img_path = json_path.replace('.json', '.jpg')
    img_dst = dst_path.replace('.json', '.jpg')
    shutil.copyfile(json_path, dst_path)
    shutil.copyfile(img_path, img_dst)


def batch_left(json_dir, dst_dir):
    json_bases = [file for file in os.listdir(json_dir) if file.endswith('.json')]
    for jb in json_bases:
        jp = os.path.join(json_dir, jb)
        dp = os.path.join(dst_dir, jb)
        only4left(jp, dp)


def less2out(json_path, dst_path):
    jd = getJsonDict(json_path)
    new_shapes = []
    count = 0
    for box in jd['shapes']:
        if len(box['points']) > 3:
            new_shapes.append(box)
        else:
            count += 1
    # if count>0:
    jd['shapes'] = new_shapes
    with open(dst_path, mode='w', encoding='utf8') as f:
        data = json.dumps(jd)
        f.write(data)
    shutil.copyfile(json_path.replace('.json', '.jpg'), dst_path.replace('.json', '.jpg'))


def batch_less2out(json_dir, dst_dir):
    json_bases = [file for file in os.listdir(json_dir) if file.endswith('.json')]
    for jb in json_bases:
        jp = os.path.join(json_dir, jb)
        dp = os.path.join(dst_dir, jb)
        less2out(jp, dp)


def json_name2json_file(json_dir, dst_dir, img_suffix='.jpg'):
    json_file = [file for file in os.listdir(json_dir) if file.endswith('.json')]
    for jf in json_file:
        print(jf)
        jd = getJsonDict(os.path.join(json_dir, jf))
        jd['imagePath'] = jf.replace('.json', img_suffix)
        if not osp.exists(dst_dir):
            os.mkdir(dst_dir)
        with open(os.path.join(dst_dir, jf), mode='w', encoding='utf8') as f:
            data = json.dumps(jd)
            f.write(data)


# jd['shapes'][index]['flags']['text']

def json_text2json_label(json_dir, dst_dir):
    json_file = [file for file in os.listdir(json_dir) if file.endswith('.json')]
    for jf in json_file:
        print(jf)
        jd = getJsonDict(os.path.join(json_dir, jf))
        for shp in jd['shapes']:
            shp['label'] = shp['flags']['text']

        if not osp.exists(dst_dir):
            os.mkdir(dst_dir)
        with open(os.path.join(dst_dir, jf), mode='w', encoding='utf8') as f:
            data = json.dumps(jd)
            f.write(data)


def modify_label(json_dict, label_dict):
    shapes = json_dict['shapes']
    # assert len(shapes) == len(polygon_list), f'传入标注框数目{len(polygon_list)}与json_dict[shapes]数目{len(shapes)}不匹配'
    # 标注框更新
    for i in range(len(shapes)):
        if label_dict:
            k = json_dict['shapes'][i]['label']
            json_dict['shapes'][i]['label'] = label_dict[k]


def batch_modify_label(json_dir, dst_dir, label_dict=None):
    json_file = [file for file in os.listdir(json_dir) if file.endswith('.json')]
    for jf in json_file:
        jd = getJsonDict(os.path.join(json_dir, jf))
        modify_label(jd, label_dict=label_dict)
        with open(os.path.join(dst_dir, jf), mode='w', encoding='utf8') as f:
            data = json.dumps(jd)
            f.write(data)
        img_base = jf.replace('.json', '.png') if osp.exists(
            osp.join(json_dir, jf.replace('.json', '.png'))) else jf.replace('.json', '.jpg')

        shutil.copyfile(os.path.join(json_dir, img_base), os.path.join(dst_dir, img_base))


def coco_slim_by_id(json_path, dst_dir):
    jd = getJsonDict(json_path)
    images_list = jd['images']
    new_images_list = []
    cuted_img_id = set()
    for image in images_list:
        if image['file_name'].startswith('dmRectShelter'):
            new_images_list.append(image)

        else:
            cuted_img_id.add(image['id'])
    anns = jd['annotations']
    new_ann = []
    for ann in anns:
        if ann['id'] not in cuted_img_id:
            new_ann.append(ann)
    jd['images'] = new_images_list
    jd['annotations'] = new_ann
    json_base = os.path.basename(json_path)
    with open(os.path.join(dst_dir, json_base), mode='w', encoding='utf8') as f:
        data = json.dumps(jd)
        f.write(data)


def coco_slim_by_area(json_path, dst_dir, pre='dmRectBg', threshold=20000):
    jd = getJsonDict(json_path)
    images_list = jd['images']
    new_images_list = []

    for image in images_list:
        if image['file_name'].startswith(pre):
            new_images_list.append(image)
    anns = jd['annotations']
    new_ann = []
    for ann in anns:
        if int(ann['area']) < threshold:
            new_ann.append(ann)
    jd['images'] = new_images_list
    jd['annotations'] = new_ann
    json_base = os.path.basename(json_path)
    with open(os.path.join(dst_dir, json_base), mode='w', encoding='utf8') as f:
        data = json.dumps(jd)
        f.write(data)


def second_order_couple_modify_label(src, dst, label_dict):
    '''
    * src:
        * sub1
            img1
            json1
            img2
            ...
        * sub2
    * dst:
        * sub1
        * sub2

    '''
    sd_bases = [d for d in os.listdir(src) if osp.isdir(osp.join(src, d))]
    for sdb in sd_bases:
        dst2 = osp.join(dst, sdb)
        if not osp.exists(dst2):
            os.mkdir(dst2)
        batch_modify_label(osp.join(src, sdb), dst2, label_dict=label_dict)


def abstract_data_2_onedir(src, dst):
    '''
    src:路径
    '''
    bases = os.listdir(src)
    for base in bases:
        s = osp.join(src, base)
        if osp.isdir(s):
            abstract_data_2_onedir(s, dst)
        else:
            if not osp.exists(dst):
                os.mkdir(dst)
            shutil.copyfile(s, osp.join(dst, f'{osp.basename(s)}'))


def two2four(list):
    return [[list[0], list[1]], [list[2], list[1]], [list[2], list[3]], [list[0], list[3]]]


def crop_img_ratio(img, axis, in_por=.05, out_por=.2):
    '''
    axis:【xmin,ymin,xmax,ymax】
    '''
    w = axis[2] - axis[0]
    h = axis[3] - axis[1]

    x_min = axis[0] + random.randint(int(-out_por * w), int(in_por * w))
    y_min = axis[1] + random.randint(int(-out_por * h), int(in_por * h))
    x_max = axis[2] + random.randint(int(-in_por * w), int(out_por * w))
    y_max = axis[3] + random.randint(int(-in_por * h), int(out_por * h))

    x_min = int(x_min) if x_min > 0 else 0
    y_min = int(y_min) if y_min > 0 else 0
    x_max = int(x_max) if x_max < img.shape[1] else img.shape[1]
    y_max = int(y_max) if y_max < img.shape[0] else img.shape[0]

    crop_img = img[y_min:y_max, x_min:x_max, :]
    return crop_img


def get_second_order_img(root):
    '''
    img,png
    return: img absolute paths list
    '''
    dirs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    img_paths = []
    for d in dirs:
        # if osp.basename(d) in ['A1','A10','A11','A12']:
        #     continue
        img_path = [os.path.join(d, imb) for imb in os.listdir(d) if imb.endswith(('.jpg', '.png'))]
        img_paths.extend(img_path)
    return img_paths


def split_train_val_data(src, dst, portion=.8):
    img_paths = get_second_order_img(src)
    with open(os.path.join(dst, 'train_label.txt'), mode='w', encoding='utf8') as f1, open(
            os.path.join(dst, 'test_label.txt'), mode='w', encoding='utf8') as f2:

        for index, image_file in enumerate(img_paths):
            sub_dir = os.path.basename(os.path.dirname(image_file))
            img = cv2.imread(image_file)
            if random.randint(0, 9) * .1 < portion:
                # sub_dir = os.path.basename(os.path.dirname(image_file))
                if not os.path.exists(sub1 := os.path.join(dst, 'train')):
                    os.mkdir(sub1)
                if not os.path.exists(sub := os.path.join(dst, 'train', sub_dir)):
                    os.mkdir(sub)
                out_path = os.path.join(sub, os.path.basename(image_file))
                shutil.copyfile(image_file, out_path)
                f1.write(f'{out_path} 0.000000,{float(img.shape[1]):.6f},0.000000,{float(img.shape[0]):.6f}\n')

            else:
                if not os.path.exists(sub2 := os.path.join(dst, 'test')):
                    os.mkdir(sub2)
                if not os.path.exists(sub := os.path.join(dst, 'test', sub_dir)):
                    os.mkdir(sub)
                out_path = os.path.join(sub, os.path.basename(image_file))
                shutil.copyfile(image_file, out_path)
                f2.write(f'{out_path} 0.000000,{float(img.shape[1]):.6f},0.000000,{float(img.shape[0]):.6f}\n')


def cut_img_with_json_bbox(sencond_order_root, out_dir, label_dict=None):
    '''
    * 对于存在标注的图片，按照标注切割。
    * 没有标注的图片，整个儿输出。

    '''

    inner = .2
    outer = .6
    img_paths = get_second_order_img(sencond_order_root)

    for idx, img_path in tqdm(enumerate(img_paths)):
        json_path = osp.splitext(img_path)[0] + '.json'
        if osp.exists(json_path):
            jd = getJsonDict(json_path)
            img = cv2.imread(img_path)
            # 遍历json 字典的block 获得block中的坐标框信息。
            for shape in jd['shapes']:
                t, b = shape['points']
                point_list = [t[0], t[1], b[0], b[1]]

                out_img = crop_img_ratio(img, point_list, inner, outer)

                sub_dir = shape['label']
                if label_dict:
                    sub_dir = label_dict[sub_dir]
                # sub_dir = os.path.basename(os.path.dirname(img_path))
                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)
                if not os.path.exists(sub := os.path.join(out_dir, sub_dir)):
                    os.mkdir(sub)
                s = base64.b64encode(os.urandom(2)).decode("utf8")
                s = s.replace("\\", "").replace("/", "").replace("=", "").replace("+", "")

                out_path = os.path.join(sub, f'{osp.splitext(os.path.basename(img_path))[0]}_{s}.jpg')
                cv2.imwrite(out_path, out_img)
        else:
            if not osp.exists(sub := osp.join(out_dir, osp.basename(osp.dirname(img_path)))):
                os.mkdir(sub)
            shutil.copyfile(img_path, osp.join(sub, osp.basename(img_path)))
    return out_dir

    # draw_bbox_results(img, output[0]['bbox'].tolist(), r'./img.jpg')


def to_jpg_with_json(img_path, dst_dir):
    '''
    将图像输出到目标路径中，如果是jpg直接复制到目标路径，如果是png，转为jpg。
    最后对json统一处理。
    '''
    if img_path.endswith('.png'):
        img_arr = cv2.imread(img_path)
        cv2.imwrite(osp.join(dst_dir, osp.splitext(osp.basename(img_path))[0]) + '.jpg', img_arr)
        if osp.exists(jp := (osp.splitext(img_path)[0] + '.json')):
            jd = getJsonDict(jp)
            jd['imagePath'] = osp.splitext(osp.basename(img_path))[0] + '.jpg'
            with open(osp.join(dst_dir, osp.basename(jp)), mode='w', encoding='utf8') as f:
                data = json.dumps(jd)
                f.write(data)
    else:
        json_stem = osp.splitext(osp.basename(img_path))[0]
        shutil.copyfile(img_path, osp.join(dst_dir, osp.basename(img_path)))
        if osp.exists(jp := (osp.splitext(img_path)[0] + '.json')):
            shutil.copyfile(jp, osp.join(dst_dir, json_stem + '.json'))


def all_to_jpg(second_order_root, dst_root):
    img_paths = get_second_order_img(second_order_root)
    if not osp.exists(dst_root):
        os.mkdir(dst_root)
    for img_path in img_paths:
        if not osp.exists(subdir := osp.join(dst_root, osp.basename(osp.dirname(img_path)))):
            os.mkdir(subdir)
        to_jpg_with_json(img_path, subdir)


def modifyJson_bbox_label(json_dict, bboxes, h, w, img_basename, rlt_path=''):
    '''
    根据提供的信息，对json字典完成更新，不需要返回值。
    惯例json与对应的图在一个文件夹内。 否则提供相对路径：rel_path
    :param json_dict:
    :param polygon_list:
    :param h:
    :param w:
    :param img_path:
    :param rlt_path:
    :return:
    '''
    shapes = json_dict['shapes']
    # assert len(shapes) == len(polygon_list), f'传入标注框数目{len(polygon_list)}与json_dict[shapes]数目{len(shapes)}不匹配'
    # 标注框更新
    for i in range(len(shapes)):
        json_dict['shapes'][i]['points'] = [bboxes[i][:2], bboxes[i][2:4]]
        json_dict['shapes'][i]['label'] = bboxes[i][-1]

    json_dict['imageData'] = None
    json_dict['imageHeight'] = h
    json_dict['imageWidth'] = w
    json_dict['imagePath'] = os.path.join(rlt_path, img_basename)  # 相对于json的路径


def filter_copy_tree(sencond_order_root, out_dir1, out_dir2):
    '''

    '''

    img_paths = get_second_order_img(sencond_order_root)

    for idx, img_path in tqdm(enumerate(img_paths)):
        json_path = osp.splitext(img_path)[0] + '.json'
        if len(osp.basename(img_path)) <= 2 + 8 + 4:

            if not os.path.exists(out_dir1):
                os.mkdir(out_dir1)
            sub_dir = os.path.basename(osp.dirname(img_path))
            if not os.path.exists(sub := os.path.join(out_dir1, sub_dir)):
                os.mkdir(sub)

            # out_path = os.path.join(sub, f'{osp.splitext(os.path.basename(img_path))[0]}.jpg')

            shutil.copyfile(img_path, osp.join(sub, osp.basename(img_path)))
            if osp.exists(json_path):
                shutil.copyfile(json_path, osp.join(sub, osp.basename(json_path)))

        else:

            if not os.path.exists(out_dir2):
                os.mkdir(out_dir2)
            sub_dir = os.path.basename(osp.dirname(img_path))
            if not os.path.exists(sub := os.path.join(out_dir2, sub_dir)):
                os.mkdir(sub)

            # out_path = os.path.join(sub, f'{osp.splitext(os.path.basename(img_path))[0]}.jpg')

            shutil.copyfile(img_path, osp.join(sub, osp.basename(img_path)))
            if osp.exists(json_path):
                shutil.copyfile(json_path, osp.join(sub, osp.basename(json_path)))


def img_features_to_json(second_order_img_path, core_name='from_video_aug_1000_per_class2',
                         dst_json_dir=r'E:\lxd\paddleclas_museum_dev\_datasets_\feature_from_infer_2'):
    sys.path.append(r'E:\lxd\paddleclas_museum_dev')
    from Random_Forest.rf_cnn_feature import get_recg_predictor
    '''
    jd需包含信息： image_paths,features,labels,perpose
    '''
    img_paths = get_second_order_img(second_order_img_path)
    jd = defaultdict(list)
    rec_predictor = get_recg_predictor()
    dst_json_path = osp.join(dst_json_dir, f'cnn_{core_name}_features.json')
    for img_path in tqdm(img_paths):
        img_dict = {}
        img_arr = cv2.imread(img_path)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        label = osp.basename(osp.dirname(img_path))
        feature = rec_predictor.predict(img_arr)
        img_dict['img_path'] = img_path
        img_dict['img_feature'] = feature.tolist()
        img_dict['img_label'] = label
        img_dict['source'] = core_name
        jd[label].append(img_dict)

    with open(dst_json_path, mode='w', encoding='utf8') as jf:
        data = json.dumps(jd)
        jf.write(data)


def img_features_to_pkl(second_order_img_path, core_name='from_video_aug_1000_per_class',
                        dst_pkl_dir=r'E:\lxd\paddleclas_museum_dev\_datasets_'):
    sys.path.append(r'E:\lxd\paddleclas_museum_dev')
    from Random_Forest.rf_cnn_feature import get_recg_predictor
    '''
    jd需包含信息： image_paths,features,labels,perpose
    '''
    img_paths = get_second_order_img(second_order_img_path)
    pd = defaultdict(list)
    rec_predictor = get_recg_predictor()
    dst_pkl_path = osp.join(dst_pkl_dir, f'cnn_{core_name}_features.pkl')
    for img_path in tqdm(img_paths):
        img_dict = {}
        img_arr = cv2.imread(img_path)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        label = osp.basename(osp.dirname(img_path))
        feature = rec_predictor.predict(img_arr)
        img_dict['img_path'] = img_path
        img_dict['img_feature'] = feature.tolist()
        img_dict['img_label'] = label
        img_dict['source'] = core_name
        pd[label].append(img_dict)

    with open(dst_pkl_path, mode='wb') as pf:
        data = pickle.dump(pd, pf)
        # pf.write(data)


def all_to_mum(src, dst):
    jps = [osp.join(src, jf) for jf in os.listdir(src) if jf.endswith('.json')]
    for jp in jps:

        jd = getJsonDict(jp)
        label = ch2en[jd['shapes'][0]['label']]
        if not osp.exists(dst):
            os.mkdir(dst)
        if not osp.exists(sub := osp.join(dst, label)):
            os.mkdir(sub)
        s = base64.b64encode(os.urandom(5)).decode("utf8")
        s = s.replace("\\", "").replace("/", "").replace("=", "").replace("+", "")
        shutil.copy2(jp, osp.join(sub, s + '.json'))
        shutil.copy2(osp.join(src, jd['imagePath']), osp.join(sub, s + '.jpg'))


if __name__ == '__main__':
    img_dir = r'E:\lxd_dataset\Museum\cut_out_all\cuted_data\from_video_aug_1000_per_class2'
    json_dir = r'E:\lxd_dataset\Museum\fouzhou_3D\3D_cover'
    json_path = r'E:\lxd_dataset\md1/000000uploadPic_ahdxtsg_192.168.1.66_25735308477776039_23.json'
    lp = r'F:\Datasets\key_points\tem/dmPos_000001.json'
    dst_path = r'E:\lxd_dataset\Museum\fouzhou_3D\3d_img_show'
    dst1 = r'D:\dataset\OCR\图书ocr补充数据\补充数据\有效'
    dst2 = r'D:\dataset\OCR\图书ocr补充数据\补充数据\shujutang_complement'
    img_path = r'E:\lxd_dataset\Museum\福州博物馆文物封面图\3d_img_show'
    cuted = r'E:\lxd_dataset\Museum\cut_out_all\cuted_data\from_video_aug_specify_13_14_35'
    sec_dir = r'E:\lxd_dataset\Museum\cut_out_all\cuted_data\from_video_aug_3_ext'
    # getJsonDict(r'D:\dataset\OCR\shujutang_all_0628/')
    json_text2json_label(r'D:\dataset\OCR\shujutang_all_0628/', r'D:\dataset\OCR\shujutang_json_label_modify')
    # img_features_to_json(img_dir)
    # all_to_mum(json_dir,dst_path)

    # paddle.device.set_device('gpu')

    # cor_name = osp.basename(sec_dir)
    # batch_modify_label(json_dir,dst_dir,label_dict=d)

    # second_order_couple_modify_label(json_dir,dst1,label_dict=ch2en)
    # abstract_data_2_onedir(dst1,dst2)
    # subs = [osp.join(img_path,basename) for basename in os.listdir(img_path)]
    # json_name2json_file(r'D:\dataset\OCR\图书ocr补充数据\补充数据\shujutang_complement',r'D:\dataset\OCR\图书ocr补充数据\补充数据\json')

    # for sub in subs:
    #     json_name2json_file(sub,json_dir)

    # main(dst_dir,cuted)
    # all_to_jpg(img_path,json_dir)
    # filter_copy_tree(img_dir,dst1,dst2)
    # cut_img_with_json_bbox(json_dir, cuted, ch2en)

    # trans_dirs = [
    #                 r'E:\lxd_dataset\Museum\cut_out_all\cuted_data\from_video',
    #               # r'E:\lxd_dataset\Museum\cut_out_all\cuted_data\from_video_aug_1000_per_class',
    #               # r'E:\lxd_dataset\Museum\cut_out_all\cuted_data\from_video_aug_1000_per_class2',
    #               # r'E:\lxd_dataset\Museum\cut_out_all\cuted_data\from_video_aug_1_ext',
    #               # r'E:\lxd_dataset\Museum\cut_out_all\cuted_data\from_video_aug_3_ext',
    #               # r'E:\lxd_dataset\Museum\cut_out_all\cuted_data\from_video_aug_10_ext',
    #               # r'E:\lxd_dataset\Museum\cut_out_all\cuted_data\from_video_aug_no_ext',
    #               # r'E:\lxd_dataset\Museum\cut_out_all\cuted_data\from_video_aug_specify1_13_35',
    #               # r'E:\lxd_dataset\Museum\cut_out_all\cuted_data\from_video_aug_specify2_13_14_35',
    #               # r'E:\lxd_dataset\Museum\cut_out_all\cuted_data\from_video_aug_specify3_13',
    #               # r'E:\lxd_dataset\Museum\cut_out_all\cuted_data\from_video_aug_specify4_43',
    #               # r'E:\lxd_dataset\Museum\cut_out_all\cuted_data\from_video_aug_specify5_13',
    #               # r'E:\lxd_dataset\Museum\cut_out_all\cuted_data\from_video_aug_specify6_13_14_35_300',
    #               # r'E:\lxd_dataset\Museum\cut_out_all\cuted_data\from_video_aug_specify7_13_300',
    #               r'E:\lxd_dataset\Museum\cut_out_all\cuted_data\from_video_3d_img_show',
    #               # r'E:\lxd_dataset\Museum\cut_out_all\cuted_data\cuted_0421',
    #               r'E:\lxd\paddleclas_museum_dev\_datasets_\test',
    #               r'E:\lxd_dataset\Museum\cut_out_all\cuted_data\from_image_aug_10'
    #               ]
    # for trans_path in trans_dirs:
    #     cor_name = osp.basename(trans_path)
    #     img_features_to_json(trans_path,core_name=cor_name)
    # batch_fill_img(img_dir, json_dir, dst_dir,0)

    # batch_modify_label(r'E:\lxd_dataset\dm8_borrow_1080p_0401_2',r'E:\lxd_dataset\dest_json')
    # batch_modify_label(json_dir,dst_dir)
    # fill_img(img_path,json_path,dst_dir)
    # img = cv2.imread(img_path)
    # bbox = get_boxes(json_path)
    # bbox_arr = np.array(bbox)
    # for i in range(bbox_arr.shape[0]):
    #     xs = int(bbox_arr[i, :, 1].min(axis=-1))
    #     xe = int(bbox_arr[i, :, 1].max(axis=-1))
    #     ys = int(bbox_arr[i, :, 0].min())
    #     ye = int(bbox_arr[i, :, 0].max())
    #     y_len = ye - ys
    #     x_len = xe - xs
    #     if xs - 10 - x_len > 0:
    #         img[xs:xe, ys:ye, :] = img[xs - 10 - x_len:xs - 10, ys:ye, :]
