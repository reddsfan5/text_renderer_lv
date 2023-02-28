'''


'''
import copy
import json
import cv2
import numpy as np
import random
import os
import base64

from tqdm import tqdm

from .parse_json import get_specific_file_paths, getJsonDict


def pattern_generator(png_dir):
    img_paths = get_specific_file_paths(png_dir, '.png')
    while True:
        yield str(random.choice(img_paths))


def bg_with_pattern(bg_img_arr, pattern_path_generator,box):
    '''
    :return:
    '''
    pattern = None
    while pattern is None:
        pattern = cv2.imread(next(pattern_path_generator), flags=cv2.IMREAD_UNCHANGED)

    if random.randint(0, 1) == 0:
        pattern = cv2.rotate(pattern, cv2.ROTATE_90_COUNTERCLOCKWISE)
    bg_img_arr = bg_img_arr[...,:3]
    bg_h, bg_w = bg_img_arr.shape[:2]

    min_side = min(bg_h, bg_w)
    ret = np.argwhere(pattern[..., 3:] > 0)
    xs = ret[..., 1].min()
    xe = ret[..., 1].max()
    ys = ret[..., 0].min()
    ye = ret[..., 0].max()
    pattern_cutout = pattern[ys:ye, xs:xe, :]
    box_xs,box_ys = box[0]
    box_xe,box_ye = box[2]
    box_h,box_w = box_ye-box_ys,box_xe-box_xs

    resized_pattern_core = cv2.resize(pattern_cutout, (box_w, box_h))
    # resized_pattern_core = cv2.blur(resized_pattern_core,(5,5))

    white = np.ones_like(bg_img_arr, dtype=np.uint8) * 255
    white = np.concatenate([white, np.zeros((*bg_img_arr.shape[:2], 1), np.uint8)], axis=-1)
    pys = box_ys
    pye = box_ye

    pxs = box_xs
    pxe = box_xe

    white[pys:pye, pxs:pxe, :] = resized_pattern_core

    # _,pattern_bar = cv2.threshold(white,0,255,type=cv2.THRESH_BINARY)
    # white = np.where(white[...,3]>254,0,255)
    index_table = white[..., 3] > 240
    # index_table = np.tile(index_table,(1,1,4))
    index_table = np.tile(index_table[..., None], (1, 1, 4))
    pattern_mask = np.where(index_table, 0, 255)[..., :3].astype(np.uint8)  # 整型类型与位运算
    pattern_mask_inv = cv2.bitwise_not(pattern_mask)

    bg = cv2.bitwise_and(pattern_mask, bg_img_arr)
    pattern = cv2.bitwise_and(white[..., :3], pattern_mask_inv)

    fusion = cv2.bitwise_or(bg, pattern)
    # import matplotlib.pyplot as plt
    # plt.imshow(fusion)
    # plt.show()

    return fusion


def grid_img(rows=5, columns=5, img_size=(2560, 1440), density=.5, img=None, show=False):
    bar_h_range = (170, 200)
    bar_w_range = (25, 28)
    pad = 30
    bar_adjacent_range = (20, 35)
    img_w, img_h = img_size
    h_bias = 100
    w_bias = 150
    two_barcode_shift = 5
    # 计算分割后每个网格的保守框，高
    grid_w = (img_w - w_bias) // columns
    grid_h = (img_h - h_bias) // rows
    bar_boxes = []
    for grid_h_start_index in range(rows):
        for grid_w_start__index in range(columns):
            if random.uniform(0, 1) > density:
                continue

            bar_adjacent_distance = random.randint(*bar_adjacent_range)
            grid_ws = grid_w_start__index * grid_w + w_bias // 2
            grid_hs = grid_h_start_index * grid_h + h_bias // 2

            bar_w = random.randint(*bar_w_range)
            bar_h = random.randint(*bar_h_range)
            if grid_w - max(bar_w_range) - 2 * pad <= 0 or grid_h - max(bar_h_range) * 2 - 2 * pad - max(
                    bar_adjacent_range) <= 0:
                raise ValueError('网格过小，请调整网格分割数')
            bar_ws_reletive = random.randint(pad, grid_w - bar_w - pad)
            bar_hs_reletive = random.randint(pad, grid_h - 2 * bar_h - 2 * pad - bar_adjacent_distance)
            bar1_ws = bar_ws_reletive + grid_ws
            bar1_hs = bar_hs_reletive + grid_hs
            bar2_ws = bar1_ws + random.randint(-two_barcode_shift, two_barcode_shift)
            bar2_hs = bar1_hs + bar_adjacent_distance + bar_h

            bar1_box = [[bar1_ws, bar1_hs], [bar1_ws + bar_w, bar1_hs], [bar1_ws + bar_w, bar1_hs + bar_h],
                        [bar1_ws, bar1_hs + bar_h]]
            bar2_box = [[bar2_ws, bar2_hs], [bar2_ws + bar_w, bar2_hs], [bar2_ws + bar_w, bar2_hs + bar_h],
                        [bar2_ws, bar2_hs + bar_h]]
            bar_boxes.append(bar1_box)
            bar_boxes.append(bar2_box)
    if show:
        bar_boxes = np.array(bar_boxes)
        cv2.polylines(img, bar_boxes, True, (255, 0, 0), thickness=16)
        import matplotlib.pyplot as plt
        plt.imshow(img)
        plt.show()
    return bar_boxes


def render_json_template(boxes, json_path, dst_dir):
    jd = getJsonDict(json_path)
    new_shapes = []
    shape = jd['shapes'][0]
    for box in boxes:
        shape['points'] = box
        new_shapes.append(copy.deepcopy(shape))
    jd['shapes'] = new_shapes
    s = base64.b64encode(os.urandom(4)).decode("utf8")
    s = s.replace("\\", "").replace("/", "").replace("=", "").replace("+", "")
    with open(os.path.join(dst_dir, s + '.json'), mode='w', encoding='utf8') as f:
        json.dump(jd, f)


if __name__ == '__main__':
    gen = pattern_generator(r'D:\lxd_code\OCR_SOURCE\0_filtered')
    bg = cv2.imread(r'E:\lxd\OCR_project\OCR_SOURCE\bg\6k5dwe.jpg')
    img = cv2.imread(r'D:\dataset\bar_code\a_bar\a_det\inside_bus_050_1221_6num_continue_vLo_000006.jpg')

    for i in range(10):
        fusion = bg_with_pattern(bg, gen,[[100,20],[230,20],[230,130],[100,130]])
        import matplotlib.pyplot as plt

        plt.imshow(fusion)
        plt.show()

    dst_dir = r'D:\dataset\hand_pose\indoorCVPR_09\adjacent_json'
    json_path = r'D:\dataset\hand_pose\indoorCVPR_09\box_template\airport_inside_0549_01.json'
    # for i in tqdm(range(10000)):
    #     boxes = grid_img(2, 15, density=.5)
    #     render_json_template(boxes, json_path, dst_dir)
