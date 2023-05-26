# coding=utf-8
import math
import tarfile
import cv2
import os
import argparse
import numpy as np
import copy
import json
import jsonlines
from .lmdb_saver import LmdbSaver
import copy


# from unrar import rarfile


# ChineseStreetViewTextRecognition
# -d D:\dataset\OCR\ChineseStreetViewTextRecognition -f train_images.tar.gz -l train.list -n ChineseStreetViewTextRecognition -t 4

# ICDAR2019-ArT
# -d D:\dataset\OCR\ICDAR2019-ArT -f train_images.tar.gz -l train_labels.json -n ICDAR2019-ArT_train
# -d D:\dataset\OCR\ICDAR2019-ArT -f test_part1_images.tar.gz -f test_part2_images.tar.gz -n ICDAR2019-ArT_test

# -d D:\dataset\OCR\ICDAR2019-ArT -f train_task2_images.tar.gz -l train_task2_labels.json -n ICDAR2019-ArT_task2_train
# -d D:\dataset\OCR\ICDAR2019-ArT -f test_part1_task2_images.tar.gz -f test_part2_task2_images.tar.gz -n ICDAR2019-ArT_task2_test

# ICDAR2019-LSVT
# -d D:\dataset\OCR\ICDAR2019-LSVT -f train_full_images_0.tar.gz -f train_full_images_1.tar.gz -l train_full_labels.json -n ICDAR2019-LSVT_train
# -d D:\dataset\OCR\ICDAR2019-LSVT -f train_weak_images_0.tar.gz -f train_weak_images_1.tar.gz -f train_weak_images_2.tar.gz -f train_weak_images_3.tar.gz -f train_weak_images_4.tar.gz -f train_weak_images_5.tar.gz -f train_weak_images_6.tar.gz -f train_weak_images_7.tar.gz -f train_weak_images_8.tar.gz -f train_weak_images_9.tar.gz -l train_weak_labels.json -n ICDAR2019-LSVT_weak_train
# -d D:\dataset\OCR\ICDAR2019-LSVT -f test_part1_images.tar.gz -f test_part2_images.tar.gz -n ICDAR2019-LSVT_test

# ChineseDocumentTextRecognition
# -d D:\dataset\OCR\ChineseDocumentTextRecognition -f images_train.lst -l data_label_train.txt -n ChineseDocText_train -t 2
# -d D:\dataset\OCR\ChineseDocumentTextRecognition -f images_test.lst -l data_label_test.txt -n ChineseDocText_test -t 2

def parse_args():
    parser = argparse.ArgumentParser("generic-image-rec train script")
    parser.add_argument(
        '-d',
        '--dir',
        type=str,
        default='',
        help='directory of compressed files')
    parser.add_argument(
        '-f',
        '--files',
        action='append',
        default=[],
        help='compressed list')
    parser.add_argument(
        '-l',
        '--labels',
        type=str,
        default=None,
        help='label information of images'
    )
    parser.add_argument(
        '-n',
        '--data_name',
        type=str,
        default='ICDAR2019-ArT',
        help='destine dataset name'
    )
    parser.add_argument(
        '-t',
        '--type',
        type=int,
        default=4,
        help='type of label format[4:whfl, 3:fl]'
    )
    parser.add_argument(
        "--is_pieces",
        type=bool,
        default=False,
        help="Whether to cut the multi-label to pieces.")
    args = parser.parse_args()
    return args


def untar(fname, dirs):
    t = tarfile.open(fname)
    t.extractall(path=dirs)


def tar(fname):
    t = tarfile.open(fname + ".tar.gz", "w:gz")
    for root, dir, files in os.walk(fname):
        print(root, dir, files)
        for file in files:
            fullpath = os.path.join(root, file)
            t.add(fullpath)
    t.close()


def get_label_by_file(image_path_name, labels):
    image_file_name = os.path.basename(image_path_name)
    label = None
    if image_file_name in labels:
        label = labels[image_file_name]
    else:
        image_name = ''.join(image_file_name.split('.')[:-1])
        if image_name in labels:
            label = labels[image_name]
    return label


def cal_distance(p1, p2):
    return math.sqrt(math.pow(p2[0] - p1[0], 2) + math.pow(p2[1] - p1[1], 2))


def compute_perspective_coord_map(points_in, h_dst=48, side_ratio=1.0, check=True):
    points = copy.deepcopy(points_in)
    w_ori = cal_distance(points[0], points[1])
    h_ori = cal_distance(points[1], points[2])
    if w_ori < h_ori and check:
        points = points[1:] + points[:1]
        w_ori, h_ori = h_ori, w_ori
    w_dst = int(w_ori * h_dst / max(1, h_ori))
    side = int(h_dst * side_ratio)
    pts_dst = [[side, side], [w_dst + side, side], [w_dst + side, h_dst + side], [side, side + h_dst]]
    return points, pts_dst, (w_dst + 2 * side, h_dst + 2 * side)


def cut_to_pieces(saver, label_info):
    skips = ['###', '*']
    image = label_info['image']
    labels = label_info['label']
    for i, label in enumerate(labels):
        points_ = label['points']
        check = True
        if label['transcription'] in skips:
            continue
        if 4 != len(points_):
            arr_pts = np.array(points_)
            l, r, t, b = min(arr_pts[:, 0]), max(arr_pts[:, 0]), min(arr_pts[:, 1]), max(arr_pts[:, 1])
            points_ = [[l, t], [r, t], [r, b], [l, b]]
            check = False
        if 1 == len(label['transcription']):
            check = False
            # print('possible error points number information, {} != 4, {}'.format(len(points), l['transcription']))
            # continue
        try:
            points, coord_dst, size_dst = compute_perspective_coord_map(points_, check=check)
            matrix = cv2.getPerspectiveTransform(np.float32(points), np.float32(coord_dst))
            piece = cv2.warpPerspective(image, matrix, size_dst)  # (h, w))
            saver.add({'image': piece, 'label': label['transcription'], 'points': coord_dst,
                       'file': label['file'] if 'file' in label else ''},
                      is_to_json=True)
        except:
            temp = 0


def save_single_sample(fin, name, labels, saver, no_label_num):
    if labels is not None:
        label = get_label_by_file(name, labels)
    else:
        label = {}
    if label is not None and fin is not None:
        content = fin.read()
        label = copy.deepcopy(label)
        try:
            image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
            if isinstance(label, dict):
                label['image'] = image
            else:
                label = {'image': image, 'label': label}
            if args.is_pieces:
                cut_to_pieces(saver, label)
            else:
                saver.add(label, is_to_json=True)
        except:
            print(' {} has invalid informtion'.format(name))
    else:
        print(' {} has no label information'.format(name))
        no_label_num += 1
    return no_label_num


def convert_tar_gz(compressed_file, labels, saver):
    fin = tarfile.open(compressed_file, 'r:gz')
    no_label_num = 0
    for i, member in enumerate(fin.getmembers()):
        if 0 == i % 5000:
            print('dealed {}'.format(i))
        f = fin.extractfile(member)
        no_label_num = save_single_sample(f, member.name, labels, saver, no_label_num)
    print('\n{} files have no label information'.format(no_label_num))


def emerge_in_data(img_arr,label,saver,is_pieces=True):
    label = {'image': img_arr, 'label': label}
    if is_pieces:
        cut_to_pieces(saver, label)
    else:
        saver.add(label, is_to_json=True)



def convert_images(file_list, labels, saver):
    lst = open(file_list, 'r', encoding='utf-8')
    lines = lst.readlines()
    no_label_num = 0
    for i, line in enumerate(lines):
        if 0 == i % 50000:
            print('have dealed {}'.format(i))
        line = line.strip().replace('\\', '/')
        f = open(line, 'rb')
        no_label_num = save_single_sample(f, line.split('/')[-1], labels, saver, no_label_num)
    print('\n{} files have no label information'.format(no_label_num))


def load_list_labels(label_path, label_len_type):
    fin = open(label_path, 'r', encoding='utf-8')  # label list: width, height, file name, label
    lines = fin.readlines()
    labels = {}
    for line in lines:
        label_segs = line.strip().split('\t')
        if 1 == len(label_segs):
            label_segs = label_segs[0].split(' ')
        if 4 == label_len_type:  # width, height, file, label
            labels[label_segs[2]] = {'width': int(label_segs[0]),
                                     'height': int(label_segs[1]),
                                     'file': label_segs[2],
                                     'label': ' '.join(label_segs[3:])}
        else:  # 2 == label_len_type: file, label
            labels[os.path.basename(label_segs[0])] = {'file': label_segs[0],
                                                       'label': ''.join(label_segs[1:])}
    return labels


class ConvertLabelCTW(object):
    def __init__(self, label_info):
        self.label_info = label_info

    def get_file_name(self):
        return self.label_info['file_name']

    def get_line_transcript(self, line_info):
        chars = []
        for ch in line_info:
            chars.append(ch['text'])
        return ''.join(chars)

    def get_rect_points(self, line_info):
        rects = []
        for ch in line_info:
            bb = ch['adjusted_bbox']
            rc = [[bb[0], bb[1]], [bb[0] + bb[2], bb[1]], [bb[0] + bb[2], bb[1] + bb[3]], [bb[0], bb[1] + bb[3]]]
            # rc = np.array([np.array(r) for r in rc])
            rects.append(rc)
        return rects

    def combine_horizontal_block(self, line_info):
        rects = self.get_rect_points(line_info)
        rc_combined = copy.deepcopy(rects[0])
        cost = 0
        for rc in rects[1:]:
            rc0, rc3 = rc[0], rc[3]
            rc_c1, rc_c2 = rc_combined[1], rc_combined[2]
            cost += abs(rc0[0] - rc_c1[0]) + abs(rc3[0] - rc_c2[0])
            rc_combined[1] = copy.deepcopy(rc[1])
            rc_combined[2] = copy.deepcopy(rc[2])
        return rc_combined, cost

    def combine_vertical_block(self, line_info):
        rects = self.get_rect_points(line_info)
        rc_combined = copy.deepcopy(rects[0])
        cost = 0
        for rc in rects[1:]:
            rc0, rc1 = rc[0], rc[1]
            rc_c2, rc_c3 = rc_combined[2], rc_combined[3]
            cost += abs(rc0[1] - rc_c3[1]) + abs(rc1[1] - rc_c2[1])
            rc_combined[2] = copy.deepcopy(rc[2])
            rc_combined[3] = copy.deepcopy(rc[3])
        return rc_combined, cost

    def get_icdar15_label(self):
        label = []
        annotations = self.label_info['annotations']
        for line in annotations:
            points_v, cost_v = self.combine_vertical_block(line)
            points_h, cost_h = self.combine_horizontal_block(line)
            line_label = {
                'transcription': self.get_line_transcript(line),
                'file': self.get_file_name(),
                'points': points_h if cost_h < cost_v else points_v,
                'illegibility': 0,
                'chars':self.get_rect_points(line),
            }
            label.append(line_label)
        return label


# 任务 切出图片中的每一个字符
def get_label_from_json_line(jsonfile):  # , imagepath): #, labelfile, lineimgpath, count_t, cnt):
    # 打开jsonl文件夹并读取所有信息
    labels = {}
    with open(jsonfile, "r+", encoding="utf-8") as f:
        json_inf = jsonlines.Reader(f)
        # 遍历jsonl的每一条信息
        for item in json_inf:
            # 通过每一条信息 的键找到对应的值 注释(主要是为了找坐标和单字的) 图片名称
            converter = ConvertLabelCTW(item)
            labels[converter.get_file_name()] = converter.get_icdar15_label()

    return labels


def load_labels(label_type):
    if args.labels:
        label_path = os.path.join(args.dir, args.labels)
        if '.json' == label_path[-5:]:
            with open(label_path, 'r', encoding='utf-8') as fin:
                labels = json.load(fin)
        elif label_path.find('.jsonl'):
            labels = get_label_from_json_line(label_path)
        else:
            labels = load_list_labels(label_path, label_type)
    else:
        labels = None
    return labels


# -d D:\dataset\OCR\ChineseDocumentTextRecognition -f images_test.lst -l data_label_test.txt -n ChineseDocText_test -t 2
# -d D:\dataset\OCR\ChineseDocumentTextRecognition -f Synthetic_Chinese_String_Dataset.rar -l data_train.txt -n ChineseDocText_3600k -t 2
# -d D:\dataset\OCR\ChineseDocumentTextRecognition -f images_3600k.lst -l data_label_train.txt -n ChineseDocText_3600k -t 2

# -d D:\Dataset\ocr\ICDAR2019-ArT -f train_images.tar.gz   -l train_labels.json -n ICDAR2019-ArT_train -t 2 --is_pieces True
# -d D:\dataset\OCR\ICDAR2019-LSVT -f train_full_images_0.tar.gz -f train_full_images_1.tar.gz  -l train_full_labels.json -n ICDAR2019-LSVT_train -t 2 --is_pieces True
# -d D:\Dataset\ocr\ICDAR2017-RCTW -f ICDAR2017-RCTW.lst   -l train_gts_ic15.json -n ICDAR2017-RCTW-train -t 2 --is_pieces True
# -d e:\Dataset\ocr\MTWI -f image_train.lst   -l txt_train_ic15.json -n MTWI-train -t 2 --is_pieces True

# -d E:/Dataset/ocr/ctw -f trainval.lst   -l train.jsonl -n ctw-train -t 2 --is_pieces True
# -d E:/Dataset/ocr/ctw -f trainval.lst   -l val.jsonl -n ctw-val -t 2 --is_pieces True


# -d D:\dataset\OCR\MTWI_lmdb_test -f image_train1.lst   -l txt_train_ic15.json -n MTWI-train -t 2 --is_pieces True

# -d E:\lxd\text_renderer_lv\example_data\effect_layout_image\effect_ensemble -f image_paths.lst   -l lmdb_label.json -n images -t 2 --is_pieces True


# 需要构造的输入数据：①image_train.lst ： jpg图像绝对路径文本行。
                # ②txt_train_ic15.json ：
if __name__ == '__main__':
    args = parse_args()
    labels = load_labels(args.type) # labels 是整个标注文件的内容 标注文件为：args.labels
    if args.is_pieces:
        args.data_name = args.data_name + '_pieces'
    # lmdb_path: E:\lxd\text_renderer_lv\example_data\effect_layout_image\effect_ensemble\images_pieces
    lmdb_saver = LmdbSaver({'lmdb_path': os.path.join(args.dir, args.data_name), 'cnt': 0, 'cache_capacity': 500})
    tar_lst = ['.gz', 'tar']
    for file in args.files:
        if file[-3:] in tar_lst:
            convert_tar_gz(os.path.join(args.dir, file), labels, lmdb_saver)
        # elif 'rar' == file[-3:]:
        #     convert_rar(os.path.join(args.dir, file), labels, lmdb_saver)
        else:
            convert_images(os.path.join(args.dir, file), labels, lmdb_saver)
    lmdb_saver.close()
