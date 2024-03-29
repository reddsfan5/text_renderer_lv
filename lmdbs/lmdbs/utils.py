import io

import cv2
from PIL import Image
import copy
import yaml
import base64
import numpy as np
import configparser

def b64encode_img(image):
    image = Image.fromarray(image[:, :, (2, 1, 0)])
    f = io.BytesIO()
    image.save(f, format="JPEG")
    imgbin = f.getvalue()
    # img_b64_str = base64.encodebytes(imgbin).decode("utf-8").replace("\n", "")
    b64_str = base64.b64encode(imgbin)
    # print(b64_str[:20])
    img_b64_str = base64.b64encode(imgbin).decode('utf-8')
    return img_b64_str


def b64decode_img(image_str):
    # imagebin = base64.decodebytes(image_str.encode("utf-8"))
    imagebin = base64.b64decode(image_str.encode('utf-8'))
    f = io.BytesIO()
    f.write(imagebin)
    image = Image.open(f)
    image = np.asarray(image, np.uint8)[:, :, (2, 1, 0)]
    return image


def get_ini_section_keys(config, section_name):
    try:
        opts = config.options(section_name)
    except:
        opts = []
    return opts


def get_int_section_with_weight(config, section_name):
    section_name_weight = section_name + '_weight'
    opts = get_ini_section_keys(config, section_name)
    opts_weight = get_ini_section_keys(config, section_name_weight)

    lmdb_list = []
    for opt in opts:
        weight = 1
        if opts_weight is not None and opt in opts_weight:
            weight = config.get(section_name_weight, opt)
        lmdb_list.append([config.get(section_name, opt), int(weight)])
    return lmdb_list


def get_ini_section(config, section_name):
    keys = get_ini_section_keys(config, section_name)
    data_dict = {}
    for key in keys:
        value = config.get(section_name, key)
        try:
            data_dict[key] = eval(value)
        except:
            data_dict[key] = value
    return data_dict


# read lmdb's list from config file
def load_lmdb_config(lmdb_config, mode):
    config = configparser.ConfigParser()
    config.read(lmdb_config)
    lmdb_set = get_int_section_with_weight(config, mode)
    char_dict = get_ini_section(config, 'Dict')
    # val_lmdb = get_lmdb_from_section(config, 'Val')
    return [lmdb_set, char_dict]  # , val_lmdb#, pseudo_lmdb#, ext


def combine_dict(dst, src):
    if dst is None:
        dst = {}
    if src is not None:
        for k, v in src.items():
            if k in dst:
                if isinstance(v, dict) and isinstance(dst[k], dict):
                    dst[k] = combine_dict(dst[k], v)
                elif isinstance(v, list):
                    dst[k].append(v)
            else:
                dst[k] = v
    return dst


def combine_config_types(config, dst_key, src_key):
    cfg = config.get(dst_key, {})
    if cfg is None:
        cfg = copy.deepcopy(config[src_key])
    else:
        cfg = copy.deepcopy(cfg)
        cfg = combine_dict(cfg, config[src_key])
    return cfg


class ClsLabelOperator(object):
    def __init__(self, dict_path, is_coarse=False, **kwargs):
        self.config = self.load_config(dict_path)
        self.is_coarse = is_coarse
        self.dict = combine_config_types(self.config, 'Coarse' if is_coarse else 'Fine', 'Global')
        self.char_list = self.dict['chars']
        self.char_list.sort()
        self.character = dict(zip(self.char_list, list(range(len(self.char_list)))))

    def load_config(self, dict_yaml):
        with open(dict_yaml, 'r') as fopen:
            return yaml.load(fopen, Loader=yaml.SafeLoader)
        return None

    def encode(self, label):
        if label in self.dict['equals']:
            label = self.dict['equals'][label]
        if label not in self.character:
            label = self.dict['default']
        return self.character[label]

    def decode(self, id):
        if id < len(self.char_list):
            return self.char_list[id]
        else:
            return self.dict['default']

    def get_class_num(self):
        return len(self.character)
if __name__ == '__main__':
    img_arr = cv2.imread(r'D:\dataset\layer_board_det\2_on_shelf\qinghua_easy\qinghua_69_on_shelf_0_192.168.2.34_0.jpg')
    img_str = b64encode_img(img_arr)
    print(img_str[:20])

