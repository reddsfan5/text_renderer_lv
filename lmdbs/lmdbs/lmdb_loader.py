# coding=utf-8
import numpy as np
import os
import cv2
import lmdb
import base64
import io
from PIL import Image
import json
import random
import time
import sys
__package__ = 'lmdbs.lmdbs'     # 疑问：不显式的指明该模块位置，为什么不能直接运行lmdb_combination.
# https://stackoverflow.com/questions/21233229/whats-the-purpose-of-the-package-attribute-in-python
sys.path.append(r'D:\lxd_code\text_renderer_lv\lmdbs\lmdbs')
from .utils import b64decode_img, ClsLabelOperator


class LMDBLoader(object):
    def __init__(self, lmdb_path, ratio=1.0, seed=None, rand=True, **kargs):
        super(LMDBLoader, self).__init__()
        self.env = lmdb.open(
            lmdb_path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)
        self.txn = self.env.begin(write=False)
        self.num_samples = int(self.txn.get('num-samples'.encode()).decode('utf-8'))
        self.ratio = ratio
        self.seed = seed
        self.idx = 0
        self.rand = rand
        self.decoder = None

    def open_lmdb(self, lmdb_path):
        if not os.path.exists(lmdb_path):
            raise FileExistsError(lmdb_path)
        if not os.path.exists(os.path.join(lmdb_path, "data.mdb")):
            raise FileExistsError(os.path.join(lmdb_path, "data.mdb"))
        env = lmdb.open(lmdb_path, max_readers=128, readonly=True, lock=False, readahead=False, meminit=False)
        return env

    def get_num_samples(self):
        num_samples = self.txn.get("num-samples".encode("utf-8"))
        if num_samples is None:
            raise KeyError("num-samples")
        num_samples = int(num_samples.decode("utf-8"))
        return num_samples


    def get_weighted_num(self):
        return self.weight * self.num_samples

    def __len__(self):
        return self.num_samples

    def trim_random(self, data):
        '''
        trim the data edge randomly if training, other trim the extended edge only
        :param data: source input data
        :return: trimmed data
        '''
        if data and 'rect' in data and not isinstance(data['image'], str):
            h, w = data['image'].shape[:2]
            l, t, r, b = data['rect']
            l, t = int(max(0, l)), int(max(0, t))
            r, b = int(min(r, w)), int(min(b, h))
            rc = [l, t, r, b]
            if random.random() < self.rand_ext:  # crop random for training samples
                sr = 2  # 1 + 0.5  # shrink ratio
                l1, t1 = sr * l, sr * t
                r1, b1 = w - sr*(w-r), h - sr*(h-b)
                if l1 < r1:
                    l, r = l1, r1
                if t1 < b1:
                    t, b = t1, b1
                l, t, r, b = int(l), int(t), int(r), int(b)
                l, t = random.randint(0, l), random.randint(0, t)
                r, b = random.randint(r, w), random.randint(b, h)

            data['image'] = data['image'][t:b, l: r, :]
            #h, w = data['image'].shape[:2]
            data['rect'] = [max(0, rc[0]-l), max(0, rc[1]-t), rc[2]-l, rc[3]-t]  # update the rect information
        return data


    def get_lmdb_sample_info(self, index):
        label_key = 'label-%09d'.encode() % index
        label = self.txn.get(label_key)
        if label is None:
            return None
        label = label.decode('utf-8')
        img_key = 'image-%09d'.encode() % index
        imgbuf = self.txn.get(img_key)
        return {'image': imgbuf, 'label': label}


    def b64decode_img(self, image_str):
        # imagebin = base64.decodebytes(image_str.encode("utf-8"))
        imagebin = base64.b64decode(image_str.encode('utf-8'))
        f = io.BytesIO()
        f.write(imagebin)
        image = Image.open(f)
        image = np.asarray(image, np.uint8)[:, :, (2, 1, 0)]
        return image

    def get_base64_sample_info(self, index):
        strobj = self.txn.get(('id' + "-{:09d}".format(index)).encode("utf-8"))
        if strobj is None:
            return None
        data = json.loads(strobj.decode('utf-8'))
        if 'image' not in data or data['image'] is None:
            return None
        data['image'] = self.b64decode_img(data['image'])  # self.decode_img(data['image'])
        return data

    def try_key_and_value(self):
        for i, [key, value] in enumerate(self.txn.cursor()):  # 遍历
            print(key)
            print(value)
            if i > 10:
                break
        temp = 0

    def __getitem__(self, item):
        # set the index of current sample
        idx = self.idx
        if self.rand:
            idx = random.randint(0, self.num_samples)
            if 0 == random.randint(0, 20000):
                random.seed(int(time.time()))
        # self.try_key_and_value()
        if self.decoder:
            data = self.decoder(idx)
        else:
            decoders = [self.get_base64_sample_info, self.get_lmdb_sample_info]
            for decoder in decoders:
                data = decoder(idx)
                if data is not None:
                    self.decoder = decoder
                    break

        # update the index information
        self.idx += 1
        if self.idx >= self.num_samples:
            self.idx = 0
        return data


class LMDBSLoaders(object):
    def __init__(self, data_keys=None, is_training=False, text_only=False, soft_label=False, is_coarse=False,
                 negative_thresh=0.3, rand_ext=0.0, **kargs):
        '''
        :param data_keys: information of lmdbs, such as
        :param is_training: is training set or not
        :param text_only: only get the text information or not
        :param soft_label: soft label or not
        :param is_coarse: coarse classification or not. If true, there are only two classes, negative and positive
        :param negative_thresh: it is positive sample if the prob is bigger than the thresh
        :param rand_ext: extend the edge randomly
        :param kargs:
        '''
        super(LMDBSLoaders, self).__init__()
        train_lmdbs, char_dict = data_keys
        self.lmdblist = list()
        self.train_sample_probs = list()
        self.rand_ext = rand_ext
        self.is_training = is_training
        self.text_only = text_only
        self.soft_label = soft_label
        self.negative_thresh = negative_thresh
        self.set_bgn_idx = [0]
        self.char_dict = ClsLabelOperator(char_dict['dict'], is_coarse=is_coarse)
        self.label_set = self.char_dict.char_list
        self.open_type_lmdbs(train_lmdbs)
        self.db_idx = list(range(len(self.lmdblist)))
        self.db_idx.reverse()


    def open_type_lmdbs(self, train_lmdbs):
        weighted_set_num = list()
        for i, db in enumerate(train_lmdbs):
            db_ptr = LMDBLoader(db, is_training=self.is_training, text_only=self.text_only, rand_ext=self.rand_ext)
            self.lmdblist.append(db_ptr)
            weighted_set_num.append(db_ptr.get_weighted_num())
            self.set_bgn_idx.append(self.set_bgn_idx[-1] + db_ptr.num_samples)

        weighted_set_num = np.array(weighted_set_num)
        self.train_sample_probs += list(weighted_set_num / max(1, weighted_set_num.sum()))

    def __len__(self):
        sum = 0
        for db in self.lmdblist:
            sum += len(db)
        return sum

    def get_data_label(self, data):
        return data['label'] if 'label' in data else data['name']

    def convert_label_to_id(self, data_dict, raw_label=False):
        if data_dict is None:
            return data_dict
        if self.soft_label:  # generate the soft label
            prob = np.zeros([len(self.char_dict.character)], dtype=np.float32)
            id_other = self.char_dict.encode('other')
            id_label = self.char_dict.encode(self.get_data_label(data_dict))
            if 0.0 == data_dict['prob']:
                data_dict['prob'] = 1.0
            prob[id_label] = data_dict['prob']
            if id_label != id_other:
                prob[id_other] = 1 - data_dict['prob']
            data_dict['soft_label'] = prob
        if 'prob' in data_dict and data_dict['prob'] < self.negative_thresh:
            data_dict['label'] = 'other'
            data_dict['prob'] = 1 - data_dict['prob']
        if not raw_label:
            data_dict['id'] = self.char_dict.encode(self.get_data_label(data_dict))
        return data_dict

    def __getitem__(self, item, rand=None, text_only=False, raw_label=False, need_trim=True):
        '''
        get one sample from the dataset
        :param item: index of the sample
        :param rand:  get the sample random or not
        :param text_only: get the sample's text information only
        :param raw_label: convert the label to id or not
        :param need_trim: trim the sample or not
        :return:
        '''
        rand = self.is_training if rand is None else rand
        if rand:
            if 0 == random.randint(0, 20000):
                random.seed(int(time.time()))
            idx = random.choices(list(range(len(self.lmdblist))), weights=self.train_sample_probs)[0]
            data_dict = self.lmdblist[idx].get_rand_item(-1, text_only=text_only, need_trim=need_trim)
        else:
            for idx in self.db_idx:
                if item >= self.set_bgn_idx[idx]:
                    item -= self.set_bgn_idx[idx]
                    break
            db = self.lmdblist[idx]
            data_dict = db.__getitem__(item, text_only=text_only, need_trim=need_trim)
        return self.convert_label_to_id(data_dict, raw_label)

