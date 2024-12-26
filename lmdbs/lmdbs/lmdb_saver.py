# coding:utf-8
import os
from pathlib import Path

import lmdb
import cv2
import json
from multiprocessing import Lock
import numpy as np
import traceback
__package__ = 'lmdbs.lmdbs'

from lv_tools.cores.json_io import load_json_to_dict
from .utils import b64encode_img, b64decode_img
from .visualize import Visualize


class LmdbSaverBase(object):
    def __init__(self, config, num_disp=20):
        self.env = self.open_lmdb(config["lmdb_path"])
        # max_key = self.env.stat()["entries"]
        self.lock = Lock()
        self.cnt = self.init_cnt() if config["cnt"] is None else config["cnt"]
        self.cache = dict()
        self.cache_capacity = config["cache_capacity"]
        self.keys = self.init_keys('keys') if config["cnt"] is None else set()
        self.labels = self.init_keys('labels') if config["cnt"] is None else set()
        self.num_disp = num_disp
        self.path = config['lmdb_path']

    @staticmethod
    def open_lmdb(lmdb_path):
        base_folder = os.path.split(lmdb_path)[0]
        if not os.path.exists(base_folder):
            os.mkdir(base_folder)
        if os.path.exists(os.path.join(lmdb_path, "data.mdb")):
            print("Open lmdb {}".format(lmdb_path))
        else:
            print("Create lmdb {}".format(lmdb_path))
        return lmdb.open(lmdb_path, map_size=1 * 1024 * 1024)

    def add_map_size(self, adder_size=200 * 1024 * 1024):
        self.lock.acquire()
        lmdb_path = self.env.path()
        map_size = os.path.getsize(os.path.join(lmdb_path, "data.mdb"))
        map_size += adder_size
        self.env.set_mapsize(map_size)
        self.lock.release()

    def init_keys(self, key_name):
        txn = self.env.begin()
        keys = txn.get(key_name.encode("utf-8"))
        if keys is None:
            return set()
        else:
            keys = keys.decode('utf-8')
            return eval(keys)

    def init_cnt(self):
        txn = self.env.begin()
        num_samples = txn.get("num-samples".encode("utf-8"))
        if num_samples is None:
            return 0
        else:
            return int(num_samples.decode("utf-8"))

    def get_cnt(self):
        self.lock.acquire()
        cnt = self.cnt
        self.cnt += 1
        self.lock.release()
        return cnt

    def write_cache(self):
        # print("\rbegin {}".format(self.cnt), end="")
        self.update_global_info()
        self.lock.acquire()
        txn = self.env.begin(write=True)
        for k, v in self.cache.items():
            try:
                txn.put(k, v)
            except lmdb.MapFullError:
                txn.abort()
                self.lock.release()
                return False
            except:
                print('unkonw error {}'.format(traceback.extract_stack(limit=2)))
                self.lock.release()
                return True
        try:
            txn.commit()
        except lmdb.MapFullError:
            txn.abort()
            self.lock.release()
            return False
        except:
            print('unkonw error {}'.format(traceback.extract_stack(limit=2)))

        self.lock.release()
        print("\rcomplete {}".format(self.cnt), end="")
        return True

    def save_samples(self):
        raise NotImplementedError

    def add(self, data):
        raise NotImplementedError

    def close(self):
        print('begin to close {}...'.format(self.path))
        if self.env is None:
            return
        idx = 0
        while idx < 100:
            if self.write_cache():
                break
            else:
                self.add_map_size()
            idx += 1
        self.env.close()
        self.env = None
        print('{} have {} samples'.format(self.path, self.cnt))

    def update_global_info(self):
        raise NotImplementedError

    def __del__(self):
        self.close()


class LmdbSaver(LmdbSaverBase):
    def __init__(self, config):
        super(LmdbSaver, self).__init__(config)

    def update_global_info(self):
        self.cache["type".encode("utf-8")] = "txt".encode("utf-8")
        self.cache["num-samples".encode("utf-8")] = str(self.cnt).encode("utf-8")
        self.cache['keys'.encode('utf-8')] = str(self.keys).encode('utf-8')
        self.cache['labels'.encode('utf-8')] = str(self.labels).encode('utf-8')

    def end_a_sample(self):
        if len(self.cache) >= self.cache_capacity:
            while True:
                if self.write_cache():
                    break
                else:
                    self.add_map_size()
            self.cache = dict()
        if self.cnt == self.num_disp:
            self.save_samples()

    # def encode_img2(self, img):
    #     img_str = cv2.imencode('.jpg', img)[1].tobytes()  # 将图片编码成流数据，放到内存缓存中，然后转化成string格式
    #     b64_code = base64.b64encode(img_str).decode('utf-8')
    #     return b64_code
    #
    # def decode_img2(self, img_str):
    #     str_decode = base64.b64decode(img_str.encode('utf-8'))
    #     nparr = np.fromstring(str_decode, np.uint8)
    #     img_restore = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #     return img_restore

    def __getitem__(self, idx):
        data = {}
        try:
            for key in self.keys:
                data[key] = self.cache[key + "-{:09d}".format(idx).encode("utf-8")]
                if 'image' == key:
                    data[key] = cv2.imdecode(np.fromstring(data[key], dtype=np.uint8), cv2.IMREAD_COLOR)
                else:
                    data[key] = data[key].decode('utf-8')
        except:  # for json
            strobj = self.cache[('id' + "-{:09d}".format(idx)).encode("utf-8")]
            data = json.loads(strobj.decode('utf-8'))
            if 'image' in data:
                data['image'] = b64decode_img(data['image'])
        return data

    def add_json_format(self, data_dict):
        # it will not increase the count if it is invalid image
        if 'image' in data_dict and isinstance(data_dict['image'], np.ndarray):
            data_dict['image'] = b64encode_img(data_dict['image'])

        cnt = self.get_cnt()
        for key, value in data_dict.items():
            self.keys.add(key)
        self.cache["id-{:09d}".format(cnt).encode("utf-8")] = json.dumps(data_dict).encode('utf-8')
        # self.cache.update({"id-{:09d}".format(cnt).encode("utf-8"): json.dumps(data_dict).encode('utf-8')})

    def add_dict_data(self, data_dict):
        cnt = self.get_cnt()
        for key, value in data_dict.items():
            self.keys.add(key)
            if 'image' == key:
                value = cv2.imencode(".jpg", value)[1]
            else:
                if not isinstance(value, str):
                    value = str(value)
                value = value.encode('utf-8')
            self.cache[(key + '-{:09d}'.format(cnt)).encode('utf-8')] = value

    def add(self, data_dict, is_to_json=False):
        try:
            if is_to_json:
                self.add_json_format(data_dict)
            else:
                self.add_dict_data(data_dict)
            if 'label' in data_dict and isinstance(data_dict['label'], str) and len(self.labels) < 100:
                self.labels.add(data_dict['label'])
            self.end_a_sample()
        except:
            print('Failure to add sample {}!!!!!!!!!!!'.format(self.cnt))
            pass

    def save_samples(self):
        sample_path = os.path.join(self.env.path(), "samples")
        if not os.path.exists(sample_path):
            os.makedirs(sample_path)
        display_num = min(self.cnt, min(100, self.cache_capacity))
        vis = Visualize()
        for i in range(display_num):
            data = self.__getitem__(i)

            img_name = os.path.join(sample_path, "{}".format(i))
            if 'image' in data:
                vis.display_infor(data, file_name=img_name + '.jpg')
            else:
                with open(img_name + '.txt', 'w', encoding='utf-8') as f:
                    f.write(json.dumps(data))


# class LmdbSaverPool(LmdbSaver,Process):
#     stop_token = "kill"
#
#     def run(self):
#         num_image = self.generator_cfg.num_image
#         save_dir = self.generator_cfg.save_dir
#         log_period = max(1, int(self.log_period / 100 * num_image))
#         try:
#             with self.dataset_cls(str(save_dir)) as db:
#                 exist_count = db.read_count()
#                 count = 0
#
#                 while True:
#                     m = self.data_queue.get()
#                     if m == stop_token:
#                         logger.info("DBWriterProcess receive stop token")
#                         break
#
#                     name = "{:09d}".format(exist_count + count)
#                     db.write('id-' + name, m["image"], m["label"], m["bbox"])
#                     count += 1
#                     if count % log_period == 0:
#                         logger.info(
#                             f"{(count / num_image) * 100:.2f}%({count}/{num_image}) {log_period / (time.time() - start + 1e-8):.1f} img/s"
#                         )
#                         start = time.time()
#                 db.write_count(count + exist_count)
#                 logger.info(f"{(count / num_image) * 100:.2f}%({count}/{num_image})")
#                 logger.info(f"Finish generate: {count}. Total: {exist_count + count}")
#         except Exception as e:
#             logger.exception("DBWriterProcess error")
#             raise e


if __name__ == '__main__':


    count = 0
    json_root = r'F:\D\dataset\OCR\need_multi_core_rec\20240208_media_hushi_cuted\rec_piece\20240208_on_shelf_hushi'
    dst_path = json_root + '_test_lmdb'
    lmdb_saver = LmdbSaver({'lmdb_path': dst_path, 'cnt': 0, 'cache_capacity': 5})
    for img_path in list(Path(json_root).glob('**/*.jpg'))[:500]:

        json_path = img_path.with_suffix('.json')
        jd = load_json_to_dict(str(json_path))
        img_arr = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), 1)
        img_base = f'id-{count:09}'  # img_path.name
        print(img_base)
        label = jd['shapes'][0]['label']
        points = jd['shapes'][0]['points']
        points = np.array(points, np.int32).tolist()
        scores = jd['shapes'][0]['scores']
        img_byte = b64encode_img(img_arr)
        h, w = img_arr.shape[:2]
        bbox_tuple = (f'{img_base}', [
            {
                "transcription": f'{label}',
                "illegibility": False,
                "points": points,
                "scores":scores
            }
        ])
        imgp, trans = bbox_tuple
        trans[0]['image'] = img_arr
        count += 1
        lmdb_saver.add({img_base:bbox_tuple[1][0]}, is_to_json=False)
