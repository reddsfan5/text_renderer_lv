import cProfile
import json
import multiprocessing
import os
import pstats
import statistics
import traceback
from abc import abstractmethod
from multiprocessing import Lock, Queue
from pathlib import Path
from typing import Dict, Tuple, Literal, List

import cv2
import lmdb
import numpy as np
from tqdm import tqdm

from lv_tools.cores.json_io import load_json_to_dict
from lmdbs.lmdbs.utils import b64encode_img


class Dataset:
    def __init__(self, data_dir: str, jpg_quality: int = 95):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        self.jpg_quality = jpg_quality

    def encode_param(self):
        return [int(cv2.IMWRITE_JPEG_QUALITY), self.jpg_quality]

    @abstractmethod
    def write(self, name: str, image: np.ndarray, label: str):
        pass

    @abstractmethod
    def read(self, name) -> Dict:
        """

        Parameters
        ----------
            name : str
                000000001

        Returns
        -------
            dict :

                .. code-block:: bash

                    {
                        "image": ndarray,
                        "label": "label",
                        "size": [int_width, int_height]
                    }
        """
        pass

    @abstractmethod
    def read_count(self) -> int:
        pass

    @abstractmethod
    def write_count(self, count: int):
        pass

    @abstractmethod
    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class ImgDataset(Dataset):
    """
    Save generated image as jpg file, save label and meta in json
    json file format:

    .. code-block:: bash

        {
             "labels": {
                "000000000": "test",
                "000000001": "text2"
             },
             "sizes": {
                "000000000": [width, height],
                "000000001": [width, height],
             }
             "num-samples": 2,
        }
    """

    LABEL_NAME = "labels.json"

    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self._img_dir = os.path.join(data_dir, "images")
        if not os.path.exists(self._img_dir):
            os.makedirs(self._img_dir)
        self._label_path = os.path.join(data_dir, self.LABEL_NAME)

        self._data = {"num-samples": 0, "labels": {}, "sizes": {}, "bboxes": {}}
        if os.path.exists(self._label_path):
            with open(self._label_path, "r", encoding="utf-8") as f:
                self._data = json.load(f)

    def write(self, name: str, image: np.ndarray, label: str, bbox=None):
        img_path = os.path.join(self._img_dir, name + ".jpg")
        cv2.imwrite(img_path, image, self.encode_param())
        self._data["labels"][name] = label

        height, width = image.shape[:2]
        self._data["sizes"][name] = (width, height)
        # todo lvixaodong 增加坐标
        if bbox:
            if 'bboxes' not in self._data.keys():
                self._data['bboxes'] = {}
            self._data['bboxes'][name] = bbox

    def read(self, name: str) -> Dict:
        img_path = os.path.join(self._img_dir, name + ".jpg")
        image = cv2.imread(img_path)
        label = self._data["labels"][name]
        size = self._data["sizes"][name]
        return {"image": image, "label": label, "size": size}

    def read_size(self, name: str) -> [int, int]:
        return self._data["sizes"][name]

    def read_count(self) -> int:
        return self._data.get("num-samples", 0)

    def write_count(self, count: int):
        self._data["num-samples"] = count

    def close(self):
        with open(self._label_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)


class LmdbDataset(Dataset):
    """
    Save generated image into lmdb. Compatible with https://github.com/PaddlePaddle/PaddleOCR
    Keys in lmdb:

        - image-000000001: image raw bytes
        - label-000000001: string
        - size-000000001: "width,height"

    """

    def __init__(self, data_dir: str, map_size=20 * 1024 * 1024 * 1024):  # 默认初始大小
        super().__init__(data_dir)
        self._lmdb_env = lmdb.open(self.data_dir, map_size=map_size)
        self._lmdb_txn = self._lmdb_env.begin(write=True)
        self.increment = 500 * 1024 * 1024  # 增加大小
        self.lock = Lock()
        self.count = 0
        self.cache_num = 500
        self.cache = []
        self.total = 0

    def write(self, name: str, image: np.ndarray, label: str, bbox=None, font_base=None, **kwargs):
        img_base = name
        img_arr = image
        img_byte = b64encode_img(img_arr)
        h, w = img_arr.shape[:2]
        points = [[0, 0], [w, 0], [w, h], [0, h]] if not bbox else bbox
        bbox_tuple = (f'{img_base}', [
            {
                "transcription": f'{label}',
                "illegibility": False,
                "points": points,
                'font': font_base
            }
        ])
        imgp, trans = bbox_tuple
        trans[0]['image'] = img_byte
        if kwargs:
            for k, v in kwargs.items():
                trans[0][k] = v

        # self._lmdb_txn = self._lmdb_env.begin(write=True)
        self.cache.append((img_base.encode(), json.dumps(trans[0]).encode('utf8')))
        if self.count > self.cache_num:

            try:
                for item in self.cache:
                    self._lmdb_txn.put(*item)
            except lmdb.MapFullError:
                # 扩大 map_size
                self._lmdb_txn.abort()
                lmdb_path = self.data_dir
                map_size = os.path.getsize(os.path.join(lmdb_path, "data.mdb"))
                map_size += self.increment
                self._lmdb_env.set_mapsize(map_size)
                # self.lock.release()
                print(f"Increased map_size to: {self.increment}")

                # 重新开始事务并重试
                self._lmdb_txn = self._lmdb_env.begin(write=True)
                # self._lmdb_txn.put(img_base.encode(), json.dumps(trans[0]).encode('utf8'))
            else:
                self._lmdb_txn.commit()
                self._lmdb_txn = self._lmdb_env.begin(write=True)
                self.cache = []
                self.count = 0

        self.count += 1
        self.total += 1

    def read(self, name: str) -> Dict:
        label = self._lmdb_txn.get(self.label_key(name)).decode()
        size_str = self._lmdb_txn.get(self.size_key(name)).decode()
        size = [int(it) for it in size_str.split(",")]

        image_bytes = self._lmdb_txn.get(self.image_key(name))
        image_buf = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_buf, cv2.IMREAD_UNCHANGED)

        return {"image": image, "label": label, "size": size}

    def read_size(self, name: str) -> [int, int]:
        """

        Args:
            name:

        Returns: (width, height)

        """
        size_key = f"size_{name}"

        size = self._lmdb_txn.get(size_key.encode()).decode()
        width = int(size.split[","][0])
        height = int(size.split[","][1])

        return width, height

    def read_count(self) -> int:
        count = self._lmdb_txn.get("num-samples".encode())
        if count is None:
            return 0
        return int(count)

    def write_count(self, count: int):
        self._lmdb_txn.put("num-samples".encode(), str(count).encode())

    def image_key(self, name: str):
        return f"image-{name}".encode()

    def label_key(self, name: str):
        return f"label-{name}".encode()

    def size_key(self, name: str):
        return f"size-{name}".encode()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        while True:
            try:
                for item in self.cache:
                    self._lmdb_txn.put(*item)
            except lmdb.MapFullError:
                # 提交当前事务
                # self._lmdb_txn.commit()
                # self.lock.acquire()
                # 扩大 map_size
                self._lmdb_txn.abort()
                lmdb_path = self.data_dir
                map_size = os.path.getsize(os.path.join(lmdb_path, "data.mdb"))
                map_size += 2 * 1024 * 1024
                self._lmdb_env.set_mapsize(map_size)
                # self.lock.release()
                print(f"Increased map_size to: {self.increment}")

                # 重新开始事务并重试
                self._lmdb_txn = self._lmdb_env.begin(write=True)
                # self._lmdb_txn.put(img_base.encode(), json.dumps(trans[0]).encode('utf8'))
            else:
                # self._lmdb_txn.put("num-samples".encode(), str(self.total).encode())
                self._lmdb_txn.commit()
                self._lmdb_txn = self._lmdb_env.begin(write=True)
                break

        self._lmdb_txn.__exit__(exc_type, exc_value, traceback)
        self._lmdb_env.close()

    def close(self):
        if self._lmdb_txn:
            self._lmdb_txn.commit()
        self._lmdb_env.close()


def contain_any(label: str, filter_strs: Tuple[str, ...]) -> bool:
    return any(filter_str in label for filter_str in filter_strs)


def filter_file_by_criteria(min_letter_num: int, label: str,
                            filter_strs: Tuple[str, ...] = ()) -> bool:
    # return (not filter_strs or not any(filter_str in label and min_letter_num<=len(filter_str) for filter_str in filter_strs))
    return (not filter_strs or min_letter_num <= len(label) and not any(
        filter_str in label for filter_str in filter_strs))


def filter_by_language(language: Literal['ch', 'en'], label):
    return language == 'ch' and filter_file_by_criteria(min_letter_num=1, label=label, filter_strs=(
        '出版', '版社')) or language == 'en' and filter_file_by_criteria(min_letter_num=4, label=label)


def gen_ocr_rec_lmdb_from_pieces(language: Literal['ch', 'en'],
                                 json_root: str):
    # json_root = rf'F:\D\dataset\OCR\need_multi_core_rec\heilongjiang0130\splited_img\piece\{language}'

    lmdb_dir = json_root + f'_lmdb'
    if not os.path.exists(lmdb_dir):
        os.mkdir(lmdb_dir)
    with LmdbDataset(lmdb_dir, map_size=100 * 1024 * 1024) as writer:
        count = 0
        total = 0
        for _ in Path(json_root).glob('**/*.jpg'):
            total += 1

        for img_path in tqdm(Path(json_root).glob('**/*.jpg'), total=total, desc='lmdb生成中：'):

            json_path = img_path.with_suffix('.json')
            if not os.path.exists(str(json_path)):
                continue
            jd = load_json_to_dict(str(json_path))
            try:
                if not jd.get('shapes', None):
                    continue
                label = jd['shapes'][0]['label']
                points = jd['shapes'][0]['points']
                if filter_by_language(language, label):
                    count += 1
                    # s = time.time()
                    img_base = f'id-{count:09}'  # img_path.name
                    if count % 1000 == 0:
                        print(img_base)
                    img_arr = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), 1)
                    points = np.array(points, np.int32).tolist()
                    # print(f'预处理时间：{time.time() - s}')

                    # 逐个写入影响效率，使用了 缓存 + 批量写入 的逻辑 ,预处理时间有点长，写到进程池，加速
                    # s2 = time.time()
                    if jd['shapes'][0].get('scores'):
                        writer.write(img_base, img_arr, label, bbox=points, scores=jd['shapes'][0]['scores'])
                    else:
                        writer.write(img_base, img_arr, label, bbox=points)
         # print(f'写入时间：{time.time() - s2}')
            except:
                print('出错文件：', json_path)
                traceback.print_exc()
            # writer.write(img_base, img_arr, label, bbox=points)
        writer.write_count(count)
        print(writer.read_count())


def filter_lmdb(lmdb_path: str, language: Literal['en', 'ch'] = 'ch'):
    map_size = os.path.getsize(os.path.join(lmdb_path, "data.mdb"))
    with lmdb.open(lmdb_path, map_size=map_size) as env:
        with env.begin(write=True) as txn:
            keys_to_delete = []
            stats = txn.stat()
            # 获取总条目数
            total_entries = stats['entries']
            count = 0
            for key, value in tqdm(txn.cursor(), total=total_entries):
                if count > 5000000:
                    break
                try:
                    count += 1
                    jd = json.loads(value.decode())
                    print(key.decode())
                    label = jd['transcription']
                    if not filter_by_language(language, label):
                        keys_to_delete.append(key)
                except Exception as e:
                    print(e)
                    continue
            # for key in keys_to_delete:
            #     txn.delete(key)


def lmdb_key_normalize(lmdb_path: str, language: Literal['ch', 'en'], cache: int = 5000,
                       increment: int = 500 * 1024 ** 2, mean_score_threshold=.85):
    '''
    原地修改风险大，不如写入到另一个lmdb文件。

    Parameters
    ----------
    lmdb_path

    Returns
    -------

    '''
    dst_lmdb_path = lmdb_path + '_normalized'
    map_size = os.path.getsize(os.path.join(lmdb_path, "data.mdb"))
    item_to_add = {}
    # with lmdb.open(lmdb_path, map_size=map_size) as env_read,lmdb.open(dst_lmdb_path, map_size=map_size+200*1024**2) as env_write:
    with lmdb.open(lmdb_path, map_size=map_size) as env_read, lmdb.open(dst_lmdb_path,
                                                                        map_size=1024 ** 3) as env_write:
        with env_read.begin() as txn:
            txn2 = env_write.begin(write=True)
            count = 1
            add_flag = 0
            for key, value in tqdm(txn.cursor()):
                if key.decode().startswith('id-'):
                    value_dict = json.loads(value.decode())
                    label = value_dict.get('transcription').replace(' ', '')
                    scores = [v for v in value_dict.get('scores').values() if isinstance(v, float) and v > 0]
                    mean_score = statistics.mean(scores)
                    if not filter_by_language(language, label) or mean_score < mean_score_threshold:
                        continue

                    new_key = ('id-' + str(count).zfill(9)).encode()
                    count += 1
                    add_flag += 1
                    item_to_add[new_key] = value
                    try:
                        if add_flag >= cache:
                            for k2, v2 in item_to_add.items():
                                txn2.put(k2, v2)
                            txn2.commit()
                            txn2 = env_write.begin(write=True)
                            add_flag = 0
                            item_to_add = {}
                    except lmdb.MapFullError:
                        # 扩大 map_size
                        txn2.abort()
                        map_size = os.path.getsize(os.path.join(dst_lmdb_path, "data.mdb"))
                        map_size += increment
                        env_write.set_mapsize(map_size)

                        # 重新开始事务并重试
                        txn2 = env_write.begin(write=True)

            txn2.put('num-samples'.encode(), str(count).encode())
            txn2.commit()  # 确保最后的更改被提交


'''
多进程任务明确：
1. 生产者不断地往池子里储备要写入lmdb的数据
2. 消费者从池子里往外拿，然后写入lmdb

'''


def rec_data_producer(queue: Queue, language: Literal['ch', 'en'], img_path: Path):
    json_path = img_path.with_suffix('.json')
    jd = load_json_to_dict(str(json_path))
    try:
        if not jd.get('shapes', None):
            return
        label = jd['shapes'][0]['label']
        points = jd['shapes'][0]['points']
        if filter_by_language(language, label):
            # s = time.time()
            img_arr = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), 1)
            points = np.array(points, np.int32).tolist()
            # print(f'预处理时间：{time.time() - s}')
            # 逐个写入影响效率，使用了 缓存 + 批量写入 的逻辑 ,预处理时间有点长，写到进程池，加速
            # s2 = time.time()

            queue.put((img_arr, label, {'bbox': points}, {'scores': jd['shapes'][0]['scores']}))
            # print(f'写入时间：{time.time() - s2}')
    except:
        print('出错文件：', json_path)
        traceback.print_exc()


def multi_producer(queue: Queue, lock: Lock, language: Literal['ch', 'en'], img_paths: List[Path]):
    while True:
        with lock:
            if img_paths:
                img_path = img_paths.pop()
            else:
                break
        json_path = img_path.with_suffix('.json')
        jd = load_json_to_dict(str(json_path))
        try:
            if not jd.get('shapes', None):
                return
            label = jd['shapes'][0]['label']
            points = jd['shapes'][0]['points']
            # if filter_by_language(language, label):
            if True:
                # s = time.time()
                img_arr = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), 1)
                points = np.array(points, np.int32).tolist()
                # print(f'预处理时间：{time.time() - s}')
                # 逐个写入影响效率，使用了 缓存 + 批量写入 的逻辑 ,预处理时间有点长，写到进程池，加速
                # s2 = time.time()

                scores = jd['shapes'][0].get('scores',1)

                queue.put({'image': img_arr, 'label': label, 'bbox': points, 'scores': scores,'file':img_path.name})
                # print(f'写入时间：{time.time() - s2}')
        except:
            print('出错文件：', json_path)
            traceback.print_exc()


def rec_data_consumer(queue: Queue, lmdb_dir):
    count = 0
    with LmdbDataset(lmdb_dir, map_size=200 * 1024 * 1024) as writer:
        pbar = tqdm()
        while True:
            data = queue.get()
            if data is None:
                break
            pbar.update(1)
            img_base = f'id-{count:09}'  # img_path.name
            if count % 1000 == 0:
                print(img_base)

            count += 1
            writer.write(img_base, **data)

        writer.write_count(count)
        print(writer.read_count())
        pbar.close()


def main_async(language: Literal['ch', 'en'], json_root: str, max_workers: int = 8):
    lmdb_dir = json_root + '_lmdb'
    with multiprocessing.Manager() as manager:
        img_paths = manager.list(list(Path(json_root).glob('**/*.jpg')))
        lock = manager.Lock()

        queue = Queue()
        prods = []
        for _ in range(max_workers):
            p = multiprocessing.Process(target=multi_producer,
                                        kwargs={'queue': queue, 'lock': lock, 'language': language,
                                                'img_paths': img_paths})
            prods.append(p)
            p.start()
        cons = multiprocessing.Process(target=rec_data_consumer, args=(queue, lmdb_dir))

        cons.start()

        for prod in prods:
            prod.join()

        queue.put(None)

        cons.join()


if __name__ == "__main__":
    root = r'F:\dataset\OCR\3.text_rec\0_hard_data_increament\2_hard_data_rotated_increment'

    lmdb_dir = r'F:\D\dataset\OCR\need_multi_core_rec\hard_data_increament\hard_data_calibrated_liu_li_lv_rotated_increment_lmdb'



    cProfile.run("gen_ocr_rec_lmdb_from_pieces('ch', root)",sort='cumtime',filename='time_analysis.prof')
    # main_async('ch',root)
    # main_async('ch',root)

    # p = pstats.Stats('main_async_time_analysis.prof')
    # p.sort_stats('time').print_stats()

    # filter_lmdb(lmdb_path=lmdb_dir,language='ch')
    # lmdb_key_normalize(lmdb_dir, 'ch')
    # with LmdbDataset(r"D:\dataset\OCR\1\ori_img\lmdb_data") as ld:
    #     data = ld.read("469d91a68c1c722b373d69568a5e08145.jpg")
    #     print(data)
    #     cv2.imwrite("test.jpg", data["image"])

    # with lmdb.open(lmdb_dir,readonly=True) as env:
    #     txn = env.begin()
    #     for item in txn.cursor():
    #         print(item)

    # json_root = r'F:\D\dataset\OCR\need_multi_core_rec\sources_hulu_book_55k\rec_piece\rotated\80_rotated_more_letter\4_letter_ch'
    # for img_path in list(Path(json_root).glob('**/*.jpg')):
    #     json_path = img_path.with_suffix('.json')
    #     jd = load_json_to_dict(str(json_path))
    #     label = jd['shapes'][0]['label']
    #     points = jd['shapes'][0]['points']
    #     print(label)
    #     print(filter_file_by_criteria(json_path,min_letter_num=2,language='ch',label=label, filter_strs=('出版', '版社')))
    # print(filter_file_by_criteria(json_path, min_letter_num=4, language='en', label=label))
