import os
import json
from pathlib import Path
from typing import Dict
from abc import abstractmethod
from costum_utils.parse_json import getJsonDict
from lmdbs.lmdbs.utils import b64encode_img
import lmdb
import cv2
import numpy as np


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


class LmdbDataset(Dataset):
    """
    Save generated image into lmdb. Compatible with https://github.com/PaddlePaddle/PaddleOCR
    Keys in lmdb:

        - image-000000001: image raw bytes
        - label-000000001: string
        - size-000000001: "width,height"

    """

    def __init__(self, data_dir: str, map_size=1 * 1024 * 1024 * 1024):  # 默认初始大小
        super().__init__(data_dir)
        self._lmdb_env = lmdb.open(self.data_dir, map_size=map_size)
        self._lmdb_txn = self._lmdb_env.begin(write=True)
        self.increment = 300 * 1024 * 1024  # 增加大小



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

        try:
            self._lmdb_txn.put(img_base.encode(), json.dumps(trans[0]).encode('utf8'))
        except lmdb.MapFullError:
            # 提交当前事务
            # self._lmdb_txn.commit()
            # 扩大 map_size

            lmdb_path = self.data_dir
            map_size = os.path.getsize(os.path.join(lmdb_path, "data.mdb"))
            map_size += 200 * 1024 * 1024
            self._lmdb_env.set_mapsize(map_size)

            print(f"Increased map_size to: {self.increment}")

            # 重新开始事务并重试
            with self._lmdb_env.begin(write=True) as txn:

                txn.put(img_base.encode(), json.dumps(trans[0]).encode('utf8'))


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
        self._lmdb_txn.__exit__(exc_type, exc_value, traceback)
        self._lmdb_env.close()
    def close(self):
        if self._lmdb_txn:
            self._lmdb_txn.commit()
        self._lmdb_env.close()

if __name__ == "__main__":
    # image = cv2.imread("f_004.jpg")
    # label = "test"
    json_root = r'F:\D\dataset\OCR\need_multi_core_rec\20240208_media_hushi_cuted\rec_piece\20240208_on_shelf_hushi'
    lmdb_dir = json_root + '_lmdb222'
    if not os.path.exists(lmdb_dir):
        os.mkdir(lmdb_dir)
    with LmdbDataset(lmdb_dir,map_size=100*1024) as writer:
        count = 0
        for img_path in Path(json_root).glob('**/*.jpg'):
            count += 1
            json_path = img_path.with_suffix('.json')
            jd = getJsonDict(str(json_path))
            img_arr = cv2.imdecode(np.fromfile(str(img_path),dtype=np.uint8),1)
            img_base = f'id-{count:09}' #img_path.name
            print(img_base)
            label = jd['shapes'][0]['label']
            points = jd['shapes'][0]['points']
            points = np.array(points,np.int32).tolist()
            writer.write(img_base, img_arr, label,bbox=points,scores=jd['shapes'][0]['scores'])
            # writer.write(img_base, img_arr, label, bbox=points)
        writer.write_count(count)
        print(writer.read_count())