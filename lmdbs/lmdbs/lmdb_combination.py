import os
import time
import cv2
import lmdb
import argparse
import sys
sys.path.append(r'D:\lxd_code\text_renderer_lv\lmdbs\lmdbs')
from lmdb_loader import LMDBLoader
from lmdb_saver import LmdbSaver
from os import path as osp

def parse_args() -> object:
    parser = argparse.ArgumentParser("generic-image-rec train script")
    parser.add_argument(
        '--lmdb_dir',
        type=str,
        default=r'D:\dataset\OCR\lmdb_0506',
        help='path of source sub-lmdb directory')

    args = parser.parse_args()
    return args


def save_hierarchical_lmdb_dataset(lmdb_saver, data_dir,filter_invalid_bar=True,tem_calibration=False):

    # count = 0
    for dirpath, dirnames, filenames in os.walk(data_dir + '/'):
        try:
            lmdb_loader = LMDBLoader(dirpath, rand=False)
            print('deal with {}'.format(dirpath))
            for i in range(lmdb_loader.num_samples):
                if (0 == i % 1000):
                    print('dealed samples {}'.format(i))
                data = lmdb_loader.__getitem__(i)

                if filter_invalid_bar:
                    image = data['image']
                    h, w = image.shape[:2]
                    if w/h<1.5:
                        cv2.imwrite(rf'D:\dataset\bar_code\b_show\bar_ex/{i}.jpg',image)
                        continue
                    test_area = image[int(h * .2):int(h * .8), int(w * .2):int(w * .8), 0]
                    if test_area.max() - test_area.min() < 20:
                        continue
                # if tem_calibration:
                #     image = data['image']
                #     h, w = image.shape[:2]
                #     pad_unit = h // 3
                #     box = [[5 * pad_unit, pad_unit], [w - 5 * pad_unit, pad_unit], [w - 5 * pad_unit, 2 * pad_unit],
                #            [5 * pad_unit, 2 * pad_unit]]
                #     data['points'] = box

                lmdb_saver.add(data, is_to_json=True)
                # count += 1
                # if count>10000:
                #     return
        except:
            continue

'''
h,w = image.shape[:2]
pad_unit = h//3
box = [[5*pad_unit,pad_unit],[w-5*pad_unit,pad_unit],[w-5*pad_unit,2*pad_unit],[5*pad_unit,2*pad_unit]]

'''



if __name__ == '__main__':
    args = parse_args()
    local_time = time.localtime()
    mon,day,hour = local_time.tm_mon,local_time.tm_mday,local_time.tm_hour
    dst_path = osp.join(osp.dirname(args.lmdb_dir),f'synthesis_bookname_{mon:02}_{day:02}_{hour:02}')
    lmdb_saver = LmdbSaver({'lmdb_path': dst_path, 'cnt': 0, 'cache_capacity': 500})
    save_hierarchical_lmdb_dataset(lmdb_saver, args.lmdb_dir,filter_invalid_bar=False)
    lmdb_saver.close()
