import json
from os import path as osp
import cv2
from typing import List

from parse_json import getJsonDict
def lmdb_ocr_dict(img_path:str,label:str,points:List[List]=None):
    '''
    适用于一图一框
    '''
    img_base = osp.basename(img_path)
    img_arr = cv2.imread(img_path)
    h,w = img_arr.shape[:2]
    points = [[0,0],[w,0],[w,h],[0,h]] if not points else points
    bbox_tuple = (f'{img_base}', [
        {
            "transcription": f'{label}',
            "illegibility": 0,
            "points": points,
            "file": f"{img_path}"
        }
    ])

    return bbox_tuple

def lmdb_json_lst(json_path,tar_json_path,tar_lst_path):
    jd = getJsonDict(json_path)
    tar_dict = {}
    with open(tar_lst_path,mode='w',encoding='utf8') as lst:
        for index,item in enumerate(jd['labels']):
            img_path = f'{osp.join(osp.dirname(json_path),"images",item+".jpg")}'
            transcription = jd['labels'][item]
            imgp,trans = lmdb_ocr_dict(img_path,transcription,jd['bboxes'][item])
            tar_dict[imgp] = trans
            lst.write(img_path)
            if index<=len(jd['labels']):
                lst.write('\n')
    with open(tar_json_path,mode='w',encoding='utf8') as f:
        data = json.dumps(tar_dict)
        f.write(data)


if __name__ == '__main__':
    img_path = r"D:\dataset\OCR\MTWI_lmdb_test\image_train\T1._WBXtXdXXXXXXXX_!!0-item_pic.jpg.jpg"
    json_path = r'E:\lxd\text_renderer_lv\example_data\effect_layout_image\effect_ensemble/labels.json'
    label = r'哈佛成长课'
    tar_json_path = r'E:\lxd\text_renderer_lv\example_data\effect_layout_image\effect_ensemble/lmdb_label.json'
    tar_lst = r'E:\lxd\text_renderer_lv\example_data\effect_layout_image\effect_ensemble/image_paths.lst'
    lmdb_json_lst(json_path,tar_json_path,tar_lst)
    # lmdb_ocr_dict(img_path,label)
    # jd = getJsonDict(json_path)
    # tar_dict = {}
    # with open(tar_lst, mode='w', encoding='utf8') as f:
    #     for index,item in enumerate(jd['labels']):
    #         img_path = f'{osp.join(osp.dirname(json_path),"images",item+".jpg")}'
    #         f.write(img_path)
    #         if index<=len(jd['labels']):
    #             f.write('\n')

    # print(jd.keys())

