import os
import random
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from lv_tools.cores.json_io import load_json_to_dict, save_json
from lv_tools.cores.str_utils import rand_str
from lv_tools.data_parsing.labelme_json_ops import json_points_resize
from lv_tools.img_tools.bg_gen import BgGener
from lv_tools.img_tools.img_resize import limit_hw
from lv_tools.img_tools.rotate_box import rotate_image_arr_and_polygons
from lv_tools.task_book_info_gen.book_info_content import BookSpineContent
from lv_tools.task_book_info_gen.book_info_template import BookInfoTemplate
from text_renderer.config import get_cfg
from text_renderer.render_book_spine import CallnumberRender

if __name__ == '__main__':
    config_path = r'D:\lxd_code\text_renderer_lv\example_data\config_for_book_spine_info_gen.py'
    # book_excel_path = r'F:\dataset\OCR\图书目录\书脊分类信息生成数据\格式模板-现货数据-mini.xlsx' # 仅供调试用
    callnumber_excel_path = r'F:\dataset\OCR\图书目录\书脊分类信息生成数据\分类号-种次号-等宽索书号.xlsx'
    # callnumber_excel_path = r'F:\dataset\OCR\callnumber_gen\索书号.xlsx'
    # book_spine_temp_json_path = r'F:\dataset\OCR\3-2.book_info_classes\ocr_book_info_classes\book_spine_index\v2\book_spine_index.json'
    callnumber_temp_json_path = r'F:\dataset\OCR\callnumber_gen\callnumber_index\callnumber_index.json'
    # bg_dir = r'D:\lxd_code\bar_dm\dm_bar_base\indoorCVPR_09\Images'
    bg_dir = r'F:\dataset\OCR\callnumber_gen\callnumber_bg_normal'
    dst_dir = r'F:\dataset\OCR\3-2.book_info_classes\data_callnumber\250110'
    callnumber_bg_mimicer = BgGener(bg_dir)
    callnumber_template = BookInfoTemplate(callnumber_temp_json_path)
    book_spine_content = BookSpineContent(callnumber_excel_path, callnumber_max_len=2)

    generator_cfg = get_cfg(config_path)[0]

    callnumber_render = CallnumberRender(generator_cfg.render_cfg, dst_dir)

    s = time.time()
    limit_h = 200
    for i in tqdm(range(3 * 10 ** 1)):
        try:

            item = book_spine_content.choice()

            callnumber_list = item['7索书号']
            callnumber_num = len(callnumber_list)

            book_info_template_path = callnumber_template.get_template_json_path_by_callnumber_part_num(callnumber_num)

            # 核心逻辑

            template_jd = load_json_to_dict(book_info_template_path)

            h, w = template_jd['imageHeight'], template_jd['imageWidth']

            bg_h, bg_w = limit_hw(h, w, target_max_h=limit_h)
            json_points_resize(template_jd, (bg_h, bg_w))

            bg = callnumber_bg_mimicer((bg_h, bg_w))
            bg, dst_jd = callnumber_render(template_jd, bg, item)

            # filter

            h_list = []
            w_list = []
            for shape in dst_jd['shapes']:
                points = np.array(shape['points'], np.int32)
                _, (w, h), _ = cv2.minAreaRect(points)
                h_list.append(h)
                w_list.append(w)
            if not (len(h_list) == len(w_list) == 2 and abs(h_list[0] - h_list[1]) < 2 and abs(
                    w_list[0] - w_list[1]) < 2):
                continue

            time_str = datetime.now().strftime("%Y%m%d-%H%M%S%f")

            template_img_path = book_info_template_path.replace('.json', '.jpg')

            s = rand_str(5)
            img_base = f'{s}_{Path(template_img_path).stem}_{time_str}.jpg'

            if (flag := random.randint(0, 1)):
                dst_jd, bg = rotate_image_arr_and_polygons(dst_jd, bg, -90 + random.randint(-10, 10))
            else:
                dst_jd, bg = rotate_image_arr_and_polygons(dst_jd, bg, random.randint(-10, 10))

            save_dir = dst_dir + f'_{str(flag).zfill(2)}'
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            cv2.imwrite(rf'{save_dir}\{img_base}',
                        bg)
            dst_jd['imagePath'] = img_base
            save_json(os.path.join(save_dir, img_base[:-3] + 'json'), dst_jd)



        except:
            traceback.print_exc()
            continue
