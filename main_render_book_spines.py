import os
import time
import traceback
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from lv_tools.cores.json_io import load_json_to_dict
from lv_tools.cores.str_utils import rand_str
from lv_tools.cores.timestemp import get_datetime
from lv_tools.data_parsing.labelme_json_ops import json_points_resize
from lv_tools.dataset_io.data_saver import LmdbJsonLSaver, LmdbSaver, JsonLSaver
from lv_tools.img_tools.bg_gen import BgGener
from lv_tools.img_tools.img_resize import limit_hw
from lv_tools.task_book_info_gen.book_info_content import BookSpineContent
from lv_tools.task_book_info_gen.book_info_template import BookInfoTemplate
from text_renderer.config import get_cfg
from text_renderer.render_book_spine import BookSpineRender


def item_title_len(item: dict) -> int:
    title_text_num = sum([len(title_piece) for title_piece in item['0主标题']])
    return title_text_num


def text_render():
    render = BookSpineRender(generator_cfg.render_cfg, book_spine_content, book_spine_template, spine_mimicer, dst_dir)
    render()


if __name__ == '__main__':
    config_path = r'.\example_data\config_for_book_spine_info_gen.py'
    # book_excel_path = r'F:\dataset\OCR\图书目录\书脊分类信息生成数据\格式模板-现货数据-mini.xlsx' # 仅供调试用
    book_excel_path = r'F:\dataset\OCR\图书目录\书脊分类信息生成数据\格式模板-现货数据-mini.xlsx'
    # callnumber_excel_path = r'F:\dataset\OCR\callnumber_gen\索书号.xlsx'
    book_spine_temp_json_path = r'F:\dataset\OCR\3-2.book_info_classes\ocr_book_info_classes\book_spine_index\book_spine_index.json'
    # callnumber_temp_json_path = r'F:\dataset\OCR\callnumber_gen\callnumber_index\callnumber_index.json'
    bg_dir = r'D:\lxd_code\bar_dm\dm_bar_base\indoorCVPR_09\Images'
    now = get_datetime()
    dst_dir = rf'F:\dataset\OCR\3-2.book_info_classes\book_spine_info_{now}_lmdb'
    spine_mimicer = BgGener(bg_dir)
    book_spine_template = BookInfoTemplate(book_spine_temp_json_path)
    book_spine_content = BookSpineContent(book_excel_path, callnumber_max_len=3)

    generator_cfg = get_cfg(config_path)[0]
    s = time.time()

    book_spine_render = BookSpineRender(generator_cfg.render_cfg, dst_dir)

    lmdb_saver = LmdbSaver(dst_dir)
    jsonl_saver = JsonLSaver(os.path.join(dst_dir, os.path.basename(dst_dir).replace('_lmdb', '') + '.jsonl'))

    lmdb_jsonl_saver = LmdbJsonLSaver(lmdb_saver, jsonl_saver)
    # jpg_json_saver = JpgJsonSaver(dst_dir)
    # with  LmdbJsonLSaver(lmdb_saver, jsonl_saver) as lmdb_jsonl_saver:

    for i in tqdm(range(1 * 10 ** 2)):
        try:
            # text_render()

            item = book_spine_content.get_one_item_with_title()
            book_title_len = item_title_len(item)
            book_info_template_path = book_spine_template.get_template_json_path_by_book_name_length(book_title_len)

            # 核心逻辑

            template_jd = load_json_to_dict(book_info_template_path)

            h, w = template_jd['imageHeight'], template_jd['imageWidth']

            # ⭐对目标书籍的大小进行限制，并修改对应json相关信息,掠过索书号，给下一阶段去贴

            bg_h, bg_w = limit_hw(h, w)

            json_points_resize(template_jd, (bg_h, bg_w))

            bg = spine_mimicer((bg_h, bg_w))
            # 文本渲染逻辑
            bg, dst_jd = book_spine_render(template_jd, bg, item)

            time_str = datetime.now().strftime("%Y%m%d-%H%M%S%f")

            template_img_path = book_info_template_path.replace('.json', '.jpg')

            s = rand_str(5)
            img_base = f'{s}_{Path(template_img_path).stem}_{time_str}.jpg'

            dst_jd['image_path'] = img_base

            lmdb_jsonl_saver.put(bg, dst_jd)
            # jpg_json_saver.put(bg,dst_jd)

            # BookSpineRender(generator_cfg.render_cfg, book_spine_content, book_spine_template, spine_mimicer, dst_dir)()
            # CallnumberRender(generator_cfg.render_cfg, book_spine_content, book_spine_template, mimic_spine, dst_dir)()
        except:
            traceback.print_exc()
            continue
    lmdb_jsonl_saver.close()
