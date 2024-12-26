import time
import traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from tqdm import tqdm

from lv_tools.task_book_info_gen.bg_gen import MimicSpineBg
from lv_tools.task_book_info_gen.book_info_content import BookSpineContent
from lv_tools.task_book_info_gen.book_info_template import BookInfoTemplate
from text_renderer.config import get_cfg
from text_renderer.render_book_spine import BookSpineRender, CallnumberRender


def text_render():
    render = BookSpineRender(generator_cfg.render_cfg, book_spine_content, book_spine_template, mimic_spine, dst_dir)
    render()


if __name__ == '__main__':
    config_path = r'D:\lxd_code\text_renderer_lv\example_data\config_for_book_spine_info_gen.py'
    # book_excel_path = r'F:\dataset\OCR\图书目录\书脊分类信息生成数据\格式模板-现货数据-mini.xlsx' # 仅供调试用
    callnumber_excel_path = r'F:\dataset\OCR\图书目录\书脊分类信息生成数据\中科大-山西-绵阳-去重-索书号.xlsx'
    # callnumber_excel_path = r'F:\dataset\OCR\callnumber_gen\索书号.xlsx'
    # book_spine_temp_json_path = r'F:\dataset\OCR\3-2.book_info_classes\ocr_book_info_classes\book_spine_index\v2\book_spine_index.json'
    callnumber_temp_json_path = r'F:\dataset\OCR\callnumber_gen\callnumber_index\callnumber_index.json'
    bg_dir = r'D:\lxd_code\bar_dm\dm_bar_base\indoorCVPR_09\Images'
    dst_dir = r'F:\dataset\OCR\3-2.book_info_classes\data_callnumber\v2'
    mimic_spine = MimicSpineBg(bg_dir)
    book_spine_template = BookInfoTemplate(callnumber_temp_json_path)
    book_spine_content = BookSpineContent(callnumber_excel_path, callnumber_max_len=3)

    generator_cfg = get_cfg(config_path)[0]
    s = time.time()
    # 目前多进程存在问题，生产的数据重复的。
    # future_list = []
    # with ProcessPoolExecutor(max_workers=2) as exe:
    #
    #     for i in tqdm(range(1 * 10 ** 2)):
    #         try:
    #             # text_render()
    #             future_list.append(exe.submit(BookSpineRender(generator_cfg.render_cfg, book_spine_content, book_spine_template, mimic_spine,
    #                             dst_dir)))
    #             # CallnumberRender(generator_cfg.render_cfg, book_spine_content, book_spine_template, mimic_spine, dst_dir)()
    #         except:
    #             traceback.print_exc()
    #             continue
    # for result in as_completed(future_list):
    #     print(result.result())


    for i in tqdm(range(3*10)):
        try:
            # text_render()
            # BookSpineRender(generator_cfg.render_cfg, book_spine_content, book_spine_template, mimic_spine, dst_dir)()
            CallnumberRender(generator_cfg.render_cfg, book_spine_content, book_spine_template, mimic_spine, dst_dir)()
        except:
            traceback.print_exc()
            continue
