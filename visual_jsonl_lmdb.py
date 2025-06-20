
from lv_tools.dataset_io.data_loader import LmdbJsonLLoader
from lv_tools.task_ocr.visual_alva_data import VisualAlvaDataLmdb
import os
if __name__ == '__main__':

    lmdb_path = r'F:\dataset\OCR\3-2.book_info_classes\syn_entire_img_rec_book_info\syn_en_entire_books_v2_20250512-163344'
    # jsonl_path = os.path.join(lmdb_path, os.path.basename(lmdb_path).rsplit('_', maxsplit=1)[0] + '.jsonl')
    jsonl_path = os.path.join(lmdb_path, os.path.basename(lmdb_path) + '.jsonl')
    json_dir = r'F:\dataset\OCR\3-2.book_info_classes\@ocr_book_info_classes\book_spine_rec_shujutang\final_alva_json'
    img_dir = r'F:\dataset\OCR\3-2.book_info_classes\@ocr_book_info_classes\book_spine_rec_shujutang\ori_img'

    alva_json_path = r'F:\dataset\OCR\3-2.book_info_classes\ocr_book_info_classes\book_spine_rec_shujutang\116_bookseg_0502_191457_0_IMG20220501113949.json'
    img_path = alva_json_path[:-5] + '.jpg'
    # ⭐1.lmdb格式数据可视化⭐
    lmdb_jsonl_loader = LmdbJsonLLoader(lmdb_path, jsonl_paths=jsonl_path)
    visual_alva = VisualAlvaDataLmdb(lmdb_jsonl_loader, dst_dir=lmdb_path)
    visual_alva.show(80)
    # ⭐图像json对儿的数据可视化⭐
    # img_arr_jd_loader = ImgJsonLoader(
    #     r'F:\dataset\OCR\3.text_rec\hard_for_debug\250428', img_parent='ori_img',
    #     json_parent='final_alva_json')
    # visual_alva = VisualAlvaDataLmdb(img_arr_jd_loader, dst_dir=img_dir)
    # visual_alva.show(2)