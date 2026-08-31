
from lv_tools.lmdbs.visualize import (
    Visualize,
    Visualizer,
    get_ori_data,
    show_samples,
    show_single,
)


def visualization_some_sample(lmdb_path, num=100, lan="ch"):
    """Compatibility wrapper for the pre-refactor visualization entry point."""
    return show_samples(lmdb_path, count=num)

if __name__ == '__main__':
    visualization_some_sample(
        r'F:\dataset\OCR\3.text_rec\synthesis\rec_pieces\lmdb_datatest_083109_48_14\effect_layout_image\effect_ensembleauthor')
    # cvt_data()
    # lmdb_info(r'\\192.168.1.11\dataset\bar_code\a_bar\a_det\synthesis\bar_spine_1226_6num_remainder_v4lmdb')
    # li = [str(i).zfill(6) for i in range(10**6)]
    # print(li[:10])
    # lmdb_path = r'F:\dataset\OCR\2.text_det\syn_digit_series_with_parenthesis_lmdb_num_5000'
    # with lmdb.open(lmdb_path) as env:
    #     with env.begin(write=False) as txn:
    #         for k,v in txn.cursor():
    #             if not k.decode('utf8').startswith('id-'):
    #                 print(f"{k.decode('utf8')}---{v.decode('unicode_escape')}")
