
from pathlib import Path

import lmdb

from lv_tools.lmdbs.visualize import (
    Visualize,
    Visualizer,
    get_ori_data,
    show_samples,
    show_single,
)


def ensure_num_samples(lmdb_path):
    """Add missing num-samples metadata without changing an existing value."""
    data_path = Path(lmdb_path) / "data.mdb"
    map_size = max(data_path.stat().st_size + 1024 ** 2, 10 * 1024 ** 2)
    env = lmdb.open(str(lmdb_path), map_size=map_size, create=False)
    try:
        with env.begin(write=True) as txn:
            if txn.get(b"num-samples") is not None:
                return False

            sample_key_counts = {b"id-": 0, b"image-": 0, b"label-": 0}
            with txn.cursor() as cursor:
                for key, _ in cursor:
                    for prefix in sample_key_counts:
                        if key.startswith(prefix) and key[len(prefix):].isdigit():
                            sample_key_counts[prefix] += 1
                            break

            count = max(sample_key_counts.values())
            if count == 0:
                count = txn.stat()["entries"]
            txn.put(b"num-samples", str(count).encode("utf-8"))
    finally:
        env.close()

    print(f"num-samples missing; added value: {count}")
    return True


def visualization_some_sample(lmdb_path, num=100, lan="ch"):
    """Compatibility wrapper for the pre-refactor visualization entry point."""
    ensure_num_samples(lmdb_path)
    return show_samples(lmdb_path, count=num)

if __name__ == '__main__':
    visualization_some_sample(
        r'F:\dataset\OCR\3.text_rec\synthesis\rec_pieces\lmdb_yaml_090110_38_36\callnumber')
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
