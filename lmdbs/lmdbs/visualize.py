"""LMDB 可视化入口：复用 lv_tools，并补齐本仓库的 num-samples 修复。

用法
----
  # 一键：导出原图+标注 map.txt，并渲染带标注预览图（最常用）
  python -m lmdbs.lmdbs.visualize full --lmdb F:\dataset\OCR\3.text_rec\synthesis\rec_pieces\test --num 20

  # 仅渲染预览图到 <lmdb>/samples
  python -m lmdbs.lmdbs.visualize samples --lmdb F:/path/to/lmdb --num 30

  # 仅导出原图与 map.txt 到 <lmdb>/ori_data
  python -m lmdbs.lmdbs.visualize ori-data --lmdb F:/path/to/lmdb --num 20
"""

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
                        if key.startswith(prefix) and key[len(prefix) :].isdigit():
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


def visualize_full(lmdb_path, num=20):
    """导出原图+标注，并渲染预览图（对应 lv_tools ``full``）。"""
    ensure_num_samples(lmdb_path)
    get_ori_data(lmdb_path, num)
    show_samples(lmdb_path, num)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    full_p = sub.add_parser("full", help="导出原图+标注，并渲染预览图")
    full_p.add_argument("--lmdb", required=True, help="LMDB 目录路径")
    full_p.add_argument("--num", type=int, default=20, help="采样数量（默认 20）")

    samples_p = sub.add_parser("samples", help="仅渲染预览图到 <lmdb>/samples")
    samples_p.add_argument("--lmdb", required=True, help="LMDB 目录路径")
    samples_p.add_argument("--num", type=int, default=50, help="展示数量（默认 50）")

    ori_p = sub.add_parser("ori-data", help="导出原图与 map.txt 到 <lmdb>/ori_data")
    ori_p.add_argument("--lmdb", required=True, help="LMDB 目录路径")
    ori_p.add_argument("--num", type=int, default=20, help="抽取数量（默认 20）")

    show_p = sub.add_parser("show", help="查看单条记录原文")
    show_p.add_argument("--lmdb", required=True, help="LMDB 目录路径")
    show_p.add_argument("--id", type=int, default=0, help="记录 id")

    args = parser.parse_args(argv)
    ensure_num_samples(args.lmdb)

    if args.command == "full":
        visualize_full(args.lmdb, args.num)
    elif args.command == "samples":
        show_samples(args.lmdb, args.num)
    elif args.command == "ori-data":
        get_ori_data(args.lmdb, args.num)
    elif args.command == "show":
        show_single(args.lmdb, args.id)


if __name__ == "__main__":
    main()
