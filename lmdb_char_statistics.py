"""Count characters in recognition labels stored in an LMDB dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterator, Tuple

import lmdb
from tqdm import tqdm


def _find_key_prefix(txn) -> Tuple[bytes, str]:
    """Detect the recognition-label storage format used by the LMDB."""
    cursor = txn.cursor()
    for prefix, storage_format in (
        (b"id-", "json"),
        (b"label-", "plain_text"),
    ):
        if cursor.set_range(prefix) and cursor.key().startswith(prefix):
            return prefix, storage_format

    raise ValueError("LMDB 中未找到 id-* 或 label-* 识别标签")


def _iter_labels(txn, prefix: bytes, storage_format: str) -> Iterator[str]:
    cursor = txn.cursor()
    if not cursor.set_range(prefix):
        return

    for key, value in cursor:
        if not key.startswith(prefix):
            break

        if storage_format == "json":
            try:
                record = json.loads(value.decode("utf-8"))
                label = record["transcription"]
            except (UnicodeDecodeError, json.JSONDecodeError, KeyError) as error:
                raise ValueError(
                    "无法读取记录 {} 的 transcription".format(
                        key.decode("utf-8", errors="replace")
                    )
                ) from error
        else:
            try:
                label = value.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(
                    "无法以 UTF-8 解码记录 {}".format(
                        key.decode("utf-8", errors="replace")
                    )
                ) from error

        if not isinstance(label, str):
            raise ValueError(
                "记录 {} 的 transcription 不是字符串".format(
                    key.decode("utf-8", errors="replace")
                )
            )
        yield label


def collect_character_statistics(lmdb_path: Path):
    """Return sorted character counts and basic dataset statistics."""
    if not lmdb_path.exists():
        raise FileNotFoundError("LMDB 路径不存在: {}".format(lmdb_path))

    counter = Counter()
    sample_count = 0

    env = lmdb.open(
        str(lmdb_path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    try:
        with env.begin(write=False) as txn:
            prefix, storage_format = _find_key_prefix(txn)
            raw_sample_count = txn.get(b"num-samples")
            expected_sample_count = None
            if raw_sample_count is not None:
                try:
                    expected_sample_count = int(raw_sample_count)
                except ValueError:
                    pass

            labels = _iter_labels(txn, prefix, storage_format)
            for label in tqdm(
                labels,
                total=expected_sample_count,
                desc="LMDB",
                unit="sample",
                dynamic_ncols=True,
                ascii=True,
            ):
                counter.update(label)
                sample_count += 1
    finally:
        env.close()

    sorted_counts = sorted(
        counter.items(),
        key=lambda item: (-item[1], item[0]),
    )
    return sorted_counts, sample_count, storage_format


def collect_labelme_coverage(labelme_dir: Path, training_characters):
    """Check whether every character in LabelMe labels exists in the LMDB."""
    if not labelme_dir.is_dir():
        raise NotADirectoryError(
            "LabelMe 标注目录不存在: {}".format(labelme_dir)
        )

    json_paths = sorted(labelme_dir.rglob("*.json"))
    if not json_paths:
        raise ValueError(
            "LabelMe 标注目录中未找到 JSON: {}".format(labelme_dir)
        )

    annotation_counter = Counter()
    missing_counter = Counter()
    shape_count = 0
    affected_files = []

    for json_path in tqdm(
        json_paths,
        desc="LabelMe",
        unit="file",
        dynamic_ncols=True,
        ascii=True,
    ):
        try:
            with json_path.open(encoding="utf-8") as file:
                annotation = json.load(file)
            shapes = annotation["shapes"]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError) as error:
            raise ValueError(
                "无法读取 LabelMe 标注: {}".format(json_path)
            ) from error

        problem_labels = []
        for shape_index, shape in enumerate(shapes):
            label = shape.get("label")
            if not isinstance(label, str):
                raise ValueError(
                    "{} 的 shapes[{}].label 不是字符串".format(
                        json_path,
                        shape_index,
                    )
                )

            shape_count += 1
            annotation_counter.update(label)
            missing_characters = sorted(
                {character for character in label if character not in training_characters}
            )
            if missing_characters:
                problem_labels.append(
                    {
                        "label": label,
                        "missing_characters": missing_characters,
                    }
                )
                missing_counter.update(
                    character
                    for character in label
                    if character not in training_characters
                )

        if problem_labels:
            affected_files.append(
                {
                    "file": str(json_path.relative_to(labelme_dir)),
                    "problem_labels": problem_labels,
                }
            )

    sorted_missing_counts = sorted(
        missing_counter.items(),
        key=lambda item: (-item[1], item[0]),
    )
    return {
        "labelme_dir": str(labelme_dir.resolve()),
        "is_fully_covered": not missing_counter,
        "annotation_file_count": len(json_paths),
        "annotation_shape_count": shape_count,
        "annotation_character_count": sum(annotation_counter.values()),
        "annotation_unique_character_count": len(annotation_counter),
        "missing_character_count": sum(missing_counter.values()),
        "missing_unique_character_count": len(missing_counter),
        "affected_file_count": len(affected_files),
        "missing_character_counts": {
            character: count
            for character, count in sorted_missing_counts
        },
        "files_with_missing_characters": affected_files,
    }


def load_statistics(statistics_path: Path):
    """Load character statistics without scanning the LMDB again."""
    try:
        with statistics_path.open(encoding="utf-8") as file:
            result = json.load(file)

        if "character_counts" in result:
            sorted_counts = list(result["character_counts"].items())
        else:
            sorted_counts = [
                (item["character"], item["count"])
                for item in result["characters"]
            ]

        sorted_counts = sorted(
            ((character, int(count)) for character, count in sorted_counts),
            key=lambda item: (-item[1], item[0]),
        )
        lmdb_path = Path(result["lmdb_path"])
        sample_count = int(result["sample_count"])
        storage_format = result["storage_format"]
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            "无法读取字符统计 JSON: {}".format(statistics_path)
        ) from error

    return lmdb_path, sorted_counts, sample_count, storage_format


def write_statistics(
    output_path: Path,
    lmdb_path: Path,
    sorted_counts,
    sample_count: int,
    storage_format: str,
    labelme_coverage=None,
) -> None:
    """Write the ordered statistics as UTF-8 JSON."""
    result = {
        "lmdb_path": str(lmdb_path.resolve()),
        "storage_format": storage_format,
        "sample_count": sample_count,
        "total_character_count": sum(count for _, count in sorted_counts),
        "unique_character_count": len(sorted_counts),
        "character_counts": {
            character: count
            for character, count in sorted_counts
        },
        "characters": [
            {"character": character, "count": count}
            for character, count in sorted_counts
        ],
    }
    if labelme_coverage is not None:
        result["labelme_coverage"] = labelme_coverage

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="统计识别 LMDB 标签中的字符频次，并按频次降序写入 JSON",
    )
    parser.add_argument(
        "lmdb_path",
        type=Path,
        nargs="?",
        help="待统计的 LMDB 路径",
    )
    parser.add_argument(
        "--statistics-json",
        type=Path,
        help="读取已有统计 JSON，跳过 LMDB 扫描",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="结果路径；默认写入 <LMDB名称>_char_counts.json",
    )
    parser.add_argument(
        "--labelme-dir",
        type=Path,
        help="可选：检查该目录下 LabelMe 标签字符是否被 LMDB 覆盖",
    )
    args = parser.parse_args()

    if (args.lmdb_path is None) == (args.statistics_json is None):
        parser.error("必须且只能指定 lmdb_path 或 --statistics-json")

    if args.statistics_json is not None:
        statistics_path = args.statistics_json.resolve()
        (
            lmdb_path,
            sorted_counts,
            sample_count,
            storage_format,
        ) = load_statistics(statistics_path)
    else:
        statistics_path = None
        lmdb_path = args.lmdb_path.resolve()
        (
            sorted_counts,
            sample_count,
            storage_format,
        ) = collect_character_statistics(lmdb_path)

    output_path = args.output
    if output_path is None:
        if statistics_path is not None:
            output_path = statistics_path.with_name(
                "{}_with_coverage.json".format(statistics_path.stem)
            )
        else:
            output_path = lmdb_path.parent / "{}_char_counts.json".format(
                lmdb_path.name
            )
    else:
        output_path = output_path.resolve()

    labelme_coverage = None
    if args.labelme_dir is not None:
        labelme_coverage = collect_labelme_coverage(
            args.labelme_dir.resolve(),
            {character for character, _ in sorted_counts},
        )
    write_statistics(
        output_path,
        lmdb_path,
        sorted_counts,
        sample_count,
        storage_format,
        labelme_coverage,
    )

    print("统计完成: {}".format(output_path))
    print("样本数: {}".format(sample_count))
    print("字符总数: {}".format(sum(count for _, count in sorted_counts)))
    print("不同字符数: {}".format(len(sorted_counts)))
    if labelme_coverage is not None:
        print(
            "LabelMe 字符是否全部覆盖: {}".format(
                "是" if labelme_coverage["is_fully_covered"] else "否"
            )
        )
        print(
            "缺失字符种类数: {}".format(
                labelme_coverage["missing_unique_character_count"]
            )
        )
        print(
            "受影响标注文件数: {}".format(
                labelme_coverage["affected_file_count"]
            )
        )


if __name__ == "__main__":
    main()
