"""Build a balanced Korean OCR corpus for missing Hangul syllables."""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

from tqdm import tqdm


QUALIFIERS = (
    "오늘의",
    "새로운",
    "첫",
    "주요",
    "짧은",
    "긴",
    "쉬운",
    "어려운",
    "기본",
    "추천",
    "선택한",
    "읽을",
    "찾은",
    "기록할",
    "학습할",
    "연습할",
    "확인할",
    "검색할",
    "소개할",
    "검토할",
)

NOUNS = (
    "표현",
    "단어",
    "문구",
    "제목",
    "자료",
    "내용",
    "목록",
    "문장",
    "표기",
    "발음",
    "이름",
    "주제",
    "글자",
    "음절",
    "기록",
    "정보",
    "소개",
    "설명",
    "예문",
    "항목",
)

ALLOWED_PUNCTUATION = frozenset("！？")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = "".join(
        character
        for character in text
        if not (
            unicodedata.category(character).startswith("P")
            and character not in ALLOWED_PUNCTUATION
        )
    )
    return re.sub(r"\s+", " ", text).strip()


def is_hangul_syllable(character: str) -> bool:
    return len(character) == 1 and "\uac00" <= character <= "\ud7a3"


def split_text(text: str, max_length: int):
    """Split text at word boundaries, falling back to fixed-width chunks."""
    chunks = []
    current = ""
    for word in normalize_text(text).split(" "):
        while len(word) > max_length:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(word[:max_length])
            word = word[max_length:]

        candidate = word if not current else "{} {}".format(current, word)
        if len(candidate) <= max_length:
            current = candidate
        else:
            chunks.append(current)
            current = word

    if current:
        chunks.append(current)
    return chunks


def extract_target_seeds(text: str, character: str, max_length: int):
    """Return short, meaningful pieces that preserve the target character."""
    seeds = [chunk for chunk in split_text(text, max_length) if character in chunk]
    for word in normalize_text(text).split(" "):
        if character not in word:
            continue
        if len(word) <= 4:
            anchor = word
        else:
            index = word.index(character)
            start = min(max(0, index - 1), len(word) - 4)
            anchor = word[start:start + 4]
        if anchor not in seeds:
            seeds.append(anchor)
    return seeds


def load_targets_and_labelme_seeds(report_path: Path, max_text_length: int):
    with report_path.open(encoding="utf-8") as file:
        report = json.load(file)

    coverage = report["labelme_coverage"]
    targets = sorted(
        character
        for character in coverage["missing_character_counts"]
        if is_hangul_syllable(character)
    )
    seeds = defaultdict(list)
    seed_sets = defaultdict(set)

    for file_info in coverage["files_with_missing_characters"]:
        for problem in file_info["problem_labels"]:
            text = normalize_text(problem["label"])
            if not text:
                continue
            for character in targets:
                if character not in text:
                    continue
                for seed in extract_target_seeds(
                    text,
                    character,
                    max_text_length,
                ):
                    if seed not in seed_sets[character]:
                        seeds[character].append((seed, "labelme"))
                        seed_sets[character].add(seed)

    return targets, seeds, seed_sets


def add_local_corpus_seeds(
    corpus_path: Path,
    targets,
    seeds,
    seed_sets,
    max_seeds_per_character: int,
    max_text_length: int,
) -> None:
    target_set = set(targets)
    with corpus_path.open(encoding="utf-8", errors="ignore") as file:
        for line in tqdm(
            file,
            desc="Source corpus",
            unit="line",
            dynamic_ncols=True,
            ascii=True,
        ):
            text = normalize_text(line)
            if not text:
                continue

            for character in set(text) & target_set:
                for seed in extract_target_seeds(
                    text,
                    character,
                    max_text_length,
                ):
                    if len(seeds[character]) >= max_seeds_per_character:
                        break
                    if seed in seed_sets[character]:
                        continue
                    seeds[character].append((seed, "local_corpus"))
                    seed_sets[character].add(seed)


def build_corpus(
    targets,
    seeds,
    n: int,
    seed: int,
    min_text_length: int,
    max_text_length: int,
):
    rng = random.Random(seed)
    records = []
    used_texts = set()
    assigned_counts = Counter()
    source_counts = Counter()
    template_combinations = list(product(QUALIFIERS, NOUNS))

    for character in tqdm(
        targets,
        desc="Build corpus",
        unit="char",
        dynamic_ncols=True,
        ascii=True,
    ):
        character_seeds = list(seeds[character])
        rng.shuffle(character_seeds)

        for text, source in character_seeds:
            text = normalize_text(text)
            if len(text) < min_text_length:
                continue
            if text in used_texts:
                continue
            records.append(
                {
                    "target_char": character,
                    "text": text,
                    "source": source,
                    "generated": False,
                }
            )
            used_texts.add(text)
            assigned_counts[character] += 1
            source_counts[source] += 1
            if assigned_counts[character] >= n:
                break

        if assigned_counts[character] >= n:
            continue
        if not character_seeds:
            raise RuntimeError(
                "目标字符 {!r} 没有可用于扩增的真实种子文本".format(character)
            )

        anchors = [text for text, _ in character_seeds]
        rng.shuffle(template_combinations)
        for qualifier, noun in template_combinations:
            for anchor in anchors:
                text = normalize_text("{} {} {}".format(qualifier, noun, anchor))
                if not min_text_length <= len(text) <= max_text_length:
                    continue
                if text in used_texts:
                    continue
                records.append(
                    {
                        "target_char": character,
                        "text": text,
                        "source": "template",
                        "generated": True,
                        "seed_text": anchor,
                    }
                )
                used_texts.add(text)
                assigned_counts[character] += 1
                source_counts["template"] += 1
                if assigned_counts[character] >= n:
                    break
            if assigned_counts[character] >= n:
                break

        if assigned_counts[character] < n:
            raise RuntimeError(
                "目标字符 {!r} 仅生成 {} 条，未达到 {} 条".format(
                    character,
                    assigned_counts[character],
                    n,
                )
            )

    return records, assigned_counts, source_counts


def validate_corpus(
    records,
    targets,
    n: int,
    min_text_length: int,
    max_text_length: int,
):
    texts = [record["text"] for record in records]
    if len(texts) != len(set(texts)):
        raise ValueError("生成结果中存在重复文本")

    assigned_counts = Counter(record["target_char"] for record in records)
    for record in records:
        if record["target_char"] not in record["text"]:
            raise ValueError("文本未包含目标字符: {}".format(record))
        if record["text"] != normalize_text(record["text"]):
            raise ValueError("文本未满足规范化约束: {}".format(record["text"]))
        if not min_text_length <= len(record["text"]) <= max_text_length:
            raise ValueError("文本长度不在限制范围内: {}".format(record["text"]))

    for character in targets:
        if assigned_counts[character] != n:
            raise ValueError(
                "目标字符 {!r} 的文本数为 {}，预期为 {}".format(
                    character,
                    assigned_counts[character],
                    n,
                )
            )


def append_labelme_records(
    labelme_dir: Path,
    records,
    min_text_length: int,
    max_text_length: int,
    source_counts,
):
    """Append every LabelMe label, preserving repeated annotations."""
    json_paths = sorted(labelme_dir.rglob("*.json"))
    stats = {
        "labelme_dir": str(labelme_dir.resolve()),
        "file_count": len(json_paths),
        "shape_count": 0,
        "nonempty_label_count": 0,
        "empty_label_count": 0,
        "chunk_count": 0,
        "discarded_short_chunk_count": 0,
    }

    for json_path in tqdm(
        json_paths,
        desc="Append LabelMe",
        unit="file",
        dynamic_ncols=True,
        ascii=True,
    ):
        try:
            with json_path.open(encoding="utf-8") as file:
                shapes = json.load(file)["shapes"]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError) as error:
            raise ValueError(
                "无法读取 LabelMe 标注: {}".format(json_path)
            ) from error

        for shape_index, shape in enumerate(shapes):
            stats["shape_count"] += 1
            label = shape.get("label")
            if not isinstance(label, str):
                raise ValueError(
                    "{} 的 shapes[{}].label 不是字符串".format(
                        json_path,
                        shape_index,
                    )
                )

            chunks = split_text(label, max_text_length)
            if not chunks:
                stats["empty_label_count"] += 1
                continue

            stats["nonempty_label_count"] += 1
            for chunk_index, text in enumerate(chunks):
                if len(text) < min_text_length:
                    stats["discarded_short_chunk_count"] += 1
                    continue
                records.append(
                    {
                        "target_char": None,
                        "text": text,
                        "source": "labelme_all",
                        "generated": False,
                        "labelme_file": str(json_path.relative_to(labelme_dir)),
                        "shape_index": shape_index,
                        "chunk_index": chunk_index,
                    }
                )
                source_counts["labelme_all"] += 1
                stats["chunk_count"] += 1

    return stats


def validate_output_records(
    records,
    min_text_length: int,
    max_text_length: int,
) -> None:
    for record in records:
        text = record["text"]
        if not min_text_length <= len(text) <= max_text_length:
            raise ValueError("文本长度不在限制范围内: {}".format(text))
        if text != normalize_text(text):
            raise ValueError("文本未满足规范化约束: {}".format(text))


def write_outputs(
    output_dir: Path,
    records,
    targets,
    n,
    source_counts,
    min_text_length,
    max_text_length,
    labelme_stats=None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    text_path = output_dir / "korean_missing_chars_corpus.txt"
    jsonl_path = output_dir / "korean_missing_chars_corpus.jsonl"
    summary_path = output_dir / "korean_missing_chars_summary.json"

    with text_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(record["text"] + "\n")

    with jsonl_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    actual_coverage = {
        character: sum(character in record["text"] for record in records)
        for character in targets
    }
    summary = {
        "target_character_count": len(targets),
        "n_per_character": n,
        "min_text_length": min_text_length,
        "max_text_length": max_text_length,
        "total_text_count": len(records),
        "all_texts_unique": len(records) == len({r["text"] for r in records}),
        "source_counts": dict(source_counts),
        "target_characters": targets,
        "actual_coverage_counts": actual_coverage,
        "text_corpus": str(text_path.resolve()),
        "metadata_corpus": str(jsonl_path.resolve()),
    }
    if labelme_stats is not None:
        summary["appended_labelme"] = labelme_stats
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="为训练集缺失的韩文音节构建均衡 OCR corpus",
    )
    parser.add_argument("coverage_report", type=Path, help="字符覆盖报告 JSON")
    parser.add_argument("source_corpus", type=Path, help="真实韩国语文本语料")
    parser.add_argument("--output-dir", type=Path, default=Path("korean_coverage_corpus"))
    parser.add_argument(
        "--labelme-dir",
        type=Path,
        help="递归追加该目录下所有 LabelMe shapes[*].label",
    )
    parser.add_argument("--n", type=int, default=200, help="每个目标字符的文本数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--min-text-length", type=int, default=2)
    parser.add_argument("--max-text-length", type=int, default=12)
    args = parser.parse_args()

    if args.n <= 0:
        parser.error("--n 必须大于 0")
    if args.min_text_length < 2:
        parser.error("--min-text-length 不能小于 2")
    if args.min_text_length > args.max_text_length:
        parser.error("--min-text-length 不能大于 --max-text-length")
    if args.max_text_length < 8:
        parser.error("--max-text-length 不能小于 8")

    report_path = args.coverage_report.resolve()
    corpus_path = args.source_corpus.resolve()
    if not report_path.is_file():
        parser.error("覆盖报告不存在: {}".format(report_path))
    if not corpus_path.is_file():
        parser.error("真实语料不存在: {}".format(corpus_path))

    targets, seeds, seed_sets = load_targets_and_labelme_seeds(
        report_path,
        args.max_text_length,
    )
    if not targets:
        parser.error("覆盖报告中没有缺失的韩文音节")

    add_local_corpus_seeds(
        corpus_path,
        targets,
        seeds,
        seed_sets,
        max_seeds_per_character=args.n,
        max_text_length=args.max_text_length,
    )
    records, _, source_counts = build_corpus(
        targets,
        seeds,
        n=args.n,
        seed=args.seed,
        min_text_length=args.min_text_length,
        max_text_length=args.max_text_length,
    )
    validate_corpus(
        records,
        targets,
        args.n,
        args.min_text_length,
        args.max_text_length,
    )
    labelme_stats = None
    if args.labelme_dir is not None:
        labelme_dir = args.labelme_dir.resolve()
        if not labelme_dir.is_dir():
            parser.error("LabelMe 目录不存在: {}".format(labelme_dir))
        labelme_stats = append_labelme_records(
            labelme_dir,
            records,
            args.min_text_length,
            args.max_text_length,
            source_counts,
        )
    validate_output_records(
        records,
        args.min_text_length,
        args.max_text_length,
    )
    write_outputs(
        args.output_dir.resolve(),
        records,
        targets,
        args.n,
        source_counts,
        args.min_text_length,
        args.max_text_length,
        labelme_stats,
    )

    print("Corpus generated: {}".format(args.output_dir.resolve()))
    print("Target Hangul syllables: {}".format(len(targets)))
    print("Texts per syllable: {}".format(args.n))
    print("Minimum text length: {}".format(args.min_text_length))
    print("Maximum text length: {}".format(args.max_text_length))
    print("Total texts: {}".format(len(records)))
    print("Sources: {}".format(dict(source_counts)))
    if labelme_stats is not None:
        print("Appended LabelMe: {}".format(labelme_stats))


if __name__ == "__main__":
    main()
