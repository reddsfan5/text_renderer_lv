"""Supplement an existing corpus for characters underrepresented in LMDB."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

from tqdm import tqdm

from scripts.corpus.build_korean_coverage_corpus import (
    NOUNS,
    QUALIFIERS,
    extract_target_seeds,
    normalize_text,
)


def load_training_counts(statistics_path: Path) -> Counter:
    with statistics_path.open(encoding="utf-8") as file:
        statistics = json.load(file)

    if "character_counts" in statistics:
        items = statistics["character_counts"].items()
    else:
        items = (
            (item["character"], item["count"])
            for item in statistics["characters"]
        )
    return Counter({character: int(count) for character, count in items})


def load_corpus(corpus_dir: Path):
    jsonl_path = corpus_dir / "korean_missing_chars_corpus.jsonl"
    summary_path = corpus_dir / "korean_missing_chars_summary.json"
    with jsonl_path.open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]
    with summary_path.open(encoding="utf-8") as file:
        summary = json.load(file)
    return records, summary


def clean_records(records, min_text_length):
    cleaned_records = []
    changed_text_count = 0
    discarded_short_text_count = 0
    for record in records:
        original_text = record["text"]
        text = normalize_text(original_text)
        if text != original_text:
            changed_text_count += 1
        if len(text) < min_text_length:
            discarded_short_text_count += 1
            continue
        record["text"] = text
        cleaned_records.append(record)
    return cleaned_records, changed_text_count, discarded_short_text_count


def collect_annotation_profile(records):
    annotation_counts = Counter()
    annotation_records = []
    for record in records:
        if record.get("source") != "labelme_all":
            continue
        annotation_counts.update(record["text"])
        annotation_records.append(record)
    return annotation_counts, annotation_records


def build_seed_index(annotation_records, target_characters, max_text_length):
    seeds = defaultdict(list)
    seed_sets = defaultdict(set)
    target_set = set(target_characters)
    for record in annotation_records:
        text = record["text"]
        for character in set(text) & target_set:
            for seed in extract_target_seeds(text, character, max_text_length):
                if seed in seed_sets[character]:
                    continue
                seeds[character].append(seed)
                seed_sets[character].add(seed)
    return seeds


def supplement_records(
    records,
    training_counts,
    annotation_counts,
    annotation_records,
    min_count,
    min_text_length,
    max_text_length,
    random_seed,
):
    annotation_characters = {
        character
        for character in annotation_counts
        if not character.isspace()
    }
    missing_characters = {
        character
        for character in annotation_characters
        if training_counts[character] == 0
    }
    low_coverage_characters = {
        character
        for character in annotation_characters
        if 0 < training_counts[character] < min_count
    }

    corpus_counts_before = Counter()
    for record in records:
        corpus_counts_before.update(record["text"])
    projected_counts = training_counts + corpus_counts_before
    target_characters = sorted(
        character
        for character in missing_characters | low_coverage_characters
        if projected_counts[character] < min_count
    )

    seeds = build_seed_index(
        annotation_records,
        target_characters,
        max_text_length,
    )
    used_texts = {record["text"] for record in records}
    supplement_counts = Counter()
    added_records = []
    rng = random.Random(random_seed)
    contexts = [
        (qualifier, noun, position)
        for qualifier, noun, position in product(
            QUALIFIERS,
            NOUNS,
            ("suffix", "prefix"),
        )
    ]

    for character in tqdm(
        target_characters,
        desc="Supplement coverage",
        unit="char",
        dynamic_ncols=True,
        ascii=True,
    ):
        if projected_counts[character] >= min_count:
            continue
        character_seeds = list(seeds[character])
        if not character_seeds:
            raise RuntimeError(
                "字符 {!r} 没有可用于补齐的 LabelMe 种子".format(character)
            )

        rng.shuffle(character_seeds)
        rng.shuffle(contexts)
        for qualifier, noun, position in contexts:
            for seed in character_seeds:
                if position == "suffix":
                    text = "{} {} {}".format(qualifier, noun, seed)
                else:
                    text = "{} {} {}".format(seed, qualifier, noun)
                text = normalize_text(text)
                if not min_text_length <= len(text) <= max_text_length:
                    continue
                if text in used_texts:
                    continue

                character_counts = Counter(text)
                record = {
                    "target_char": character,
                    "text": text,
                    "source": "coverage_supplement",
                    "generated": True,
                    "seed_text": seed,
                    "lmdb_count_before": training_counts[character],
                }
                records.append(record)
                added_records.append(record)
                used_texts.add(text)
                projected_counts.update(character_counts)
                supplement_counts.update(character_counts)
                if projected_counts[character] >= min_count:
                    break
            if projected_counts[character] >= min_count:
                break

        if projected_counts[character] < min_count:
            raise RuntimeError(
                "字符 {!r} 补齐后仍只有 {} 次，目标为 {} 次".format(
                    character,
                    projected_counts[character],
                    min_count,
                )
            )

    remaining_undercovered = {
        character: projected_counts[character]
        for character in annotation_characters
        if projected_counts[character] < min_count
    }
    if remaining_undercovered:
        raise RuntimeError(
            "仍有 {} 个字符未达到覆盖阈值".format(len(remaining_undercovered))
        )

    details = []
    for character in sorted(
        missing_characters | low_coverage_characters,
        key=lambda item: (training_counts[item], item),
    ):
        details.append(
            {
                "character": character,
                "codepoint": "U+{:04X}".format(ord(character)),
                "lmdb_count": training_counts[character],
                "labelme_count": annotation_counts[character],
                "corpus_count_before": corpus_counts_before[character],
                "supplement_count": supplement_counts[character],
                "projected_count": projected_counts[character],
            }
        )

    report = {
        "min_coverage_count": min_count,
        "annotation_unique_non_whitespace_count": len(annotation_characters),
        "lmdb_missing_unique_count": len(missing_characters),
        "lmdb_low_coverage_unique_count": len(low_coverage_characters),
        "characters_requiring_supplement_count": len(target_characters),
        "generated_text_count": len(added_records),
        "remaining_undercovered_count": 0,
        "character_details": details,
    }
    return added_records, report


def validate_records(records, min_text_length, max_text_length):
    for record in records:
        text = record["text"]
        if not min_text_length <= len(text) <= max_text_length:
            raise ValueError("文本长度不在限制范围内: {}".format(text))
        if text != normalize_text(text):
            raise ValueError("文本未满足规范化约束: {}".format(text))


def write_outputs(corpus_dir: Path, records, summary, report):
    text_path = corpus_dir / "korean_missing_chars_corpus.txt"
    jsonl_path = corpus_dir / "korean_missing_chars_corpus.jsonl"
    summary_path = corpus_dir / "korean_missing_chars_summary.json"
    report_path = corpus_dir / "korean_coverage_supplement_report.json"

    source_counts = Counter(record["source"] for record in records)
    summary["total_text_count"] = len(records)
    summary["min_text_length"] = report["min_text_length"]
    summary["max_text_length"] = report["max_text_length"]
    summary["all_texts_unique"] = len(records) == len(
        {record["text"] for record in records}
    )
    summary["source_counts"] = dict(source_counts)
    summary["coverage_supplement"] = {
        "min_coverage_count": report["min_coverage_count"],
        "generated_text_count": source_counts["coverage_supplement"],
        "latest_generated_text_count": report["generated_text_count"],
        "report": str(report_path.resolve()),
    }
    summary["text_cleanup"] = {
        "allowed_punctuation": "！？",
        "changed_text_count": report["changed_text_count"],
        "discarded_short_text_count": report["discarded_short_text_count"],
        "strip_edge_whitespace": True,
    }

    temporary_paths = {
        text_path: text_path.with_suffix(text_path.suffix + ".tmp"),
        jsonl_path: jsonl_path.with_suffix(jsonl_path.suffix + ".tmp"),
        summary_path: summary_path.with_suffix(summary_path.suffix + ".tmp"),
        report_path: report_path.with_suffix(report_path.suffix + ".tmp"),
    }
    with temporary_paths[text_path].open("w", encoding="utf-8") as file:
        for record in records:
            file.write(record["text"] + "\n")
    with temporary_paths[jsonl_path].open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    with temporary_paths[summary_path].open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")
    with temporary_paths[report_path].open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")

    for final_path, temporary_path in temporary_paths.items():
        temporary_path.replace(final_path)


def main():
    parser = argparse.ArgumentParser(
        description="基于完整 LabelMe 字符需求补齐低覆盖训练字符",
    )
    parser.add_argument("statistics_json", type=Path, help="LMDB 字符统计 JSON")
    parser.add_argument("corpus_dir", type=Path, help="待补齐的 corpus 目录")
    parser.add_argument("--min-count", type=int, default=200)
    parser.add_argument("--min-text-length", type=int, default=2)
    parser.add_argument("--max-text-length", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.min_count <= 0:
        parser.error("--min-count 必须大于 0")
    if args.min_text_length < 2:
        parser.error("--min-text-length 不能小于 2")
    if args.min_text_length > args.max_text_length:
        parser.error("--min-text-length 不能大于 --max-text-length")
    if args.max_text_length < 8:
        parser.error("--max-text-length 不能小于 8")

    statistics_path = args.statistics_json.resolve()
    corpus_dir = args.corpus_dir.resolve()
    if not statistics_path.is_file():
        parser.error("字符统计 JSON 不存在: {}".format(statistics_path))
    if not corpus_dir.is_dir():
        parser.error("corpus 目录不存在: {}".format(corpus_dir))

    training_counts = load_training_counts(statistics_path)
    records, summary = load_corpus(corpus_dir)
    records, changed_text_count, discarded_short_text_count = clean_records(
        records,
        args.min_text_length,
    )
    annotation_counts, annotation_records = collect_annotation_profile(records)
    added_records, report = supplement_records(
        records,
        training_counts,
        annotation_counts,
        annotation_records,
        args.min_count,
        args.min_text_length,
        args.max_text_length,
        args.seed,
    )
    report["min_text_length"] = args.min_text_length
    report["max_text_length"] = args.max_text_length
    report["allowed_punctuation"] = "！？"
    report["changed_text_count"] = changed_text_count
    report["discarded_short_text_count"] = discarded_short_text_count
    validate_records(records, args.min_text_length, args.max_text_length)
    write_outputs(corpus_dir, records, summary, report)

    print("Coverage supplement complete: {}".format(corpus_dir))
    print("LMDB missing characters: {}".format(report["lmdb_missing_unique_count"]))
    print(
        "LMDB low-coverage characters: {}".format(
            report["lmdb_low_coverage_unique_count"]
        )
    )
    print("Generated texts: {}".format(len(added_records)))
    print("Cleaned texts: {}".format(changed_text_count))
    print("Discarded short texts: {}".format(discarded_short_text_count))
    print("Remaining undercovered characters: 0")


if __name__ == "__main__":
    main()
