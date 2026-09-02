"""Find fonts whose uppercase and lowercase Latin letters are indistinguishable."""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from fontTools.pens.recordingPen import RecordingPen
from fontTools.ttLib import TTCollection, TTFont


FONT_EXTENSIONS = {".ttf", ".otf", ".ttc"}


def _outline(font: TTFont, glyph_name: str):
    try:
        pen = RecordingPen()
        font.getGlyphSet()[glyph_name].draw(pen)
        return tuple(pen.value), tuple(font["hmtx"][glyph_name][:2])
    except Exception:
        return None


def _face_count(path: Path) -> int:
    if path.suffix.lower() != ".ttc":
        return 1
    collection = TTCollection(str(path))
    try:
        return len(collection.fonts)
    finally:
        collection.close()


def _face_pair_counts(path: Path, face_index: int) -> Tuple[int, int, int, List[str]]:
    font = TTFont(str(path), ignoreDecompileErrors=True, fontNumber=face_index)
    try:
        cmap = {}
        for table in font["cmap"].tables:
            if table.isUnicode():
                cmap.update(table.cmap)

        pairs = same_glyph = same_outline = 0
        same_pairs = []
        for offset in range(26):
            upper = ord("A") + offset
            lower = ord("a") + offset
            if upper not in cmap or lower not in cmap:
                continue
            pairs += 1
            upper_glyph, lower_glyph = cmap[upper], cmap[lower]
            if upper_glyph == lower_glyph:
                same_glyph += 1
                same_pairs.append(chr(upper) + chr(lower))
            elif _outline(font, upper_glyph) == _outline(font, lower_glyph):
                same_outline += 1
                same_pairs.append(chr(upper) + chr(lower))
        return pairs, same_glyph, same_outline, same_pairs
    finally:
        font.close()


def scan_directory(directory: Path, min_pairs: int = 5, recursive: bool = False) -> List[Dict]:
    """Return fonts with at least ``min_pairs`` indistinguishable letter pairs."""
    if min_pairs < 1 or min_pairs > 26:
        raise ValueError("min_pairs must be between 1 and 26")
    iterator: Iterable[Path] = directory.rglob("*") if recursive else directory.iterdir()
    fonts = sorted(
        path for path in iterator if path.is_file() and path.suffix.lower() in FONT_EXTENSIONS
    )
    results = []
    for path in fonts:
        try:
            faces = []
            for face_index in range(_face_count(path)):
                pairs, same_glyph, same_outline, same_pairs = _face_pair_counts(path, face_index)
                faces.append(
                    {
                        "face": face_index,
                        "available_pairs": pairs,
                        "same_glyph_pairs": same_glyph,
                        "same_outline_pairs": same_outline,
                        "same_pairs": same_pairs,
                    }
                )
            if any(
                face["same_glyph_pairs"] + face["same_outline_pairs"] >= min_pairs
                for face in faces
            ):
                results.append({"path": str(path), "faces": faces})
        except Exception as error:
            results.append({"path": str(path), "error": str(error), "faces": []})
    return results


def move_results(results: List[Dict], destination: Path) -> None:
    """Move scan results, refusing collisions before changing any file."""
    destination.mkdir(parents=True, exist_ok=True)
    source_paths = [Path(item["path"]) for item in results if "error" not in item]
    collisions = [destination / path.name for path in source_paths if (destination / path.name).exists()]
    if collisions:
        raise FileExistsError("Destination already contains: " + ", ".join(map(str, collisions)))
    for path in source_paths:
        shutil.move(str(path), str(destination / path.name))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing font files")
    parser.add_argument("--min-pairs", type=int, default=5, help="Minimum indistinguishable pairs (default: 5)")
    parser.add_argument("--recursive", action="store_true", help="Scan subdirectories recursively")
    parser.add_argument("--move-to", type=Path, help="Move matching fonts to this directory")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Print JSON instead of a table")
    args = parser.parse_args()

    if not args.directory.is_dir():
        parser.error(f"directory does not exist: {args.directory}")
    if args.move_to and args.move_to.resolve() == args.directory.resolve():
        parser.error("--move-to must differ from the input directory")

    results = scan_directory(args.directory, args.min_pairs, args.recursive)
    if args.move_to:
        move_results(results, args.move_to)

    if args.as_json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        action = "matched" if not args.move_to else "moved"
        print(f"{action}: {len(results)}")
        for item in results:
            if "error" in item:
                print(f"ERROR\t{item['path']}\t{item['error']}")
                continue
            best = max(
                item["faces"],
                key=lambda face: face["same_glyph_pairs"] + face["same_outline_pairs"],
            )
            score = best["same_glyph_pairs"] + best["same_outline_pairs"]
            print(f"{score:2d}\t{item['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
