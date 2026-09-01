"""Check whether a font safely covers a language character set."""

import argparse
import json
from pathlib import Path

from text_renderer.font_manager import validate_font_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("font", type=Path, help="TTF, OTF, or TTC file")
    parser.add_argument(
        "--charset",
        type=Path,
        help="UTF-8 file containing one required character per line",
    )
    parser.add_argument("--min-coverage", type=float, default=0.95)
    args = parser.parse_args()

    required = ()
    if args.charset:
        required = args.charset.read_text(encoding="utf-8-sig").splitlines()
    result = validate_font_file(args.font, required, args.min_coverage)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["safe"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
