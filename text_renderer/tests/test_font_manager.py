from pathlib import Path

from text_renderer.font_manager import validate_font_file


FONT_PATH = Path(__file__).resolve().parents[2] / "lmdbs/lmdbs/simhei.ttf"


def test_validate_font_reports_coverage_and_missing_characters():
    result = validate_font_file(FONT_PATH, ["中", "\U0010ffff"], min_coverage=0.5)

    assert result["faces"][0]["covered_count"] == 1
    assert result["faces"][0]["missing_characters"] == ["\U0010ffff"]
    assert result["faces"][0]["coverage"] == 0.5


def test_validate_font_rejects_insufficient_language_coverage():
    result = validate_font_file(FONT_PATH, ["中", "\U0010ffff"], min_coverage=1.0)

    assert result["safe"] is False
