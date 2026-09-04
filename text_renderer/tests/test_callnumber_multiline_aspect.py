from text_renderer.corpus.call_number_generator import PatternConfig
from text_renderer.pipelines.callnumber_multiline_lmdb import (
    default_patterns_for_direction,
    filter_patterns_by_aspect,
    layout_aspect_from_lengths,
    layout_aspect_from_lines,
)


def test_layout_aspect_row_column_duality():
    assert layout_aspect_from_lengths([1, 1, 1], "row") == 3.0
    assert abs(layout_aspect_from_lengths([1, 1, 1], "column") - 1 / 3) < 1e-9
    assert abs(layout_aspect_from_lengths([3, 3, 0], "row") - 2 / 3) < 1e-9
    assert layout_aspect_from_lengths([3, 3, 0], "column") == 1.5


def test_layout_aspect_from_lines_matches_lengths():
    assert layout_aspect_from_lines(["AB", "CD", "EF"], "row") == 1.5
    assert abs(layout_aspect_from_lines(["AB", "CD", "EF"], "column") - 2 / 3) < 1e-9


def test_default_patterns_are_direction_specific():
    row = default_patterns_for_direction("row")
    column = default_patterns_for_direction("column")
    assert row != column
    assert all(
        layout_aspect_from_lengths(item["lengths"], "row") <= 1.2 + 1e-9 for item in row
    )
    assert all(
        layout_aspect_from_lengths(item["lengths"], "column") <= 1.2 + 1e-9
        for item in column
    )


def test_filter_patterns_by_aspect_drops_invalid_for_direction():
    patterns = [
        PatternConfig(lengths=(1, 1, 1), weight=1),
        PatternConfig(lengths=(3, 3, 0), weight=1),
    ]
    kept_row, dropped_row = filter_patterns_by_aspect(patterns, 1.2, "row")
    assert [p.lengths for p in kept_row] == [(3, 3, 0)]
    assert [p.lengths for p, _ in dropped_row] == [(1, 1, 1)]

    kept_col, dropped_col = filter_patterns_by_aspect(patterns, 1.2, "column")
    assert [p.lengths for p in kept_col] == [(1, 1, 1)]
    assert [p.lengths for p, _ in dropped_col] == [(3, 3, 0)]
