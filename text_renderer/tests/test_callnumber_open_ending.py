import random

from text_renderer.corpus.call_number_generator import (
    BuiltInDistributionCallNumberGenerator,
    PatternConfig,
    _BUILTIN_LINE1_TEMPLATES,
    _OPEN_LINE_ENDING_CHARS,
    build_call_number_generator,
    ends_with_open_symbol,
    forbid_open_line_ending,
)


def test_forbid_open_line_ending_replaces_trailing_dot():
    rng = random.Random(0)
    assert forbid_open_line_ending("I123.", rng)[-1] not in _OPEN_LINE_ENDING_CHARS
    assert forbid_open_line_ending("I12.3", rng) == "I12.3"
    assert not ends_with_open_symbol("I12.3")
    assert ends_with_open_symbol("I12.")


def test_line1_templates_do_not_end_with_open_symbol():
    for templates in _BUILTIN_LINE1_TEMPLATES.values():
        for template in templates:
            assert template[-1] not in _OPEN_LINE_ENDING_CHARS, template


def test_learned_generator_never_ends_line_with_open_symbol():
    generator = BuiltInDistributionCallNumberGenerator(
        patterns=[
            PatternConfig(lengths=(5, 4, 4), weight=1),
            PatternConfig(lengths=(7, 6, 0), weight=1),
            PatternConfig(lengths=(10, 8, 4), weight=1),
        ],
        rng=random.Random(42),
    )
    for _ in range(500):
        for line in generator.generate():
            assert line
            assert line[-1] not in _OPEN_LINE_ENDING_CHARS, line


def test_grammar_generator_never_ends_line_with_open_symbol():
    generator = build_call_number_generator(
        "grammar",
        patterns=[PatternConfig(lengths=(5, 4, 4), weight=1)],
        rng=random.Random(7),
    )
    for _ in range(100):
        for line in generator.generate():
            assert line[-1] not in _OPEN_LINE_ENDING_CHARS, line
