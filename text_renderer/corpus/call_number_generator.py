"""Grammar and embedded-distribution strategies for unbounded call numbers."""

from __future__ import annotations

import random
import string
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from numbers import Real


# Characters that must not end a call-number line (open / unfinished marks).
_OPEN_LINE_ENDING_CHARS = frozenset(".-:=(/")


def ends_with_open_symbol(text: str) -> bool:
    """Return True if text ends with an unfinished punctuation mark."""
    return bool(text) and text[-1] in _OPEN_LINE_ENDING_CHARS


def forbid_open_line_ending(text: str, rng=None) -> str:
    """Replace any trailing open-ended symbols with digits."""
    if not text or not ends_with_open_symbol(text):
        return text
    rng = rng or random.Random()
    chars = list(text)
    while chars and chars[-1] in _OPEN_LINE_ENDING_CHARS:
        chars[-1] = rng.choice(string.digits)
    return "".join(chars)


@dataclass(frozen=True)
class PatternConfig:
    """One call-number layout pattern and its sampling weight."""

    lengths: tuple[int, ...]
    weight: float = 1.0

    def __post_init__(self):
        if not isinstance(self.lengths, tuple) or not 1 <= len(self.lengths) <= 3:
            raise ValueError("lengths must contain 1 to 3 line lengths")

        for line_index, length in enumerate(self.lengths):
            if isinstance(length, bool) or not isinstance(length, int):
                raise ValueError("line lengths must be integers")

            if length < 0:
                raise ValueError(
                    f"line {line_index + 1} length must be non-negative, "
                    f"got {length}"
                )

        if not any(self.lengths):
            raise ValueError("at least one line length must be greater than zero")

        if (
            isinstance(self.weight, bool)
            or not isinstance(self.weight, Real)
            or self.weight < 0
        ):
            raise ValueError("weight must be a non-negative number")


class CallNumberGenerator(ABC):
    """Common strategy interface for infinite call-number generation."""

    @abstractmethod
    def generate(self):
        """Generate one call number as a list of non-empty lines."""

    @abstractmethod
    def summary(self):
        """Return a JSON-serializable description of the active strategy."""

    def __iter__(self):
        return self

    def __next__(self):
        return self.generate()


def _validate_pattern_configs(patterns):
    if not isinstance(patterns, (list, tuple)) or not patterns:
        raise ValueError("patterns must be a non-empty list")

    if not all(isinstance(pattern, PatternConfig) for pattern in patterns):
        raise ValueError("every pattern must be a PatternConfig instance")

    if not any(pattern.weight > 0 for pattern in patterns):
        raise ValueError("at least one pattern weight must be positive")

    return tuple(patterns)


class InfiniteCallNumberGenerator(CallNumberGenerator):
    """Generate call numbers from configurable grammar patterns.

    Grammar:

    line 1:
        1-2 uppercase letters followed by enough digits to reach the
        configured visible length

    line 2:
        the configured number of digits

    line 3:
        the configured number of visible characters

        length 1:
            digit only

        length 2:
            may occasionally start with "v"

        length 3 or greater:
            may occasionally start with "v" or ".v"

    A zero length omits that line while preserving the grammar of the other
    configured line positions.
    """

    def __init__(
        self,
        patterns: list[PatternConfig],
        line3_prefix_probability: float = 0.20,
        line3_dot_v_probability: float = 0.50,
        rng=None,
    ):
        self._rng = rng or random.Random()

        self._patterns = self._validate_patterns(patterns)

        self.line3_prefix_probability = self._validate_probability(
            "line3_prefix_probability",
            line3_prefix_probability,
        )

        self.line3_dot_v_probability = self._validate_probability(
            "line3_dot_v_probability",
            line3_dot_v_probability,
        )

    @staticmethod
    def _validate_patterns(patterns):
        return _validate_pattern_configs(patterns)

    @staticmethod
    def _validate_probability(name, value):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not 0 <= value <= 1
        ):
            raise ValueError(
                f"{name} must be between 0 and 1"
            )

        return float(value)

    def _random_chars(self, alphabet, count):
        return "".join(
            self._rng.choice(alphabet)
            for _ in range(count)
        )

    def _line1(self, total_length):
        feasible_letter_counts = [
            letter_count
            for letter_count in (1, 2)
            if letter_count <= total_length
        ]

        letter_count = self._rng.choice(
            feasible_letter_counts
        )

        digit_count = total_length - letter_count

        return (
            self._random_chars(
                string.ascii_uppercase,
                letter_count,
            )
            + self._random_chars(
                string.digits,
                digit_count,
            )
        )

    def _line2(self, length):
        return self._random_chars(
            string.digits,
            length,
        )

    def _line3(self, total_length):
        prefix = ""

        if (
            total_length >= 2
            and self._rng.random()
            < self.line3_prefix_probability
        ):
            if (
                total_length >= 3
                and self._rng.random()
                < self.line3_dot_v_probability
            ):
                prefix = ".v"

            else:
                prefix = "v"

        digit_count = total_length - len(prefix)

        return (
            prefix
            + self._random_chars(
                string.digits,
                digit_count,
            )
        )

    def _select_pattern(self):
        return self._rng.choices(
            self._patterns,
            weights=[
                pattern.weight
                for pattern in self._patterns
            ],
            k=1,
        )[0]

    def generate(self):
        """Generate one call number."""

        pattern = self._select_pattern()

        lengths = pattern.lengths

        line_generators = (
            self._line1,
            self._line2,
            self._line3,
        )

        lines = [
            generator(length)
            for generator, length in zip(line_generators, lengths)
            if length > 0
        ]

        return lines

    def summary(self):
        return {
            "strategy": "grammar",
            "grammar": {
                "line1": "[A-Z]{1,2}[0-9]*, exact configured length",
                "line2": "[0-9]+, exact configured length",
                "line3": (
                    "[0-9]+, exact configured length; "
                    "length 2 may start with v; "
                    "length 3+ may start with v or .v"
                ),
            },
            "patterns": [
                {
                    "lengths": list(pattern.lengths),
                    "weight": pattern.weight,
                }
                for pattern in self._patterns
            ],
            "zero_length_semantics": "the configured line is omitted",
            "line3_prefix_probability":
                self.line3_prefix_probability,
            "line3_dot_v_probability":
                self.line3_dot_v_probability,
        }

_BUILTIN_LINE_COUNT_COUNTER = Counter({3: 115429, 2: 3909})
_BUILTIN_LINE_LENGTH_COUNTERS = [
    Counter({7: 64107, 6: 36067, 4: 11172, 3: 4473, 8: 2044, 11: 368,
             10: 940, 9: 81, 5: 25, 2: 59, 12: 2}),
    Counter({7: 46206, 6: 31525, 5: 12010, 8: 10419, 4: 3722,
             9: 9137, 10: 3578, 12: 391, 3: 103, 17: 3, 16: 5,
             11: 1991, 15: 43, 13: 196, 2: 1, 14: 8}),
    Counter({4: 113432, 5: 1982, 3: 11, 10: 4}),
]

# Character masks distilled from the workbook.  These are deliberately
# stored as code constants: generation never opens the source workbook.
_BUILTIN_LINE1_TEMPLATES = {
    2: Counter({"LD": 1}),
    3: Counter({"LDD": 1}),
    4: Counter({"LDDD": 1}),
    # Trailing "." is forbidden (open-ended symbol); keep decimals mid-string only.
    5: Counter({"LD-DD": 3, "LLDDD": 1, "LD.DD": 1}),
    6: Counter({"LDDD.D": 4, "LDDDDD": 1}),
    7: Counter({"LDDD.DD": 7, "LDDDDDD": 1}),
    8: Counter({"LDDD.DDD": 8, "LDDDDDDD": 1}),
    9: Counter({"LDDD.D-DD": 4, "LDDD.DDDD": 4}),
    10: Counter({"LDDD.DDD.D": 4, "LDDD.DD-DD": 3}),
    11: Counter({"LDDD.DDD.DD": 4}),
    12: Counter({"LDDD.DDD.DDD": 4}),
}
_BUILTIN_LINE2_TEMPLATES = {
    2: Counter({"DD": 1}),
    3: Counter({"D=D": 1}),
    4: Counter({"DD=D": 2, "DD()": 1, "D=DD": 1}),
    5: Counter({"DDD=D": 2, "DD=DD": 1, "DDD()": 1}),
    6: Counter({"DDDD=D": 2, "DDD(D)": 1, "DDD=DD": 1}),
    7: Counter({"DDDD(D)": 2, "DDDD=DD": 1, "DDD=D-D": 1}),
    8: Counter({"DDDD(DD)": 2, "DDDD=D-D": 1, "DDD=D-DD": 1}),
    9: Counter({"DDDD=D(D)": 2, "DDDD-D(D)": 1, "DDDD(DD:D)": 1}),
    10: Counter({"DDDD(DDDD)": 2, "DDDD=D-DDD": 1}),
    11: Counter({"DDDD(DDDDD)": 1}),
    12: Counter({"DDDD(DDDDDD)": 1}),
    13: Counter({"DDDDD(DDDDD)": 1}),
    14: Counter({"DDDDD(DDDDDD)": 1}),
    15: Counter({"DDDDDD(DDDDDDD)": 1}),
    16: Counter({"DDDDDD(DDDDDDDD)": 1}),
    17: Counter({"DDDDDDD(DDDDDDDD)": 1}),
}
_BUILTIN_LINE1_LETTERS = Counter({
    "I": 102588, "K": 15957, "H": 216, "J": 140, "G": 106,
    "A": 65, "B": 55, "C": 50, "D": 45, "E": 40, "F": 35,
    "L": 30, "M": 25, "N": 25, "O": 20, "P": 20, "Q": 10,
    "R": 20, "S": 20, "T": 20, "U": 15, "V": 15, "W": 10,
    "X": 8, "Y": 8, "Z": 8,
})
_BUILTIN_YEAR_DECADES = Counter({2000: 2, 2010: 5, 2020: 2})


class BuiltInDistributionCallNumberGenerator(CallNumberGenerator):
    """Generate call numbers from a baked-in distribution model."""

    def __init__(
        self,
        patterns=None,
        line1_letter_mode="uniform",
        rng=None,
    ):
        self._rng = rng or random.Random()
        self._patterns = (
            None
            if patterns is None
            else _validate_pattern_configs(patterns)
        )
        self.line1_letter_mode = self._validate_line1_letter_mode(
            line1_letter_mode
        )
        self._line_count_counter = _BUILTIN_LINE_COUNT_COUNTER.copy()
        self._line_length_counters = [counter.copy() for counter in _BUILTIN_LINE_LENGTH_COUNTERS]
        self._template_counters = [
            {length: counter.copy() for length, counter in templates.items()}
            for templates in (_BUILTIN_LINE1_TEMPLATES, _BUILTIN_LINE2_TEMPLATES)
        ]
        self._line1_letters = _BUILTIN_LINE1_LETTERS.copy()
        self._line1_prefix_lengths = Counter({1: 99, 2: 1})
        self._line1_decimal_probability = 0.877
        self._year_decades = _BUILTIN_YEAR_DECADES.copy()
        self._year_min = 2005
        self._year_max = 2024

    @staticmethod
    def _validate_line1_letter_mode(mode):
        if mode not in {"uniform", "learned"}:
            raise ValueError(
                "line1_letter_mode must be 'uniform' or 'learned'"
            )
        return mode

    def _weighted_choice(self, counter):
        return self._rng.choices(
            list(counter.keys()),
            weights=list(counter.values()),
            k=1,
        )[0]

    def _sample_template(self, line_index, length):
        if line_index >= len(self._template_counters):
            return None
        templates = self._template_counters[line_index].get(length)
        if not templates:
            return None
        return self._weighted_choice(templates)

    def _sample_line1_letter(self):
        if self.line1_letter_mode == "learned" and self._line1_letters:
            return self._weighted_choice(self._line1_letters)
        return self._rng.choice(string.ascii_uppercase)

    def _render_template(self, template, line_index):
        output = []
        for character in template:
            if character == "D":
                output.append(self._rng.choice(string.digits))
            elif character == "L":
                output.append(
                    self._sample_line1_letter()
                    if line_index == 0
                    else self._rng.choice(string.ascii_uppercase)
                )
            else:
                output.append(character)
        return "".join(output)

    def _generate_first_line(self, length):
        template = self._sample_template(0, length)
        if template:
            return self._render_template(template, line_index=0)
        if length == 1:
            return self._sample_line1_letter()

        prefix_length = 1
        if self._line1_prefix_lengths:
            prefix_length = min(
                self._weighted_choice(self._line1_prefix_lengths),
                length,
            )
        prefix = "".join(
            self._sample_line1_letter()
            for _ in range(prefix_length)
        )
        body_length = length - prefix_length
        body = [self._rng.choice(string.digits) for _ in range(body_length)]
        # Keep "." mid-body only; never first or last char of the numeric body.
        if (
            body_length >= 3
            and self._rng.random() < self._line1_decimal_probability
        ):
            body[self._rng.randint(1, body_length - 2)] = "."
        return prefix + "".join(body)

    def _generate_second_line(self, length):
        template = self._sample_template(1, length)
        if template:
            return self._render_template(template, line_index=1)
        return "".join(self._rng.choice(string.digits) for _ in range(length))

    def _generate_year(self):
        decade = self._weighted_choice(self._year_decades)
        year = decade + self._rng.randint(0, 9)
        if self._year_min is not None:
            year = max(self._year_min, year)
        if self._year_max is not None:
            year = min(self._year_max, year)
        return str(year)

    def _generate_third_line(self, length):
        if length == 4 and self._year_decades:
            return self._generate_year()
        return "".join(self._rng.choice(string.digits) for _ in range(length))

    def _generate_line(self, line_index, length):
        if line_index == 0:
            line = self._generate_first_line(length)
        elif line_index == 1:
            line = self._generate_second_line(length)
        else:
            line = self._generate_third_line(length)
        return forbid_open_line_ending(line, self._rng)

    def _sample_lengths(self):
        line_count = self._weighted_choice(self._line_count_counter)
        return tuple(
            self._weighted_choice(self._line_length_counters[line_index])
            for line_index in range(line_count)
        )

    def generate(self):
        if self._patterns is None:
            lengths = self._sample_lengths()
        else:
            pattern = self._rng.choices(
                self._patterns,
                weights=[item.weight for item in self._patterns],
                k=1,
            )[0]
            lengths = pattern.lengths

        return [
            self._generate_line(line_index, length)
            for line_index, length in enumerate(lengths)
            if length > 0
        ]

    def summary(self):
        top_templates = []
        for line_index, length_counter in enumerate(self._template_counters):
            for length, templates in length_counter.items():
                top_templates.append({
                    "line": line_index + 1,
                    "length": length,
                    "templates": templates.most_common(3),
                })
        return {
            "strategy": "builtin_distribution",
            "source": "embedded call-number distribution model",
            "line_count_distribution": dict(self._line_count_counter),
            "line_length_distributions": [
                dict(counter) for counter in self._line_length_counters
            ],
            "year_range": [self._year_min, self._year_max],
            "line3_policy": (
                "numeric; four-character values sample learned publication-year decades"
            ),
            "line1_letter_mode": self.line1_letter_mode,
            "patterns": None if self._patterns is None else [
                {"lengths": list(pattern.lengths), "weight": pattern.weight}
                for pattern in self._patterns
            ],
            "template_examples": top_templates[:30],
            "sampling": "model-based infinite sampling; no external files are read",
        }


def build_call_number_generator(
    strategy,
    *,
    patterns=None,
    line3_prefix_probability=0.20,
    line3_dot_v_probability=0.50,
    line1_letter_mode="uniform",
    rng=None,
):
    """Construct one configured call-number generation strategy."""
    if strategy == "grammar":
        return InfiniteCallNumberGenerator(
            patterns=patterns,
            line3_prefix_probability=line3_prefix_probability,
            line3_dot_v_probability=line3_dot_v_probability,
            rng=rng,
        )
    if strategy == "learned":
        return BuiltInDistributionCallNumberGenerator(
            patterns=patterns,
            line1_letter_mode=line1_letter_mode,
            rng=rng,
        )
    raise ValueError(
        f"unknown call-number generation strategy: {strategy!r}"
    )


if __name__ == "__main__":

    generator = InfiniteCallNumberGenerator(
        patterns=[
            PatternConfig(
                lengths=(2, 2, 2),
                weight=0.5,
            ),
            PatternConfig(
                lengths=(1, 1, 1),
                weight=0.2,
            ),
            PatternConfig(
                lengths=(3, 3, 3),
                weight=0.3,
            ),
        ],
        line3_prefix_probability=0.25,
        line3_dot_v_probability=0.5,
    )

    for _ in range(10):
        print(generator.generate())
