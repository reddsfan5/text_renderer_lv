from argparse import Namespace
from pathlib import Path

import pytest

from text_renderer.cli import apply_cli_overrides, generate_samples, parse_args
from text_renderer.config import (
    AppCfg,
    FixedPerspectiveTransformCfg,
    GeneratorCfg,
    RenderCfg,
)
from text_renderer.config.yaml_config import (
    _apply_length_policy,
    _build_v2_job_roots,
    _clean_line,
    _load_corpus_items,
    _load_v1_root,
)
from text_renderer.corpus.corpus import Corpus
from text_renderer.runner import iter_task_batch_sizes


def test_clean_line_keeps_only_allowed_punctuation():
    text = _clean_line(
        "  한글,  문장！？  ",
        charset=None,
        strip=True,
        normalize_spaces=True,
        remove_punctuation=True,
        allowed_punctuation="！？",
    )
    assert text == "한글 문장！？"


def test_split_long_text_respects_min_and_max_length():
    assert _apply_length_policy("가나다라마", 2, 2, "split") == ["가나", "다라"]


def test_corpus_loader_caps_items_to_generation_capacity(tmp_path):
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text(
        "\n".join(f"item-{index}" for index in range(10)), encoding="utf-8"
    )

    items = _load_corpus_items(
        {
            "path": str(corpus_path),
            "min_length": 1,
            "max_length": 20,
            "capacity_multiplier": 1.5,
            "remove_punctuation": False,
        },
        num_images=2,
    )

    assert items == ["item-0", "item-1", "item-2"]


def test_fixed_perspective_uses_configured_angles():
    cfg = FixedPerspectiveTransformCfg(x=3, y=4, z=5)
    assert cfg.get_xyz() == (3, 4, 5)


def test_corpus_punctuation_policy_keeps_configured_characters():
    corpus = Corpus.__new__(Corpus)
    corpus.cfg = type(
        "Cfg",
        (),
        {"remove_punctuation": True, "allowed_punctuation": "！？"},
    )()
    assert corpus.remove_punctuation("韩文,！@？") == "韩文！？"


def test_cli_values_override_loaded_config():
    corpus = type("Corpus", (), {"texts": ["a"]})()
    app_cfg = AppCfg(
        generators=[
            GeneratorCfg(
                num_image=1,
                save_dir=Path("old"),
                render_cfg=RenderCfg(corpus=corpus),
            )
        ]
    )
    args = Namespace(
        num_images=3,
        num_processes=0,
        log_period=2,
        dataset="lmdb",
        output_dir="new",
    )

    apply_cli_overrides(app_cfg, args)

    assert app_cfg.generators[0].num_image == 3
    assert app_cfg.generators[0].save_dir == Path("new")
    assert len(corpus.texts) >= 3
    assert app_cfg.runtime.num_processes == 0
    assert app_cfg.storage.dataset == "lmdb"


def test_cli_sample_count_is_non_negative():
    assert parse_args(["--sample-count", "0"]).sample_count == 0
    with pytest.raises(ValueError, match="sample-count"):
        apply_cli_overrides(
            AppCfg(generators=[]),
            Namespace(
                num_images=None,
                num_processes=None,
                log_period=None,
                sample_count=-1,
                dataset=None,
                output_dir=None,
            ),
        )


def test_generate_samples_calls_visualizer_for_each_lmdb_generator(monkeypatch, tmp_path):
    calls = []

    monkeypatch.setattr(
        "text_renderer.cli._visualize_lmdb",
        lambda path, count: calls.append((path, count)),
    )
    app_cfg = AppCfg(
        generators=[
            GeneratorCfg(1, tmp_path / "one", RenderCfg(corpus=[]), name="one"),
            GeneratorCfg(1, tmp_path / "two", RenderCfg(corpus=[]), name="two"),
        ]
    )
    app_cfg.storage.dataset = "lmdb"

    generate_samples(app_cfg, sample_count=3)

    assert calls == [(tmp_path / "one", 3), (tmp_path / "two", 3)]


def test_v2_language_profile_keeps_corpus_font_and_text_together():
    root = {
        "version": 2,
        "generator": {
            "active_languages": ["korean"],
            "num_images": 10,
            "output_dir": "output/{language}",
        },
        "background": {"dir": "background"},
        "languages": {
            "korean": {
                "corpus": {"path": "korean.txt"},
                "font": {"dir": "korean-fonts", "size": [30, 36]},
            },
            "tibetan": {
                "corpus": {"path": "tibetan.txt", "min_length": 2},
                "font": {
                    "active": "book",
                    "variants": {
                        "book": {"dir": "tibetan-fonts", "size": [28, 34]},
                        "display": {
                            "dir": "tibetan-display-fonts",
                            "size": [30, 38],
                        },
                    },
                },
                "text": {
                    "orientation": "horizontal",
                    "rotate_characters": False,
                },
            },
        },
    }

    jobs = _build_v2_job_roots(root, ["tibetan"])

    assert len(jobs) == 1
    language, job = jobs[0]
    assert language == "tibetan"
    assert job["corpus"]["path"] == "tibetan.txt"
    assert job["corpus"]["orientation"] == "horizontal"
    assert job["font"]["dir"] == "tibetan-fonts"
    assert job["generator"]["output_dir"] == "output/tibetan"


def test_repeated_language_cli_argument_selects_multiple_profiles():
    args = parse_args(["--language", "chinese", "--language", "korean"])
    assert args.selected_languages == ["chinese", "korean"]


def test_output_override_uses_language_names_for_multiple_generators():
    app_cfg = AppCfg(
        generators=[
            GeneratorCfg(1, Path("old-a"), RenderCfg(corpus=[]), name="chinese"),
            GeneratorCfg(1, Path("old-b"), RenderCfg(corpus=[]), name="korean"),
        ]
    )
    args = Namespace(
        num_images=None,
        num_processes=None,
        log_period=None,
        dataset=None,
        output_dir="output",
    )

    apply_cli_overrides(app_cfg, args)

    assert app_cfg.generators[0].save_dir == Path("output/chinese")
    assert app_cfg.generators[1].save_dir == Path("output/korean")


def test_task_batches_cover_every_generation_attempt():
    assert list(iter_task_batch_sizes(18, 8)) == [8, 8, 2]
    assert list(iter_task_batch_sizes(3, 8)) == [3]


@pytest.mark.parametrize("field", ["queue_size", "task_batch_size"])
def test_runtime_performance_values_must_be_positive(field):
    root = {
        "runtime": {field: 0},
        "generator": {"num_images": 1, "output_dir": "output"},
        "corpus": {"path": "unused.txt"},
        "font": {"dir": "unused-fonts", "size": [30, 36]},
        "background": {"dir": "unused-backgrounds"},
    }

    with pytest.raises(ValueError, match=field):
        _load_v1_root(root)
