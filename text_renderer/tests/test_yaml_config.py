from argparse import Namespace
from pathlib import Path

from main import apply_cli_overrides, parse_args
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
)
from text_renderer.corpus.corpus import Corpus


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
                "font": {"dir": "tibetan-fonts", "size": [28, 34]},
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
