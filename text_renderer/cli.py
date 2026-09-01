"""多语言 OCR 文本识别数据生成入口。

常用命令
--------
使用 YAML 中 ``generator.active_languages`` 指定的语言：

    python main.py --config configs/multilingual.yaml

临时选择一个或多个语言：

    python main.py --language korean
    python main.py --language chinese --language korean

命令行参数优先于 YAML。语言对应的 corpus、charset、font 和文字方向配置在
``configs/multilingual.yaml`` 的 ``languages`` 节点中。
"""

import argparse
import multiprocessing as mp
from pathlib import Path

from text_renderer.config import AppCfg, get_app_cfg
from text_renderer.runner import run


def parse_args(arg_list: list = None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        default="./configs/multilingual.yaml",
        help="YAML config path; legacy Python configs are also supported",
    )
    parser.add_argument(
        "--language",
        dest="selected_languages",
        action="append",
        help="Select a language profile; repeat to generate multiple languages",
    )
    parser.add_argument("--dataset", choices=["lmdb", "img"], default=None)
    parser.add_argument("--num-processes", "--num_processes", type=int, default=None)
    parser.add_argument("--num-images", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--log-period", "--log_period", type=float, default=None)
    return parser.parse_args(arg_list)


def apply_cli_overrides(app_cfg: AppCfg, args):
    if args.num_images is not None and args.num_images <= 0:
        raise ValueError("--num-images must be > 0")
    if args.num_processes is not None and args.num_processes < 0:
        raise ValueError("--num-processes must be >= 0")
    if args.log_period is not None and args.log_period <= 0:
        raise ValueError("--log-period must be > 0")

    if args.num_processes is not None:
        app_cfg.runtime.num_processes = args.num_processes
    if args.log_period is not None:
        app_cfg.runtime.log_period = args.log_period
    if args.dataset is not None:
        app_cfg.storage.dataset = args.dataset

    for generator_cfg in app_cfg.generators:
        if args.num_images is not None:
            generator_cfg.num_image = args.num_images
            corpus = generator_cfg.render_cfg.corpus
            corpora = corpus if isinstance(corpus, list) else [corpus]
            for corpus_item in corpora:
                texts = getattr(corpus_item, "texts", None)
                if isinstance(texts, list) and texts and len(texts) < args.num_images:
                    repeat = (args.num_images + len(texts) - 1) // len(texts)
                    corpus_item.texts *= repeat

        if args.output_dir is not None:
            output_dir = Path(args.output_dir)
            if len(app_cfg.generators) > 1:
                output_dir = output_dir / generator_cfg.name
            generator_cfg.save_dir = output_dir
    return app_cfg


def main(arg_list: list = None):
    args = parse_args(arg_list)
    app_cfg = apply_cli_overrides(
        get_app_cfg(args.config, args.selected_languages), args
    )
    mp.set_start_method("spawn", force=True)
    return run(app_cfg)


if __name__ == "__main__":
    main()
