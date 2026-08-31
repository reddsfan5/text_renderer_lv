"""多语言 OCR 文本识别数据生成入口。

常用命令
--------
使用 YAML 中 ``generator.active_languages`` 指定的语言：

    python main.py --config example_data/multilingual.yaml

临时选择一个或多个语言：

    python main.py --language korean
    python main.py --language chinese --language korean

覆盖生成数量、输出目录和进程数：

    python main.py --language korean --num-images 10000 --num-processes 4

调试时使用单进程，异常堆栈最清楚：

    python main.py --language korean --num-images 10 --num-processes 0

配置说明
--------
命令行参数优先于 YAML。语言对应的 corpus、charset、font 和文字方向配置在
``example_data/multilingual.yaml`` 的 ``languages`` 节点中。多进程性能参数位于
``runtime``：``num_processes`` 是渲染进程数，``queue_size`` 控制待写图片的
内存上限，``task_batch_size`` 控制每个进程池任务连续生成的图片数。

建议先从物理 CPU 核心数的一半开始测试。字体初始化成本较高，小批量任务可能
单进程更快；批量生成时再逐步增加进程数。每个渲染进程都会持有字体缓存和预加载
背景，内存不足时应降低 ``num_processes`` 或关闭背景预加载。
"""

import argparse
import multiprocessing as mp
import os
import queue
import sys
import time
from multiprocessing.context import Process
from pathlib import Path
from typing import Iterable

sys.path.append("D:/lxd_code/lv_tools/src")

import cv2
from loguru import logger

from text_renderer.config import AppCfg, GeneratorCfg, get_app_cfg
from text_renderer.dataset import ImgDataset, LmdbDataset
from text_renderer.render import Render
from text_renderer.utils.draw_utils import Imgerror

cv2.setNumThreads(1)

STOP_TOKEN = "kill"

# 单进程模式和进程池 worker 都通过 process_setup 初始化这两个对象。
render: Render
worker_data_queue = None


class DBWriterProcess(Process):
    def __init__(
        self,
        dataset_cls: LmdbDataset,
        data_queue,
        generator_cfg: GeneratorCfg,
        log_period: float = 1,
        dataset_kwargs=None,
    ):
        super().__init__()
        self.dataset_cls = dataset_cls
        self.data_queue = data_queue
        self.generator_cfg = generator_cfg
        self.log_period = log_period
        self.dataset_kwargs = dataset_kwargs or {}

    def run(self):
        num_image = self.generator_cfg.num_image
        save_dir = self.generator_cfg.save_dir
        log_interval = max(1, int(self.log_period / 100 * num_image))
        try:
            with self.dataset_cls(str(save_dir), **self.dataset_kwargs) as db:
                exist_count = db.read_count()
                count = 0
                logger.info("Exist image count in {}: {}", save_dir, exist_count)
                start = time.time()
                while True:
                    data = self.data_queue.get()
                    if data == STOP_TOKEN:
                        break

                    name = "{:09d}".format(exist_count + count)
                    db.write(
                        "id-" + name,
                        data["image"],
                        data["label"],
                        data["bbox"],
                        data["font"],
                    )
                    count += 1
                    if count % log_interval == 0:
                        elapsed = time.time() - start
                        logger.info(
                            "{:.2f}%({}/{}) {:.1f} img/s",
                            count / num_image * 100,
                            count,
                            num_image,
                            log_interval / (elapsed + 1e-8),
                        )
                        start = time.time()

                db.write_count(count + exist_count)
                logger.info(
                    "Finish generate: {}. Total in dataset: {}",
                    count,
                    exist_count + count,
                )
        except Exception:
            logger.exception("DBWriterProcess error")
            raise


def generate_img() -> bool:
    """Render one image and put it into the writer queue."""
    if worker_data_queue is None:
        raise RuntimeError("Generator worker has not been initialized")

    try:
        data = render()
    except Imgerror as error:
        logger.warning("Skip image after render retries: {}", error)
        return False

    worker_data_queue.put(
        {
            "image": data[0],
            "label": data[1],
            "bbox": data[2],
            "font": data[3],
        }
    )
    return True


def generate_batch(num_attempts: int) -> int:
    """Generate a small batch to reduce process-pool scheduling overhead."""
    return sum(generate_img() for _ in range(num_attempts))


def process_setup(render_cfg, data_queue):
    """Initialize process-local render state and its shared output queue."""
    global render, worker_data_queue
    import numpy as np

    np.random.seed()
    render = Render(render_cfg)
    worker_data_queue = data_queue
    logger.info("Finish setup image generate process: {}", os.getpid())


def iter_task_batch_sizes(total: int, batch_size: int) -> Iterable[int]:
    """Yield batch sizes whose sum equals total."""
    full_batches, remainder = divmod(total, batch_size)
    for _ in range(full_batches):
        yield batch_size
    if remainder:
        yield remainder


def parse_args(arg_list: list = None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        default="./example_data/multilingual.yaml",
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


def get_dataset_kwargs(app_cfg: AppCfg):
    storage = app_cfg.storage
    if storage.dataset == "lmdb":
        return {
            "map_size": storage.lmdb_map_size_mb * 1024 * 1024,
            "cache_num": storage.lmdb_cache_size,
            "map_increment": storage.lmdb_map_increment_mb * 1024 * 1024,
            "jpg_quality": storage.jpg_quality,
        }
    return {"jpg_quality": storage.jpg_quality}


def join_writer_process(process):
    process.join()
    if process.exitcode != 0:
        raise RuntimeError(
            f"Dataset writer process failed with exit code {process.exitcode}"
        )


def stop_writer_after_error(writer_process, data_queue):
    """Best-effort writer shutdown without hiding the original worker error."""
    if writer_process.is_alive():
        try:
            data_queue.put(STOP_TOKEN, timeout=5)
        except queue.Full:
            writer_process.terminate()
    writer_process.join(timeout=30)
    if writer_process.is_alive():
        writer_process.terminate()
        writer_process.join()


def run_generator(
    generator_cfg: GeneratorCfg,
    app_cfg: AppCfg,
    dataset_cls,
    dataset_kwargs,
) -> int:
    """Generate one configured dataset and return the successful image count."""
    runtime = app_cfg.runtime
    data_queue = mp.Queue(maxsize=runtime.queue_size)
    writer_process = DBWriterProcess(
        dataset_cls,
        data_queue,
        generator_cfg,
        runtime.log_period,
        dataset_kwargs,
    )
    writer_process.start()
    completed = False
    try:
        if runtime.num_processes == 0:
            process_setup(generator_cfg.render_cfg, data_queue)
            generated_count = generate_batch(generator_cfg.num_image)
        else:
            batches = iter_task_batch_sizes(
                generator_cfg.num_image, runtime.task_batch_size
            )
            pool = mp.Pool(
                processes=runtime.num_processes,
                initializer=process_setup,
                initargs=(generator_cfg.render_cfg, data_queue),
            )
            try:
                generated_count = sum(
                    pool.imap_unordered(generate_batch, batches, chunksize=1)
                )
            except BaseException:
                pool.terminate()
                raise
            else:
                # Graceful shutdown lets every worker flush its Queue feeder thread.
                pool.close()
            finally:
                pool.join()

        if writer_process.exitcode is not None:
            join_writer_process(writer_process)
        data_queue.put(STOP_TOKEN)
        join_writer_process(writer_process)
        completed = True
    except BaseException:
        stop_writer_after_error(writer_process, data_queue)
        raise
    finally:
        if not completed:
            data_queue.cancel_join_thread()
        data_queue.close()
        if completed:
            data_queue.join_thread()

    skipped = generator_cfg.num_image - generated_count
    logger.info(
        "Generator '{}': {} successful, {} skipped",
        generator_cfg.name,
        generated_count,
        skipped,
    )
    return generated_count


def run(app_cfg: AppCfg):
    dataset_cls = LmdbDataset if app_cfg.storage.dataset == "lmdb" else ImgDataset
    dataset_kwargs = get_dataset_kwargs(app_cfg)
    return [
        run_generator(generator_cfg, app_cfg, dataset_cls, dataset_kwargs)
        for generator_cfg in app_cfg.generators
    ]


def main(arg_list: list = None):
    args = parse_args(arg_list)
    app_cfg = apply_cli_overrides(
        get_app_cfg(args.config, args.selected_languages), args
    )
    mp.set_start_method("spawn", force=True)
    return run(app_cfg)


if __name__ == "__main__":
    main()
