"""Multiprocess rendering and dataset writing runtime."""

import multiprocessing as mp
import os
import queue
import sys
import time
from multiprocessing.context import Process
from typing import Iterable

sys.path.append("D:/lxd_code/lv_tools/src")

import cv2
from loguru import logger

from text_renderer.config import AppCfg, GeneratorCfg
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
