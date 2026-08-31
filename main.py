import argparse
import multiprocessing as mp
import os
import time
from multiprocessing import Value
from multiprocessing.context import Process
from pathlib import Path
import sys
sys.path.append('D:/lxd_code/lv_tools/src')
import cv2
from loguru import logger

from text_renderer.config import AppCfg, GeneratorCfg, get_app_cfg
from text_renderer.dataset import LmdbDataset, ImgDataset
from text_renderer.render import Render
from text_renderer.utils.draw_utils import Imgerror
"""
需要暴露的一些参数：
1. 语言类型 ： 中文、韩文、日文、法文、德文、西班牙文、阿拉伯文、拉丁文、藏文
    * 用于自动适配font，corpus
2. 文字翻转类配置
    * 用于配置是否翻转文字，哪些文字需要翻转等。




"""
cv2.setNumThreads(1)

STOP_TOKEN = "kill"
index = Value('i', 0)

# each child process will initialize Render in process_setup
render: Render


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
        log_period = max(1, int(self.log_period / 100 * num_image))
        try:
            with self.dataset_cls(str(save_dir), **self.dataset_kwargs) as db:
                exist_count = db.read_count()
                count = 0
                logger.info(f"Exist image count in {save_dir}: 【{exist_count}】")
                start = time.time()
                while True:
                    m = self.data_queue.get()
                    if m == STOP_TOKEN:
                        logger.info("DBWriterProcess receive stop token")
                        break

                    name = "{:09d}".format(exist_count + count)
                    db.write('id-' + name, m["image"], m["label"], m["bbox"], m["font"])
                    count += 1
                    if count % log_period == 0:
                        logger.info(
                            f"{(count / num_image) * 100:.2f}%({count}/{num_image}) {log_period / (time.time() - start + 1e-8):.1f} img/s"
                        )
                        start = time.time()
                db.write_count(count + exist_count)
                logger.info(f"{(count / num_image) * 100:.2f}%({count}/{num_image})")
                logger.info(f"Finish generate: {count}. Total: {exist_count + count}")
        except Exception as e:
            logger.exception("DBWriterProcess error")
            raise e


def generate_img(data_queue):
    try:
        # print('多进程断点处')
        data = render()
    except Imgerror as error:
        data = None
        logger.warning("Skip image after render retries: {}", error)

    if data is not None:
        data_queue.put({"image": data[0], "label": data[1], "bbox": data[2], 'font': data[3]})


def process_setup(*args):
    # render 的配置部分
    global render
    import numpy as np

    # Make sure different process has different random seed
    np.random.seed()

    render = Render(args[0])
    logger.info(f"Finish setup image generate process: {os.getpid()}")


def parse_args(arg_list: list = None):
    parser = argparse.ArgumentParser(description="Generate OCR text images")
    parser.add_argument(
        "--config",
        default="./example_data/korean_vertical.yaml",
        help="YAML config path; legacy Python configs are also supported",
    )
    parser.add_argument("--dataset", choices=["lmdb", "img"], default=None)
    parser.add_argument("--num-processes", "--num_processes", type=int, default=None)
    parser.add_argument("--num-images", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--log-period", "--log_period", type=float, default=None)
    if arg_list is None:
        return parser.parse_args()
    else:
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

    for index, generator_cfg in enumerate(app_cfg.generators):
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
                output_dir = output_dir / f"generator_{index}"
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



if __name__ == "__main__":
    '''
    单进程，多进程，和书脊渲染的逻辑，存在不少冗余，比如颜色变换，存在三处。需要合并逻辑
    
    
    '''

    r'''
    当前困惑：不支持字体是如何使程序的计数出现问题，导致 data_queue.put(STOP_TOKEN) 没有执行的。
    
    40 extra bytes in post.stringData array:
    
    This problem is reported by fontTools, and it seems to be caused to a font file containing extra data. 
    It’s just a warning, so if your PDF document doesn’t have any errors, you can safely ignore it.
    
    
    font show: E:\lxd\OCR_project\OCR_SOURCE\font\font_show
    font not suport: E:\lxd\OCR_project\OCR_SOURCE\font
    '''

    args = parse_args()
    app_cfg = apply_cli_overrides(get_app_cfg(args.config), args)
    dataset_cls = LmdbDataset if app_cfg.storage.dataset == "lmdb" else ImgDataset
    dataset_kwargs = get_dataset_kwargs(app_cfg)
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    # using multiprocessing.Manager().Queue() is fine in every case and less troublesome.
    # pay particular attention when using multiprocessing.Queue() because it can have undesired effects
    data_queue = manager.Queue()

    for generator_cfg in app_cfg.generators:
        db_writer_process = DBWriterProcess(
            dataset_cls,
            data_queue,
            generator_cfg,
            app_cfg.runtime.log_period,
            dataset_kwargs,
        )
        db_writer_process.start()

        if app_cfg.runtime.num_processes == 0:
            process_setup(generator_cfg.render_cfg)
            for _ in range(generator_cfg.num_image):
                generate_img(data_queue)
            data_queue.put(STOP_TOKEN)
            join_writer_process(db_writer_process)
        else:
            with mp.Pool(
                    processes=app_cfg.runtime.num_processes,
                    initializer=process_setup,
                    initargs=(generator_cfg.render_cfg,),
            ) as pool:
                for _ in range(generator_cfg.num_image):
                    pool.apply_async(generate_img, args=(data_queue,))

                pool.close()
                pool.join()

            data_queue.put(STOP_TOKEN)
            join_writer_process(db_writer_process)
