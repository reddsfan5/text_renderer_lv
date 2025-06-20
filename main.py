import argparse
import multiprocessing as mp
import os
import time
import traceback
from multiprocessing import Value
from multiprocessing.context import Process
import sys
sys.path.append('D:\lxd_code\lv_tools\src')
import cv2
from loguru import logger

from text_renderer.config import get_cfg, GeneratorCfg
from text_renderer.dataset import LmdbDataset, ImgDataset
from text_renderer.render import Render
from text_renderer.utils.draw_utils import Imgerror

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
    ):
        super().__init__()
        self.dataset_cls = dataset_cls
        self.data_queue = data_queue
        self.generator_cfg = generator_cfg
        self.log_period = log_period

    def run(self):
        num_image = self.generator_cfg.num_image
        save_dir = self.generator_cfg.save_dir
        log_period = max(1, int(self.log_period / 100 * num_image))
        try:
            with self.dataset_cls(str(save_dir)) as db:
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
    except Imgerror:
        data = None
        traceback.print_exc()

    if data is not None:
        data_queue.put({"image": data[0], "label": data[1], "bbox": data[2], 'font': data[3]})


def process_setup(*args):
    global render
    import numpy as np

    # Make sure different process has different random seed
    np.random.seed()

    render = Render(args[0])
    logger.info(f"Finish setup image generate process: {os.getpid()}")


def parse_args(arg_list: list = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="python file path")
    parser.add_argument("--dataset", default="img", choices=["lmdb", "img"])
    parser.add_argument("--num_processes", type=int, default=2)
    parser.add_argument("--log_period", type=float, default=10)
    if arg_list is None:
        return parser.parse_args()
    else:
        return parser.parse_args(arg_list)



if __name__ == "__main__":
    '''
    单进程，多进程，和书脊渲染的逻辑，存在不少冗余，比如颜色变换，存在三处。需要合并逻辑
    
    
    '''


    '''
    # 代码都是针对写入lmdb 的逻辑进行改写的，没有维护写入img的逻辑，所以在写入img时，会报错。
    
    --config .\example_data/effect_layout_example.py --dataset lmdb --num_processes 1 --log_period 2
    
    --config .\example_data/example.py --dataset lmdb --num_processes 1 --log_period 2
    '''
    r'''
    当前困惑：不支持字体是如何使程序的计数出现问题，导致 data_queue.put(STOP_TOKEN) 没有执行的。
    
    40 extra bytes in post.stringData array:
    
    This problem is reported by fontTools, and it seems to be caused to a font file containing extra data. 
    It’s just a warning, so if your PDF document doesn’t have any errors, you can safely ignore it.
    
    
    font show: E:\lxd\OCR_project\OCR_SOURCE\font\font_show
    font not suport: E:\lxd\OCR_project\OCR_SOURCE\font
    '''

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    # using multiprocessing.Manager().Queue() is fine in every case and less troublesome.
    # pay particular attention when using multiprocessing.Queue() because it can have undesired effects
    data_queue = manager.Queue()
    arg_list = '--config .\example_data/effect_layout_example.py --dataset lmdb --num_processes 1 --log_period 2'.split(' ')

    if not sys.argv[1:]:
        args = parse_args(arg_list)
    else:
        args = parse_args()
    # args = parse_args()

    dataset_cls = LmdbDataset if args.dataset == "lmdb" else ImgDataset

    generator_cfgs = get_cfg(args.config)

    for generator_cfg in generator_cfgs:
        db_writer_process = DBWriterProcess(
            dataset_cls, data_queue, generator_cfg, args.log_period
        )
        db_writer_process.start()

        if args.num_processes == 0:
            process_setup(generator_cfg.render_cfg)
            for _ in range(generator_cfg.num_image):
                generate_img(data_queue)
            data_queue.put(STOP_TOKEN)
            db_writer_process.join()
        else:
            with mp.Pool(
                    processes=args.num_processes,
                    initializer=process_setup,
                    initargs=(generator_cfg.render_cfg,),
            ) as pool:
                for _ in range(generator_cfg.num_image):
                    pool.apply_async(generate_img, args=(data_queue,))

                pool.close()
                pool.join()

            data_queue.put(STOP_TOKEN)
            db_writer_process.join()
