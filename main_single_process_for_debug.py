import argparse
import os
import traceback
from multiprocessing import Value

import cv2
from loguru import logger

from text_renderer.config import get_cfg, GeneratorCfg
from text_renderer.dataset import LmdbDataset, ImgDataset
from text_renderer.render import RenderOne,Render
from text_renderer.utils.draw_utils import Imgerror

cv2.setNumThreads(1)

index = Value('i', 0)

render: RenderOne


class DBWriterProcess:
    def __init__(
            self,
            dataset_cls: LmdbDataset,
            generator_cfg: GeneratorCfg,
            log_period: float = 1,
    ):
        super().__init__()
        self.dataset_cls = dataset_cls
        self.generator_cfg = generator_cfg
        self.log_period = log_period

    def gen_data(self, m):
        save_dir = self.generator_cfg.save_dir
        try:
            with self.dataset_cls(str(save_dir)) as db:
                exist_count = db.read_count()
                logger.info(f"Exist image count in {save_dir}: 【{exist_count}】")
                name = "{:09d}".format(exist_count + 1)
                db.write('id-' + name, m["image"], m["label"], m["bbox"], m["font"])
                db.write_count(1 + exist_count)
        except Exception as e:
            logger.exception("DBWriterProcess error")
            raise e

def generate_img(text:str):
    try:
        data = render(text)
    except Imgerror:
        data = None
        traceback.print_exc()

    if data is not None:
        return {"image": data[0], "label": data[1], "bbox": data[2], 'font': data[3]}


def process_setup(*args):
    global render
    import numpy as np

    # Make sure different process has different random seed
    np.random.seed()

    render = RenderOne(args[0])
    logger.info(f"Finish setup image generate process: {os.getpid()}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="python file path")
    parser.add_argument("--dataset", default="img", choices=["lmdb", "img"])
    parser.add_argument("--num_processes", type=int, default=2)
    parser.add_argument("--log_period", type=float, default=10)
    return parser.parse_args()


if __name__ == "__main__":
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

    args = parse_args()

    dataset_cls = LmdbDataset if args.dataset == "lmdb" else ImgDataset

    generator_cfgs = get_cfg(args.config)

    for generator_cfg in generator_cfgs:
        db_writer_process = DBWriterProcess(
            dataset_cls, generator_cfg, args.log_period
        )
        process_setup(generator_cfg.render_cfg)
        for i in range(generator_cfg.num_image):
            ret = generate_img('gfhgfhfghfghfgh')
            from matplotlib import pyplot as plt

            bbox = ret['bbox']
            (xs, ys), (xe, ye) = bbox[0], bbox[2]

            plt.imshow(ret['image'][ys:ye, xs:xe, :])
            plt.show()
            db_writer_process.gen_data(ret)
