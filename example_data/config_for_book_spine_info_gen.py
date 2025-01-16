import inspect
import os
import time
from pathlib import Path

from lv_tools.task_ocr_text_render.series_text import (
    series_text_gen)
from text_renderer.config import (
    RenderCfg,
    GeneratorCfg,
    SafeTextColorCfg,
    FixedTextColorCfg,
)
from text_renderer.corpus import *
from text_renderer.corpus.book_spine_corpus import BookSpineCorpusCfg, BookSpineCorpus

'''
1.corpus 文件要用utf8编码格式
2.各个路径不用字符串，而是Path 对象。
'''


def base_cfg(name: str):
    return GeneratorCfg(
        num_image=5,
        save_dir=DST_DIR / "effect_layout_image" / name,
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,
            corpus=EnumCorpus(
                EnumCorpusCfg(
                    items=["Hello你好"],
                    text_color_cfg=FixedTextColorCfg(),
                    **font_cfg,
                ),
            ),
        ),
    )


def book_spine_text(items=None):
    cfg = base_cfg(
        inspect.currentframe().f_code.co_name + 'author')  # inspect.currentframe().f_code.co_name 返回所在函数的字符串形式的函数名
    cfg.num_image = NUM_IMG
    cfg.render_cfg.gray = False

    cfg.render_cfg.corpus = [BookSpineCorpus(
        BookSpineCorpusCfg(
            items=items,  # Hello你好english规规
            text_color_cfg=SafeTextColorCfg(),
            **font_cfg,
        ),
    ), ]
    return cfg


FONT_SMP = Path(r'D:\lxd_code\OCR\OCR_SOURCE\font\font_set\简体-简体-低风险')
FONT_MINI = Path(r'D:\lxd_code\OCR\OCR_SOURCE\font\font_set\font_mini')
FONT_HARD = Path(r'D:\lxd_code\OCR\OCR_SOURCE\font\font_set\超个性-存在简体繁体混合使用\超个性-已更新')
FONT_ONE = Path(r'D:\lxd_code\OCR\OCR_SOURCE\font\font_set\简体-简体-低风险\单一字体\2')
# 文本统一过滤的必要不大。如果文本过大，大到超出内存限制，这种统一到列表中的做法就不可行了。
'''
目前想到的优化方案是：
1. 文本处理，
'''

len_limit = 200
# 支持的字符集，用于过滤超纲字符
with open(r'D:\lxd_code\OCR\OCR_SOURCE\corpus/chn_charset_dict_9735.txt', encoding='utf8', mode='r') as chr:
    chr_set = set(chr.read().split('\n'))
# 所有可选的书名、作者名列表。

NUM_IMG = 100
text_list = series_text_gen(data_num=NUM_IMG)

local_time = time.localtime()
mon, day, hour, minite, sec = local_time.tm_mon, local_time.tm_mday, local_time.tm_hour, local_time.tm_min, local_time.tm_sec
DST_DIR = Path(fr'D:\dataset\OCR\lmdb_datatest_{mon:02}{day:02}{hour:02}_{minite:02}_{sec:02}')
BG_DIR = Path(r'D:\lxd_code\OCR\OCR_SOURCE\bg')
CURRENT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))

font_cfg = dict(
    font_dir=FONT_ONE,
    font_size=(25, 26),  # 34,36
)

small_font_cfg = dict(
    font_dir=FONT_MINI,
    font_size=(20, 21),
)

vertical = True

if vertical:

    cfgs = [
        # effect_ensemble(),
        # effect_ensemble(text_list*2),
        book_spine_text(text_list * 2)
        # color_image(text_list),
    ]
    configs = []
    for cfg1 in cfgs:
        cfg1.render_cfg.corpus[0].cfg.horizontal = False
        configs.append(cfg1)
else:
    configs = [
        # effect_ensemble(text_list)
        # effect_ensemble()
        book_spine_text()
    ]
