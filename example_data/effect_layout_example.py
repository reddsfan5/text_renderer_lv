import inspect
import os
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import time
import random
from text_renderer.effect import *
from text_renderer.corpus import *
from text_renderer.config import (
    RenderCfg,
    NormPerspectiveTransformCfg,
    GeneratorCfg,
    SimpleTextColorCfg,
    TextColorCfg,
    FixedTextColorCfg,
    FixedPerspectiveTransformCfg,
)
from text_renderer.effect.curve import Curve
from text_renderer.layout import SameLineLayout, ExtraTextLineLayout

'''
1.corpus 文件要用utf8编码格式
2.各个路径不用字符串，而是Path 对象。
'''


def shorten_item(text_list):
    # 限制文本长度
    text_list = [text[:len_limit] for text in text_list if text]
    # 随机添加空格的文本集
    text_list_with_space = []
    for text_item in text_list:
        if not text_item:
            continue
        spaces = []
        for _ in range(len(text_item) - 1):
            space_len = random.randint(0, 8)
            spaces.append(' ' * space_len)
        cur_item = text_item[0]
        for space, text_char in zip(spaces, text_item[1:]):
            cur_item += space
            cur_item += text_char
        text_list_with_space.append(cur_item)
    text_list = text_list_with_space
    return text_list


len_limit = 20
# 支持的字符集，用于过滤超纲字符
with open(r'D:\lxd_code\OCR\OCR_SOURCE\corpus/chn_charset_dict_9735.txt', encoding='utf8', mode='r') as chr:
    chr_set = set(chr.read().split('\n'))
# 所有可选的书名、作者名列表。
txt_path = r'D:\lxd_code\OCR\OCR_SOURCE\corpus\author_bookname\filtered_author_bookname_simple.txt'
# txt_path = r'D:\lxd_code\OCR\OCR_SOURCE\corpus\digit_str/digit_text.txt'
# txt_path = r'D:\lxd_code\OCR\OCR_SOURCE\corpus\author_bookname\text_100.txt'
with open(txt_path, mode='r', encoding='utf8') as f:
    # text_list = f.read().split('\n')[:-1] # 直接截掉最后一行，这行通常为空行
    text_list = f.read().split('\n')  # 直接截掉最后一行，这行通常为空行
    # 防止空行
    text_list = [''.join(list(filter(lambda x: x in chr_set, text))) for text in text_list if text]

# text_list = shorten_item(text_list)

# # 限制文本长度
# text_list = [text[:len_limit] for text in text_list if text]
# # 随机添加空格的文本集
text_list_with_space = []
for text_item in text_list:
    if not text_item:
        continue
    spaces = []
    for _ in range(len(text_item) - 1):
        space_len = random.randint(2, 8)
        spaces.append(' ' * space_len)
    cur_item = text_item[0]
    for space, text_char in zip(spaces, text_item[1:]):
        cur_item += space
        cur_item += text_char
    text_list_with_space.append(cur_item)
text_list = text_list_with_space

# 常规文本集
# text_list = [text for text in text_list if text]

# 小数据集，多倍重复
# text_list = text_list[:100]
# text_list *= 10

# list(set(text_list)).sort()

# 大间距文字识别记录
# left = (len(text_list)//10) * 5
# right = (len(text_list)//10) * 6
part = 5
cur_index = 0
left = (len(text_list) // part) * cur_index
# right = (len(text_list) // part) * (cur_index+1)
right = 100
NUM_IMG = len(text_list[left:right])

print(text_list[left:left + 10])
print(f'目标图像数目：{NUM_IMG}')
text_list = text_list[left:right]
local_time = time.localtime()
mon, day, hour = local_time.tm_mon, local_time.tm_mday, local_time.tm_hour

DST_DIR = Path(fr'D:\dataset\OCR\lmdb_datatest_{mon:02}{day:02}{hour:02}_{left:06}_{right}')
BG_DIR = Path(r'D:\lxd_code\OCR\OCR_SOURCE\bg')
CURRENT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))

FONT_SMP = Path(r'D:\lxd_code\OCR\OCR_SOURCE\font\font_set\简体-简体-低风险')
FONT_MINI = Path(r'D:\lxd_code\OCR\OCR_SOURCE\font\font_set\font_mini')
FONT_ART = Path(r'D:\lxd_code\OCR\OCR_SOURCE\font\font_set\font_art')
FONT_SMP_TDT = Path(r'D:\lxd_code\OCR\OCR_SOURCE\font\font_set\简繁-简繁-低风险\字库齐全')
FONT_DEBUG = Path(r'D:\lxd_code\OCR\OCR_SOURCE\font\font_set\font_mini\debug')
FONT_TRADITION = Path(r'D:\lxd_code\OCR\OCR_SOURCE\font\font_set\繁体-繁体-低风险')
FONT_SIM_TRAD = Path(r'D:\lxd_code\OCR\OCR_SOURCE\font\font_set\简繁-简繁-低风险')
FONT_HARD = Path(r'D:\lxd_code\OCR\OCR_SOURCE\font\font_set\超个性存在风险字体\已更新')

font_cfg = dict(
    font_dir=FONT_HARD,
    font_size=(41, 43),  # 34,36
)

small_font_cfg = dict(
    font_dir=FONT_MINI,
    font_size=(20, 21),
)


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


def effect_ensemble(items=None):
    # print(inspect.currentframe().f_code.co_name)
    cfg = base_cfg(
        inspect.currentframe().f_code.co_name + 'author')  # inspect.currentframe().f_code.co_name 返回所在函数的字符串形式的函数名
    cfg.num_image = NUM_IMG
    cfg.render_cfg.gray = False

    # 2013-2014-商法-学生常用法规掌中宝-6
    # Hello你好english
    # Hello你好english规规
    # 中国地区间财政均等化问题研究

    cfg.render_cfg.corpus = [EnumCorpus(
        EnumCorpusCfg(
            items=items if items else ["2005中国最佳诗歌"],  # Hello你好english规规
            text_color_cfg=SimpleTextColorCfg(),
            # text_color_cfg=FixedTextColorCfg(),
            **font_cfg,
        ),
    ), ]
    # cfg.render_cfg.perspective_transform = FixedPerspectiveTransformCfg(30, 30, 1.5)
    # cfg.render_cfg.corpus.cfg.horizontal = False
    return cfg


def dropout_rand():
    cfg = base_cfg(inspect.currentframe().f_code.co_name)
    cfg.render_cfg.corpus_effects = Effects(DropoutRand(p=1, dropout_p=(0.3, 0.5)))
    return cfg


def dropout_horizontal():
    cfg = base_cfg(inspect.currentframe().f_code.co_name)
    cfg.render_cfg.corpus_effects = Effects(
        DropoutHorizontal(p=1, num_line=2, thickness=3)
    )
    return cfg


def dropout_vertical():
    cfg = base_cfg(inspect.currentframe().f_code.co_name)
    cfg.render_cfg.corpus_effects = Effects(DropoutVertical(p=1, num_line=15))
    return cfg


def line():
    poses = [
        "top",
        "bottom",
        "left",
        "right",
        "top_left",
        "top_right",
        "bottom_left",
        "bottom_right",
        "horizontal_middle",
        "vertical_middle",
    ]
    cfgs = []
    for i, pos in enumerate(poses):
        pos_p = [0] * len(poses)
        pos_p[i] = 1
        cfg = base_cfg(f"{inspect.currentframe().f_code.co_name}_{pos}")
        cfg.render_cfg.corpus_effects = Effects(
            Line(p=1, thickness=(3, 4), line_pos_p=pos_p)
        )
        cfgs.append(cfg)
    return cfgs


def padding():
    cfg = base_cfg(inspect.currentframe().f_code.co_name)
    cfg.render_cfg.corpus_effects = Effects(
        Padding(p=1, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71], center=True)
    )
    return cfg


def same_line_layout_different_font_size():
    cfg = base_cfg(inspect.currentframe().f_code.co_name)
    cfg.render_cfg.layout = SameLineLayout(h_spacing=(0.9, 0.91))
    cfg.render_cfg.corpus = [
        EnumCorpus(
            EnumCorpusCfg(
                items=["Hello "],
                text_color_cfg=FixedTextColorCfg(),
                **font_cfg,
            ),
        ),
        EnumCorpus(
            EnumCorpusCfg(
                items=[" World!"],
                text_color_cfg=FixedTextColorCfg(),
                **small_font_cfg,
            ),
        ),
    ]
    return cfg


def extra_text_line_layout():
    cfg = base_cfg(inspect.currentframe().f_code.co_name)
    cfg.render_cfg.layout = ExtraTextLineLayout(bottom_prob=1.0)
    cfg.render_cfg.corpus = [
        EnumCorpus(
            EnumCorpusCfg(
                items=["Hello world"],
                text_color_cfg=FixedTextColorCfg(),
                **font_cfg,
            ),
        ),
        EnumCorpus(
            EnumCorpusCfg(
                items=["THIS IS AN EXTRA TEXT LINE!"],
                text_color_cfg=FixedTextColorCfg(),
                **font_cfg,
            ),
        ),
    ]
    return cfg


def color_image(items=None):
    # print(inspect.currentframe().f_code.co_name)
    cfg = base_cfg(inspect.currentframe().f_code.co_name)  # inspect.currentframe().f_code.co_name 返回所在函数的字符串形式的函数名
    cfg.num_image = 1000
    cfg.render_cfg.gray = False
    cfg.render_cfg.corpus = [EnumCorpus(
        EnumCorpusCfg(
            items=items if items else ["Hello! 【你好】[english]"],
            text_color_cfg=SimpleTextColorCfg(),
            # text_color_cfg=FixedTextColorCfg(),
            **font_cfg,
        ),
    ), ]
    cfg.render_cfg.perspective_transform = FixedPerspectiveTransformCfg(30, 30, 1.5)
    # cfg.render_cfg.corpus.cfg.horizontal = False
    return cfg


def perspective_transform():
    cfg = base_cfg(inspect.currentframe().f_code.co_name)
    cfg.render_cfg.perspective_transform = FixedPerspectiveTransformCfg(30, 30, 1.5)
    return cfg


def compact_char_spacing():
    cfg = base_cfg(inspect.currentframe().f_code.co_name)
    cfg.render_cfg.corpus.cfg.char_spacing = -0.3
    return cfg


def large_char_spacing():
    cfg = base_cfg(inspect.currentframe().f_code.co_name)
    cfg.render_cfg.corpus.cfg.char_spacing = 0.5
    return cfg


def curve():
    cfg = base_cfg(inspect.currentframe().f_code.co_name)
    cfg.render_cfg.corpus_effects = Effects(
        [
            Padding(p=1, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71], center=True),
            Curve(p=1, period=180, amplitude=(4, 5)),
        ]
    )
    return cfg


def vertical_text():
    cfg = base_cfg(inspect.currentframe().f_code.co_name)
    cfg.render_cfg.corpus.cfg.horizontal = False
    # cfg.render_cfg.corpus.cfg.char_spacing = 0.1
    return cfg


def bg_and_text_mask():
    cfg = base_cfg(inspect.currentframe().f_code.co_name)
    cfg.render_cfg.perspective_transform = FixedPerspectiveTransformCfg(30, 30, 1.5)
    cfg.render_cfg.return_bg_and_mask = True
    cfg.render_cfg.height = 48
    return cfg


def emboss():
    import imgaug.augmenters as iaa

    cfg = base_cfg(inspect.currentframe().f_code.co_name)
    cfg.render_cfg.height = 48
    cfg.render_cfg.corpus_effects = Effects(
        [
            Padding(p=1, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71], center=True),
            ImgAugEffect(aug=iaa.Emboss(alpha=(0.9, 1.0), strength=(1.5, 1.6))),
        ]
    )
    return cfg


vertical = True

if vertical:

    cfgs = [
        # bg_and_text_mask()
        # emboss(),
        # vertical_text()
        # extra_text_line_layout()
        # char_spacing_compact(),
        # char_spacing_large(),
        # *line(),
        # perspective_transform(),
        # effect_ensemble(),
        effect_ensemble(text_list * 2),
        # color_image(text_list),
        # color_image(),
        # dropout_rand(),
        # dropout_horizontal(),
        # dropout_vertical(),
        # padding(),
        # same_line_layout_different_font_size(),
    ]
    configs = []
    for cfg1 in cfgs:
        cfg1.render_cfg.corpus[0].cfg.horizontal = False
        configs.append(cfg1)
else:
    configs = [
        # bg_and_text_mask()
        # emboss(),
        # vertical_text()
        # extra_text_line_layout()
        # char_spacing_compact(),
        # char_spacing_large(),
        # *line(),
        # perspective_transform(),
        # color_image(text_list),
        # effect_ensemble(text_list)
        effect_ensemble()
        # dropout_rand(),
        # dropout_horizontal(),
        # dropout_vertical(),
        # padding(),
        # same_line_layout_different_font_size(),
    ]
