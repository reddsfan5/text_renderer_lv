import inspect
import itertools
import os
import random
import time
from pathlib import Path

from lv_tools.corpus.text_preprocess import limit_text_and_add_space


from text_renderer.config import (
    RenderCfg,
    GeneratorCfg,
    SafeTextColorCfg,
    FixedTextColorCfg,
    FixedPerspectiveTransformCfg,
)
from text_renderer.corpus import *
from text_renderer.corpus.multi_line_corpus import MultiLineCorpus, MultiLineCorpusCfg
from text_renderer.effect import *
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
            space_len = random.randint(0, 6)
            spaces.append(' ' * space_len)
        cur_item = text_item[0]
        for space, text_char in zip(spaces, text_item[1:]):
            cur_item += space
            cur_item += text_char
        text_list_with_space.append(cur_item)
    return text_list_with_space


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

    cfg.render_cfg.corpus = [EnumCorpus(
        EnumCorpusCfg(
            items=items if items else ["2005中国最佳诗歌"],  # Hello你好english规规
            text_color_cfg=SafeTextColorCfg(),
            # text_color_cfg=FixedTextColorCfg(),
            **font_cfg,
        ),
    ), ]
    # cfg.render_cfg.perspective_transform = FixedPerspectiveTransformCfg(30, 30, 1.5)
    # cfg.render_cfg.corpus.cfg.horizontal = False
    return cfg

def multi_line_text(items=None):
    # print(inspect.currentframe().f_code.co_name)
    cfg = base_cfg(
        inspect.currentframe().f_code.co_name + 'author')  # inspect.currentframe().f_code.co_name 返回所在函数的字符串形式的函数名
    cfg.num_image = NUM_IMG
    cfg.render_cfg.gray = False

    cfg.render_cfg.corpus = [MultiLineCorpus(
        MultiLineCorpusCfg(
            items=items if items else ["2005中国最佳诗歌"],  # Hello你好english规规
            text_color_cfg=SafeTextColorCfg(),
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
            text_color_cfg=SafeTextColorCfg(),
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

def text_list_gen(txt_path,chr_set,is_add_space=False):

    with open(txt_path, mode='r', encoding='utf8') as f:
        # text_list = f.read().split('\n')[:-1] # 直接截掉最后一行，这行通常为空行
        text_list = f.read().split('\n')  # 直接截掉最后一行，这行通常为空行
        # 防止空行
        text_list = [''.join(list(filter(lambda x: x in chr_set, text))) for text in text_list if text]
        text_list = [text for text in text_list if text]

    text_list = limit_text_and_add_space(text_list,is_add_space=is_add_space)
    return text_list

def data_split_start_end(text_list,part=4,cur_index=0):
    left = (len(text_list) // part) * cur_index
    right = (len(text_list) // part) * (cur_index + 1)

    print(text_list[left:left + 10])

    return left,right





# 文本统一过滤的必要不大。如果文本过大，大到超出内存限制，这种统一到列表中的做法就不可行了。
'''
目前想到的优化方案是：
1. 文本处理，
'''

len_limit = 30
# 支持的字符集，用于过滤超纲字符
with open(r'D:\lxd_code\OCR\OCR_SOURCE\model\spine_rec_v2\bookridge_rec_chn_svtr_240223/chn_kor_jap_fre_rus_spa_ara_latin_tib_24188.txt', encoding='utf8', mode='r') as chr:
    chr_set = set(chr.read().split('\n'))
# 所有可选的书名、作者名列表。

# 索书号txt
# txt_path = r'D:\lxd_code\OCR\OCR_SOURCE\corpus\anhuidaxue_call_number\anhuidaxue-callnumber_splited.txt'
# 书名txt

text_path_dict = {
    'jpp':r'F:\dataset\OCR\图书目录\text\japan\open_source-book_title-japan2.txt',
    'fre':r'F:\dataset\OCR\图书目录\text\french\french_book_name_author_publisher_valid.txt',
    'spa':r'F:\dataset\OCR\图书目录\text\Spanish\valid_spanish_drop_dup_cut_long.txt',
    'rus':r'F:\dataset\OCR\图书目录\text\russian\zlib_russian.txt',
    'tib':r'F:\dataset\OCR\图书目录\text\tibetan\n-bo_normed_mini.txt'
}
txt_path =text_path_dict['tib']

text_list = text_list_gen(txt_path = txt_path,chr_set=chr_set,is_add_space=False)
print('corpus读取完毕')

# start,end = data_split_start_end(text_list)
# text_list = text_list[start:end]
NUM_IMG:int = int(8*10**2)
# text_list = series_text_gen(data_num=NUM_IMG)

local_time = time.localtime()
mon, day, hour,minite,sec = local_time.tm_mon, local_time.tm_mday, local_time.tm_hour,local_time.tm_min,local_time.tm_sec
DST_DIR = Path(fr'D:\dataset\OCR\lmdb_datatest_{mon:02}{day:02}{hour:02}_{minite:02}_{sec:02}')
# BG_DIR = Path(r'F:\dataset\OCR\callnumber_gen\callnumber_bg')
BG_DIR = Path(r'D:\lxd_code\OCR\OCR_SOURCE\bg')
# BG_DIR = Path(r'D:\lxd_code\OCR\OCR_SOURCE\bg\bg_white')
CURRENT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))

FONT_SMP = Path(r'D:\lxd_code\OCR\OCR_SOURCE\font\font_set\简体-简体-低风险')
FONT_MINI = Path(r'D:\lxd_code\OCR\OCR_SOURCE\font\font_set\font_mini')
FONT_HARD = Path(r'D:\lxd_code\OCR\OCR_SOURCE\font\font_set\超个性-存在简体繁体混合使用\超个性-已更新')
FONT_EN = Path(r'D:\lxd_code\OCR\OCR_SOURCE\font\font_set\english\手写体')
FONT_ONE = Path(r'D:\lxd_code\OCR\OCR_SOURCE\font\font_set\简体-简体-低风险\单一字体\1')
FONT_NORMAL = Path(r'D:\lxd_code\OCR\OCR_SOURCE\font\font_set\简体-简体-低风险\常规类_已更正')
FONT_KOREA= Path(r'D:\lxd_code\OCR\OCR_SOURCE\font\font_set\韩文')
FONT_JAPAN = Path(r'D:\lxd_code\OCR\OCR_SOURCE\font\font_set\日文\ttf-notdef')
FONT_FR = Path(r'D:\lxd_code\OCR\OCR_SOURCE\font\font_set\english\jinke_miaomu_done\en_fr_jinke_miaomu_done')
FONT_SPA_EN_FRE = Path(r'D:\lxd_code\OCR\OCR_SOURCE\font\font_set\english\jinke_miaomu_done\en_fr_jinke_miaomu_done\eng_fre_spa')
FONT_RUS = Path(r'D:\lxd_code\OCR\OCR_SOURCE\font\font_set\俄文\70-potryasayushhix-kirillicheskix-russkix-shriftov\selected')
FONT_TIB = Path(r'D:\lxd_code\OCR\OCR_SOURCE\font\font_set\藏文')
font_cfg = dict(
    font_dir=FONT_TIB,
    font_size=(30, 34),# 34,36
    # sp_font_excel_path=r'D:\lxd_code\OCR\OCR_SOURCE\font\索书号字体.xlsx'

)

small_font_cfg = dict(
    font_dir=FONT_ONE,
    font_size=(20, 21),
)

vertical = True
# text_iter = itertools.cycle(text_list) # 当前接口不兼容迭代器
text_list = max(1,2*(NUM_IMG//len(text_list)))*text_list



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
        effect_ensemble(text_list),
        # multi_line_text(text_list*2)
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
        cfg1.render_cfg.corpus[0].cfg.clip_length=2000
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
        # effect_ensemble()
        # multi_line_text()
        # dropout_rand(),
        # dropout_horizontal(),
        # dropout_vertical(),
        # padding(),
        # same_line_layout_different_font_size(),
    ]
