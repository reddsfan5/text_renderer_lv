from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from text_renderer.utils.errors import PanicError
from .corpus import Corpus, CorpusCfg


@dataclass
class EnumCorpusCfg(CorpusCfg):
    """
    Enum corpus config

    args:
        text_paths (List[Path]): Text file paths
        items (List[str]): Texts to choice. Only works if text_paths is empty
        num_pick (int): Random choice {count} item from texts
        filter_by_chars (bool): If True, filtering text by character set
        chars_file (Path): Character set
        filter_font (bool): Only work when filter_by_chars is True. If True, filter font file
                            by intersection of font support chars with chars file
        filter_font_min_support_chars (int): If intersection of font support chars with chars file is lower
                                             than filter_font_min_support_chars, filter this font file.
        join_str (str):

    """

    text_paths: List[Path] = field(default_factory=list)
    items: List[str] = field(default_factory=list)
    num_pick: int = 1
    filter_by_chars: bool = False
    chars_file: Path = None
    filter_font: bool = False
    filter_font_min_support_chars: int = 100
    join_str: str = ""


class EnumCorpus(Corpus):
    """
    Randomly select items from the list
    """

    def __init__(self, cfg: "CorpusCfg"):
        super().__init__(cfg)

        self.cfg: EnumCorpusCfg
        if len(self.cfg.text_paths) == 0 and len(self.cfg.items) == 0:
            raise PanicError(f"text_paths or items must not be empty")

        if len(self.cfg.text_paths) != 0 and len(self.cfg.items) != 0:
            raise PanicError(f"only one of text_paths or items can be set")

        self.texts: List[str] = []

        if len(self.cfg.text_paths) != 0:
            for text_path in self.cfg.text_paths:
                with open(str(text_path), "r", encoding="utf-8") as f:
                    for line in f.readlines():
                        self.texts.append(line.strip())

        elif len(self.cfg.items) != 0:
            self.texts = self.cfg.items

        if self.cfg.chars_file is not None:
            self.font_manager.update_font_support_chars(self.cfg.chars_file)

        if self.cfg.filter_by_chars:
            self.texts = Corpus.filter_by_chars(self.texts, self.cfg.chars_file)
            if self.cfg.filter_font:
                self.font_manager.filter_font_path(
                    self.cfg.filter_font_min_support_chars
                )

    def get_text(self) -> str:
        # todo lvxiaodong
        if not self.texts:
            with open(r'D:\lxd_code\OCR\OCR_SOURCE\corpus\chn_charset_dict_8k.txt', encoding='utf8',
                      mode='r') as chr:
                chr_set = set(chr.read().split('\n'))
            txt_path = r'D:\lxd_code\OCR\OCR_SOURCE\corpus\author_bookname\filtered_author_bookname.txt'
            with open(txt_path, mode='r', encoding='utf8') as f:
                # text_list = f.read().split('\n')[:-1] # 直接截掉最后一行，这行通常为空行
                text_list = f.read().split('\n')  # 直接截掉最后一行，这行通常为空行
                # 防止空行
                text_list = [''.join(list(filter(lambda x: x in chr_set, text))) for text in text_list if text]
                text_list = [text for text in text_list if text]
            self.texts = text_list
        # 文本遍历模式
        text = self.texts.pop()
        # todo watch text info
        # print(text)
        # text = random_choice(self.texts, self.cfg.num_pick)
        return self.cfg.join_str.join(text)
