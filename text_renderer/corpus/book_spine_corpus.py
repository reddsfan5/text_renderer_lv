import random
import typing
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

from tenacity import retry,stop_after_attempt
from text_renderer.utils.errors import PanicError, RetryError
from .corpus import Corpus, CorpusCfg
from loguru import logger

from ..utils import FontText


@dataclass
class BookSpineCorpusCfg(CorpusCfg):
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


class BookSpineCorpus(Corpus):
    """
    Randomly select items from the list
    """

    def __init__(self, cfg: "CorpusCfg"):
        super().__init__(cfg)
        self.normalized_corpus = {}
        self.cfg: BookSpineCorpusCfg
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

    def random_sample_text(self,
                           chn_charset_path: str = r'chn_charset_dict_8k.txt',
                           text_file_path: str = r'text_100.txt'
                           ) -> typing.Generator:
        if not self.texts:
            with open(chn_charset_path, encoding='utf8',mode='r') as chr:
                chr_set = set(chr.read().split('\n'))
            with open(text_file_path, mode='r', encoding='utf8') as f:
                for line in f:
                    text = line.split('\n')
                    if not text:
                        continue
                    # 防止空行
                    text = ''.join(list(filter(lambda x: x in chr_set, text)))
                    yield text

    @retry(stop=stop_after_attempt(10))
    def get_font_text(self,text:str):
        """
        This method ensures that the selected font supports all characters.

        Returns:
            FontText: A FontText object contains text and font.

        """
        # todo 文本标点过滤
        # text = self.remove_punctuation(text)
        # 最大文本长度控制
        if self.cfg.clip_length != -1 and len(text) > self.cfg.clip_length:
            text = text[: self.cfg.clip_length]

        font, support_chars, font_path = self.font_manager.get_font()
        status, intersect = self.font_manager.check_support(text, support_chars)
        if not status:

            err_msg = (
                f"{self.__class__.__name__} {font_path} not support chars: {intersect}"
            )
            # todo lvixaodong font check
            with open(r'D:\lxd_code\OCR\OCR_SOURCE\font/font_not_suport_0707.txt',mode='a+',encoding='utf8') as f:
                f.write(f'{font_path} {text}')
                f.write('\n')
            logger.debug(err_msg)
            raise RetryError(err_msg)

        return FontText(font, text, font_path, self.cfg.horizontal)


class TextCorpusGen:
    def __init__(self, txt_file_path: Union[str, Path]):
        self.txt_file_path = txt_file_path

    def corpus_gener(self):
        with open(self.txt_file_path, 'r', encoding='utf8') as f:
            # return (line for line in f if line.strip())  # ValueError: I/O operation on closed file.
            lines = [line.strip() for line in f if line.strip()]
            random.shuffle(lines)
            for line in lines:
                yield line
