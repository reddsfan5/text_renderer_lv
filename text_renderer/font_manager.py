import os
import random
from functools import lru_cache
from pathlib import Path
from typing import List, Set, Tuple, Dict

import pandas as pd
from PIL import ImageFont
from PIL.ImageFont import FreeTypeFont
from fontTools.ttLib import TTFont, TTCollection
from loguru import logger

from text_renderer.utils.utils import load_chars_file


def get_excel_font(excel_path: str):
    df = pd.read_excel(excel_path)
    font_bases = set(df['字体'])
    return font_bases


class FontManager:
    def __init__(self, font_dir: Path, font_size: Tuple[int, int], sp_font_excel_path: str = ''):
        assert font_size[0] < font_size[1]
        self.font_size_range = font_size
        self.font_dir = font_dir
        self.font_paths: List[str] = self._get_font_paths(sp_font_excel_path)
        self.font_support_chars_cache: Dict[str, Set] = {}
        # Created in self.update_font_support_chars(), used to filter font_path
        self.font_support_chars_intersection_with_chars: Dict[str, Set] = {}

        self._load_font_support_chars()

    def _get_font_paths(self, sp_font_excel_path: str = ''):
        '''
        r'D:\lxd_code\OCR\OCR_SOURCE\font\索书号字体.xlsx'
        -------

        '''
        valid_font = get_excel_font(sp_font_excel_path) if sp_font_excel_path and os.path.exists(
            sp_font_excel_path) else set()
        font_paths = []
        for font_path in self.font_dir.glob("**/*"):
            if font_path.suffix in [".ttc", ".TTC", ".ttf", ".TTF", ".otf", ".OTF"]:
                if valid_font and os.path.basename(font_path) not in self.valid_font:
                    continue
                else:
                    font_paths.append(str(font_path))

        return font_paths

    def get_font(self) -> Tuple[FreeTypeFont, Set, str]:
        font_path = random.choice(self.font_paths)
        font_size = random.randint(*self.font_size_range)

        font = self._get_font(font_path, font_size)
        font_support_chars = self.font_support_chars_cache[font_path]

        return font, font_support_chars, font_path

    def check_support(self, text: str, chars: Set) -> Tuple[bool, Set]:
        # Check whether all chars in text exist in chars

        text_set = set(text)
        intersect = text_set - chars
        status = len(intersect) == 0

        return status, intersect

    def _load_font_support_chars(self):
        for font_path in self.font_paths:
            ttf = self._load_ttfont(font_path)

            chars_int = set()
            try:
                for table in ttf["cmap"].tables:
                    for k, v in table.cmap.items():
                        chars_int.add(k)
            except AssertionError as e:
                logger.error(f"Load font file {font_path} failed, skip it. Error: {e}")
                # os.remove(font_path)

            supported_chars = set([chr(c_int) for c_int in chars_int])

            ttf.close()

            self.font_support_chars_cache[font_path] = supported_chars

    def update_font_support_chars(self, chars_file):
        """
        Although some fonts have a specific character in the cmap, the rendered text is blank on the image.

        Parameters
        ----------
        chars_file: Path
            one char per line
        """
        white_list = [" "]

        charset = load_chars_file(chars_file)
        for font_path in self.font_paths:
            removed_chars = []
            font = self._get_font(font_path, 10)
            chars = self.font_support_chars_cache[font_path].copy()
            for c in chars & charset:
                bbox = font.getmask(c).getbbox()
                if (
                        c not in white_list
                        and bbox is None
                        and c in self.font_support_chars_cache[font_path]
                ):
                    self.font_support_chars_cache[font_path].remove(c)
                    removed_chars.append(c)

            if len(removed_chars) != 0:
                if len(removed_chars) > 10:
                    logger.info(
                        f"Remove {len(removed_chars)} empty char mask from font [{font_path}]: {removed_chars[:10]}..."
                    )
                else:
                    logger.info(
                        f"Remove {len(removed_chars)} empty char mask from font [{font_path}]: {removed_chars}"
                    )

            self.font_support_chars_intersection_with_chars[font_path] = (
                    self.font_support_chars_cache[font_path] & chars
            )

    def filter_font_path(self, min_support_chars: int):
        """
        Filter font_path if intersection of font support chars with chars file is too few.
        This method will change content of self.font_paths. Must be called after update_font_support_chars.

        Parameters
        ----------
        min_support_chars: int
            if intersection of font support chars with chars file is lower than min_support_chars, filter this font

        """
        new_font_paths = []
        for font_path in self.font_paths:
            inter = len(self.font_support_chars_intersection_with_chars[font_path])
            if inter < min_support_chars:
                logger.info(
                    f"{font_path} is filtered cause intersection of font supported chars and chars file is too few: {inter}<{min_support_chars}"
                )
                continue

            new_font_paths.append(font_path)

        if len(new_font_paths) != self.font_paths:
            logger.info(
                f"Filter font path: {len(self.font_paths)} -> {len(new_font_paths)}"
            )
            self.font_paths = new_font_paths

    def _load_ttfont(self, font_path: str) -> TTFont:
        """
        Read ttc, ttf, otf font file, return a TTFont object
        """

        # ttc is collection of ttf
        if font_path.endswith(("ttc", "TTC")):
            ttc = TTCollection(font_path)
            # assume all ttfs in ttc file have same supported chars
            return ttc.fonts[0]

        if (
                font_path.endswith("ttf")
                or font_path.endswith("TTF")
                or font_path.endswith("otf")
                or font_path.endswith("OTF")
        ):
            ttf = TTFont(
                font_path, 0, allowVID=0, ignoreDecompileErrors=True, fontNumber=-1
            )

            return ttf

    @lru_cache()
    def _get_font(self, font_path: str, font_size: int) -> FreeTypeFont:
        font = ImageFont.truetype(font_path, font_size)
        return font


def font_check_rm_wrong(font_paths):
    for font_path in font_paths:
        print(font_path)
        if font_path.endswith(("ttc", "TTC")):
            ttc = TTCollection(font_path)
            # assume all ttfs in ttc file have same supported chars
            ttf = ttc.fonts[0]

        elif (
                font_path.endswith("ttf")
                or font_path.endswith("TTF")
                or font_path.endswith("otf")
                or font_path.endswith("OTF")
        ):
            ttf = TTFont(
                font_path, 0, allowVID=0, ignoreDecompileErrors=True, fontNumber=-1
            )
        else:
            continue
        try:
            for table in ttf["cmap"].tables:
                for k, v in table.cmap.items():
                    pass
                print()
        except AssertionError as e:
            logger.error(f"Load font file {font_path} failed, skip it. Error: {e}")
            # os.remove(font_path)
        ttf.close()


if __name__ == '__main__':
    font_dir_path = Path(r'E:\lxd\OCR_project\OCR_SOURCE\font\font_set\tem')
    font_file_path = Path(r'E:\lxd\OCR_project\OCR_SOURCE\font\font_set\font_file\font_file.txt')
    # font_file_path = None
    # for _ in range(10):
    #     fm =  FontManager(font_dir_path,font_file_path,(10,20))
    #
    #     print(f'{font_file_path}有问题')

    # print(fm.font_support_chars_cache)
