import shutil
from os import path as osp
import os
from pathlib import Path
from pprint import pprint

from tqdm import tqdm

FONT_SUFFIXES = ('.otf','.ttf','.ttc','.OTF','.TTF','.TTC','.woff','.woff2')
def remove_no_suport_font(font_no_suport_text = r'E:\lxd\OCR_project\OCR_SOURCE\font/font_not_suport.txt'):

    with open(font_no_suport_text,encoding='utf8',mode='r') as f:
        font_ns_list = [ns.strip().split(' ')[0] for ns in f.readlines() if ns]
        font_ns_list = list(set(font_ns_list))
        for fp in font_ns_list:
            if osp.exists(fp):
                os.remove(fp)


def font_select_with_thumb(thumb_dir):
    thumb_preserves = [osp.join(thumb_dir,file[:-4]) for file in os.listdir(thumb_dir) if file.endswith('.png')]
    font_paths = [osp.join(thumb_dir,file) for file in os.listdir(thumb_dir) if not file.endswith('.png')]

    for font_path in font_paths:
        if font_path not in thumb_preserves:
            os.remove(font_path)

def find_font_file(font_dir,out_dir):
    font_dir_path = Path(font_dir)
    font_paths = [fnt_path for fnt_path in font_dir_path.glob('**/*') if fnt_path.suffix in FONT_SUFFIXES]
    # print(len(font_paths))
    for font_path in tqdm(font_paths):
        shutil.copyfile(str(font_path),os.path.join(out_dir,font_path.name))


if __name__ == '__main__':
    # remove_no_suport_font()
    # font_select_with_thumb(r'E:\lxd\OCR_project\OCR_SOURCE\font\font_mini\mimini')
    find_font_file(r'D:\lxd_code\OCR_SOURCE\font\fonts-main\ofl',r'D:\lxd_code\OCR_SOURCE\font\font_set\english\google_handwriting')