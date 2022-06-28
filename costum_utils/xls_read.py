import os

import pandas as pd
from os import path as osp
def corpus_txt(corpus_dir, corpus_base,char_limit=25):
    with open(osp.join(corpus_dir, f'{corpus_base}_less_{char_limit}.txt'), mode='a', encoding='utf8') as f:
        '''
        图书目录:书名,出版社,作者
        booklibrary_ext: name, publisher,author
        馆藏清单20220606: 题名，出版社,责任者
        漯河市图书馆: 题名, 出版社, 著者
        天津滨海新区图书: 题名，出版社, 作者
        通借通还库分类号为I的馆藏: 题名，出版社, 作者
        
        '''

        item = '题名'
        for index in df[item].index:
            if len(str(df[item].get(index)))>char_limit:
                continue
            f.write(f"{df[item].get(index)}\n")

def image_list_txt(txt_path,bg_dir):
    rel_names = [osp.join(osp.basename(bg_dir),base) for base in os.listdir(bg_dir)]
    with open(txt_path,encoding='utf8',mode='w') as f:
        for index,rel_name in enumerate(rel_names):
            f.write(f"{rel_name}\t{index}\n")

def merge_txt(txt_dir,dst_txt_path):
    with open(dst_txt_path,mode='a',encoding='utf8') as merge_file:
        txt_paths = [osp.join(txt_dir,tp) for tp in os.listdir(txt_dir)]
        for txt_path in txt_paths:
            print(txt_path)
            with open(txt_path,mode='r',encoding='utf8') as f:
                content = f.read().strip()
                merge_file.write(content)


if __name__ == '__main__':
    xls_path = r'E:\lxd_dataset\图书目录/图书目录.xls'
    df = pd.read_excel(xls_path, header=0)
    # pd.set_option('display.width', None)
    # pd.set_option('display.max_columns', None)
    # print(df.columns.values)
    corpus_base = rf'{osp.splitext(osp.basename(xls_path))[0]}'
    corpus_dir = r'E:\lxd\PaddleOCR\StyleText\examples\corpus'
    img_list_path = r'E:\lxd\PaddleOCR\StyleText\examples/image_list_sel.txt'
    bg_dir = r'E:\lxd\PaddleOCR\StyleText\examples\style_images_selected'
    txt_dir = r'E:\lxd\PaddleOCR\StyleText\examples\corpus'
    merged_txt_path = r'E:\lxd\PaddleOCR\StyleText\examples/corpus_merged.txt'

    # corpus_txt(corpus_dir,corpus_base)

    # 列名展示 主要列名：[书名，作者，出版社]

    # image_list_txt(img_list_path,bg_dir)

    merge_txt(txt_dir,merged_txt_path)