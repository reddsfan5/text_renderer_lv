import json
import os
from pprint import pprint
import unicodedata
import pandas as pd
from os import path as osp
import string
import re

def corpus_txt(corpus_dir, corpus_base,char_limit=100):
    with open(osp.join(corpus_dir, f'{corpus_base}_less_{char_limit}.txt'), mode='a', encoding='utf8') as f:
        '''
        图书目录:书名,出版社,作者
        booklibrary_ext: name, publisher,author
        馆藏清单20220606: 题名，出版社,责任者
        漯河市图书馆: 题名, 出版社, 著者
        天津滨海新区图书: 题名，出版社, 作者
        通借通还库分类号为I的馆藏: 题名，出版社, 作者
        
        '''

        item = 'author'
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
        print(txt_paths)
        for txt_path in txt_paths:
            print(txt_path)
            with open(txt_path,mode='r',encoding='utf8') as f:
                content = f.read().strip()
                merge_file.write(content)

def txt_filter(dst_path,text_path=r'E:\lxd\OCR_project\OCR_SOURCE\corpus\merged/author_bookname_all.txt',filter_path=r'E:\lxd\OCR_project\OCR_SOURCE\corpus/chn_charset_dict_8k.txt'):
    with open(filter_path, encoding='utf8', mode='r') as chr:
        chr_set = set(chr.read().split('\n'))
    with open(text_path, mode='r', encoding='utf8') as f,open(dst_path,encoding='utf8',mode='a') as fp:
        text_list = f.read().split('\n')[:-1]  # 直接截掉最后一行，这行通常为空行
        # 防止空行
        text_list = [''.join(list(filter(lambda x: x in chr_set, text))) for text in text_list if text]
        text_list = [text for text in text_list if text and len(text)<=25]

        text_list = list(set(text_list))
        for index,f_text in enumerate(text_list):
            fp.write(f_text)
            if index<len(text_list)-1:
                fp.write('\n')

def get_rare_text(not_suport_text,output_path):
    with open(not_suport_text,encoding='utf8',mode='r') as f,open(output_path,encoding='utf8',mode='w') as w1:
        content = [i.strip() for i in f.readlines()]
        core_contents = [i.split(' ')[1] for i in content]
        # print(core_contents)
        for core in core_contents[:-1]:
            w1.write(core)
            w1.write('\n')
        w1.write(core_contents[-1])


def read_open_library_file(path, max_line = None, field='title'):
    '''
    field:  title(书名),name(作者)
    Parameters
    ----------
    path
    max_line
    field

    Returns
    -------

    '''
    dir_name,base_name = osp.split(path)
    stem,ext = osp.splitext(base_name)
    exclude = set('?=#')
    with open(path,encoding='utf8') as file,open(osp.join(dir_name,stem+f'_{field}'+'.txt'),mode='w',encoding='utf8') as f2:
        count = 0
        # # 移除指定字符
        for line in file:
            if max_line and count > max_line:
                break
            data = line.strip().split('\t')
            details = eval(data[4])  # 或者使用 ast.literal_eval() 函数
            title = details.get(field,None)
            if not title:
                continue
            title = ''.join(char for char in title if char not in exclude)
            if not contains_foreign(title) and 10<len(title)<100:
                # print(title)
                f2.write(title)
                f2.write('\n')
            count += 1
            print(count)

def read_goodreads_file(path, max_line = None, field='original_title'):
    '''
    field:  original_title(书名),name(作者)
    Parameters
    ----------
    path
    max_line
    field

    Returns
    -------

    '''
    dir_name,base_name = osp.split(path)
    stem,ext = osp.splitext(base_name)
    exclude = set('?=#')
    with open(path,encoding='utf8') as file,open(osp.join(dir_name,stem+f'_{field}'+'.txt'),mode='w',encoding='utf8') as f2:
        count = 0
        # # 移除指定字符
        jds = json.load(file)

        for jd in jds:
            if max_line and count > max_line:
                break
  # 或者使用 ast.literal_eval() 函数
            title = jd.get(field,None)
            if not title:
                continue
            title = ''.join(char for char in title if char not in exclude)
            if not contains_foreign(title) and 10<len(title)<100:
                # print(title)
                f2.write(title)
                f2.write('\n')
            count += 1
            print(count)



def contains_foreign(text):
    # 过滤掉标点符号和数字
    exclude = set(string.punctuation + string.digits)
    text = ''.join(char for char in text if char not in exclude)

    # 检查文本是否包含非英语字符
    regex = re.compile('[^a-zA-Z\s]')
    return bool(regex.search(text))


def merge_files(file1, file2, output_file):
    # 打开两个输入文件和输出文件
    with open(file1, 'r',encoding='utf8') as f1, open(file2, 'r',encoding='utf8') as f2, open(output_file, 'w',encoding='utf8') as fout:
        # 逐行读取两个输入文件并写入输出文件

            # 如果每次读取的数据量很大，可以使用缓冲区技术来提高效率
            # 例如：
            lines1 = f1.readlines()  # 读取10000个字节的数据
            lines2 = f2.readlines()  # 读取10000个字节的数据
            fout.writelines(lines1 + lines2)

if __name__ == '__main__':
    file1 = r'E:\lxd_dataset\图书目录\Open_Library_ol_dump_authors_2023-02-28_name.txt'
    file2 = r'E:\lxd_dataset\图书目录\Open_Library_ol_dump_works_2023-02-28_title.txt'
    file3 = r'E:\lxd_dataset\图书目录\Open_Library_ol_dump_title_name.txt'
    merge_files(file1,file2,file3)


    # xls_path = r'E:\lxd_dataset\图书目录/booklibrary_ext.xlsx'
    # df = pd.read_excel(xls_path, header=0)
    # pd.set_option('display.width', None)
    # pd.set_option('display.max_columns', None)
    # print(df.columns.values)
    # corpus_base = rf'bookname_{osp.splitext(osp.basename(xls_path))[0]}'
    # corpus_dir = r'E:\lxd\OCR_project\OCR_SOURCE\corpus__author'
    # img_list_path = r'E:\lxd\PaddleOCR\StyleText\examples/image_list_sel.txt'
    # bg_dir = r'E:\lxd\PaddleOCR\StyleText\examples\style_images_selected'
    # txt_dir = r'E:\lxd\OCR_project\OCR_SOURCE\corpus\merged'
    # merged_txt_path = r'E:\lxd\OCR_project\OCR_SOURCE\corpus\author\author_bookname_all.txt'
    # dst_path = r''
    # # corpus_txt(corpus_dir,corpus_base)
    #
    # # 列名展示 主要列名：[书名，作者，出版社]
    #
    # # image_list_txt(img_list_path,bg_dir)
    #
    # # merge_txt(txt_dir,merged_txt_path)
    # txt_filter(r'E:\lxd\OCR_project\OCR_SOURCE\corpus\author_bookname/filtered_author_bookname.txt')
    # read_goodreads_file(r'E:\lxd_dataset\图书目录\goodreads_book_authors.json',field='name')

