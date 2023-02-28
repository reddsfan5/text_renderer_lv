import os
import shutil
from pathlib import Path


def translate_digit(num, orders=('', '十'), digit_maps=('',) + tuple('一二三四五六七八九十')):
    num = int(num)
    num_str = ''
    for i in range(2):
        num, digit = divmod(num, 10)
        if digit:
            num_str = digit_maps[digit] + orders[i] + num_str
    print(num_str)
    return num_str


def dark_img_filter(src=r'\\192.168.1.183\Public\吕晓东\占座\linyi_v2_anti_pers_train_p2',
                    dst_dir=r'\\192.168.1.183\Public\吕晓东\占座\linyi_v2_anti_pers_train_p2\maybe_dark'):
    src_path = Path(src)
    ret = src_path.iterdir()
    file_dark = [file for file in ret if file.stem.endswith('18')]
    for file in file_dark:
        shutil.move(str(file), os.path.join(dst_dir, file.name))
    print(file_dark)


def text_gen(repeat_times = 30):
    CHARS2 = '1234567890'
    text_path = 'D:\lxd_code\OCR_SOURCE\corpus\digit_str/digit_text.txt'
    CHARS1 = 'ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ①②③④⑤⑥⑦⑧⑨⑩'
    with open(text_path,mode='w',encoding='utf8') as f:

        orders_list = [('', '十'),
                        ('', '拾')]
        digit_maps_list = [('',) + tuple('一二三四五六七八九十'),
                               ('',) + tuple('壹贰叁肆伍陆柒捌玖拾')]


        for i in range(50):
            for orders,digit_maps in zip(orders_list,digit_maps_list):
                digit_str = translate_digit(i, orders, digit_maps)
                if digit_str:
                    for _ in range(repeat_times):
                        f.write(digit_str)
                        f.write('\n')
                    package_digit_str(digit_str,f)

        for i in range(1,50):
            for _ in range(repeat_times):
                f.write(str(i))
                f.write('\n')
            package_digit_str(str(i), f)
        for i in CHARS1:
            for _ in range(repeat_times):
                f.write(str(i))
                f.write('\n')
            package_digit_str(str(i), f)

def package_digit_str(digit_str,f):
    punctuations = ['()', '[]', '<>', '【】', '《》']
    for punc in punctuations:
        closed_digit_str = punc[0] + digit_str + punc[1]
        f.write(closed_digit_str)
        f.write('\n')


if __name__ == '__main__':
    # text_gen()
    l = [i for i in range(10)]
    l *= 10
    print(l)



