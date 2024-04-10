import re
from dataclasses import dataclass

'''
参照：F:\D\dataset\OCR\series_book\有背景图案的阿拉伯数字

很多序列是没有很明显的组合关系的。不是 ”第xxx章“ 这种形式。

'''

SERIES_END_SET = ('卷', '册', '部', '篇', '章', '期', '幕', '集', '场')


def split_text_by_numbers(text):
    # 使用正则表达式将文本按数字位置切分为列表
    parts = re.findall(r'\d+|\D+', text)
    return parts


class Series:
    pass


def series_text_gen():
    '''
    武林外传 第15集
    可分为：【武林外传 第】 【15】 【集】 三部分。
    '''
    pass


@dataclass(order=True, frozen=True)
class SimpleData:
    name: str
    height: int
    hair: int


if __name__ == '__main__':
    text = "第15集"
    result = split_text_by_numbers(text)
    print(result)
