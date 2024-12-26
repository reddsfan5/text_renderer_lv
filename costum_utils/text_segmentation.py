import random
from typing import List

import jieba


def limit_text_length(text_list: List[str], len_limit=30):
    # 限制文本长度
    text_list = [text[:len_limit] for text in text_list if text]
    return text_list


def text_cut(text:str) -> List[str]:
    cuted_list = list(jieba.cut(text, cut_all=False))
    return cuted_list
def add_space(cuted_list:List[str]) -> str:
    '''
    传入一个分词后的列表，在元素间插入空格，然后输出拼合字符串。
    :param cuted_list:
    :return:
    '''
    if len(cuted_list) > 1:
        cuted_list_with_space = [text + ' ' * random.randint(0, 6) for text in cuted_list[:-1]] + [cuted_list[-1]]
        return ''.join(cuted_list_with_space)
    else:
        return cuted_list[0]


def limit_text_and_add_space(text_list, limit_len=None,is_add_space=True):
    if limit_len:
        text_list = limit_text_length(text_list, limit_len)
    text_list_with_space = []
    for text_item in text_list:
        if not text_item:
            continue
        cutted_list = text_cut(text_item)
        if is_add_space:
            text_list_with_space.append(add_space(cutted_list))
        else:
            text_list_with_space.append(cutted_list)
    return text_list_with_space
if __name__ == '__main__':

    print(limit_text_and_add_space(['给我的侄子阿尔伯特，我fsajfklsLKFj离开了这个岛，我在扑克游戏中赢了胖子哈根','哈利波特*海上传奇']))