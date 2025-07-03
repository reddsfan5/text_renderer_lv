from PIL import Image, ImageDraw,features,ImageFont
import numpy as np
import cv2
import os
import shutil
import time
from os import path as osp
from pathlib import Path
import unicodedata
import numpy as np
import opencc
import yaml
from PIL import ImageFont, Image, ImageDraw
from fontTools.misc.py23 import Tag
from fontTools.ttLib import TTCollection
from fontTools.ttLib import TTFont
from loguru import logger
from tqdm import tqdm

from lv_tools.font.font_correct import copy_glyph


def chr_in_font(chrs, font, font_name):
    try:
        uniMap = font['cmap'].getBestCmap()  # 获取最佳字符映射表
    except (AssertionError, AttributeError) as e:
        print(f'{font_name}: {e}')
        return False

    if not uniMap:
        print(f"Cmap为None字体：{font_name}")
        return False

    # 对于TTF字体
    if 'glyf' in font:
        glyf_table = font['glyf']
        for char in chrs:
            # 确保字符存在于uniMap中，并且相应的glyf表条目有实际的轮廓
            # glyf_table[uniMap[ord(char)]].numberOfContours <= 0会将空白字符（如：空格）过滤掉。
            ## if ord(char) not in uniMap or uniMap[ord(char)] not in glyf_table or glyf_table[uniMap[ord(char)]].numberOfContours <= 0:
            if ord(char) not in uniMap or uniMap[ord(char)] not in glyf_table:
                return False
    # 对于OTF字体，需要不同的检查逻辑，这里假定所有字符都有效
    elif 'CFF ' in font:
        # TODO: 添加针对OTF字体的字形存在性检查
        pass
    else:
        # 如果既没有glyf表也没有CFF表，则认为是不支持的字体类型
        return False

    return True




def chr_in_font_and_log(chrs, font, font_name):
    '''
    遍历chrs，确保每个字符都在uniMap中，且都在glyf表中（ttf）
    该方法只确保在规范字体中生效：
        如果，存在不规范字体，uniMap和glyf中都声明了某字符，但实际上该字符的字形用占位符糊弄，也会被认为是存在。
        如何判断某个汉字是不是在字体库中:https://cloud.tencent.com/developer/article/1576291
    目前方案，信赖cmap表。是否空白占位使用图像比对。因为glyf检查会将空白字符（如：空格）过滤掉。
    '''
    try:
        # print(font_name)

        # cmap = font['cmap']
        #
        # for i, table in enumerate(cmap.tables):
        #     print(f"子表 {i}:")
        #     print(f"  平台: {table.platformID}")
        #     print(f"  编码: {table.platEncID}")
        #     print(f"  格式: {table.format}")
        #     print(f"  语言: {table.language}")  # Windows 平台下的语言标识
        #     print(f"  覆盖码点范围: {list(table.cmap.keys())[:3]}...")  # 示例前3个码点


        # uniMap[23383]为uni5B57  ,hex(23383) 为 0x5b57.
        # uniMap = font['cmap'].getBestCmap()
        # uniMap = font['cmap'].tables[0].ttFont.getBestCmap()  # uniMap是一个字典，字典的 key 是这个字体库中所有字符的 unicode 码
        # 正确：合并所有 Unicode 子表



        uniMap = {}
        for table in font['cmap'].tables:
            if table.isUnicode():
                uniMap.update(table.cmap)

        # cff = font.get['CFF '] otf 字体没有glyf表，有个cff 表，但是字形映射在哪里记录着呢
    except (AssertionError, AttributeError) as e:
        print(f'{font["name"].names[3]}:{e}')
        return
    if not uniMap:
        print(f"Cmap为None字体：{font_name}")
        return None
    flag = True
    no_surport_char = []
    for char in chrs:
        try:
            # if ord(char) not in uniMap or (
            #         (glyf_map := font.get('glyf')) and len(glyf_map[uniMap[ord(char)]].getCoordinates(0)[0]) == 0):
            if ord(char) not in uniMap:
                flag = False
                no_surport_char.append(char)
        except:
            flag = False
            no_surport_char.append(char)
    return flag, no_surport_char, font_name


def auto_draw_bbox(img: ImageDraw, xy: tuple[float, float], text: str, font: TTFont, anchor: str):
    epsilon = 3
    draw = ImageDraw.Draw(img)
    x, y = xy

    bbox = draw.textbbox(xy, text, font=font, anchor=anchor)
    draw.rectangle(bbox, outline='green', width=2)
    draw.ellipse((x - epsilon, y - epsilon, x + epsilon, y + epsilon), fill='blue')


def font_thumb(font_dir, str1, str2, title_font_path):
    '''
    字体缩略图生成
    '''
    wrong_type = []
    font2 = ImageFont.truetype(title_font_path, 20)
    for font_name in os.listdir(font_dir):
        print(font_name)
        if osp.isdir(osp.join(font_dir, font_name)):
            continue
        font_path = osp.join(font_dir, font_name)

        try:
            if font_path.endswith(("ttc", 'TTC')):
                ttc = TTCollection(font_path)
                font = ImageFont.truetype(font_path, 90,layout_engine=ImageFont.RAQM)
                # assume all ttfs in ttc file have same supported chars
                font_ft = ttc.fonts[0]

            elif (
                    font_path.endswith("ttf")
                    or font_path.endswith("TTF")
                    or font_path.endswith("otf")
                    or font_path.endswith("OTF")
            ):
                font = ImageFont.truetype(font_path, 90,layout_engine=ImageFont.Layout.RAQM)
                font_ft = TTFont(
                    font_path, 0, allowVID=0, ignoreDecompileErrors=True, fontNumber=-1
                )
            else:
                continue
        except:
            wrong_type.append(font_name)
            continue

        # 背景图片
        img_arr = (np.ones((800, 2500, 3)) * 60).astype(np.uint8)
        im1 = Image.fromarray(img_arr)

        # 在图片上添加文字
        draw = ImageDraw.Draw(im1)
        # print(font_name)
        x, y = 200, 50

        draw.text((x, y), f'字体：{font_name}', (255, 25, 25), font=font2, anchor='lt')
        auto_draw_bbox(im1, (x, y), f'字体：{font_name}', font2, 'lt')

        try:
            if chr_in_font(str1, font_ft, font_name):
                x1, y1 = 60, 100
                draw.text((x1, y1), str1, fill='white', font=font, anchor='lt')
                auto_draw_bbox(im1, (x1, y1), str1, font, 'lt')
            if chr_in_font(str2, font_ft, font_name):
                x2, y2 = 60, 250
                draw.text((x2, y2), str2, fill='white', font=font, anchor='lt')
                auto_draw_bbox(im1, (x2, y2), str2, font, 'lt')
        except OSError as e:
            print(e)

        # 保存
        font_ft.close()
        im1.save(rf"{font_dir}/{font_name}.png")

    print(wrong_type)


def ttf_calibration(font_dir):
    '''
    将以ttf为后缀的ttc 文件找到，日志记录
    Returns
    -------
    '''
    font_dir_path = Path(font_dir)
    font_with_wrong_suf = []
    for font_path in tqdm(font_dir_path.glob('**/*')):
        if (font_path_str := str(font_path)).endswith(('ttf', 'TTF', 'otf', 'OTF')):
            infile = open(font_path_str, mode='rb')
            sfntVersion = Tag(infile.read(4))
            if sfntVersion == 'ttcf':
                font_with_wrong_suf.append(str(font_path))
                continue
            else:
                ttf = TTFont(
                    font_path, 0, allowVID=0, ignoreDecompileErrors=True, fontNumber=-1
                )
        elif font_path_str.endswith("ttc"):
            ttc = TTCollection(font_path)
            # assume all ttfs in ttc file have same supported chars
            ttf = ttc.fonts[0]
        else:
            continue

        try:
            for table in ttf["cmap"].tables:
                for kv in table.cmap.items():
                    pass
        except AssertionError as e:
            logger.error(f"Load font file {font_path} failed, skip it. Error: {e}")
    logger.info(f'错误后缀字体文件：{font_with_wrong_suf}')


def move_font_by_png_path(png_dir, font_dir):
    png_paths = [osp.join(png_dir, png_base) for png_base in os.listdir(png_dir) if png_base.endswith('png')]
    for png_path in png_paths:
        font_base = osp.splitext(osp.basename(png_path))[0]
        if osp.exists(font_path := osp.join(font_dir, font_base)):
            shutil.move(font_path, osp.join(png_dir, font_base))


def txt_cc_cvt(file_path, dst_file_path, cvt='t2s'):
    # cvt_file
    '''

    :param cvt: t2s or s2t
    :return:
    '''
    if not osp.exists(dst_dir := osp.dirname(dst_file_path)):
        os.mkdir(dst_dir)
    count = 0
    converter = opencc.OpenCC(f'{cvt}.json')
    with open(file_path, mode='r', encoding='utf8') as f, open(dst_file_path, mode='w', encoding='utf8') as wf:
        while line := f.readline():
            new_line = converter.convert(line)
            wf.write(new_line)
            print(count)
            count += 1


def abstract_traditional(src_file, dst_file):
    if not osp.exists(dst_dir := osp.dirname(dst_file)):
        os.mkdir(dst_dir)
    converter = opencc.OpenCC('t2s.json')
    cnt = 0
    with open(src_file, mode='r', encoding='utf8') as f, open(dst_file, mode='w', encoding='utf8') as wf:
        while line := f.readline():
            cnt += 1
            if cnt % 100000 == 0:
                print(cnt)
            cvter = converter.convert(line)
            if line != cvter:
                wf.write(line)


def filter_perfect_font():
    char_file = r'D:\lxd_code\OCR_SOURCE\corpus\book_name_data\counter1.yml'
    perfect_dir_path = r'D:\lxd_code\OCR_SOURCE\font\font_set\简繁-简繁-低风险\字库齐全'
    font_dir_path = Path(r'D:\lxd_code\OCR_SOURCE\font\font_set\简繁-简繁-低风险\常规类')
    with open(char_file, mode='r', encoding='utf8') as f:
        char_dict = yaml.safe_load(f)

        chars = ''.join([line.strip() for line in char_dict.keys() if line])

    font_paths = [fp for fp in font_dir_path.glob('**/*') if fp.suffix in ('.ttf', '.TTF')]
    for font_path in tqdm(font_paths):
        font_ft = TTFont(
            font_path, 0, allowVID=0, ignoreDecompileErrors=True, fontNumber=-1
        )
        flag, no_chars, font_name = chr_in_font_and_log(chars, font_ft, font_path.stem)
        if flag:
            shutil.move(str(font_path), os.path.join(perfect_dir_path, font_path.name))

        print(flag, no_chars, font_name)

def font_dis():
    
    # 创建空白图像（宽 400px，高 200px，白色背景）
    img = Image.new("RGB", (1200, 200), color=(255, 255, 255))

    # 创建绘制对象（绑定到图像）
    draw = ImageDraw.Draw(img)

    # 加载支持阿拉伯语的字体（如 Noto Naskh Arabic）
    arabic_font = ImageFont.truetype("/home/ubuntu/lxd/OCR_SOURCE/font/font_set/藏文/迷蓝—柔体.ttf",size=54,layout_engine=ImageFont.Layout.RAQM)

    # 绘制阿拉伯语文本
    arabic_text = "السلام عليكم ورحمة الله وبركاته" # 阿拉伯语“你好，世界”
    str_tib1 = r'ཀློང་ཆེན་སྙིང་གི་ཐིག་ལེའི་མཁའ་འགྲོ་བདེ་ཆེན་རྒྱལ་མོའི་སྒྲུབ་གཞུང་གི་འགྲེལ་པ་རྒྱུ'
    draw.text((10, 100), str_tib1, fill=(0, 0, 0), font=arabic_font)
    img_arr = np.array(img)
    print(img_arr.shape)

    import matplotlib
    from matplotlib import pyplot as plt
    matplotlib.use('Agg')
    plt.imshow(img)
    plt.show()

    cv2.imwrite('/home/ubuntu/lxd/text_renderer_lv/tib.jpg',img_arr)

if __name__ == '__main__':
    str1 = r'繁體：雙規鷟":eng!鄭板橋789'
    str2 = r'简体双规"：eng!镭忖廓籀徵疡郏琏轼567'

    str3 = r'117327'
    str4 = r'I247.53'
    str5 = 'แมนเชสเตอร์ซิตี้อัดอั้นตันใ'  # 泰语
    str_korea1 = '장백산항일련군가요'
    str_korea2= '내마음의시집27'
    str_japan1 = 'グラビアリアリティーとつくりやすさを追求してーズ'
    str_japan2 = '言説の変遷--厚生労働白書の分析から'
    str_fra1 = r'les contrats de déminage prévoient'
    str_fra2 = "à d'étranglement, en laissant de côté"
    str_spa1 ='Jesús Rodríguez Franco'
    str_spa2 ='Fanny T. Añaños Bedriñana'
    str_rus1=r'тЯБподиё'
    str_rus2 = r'беременна'
    str_tib1 = r'ཀློང་ཆེན་སྙིང་གི་ཐིག་ལེའི་མཁའ་འགྲོ་བདེ་ཆེན་རྒྱལ་མོའི་སྒྲུབ་གཞུང་གི་འགྲེལ་པ་རྒྱུ'
    str_tib2 = r'50བླ་མ་ཞི་བ་བཀའ་འདུས་ཀྱི་ཆོས་སྐོར།ུ༲uygur'
    str_ara1 = "تقرير: 37,3 مليار درهم حجم الرهن العقاري في الدولة في الربع الأول"
    str_ara2 = 'العرب يعطون واشنطن فرصة لقيادة مفاوضات غير مباشرة'
    lisao1 = r'摄提贞于孟陬兮'
    lisao2 = r'師傅'
    font_root = r'/home/ubuntu/lxd/OCR_SOURCE/font/font_set/阿拉伯'
    title_font_path = r'/home/ubuntu/lxd/OCR_SOURCE/font/font_set/font_art/【阿西】雾里看花-日文.ttf'


    # move_font_by_png_path(png_dir=r'D:\lxd_code\OCR\OCR_SOURCE\font\font_set\english\jinke_miaomu_done\english_miaomu\english_only',
    #                       font_dir=r'D:\lxd_code\OCR\OCR_SOURCE\font\font_set\english\jinke_miaomu_done\english_miaomu')

    # from PIL import Image, ImageDraw,features
    #
    # print(features.check_feature(feature="raqm"))


    font_thumb(font_root, str_ara1, str_ara2, title_font_path=title_font_path)
    

