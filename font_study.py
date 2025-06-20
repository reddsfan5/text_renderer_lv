import time

from PIL import ImageFont,ImageDraw,Image
import numpy as np

from lv_tools.font.safe_font import FontChoice
from render_block_based_book_spine import find_max_allowed_font_size, get_text_size_hv
from text_renderer.utils import FontText
if __name__ == '__main__':
    font_path = r'D:\lxd_code\OCR\OCR_SOURCE\font\font_set\韩文\NanumMyeongjoEcoBold.ttf'
    # font_dir: str = r'D:\lxd_code\OCR\OCR_SOURCE\font\font_set\english\jinke_miaomu_done\english_jinke\font'


    # font_size = 28
    # text = '(100일이면 나도)영어 천재 .2 ,영알못, 영어에 눈을 뜨는 5주의 기적편!'
    # pil = Image.new('RGBA', size=(500,100))
    #
    # draw = ImageDraw.Draw(pil)
    #
    #
    # font = ImageFont.truetype(font_path, font_size)
    # mask = font.getmask(text)
    #
    # from matplotlib import pyplot as plt
    # plt.imshow(mask, cmap='gray')
    # plt.show()



    # callnumber_font_root = r'D:\lxd_code\OCR\OCR_SOURCE\font\font_set\索书号可用字体'
    # s = time.time()
    # font_choice = FontChoice(callnumber_font_root)
    # # font_path = font_choice.choice_safe_font('sdfs$hjSFFf#df')
    # print(font_path)
    # font = ImageFont.truetype(font_path, 20)
    # print(font)



    # ttfont_path = r'D:\lxd_code\OCR\OCR_SOURCE\font\font_set\english\jinke_miaomu_done\111\Abite-Regular.ttf'
    # ttfont = ImageFont.truetype(ttfont_path,
    #                             40)
    #
    # font_text = FontText(ttfont,'dfsdfds',ttfont_path)
    # s = time.time()
    # ret = get_text_size_hv(font_text,char_spacing=.1)
    # print(time.time() - s)

    ttfont_path = r'D:\lxd_code\OCR\OCR_SOURCE\font\font_set\english\jinke_miaomu_done\111\Abite-Regular.ttf'
    font = ImageFont.truetype(ttfont_path,40)


    text = 'dfsdfds'
    print(font.getsize(text)[1])
    bbox = font.getbbox(text)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    img = Image.new('RGBA', size=(text_width, text_height))
    draw = ImageDraw.Draw(img)
    draw.text((0,0),text,font=font,anchor='lt',fill=(255,0,0))
    from matplotlib import pyplot as plt
    plt.imshow(img)
    plt.show()



