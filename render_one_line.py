import random
import colorsys
import spacy
from PIL import Image, ImageDraw, ImageFont

from render_block_based_book_spine import FontChoice


def main1():
    '''
    方案一：水平/垂直渐变颜色
    效果：书名文字颜色从一种颜色渐变到另一种颜色（如彩虹渐变）。
    实现思路：将书名拆分为字符，为每个字符分配渐变色值。

    '''

    def gradient_text(text, font_path, font_size, start_color, end_color, direction='horizontal'):
        # 创建临时图像计算文字尺寸
        font = ImageFont.truetype(font_path, font_size)
        dummy_img = Image.new('RGB', (1, 1))
        dummy_draw = ImageDraw.Draw(dummy_img)
        text_width = dummy_draw.textlength(text, font=font)
        text_height = font_size

        # 创建渐变画布
        img = Image.new('RGB', (int(text_width), text_height), 'white')
        draw = ImageDraw.Draw(img)

        # 生成渐变色
        if direction == 'horizontal':
            for x in range(int(text_width)):
                ratio = x / text_width
                r = int(start_color[0] + (end_color[0] - start_color[0]) * ratio)
                g = int(start_color[1] + (end_color[1] - start_color[1]) * ratio)
                b = int(start_color[2] + (end_color[2] - start_color[2]) * ratio)
                draw.line([(x, 0), (x, text_height)], fill=(r, g, b))
        else:  # vertical
            for y in range(text_height):
                ratio = y / text_height
                r = int(start_color[0] + (end_color[0] - start_color[0]) * ratio)
                g = int(start_color[1] + (end_color[1] - start_color[1]) * ratio)
                b = int(start_color[2] + (end_color[2] - start_color[2]) * ratio)
                draw.line([(0, y), (text_width, y)], fill=(r, g, b))

        # 用文字作为蒙版裁剪渐变
        mask = Image.new('L', img.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.text((0, 0), text, fill=255, font=font)
        img.putalpha(mask)

        return img

    # 使用示例
    title = "Healr's Touch"
    gradient_image = gradient_text(
        text=title,
        font_path="arial.ttf",
        font_size=72,
        start_color=(255, 0, 0),  # 红色
        end_color=(0, 0, 255),  # 蓝色
        direction='horizontal'
    )
    from matplotlib import pyplot as plt
    plt.imshow(gradient_image)
    plt.show()


def main2():
    '''
    方案二：字符随机颜色混合
    效果：每个字符随机分配颜色，形成多彩效果。
    实现思路：逐字符渲染并随机选择颜色。
    Returns
    -------

    '''

    def random_color_text(text, font_path, font_size, colors):
        # 初始化画布
        font = ImageFont.truetype(font_path, font_size)
        char_widths = [font.getbbox(char)[2] for char in text]
        total_width = sum(char_widths)
        img = Image.new('RGB', (total_width, font_size), 'white')
        draw = ImageDraw.Draw(img)

        # 逐字符绘制
        x = 0
        for i, char in enumerate(text):
            color = random.choice(colors)
            draw.text((x, 0), char, fill=color, font=font)
            x += char_widths[i]

        return img

    # 使用示例
    title = "Healr's Touch"
    colors = [
        (255, 0, 0),  # 红
        (0, 255, 0),  # 绿
        (0, 0, 255),  # 蓝
        (255, 128, 0)  # 橙
    ]

    random_text_image = random_color_text(
        text=title,
        font_path="arial.ttf",
        font_size=72,
        colors=colors
    )
    from matplotlib import pyplot as plt
    plt.imshow(random_text_image)
    plt.show()


def main3():
    '''
    方案三：HSV色轮动态混合
    效果：书名颜色按色相（Hue）动态变化，类似彩虹效果。
    实现思路：利用 HSV 颜色空间生成平滑过渡。

    Returns
    -------

    '''
    def hsv_color_text(text, font_path, font_size, saturation=1.0, value=1.0):
        # 初始化画布
        font = ImageFont.truetype(font_path, font_size)
        text_width = font.getlength(text)
        text_height = font_size
        img = Image.new('RGB', (int(text_width), text_height), 'white')
        draw = ImageDraw.Draw(img)

        # 生成HSV色相渐变
        for x in range(int(text_width)):
            hue = x / text_width  # 色相范围 0~1
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            draw.line([(x, 0), (x, text_height)], fill=(
                int(r * 255), int(g * 255), int(b * 255)
            ))

        # 文字蒙版裁剪
        mask = Image.new('L', img.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.text((0, 0), text, fill=255, font=font)
        img.putalpha(mask)

        return img

    # 使用示例
    hsv_text_image = hsv_color_text(
        text="Healr's Touch",
        font_path="arial.ttf",
        font_size=72,
        saturation=0.9,
        value=0.9
    )
    from matplotlib import pyplot as plt
    plt.imshow(hsv_text_image)
    plt.show()


def main4():
    '''
    方案四：多色分段混合（高级）
    效果：书名不同部分使用不同颜色规则（如名词红色、动词绿色）。
    实现思路：结合 NLP 词性标注动态分配颜色。

    Returns
    -------

    '''
    # 加载英文NLP模型
    nlp = spacy.load("en_core_web_sm")

    def pos_color_text(text, font_path, font_size):
        # 分析词性
        doc = nlp(text)
        color_map = {
            'NOUN': (255, 0, 0),  # 红色
            'VERB': (0, 255, 0),  # 绿色
            'ADJ': (0, 0, 255),  # 蓝色
            'DEFAULT': (0, 0, 0)  # 黑色
        }

        # 初始化画布
        font = ImageFont.truetype(font_path, font_size)
        text_width = font.getlength(text)
        img = Image.new('RGB', (int(text_width), font_size), 'white')
        draw = ImageDraw.Draw(img)
        space_width = font.getlength('  ')
        x = 0
        for token in doc:
            color = color_map.get(token.pos_, color_map['DEFAULT'])
            char_width = font.getlength(token.text)
            draw.text((x, 0), token.text, fill=color, font=font)
            x += char_width + space_width

        return img

    # 使用示例
    pos_text_image = pos_color_text(
        text="This is a sample text that needs to be split into multiple lines without breaking words.",
        font_path="arial.ttf",
        font_size=72
    )
    from matplotlib import pyplot as plt

    plt.imshow(pos_text_image)
    plt.show()


'''

扩展技巧
添加阴影/描边：使用 stroke_width 参数增强可读性：

draw.text((x, y), text, fill=color, font=font, stroke_width=2, stroke_fill='black')

叠加纹理：将文字与材质图片混合：

texture = Image.open("texture.jpg").resize(img.size)
img = Image.blend(img, texture, alpha=0.3)


'''



if __name__ == '__main__':
    main3()
