
# https://blog.csdn.net/feng_601/article/details/83857566
#功能：竖排文字 通过模板图片 写入文字到指定位置，并分别保存成新的图片
#功能说明：根据","换行（也可以根据"\n"换行）
#环境：PyDev 6.5.0   Python3.5.2

#python2与python3共存配置方法https://www.cnblogs.com/thunderLL/p/6643022.html

import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

#初始化字符串
strs = "往后余生,风雪是你,平淡是你,清贫也是你\n荣华是你,心底温柔是你,目光所致,也是你" #"包长荣,董亚静;包良荣,王林香;李发宁,靳海燕;王秉安;魏耀鑫"
#模板图片
imageFile = "./img/breno_unsplash.jpg"#"F:\\family\\请柬模板.JPG"
#新文件保存路径
file_save_dir = "./img/"

#初始化参数
x = 300   #横坐标
y = 20   #纵坐标
word_size = 50 #文字大小
word_css  = r"E:\lxd\PaddleOCR\StyleText\fonts/ch_standard.ttf" #字体文件   行楷
#STXINGKA.TTF华文行楷   simkai.ttf 楷体  SIMLI.TTF隶书  minijianhuangcao.ttf  迷你狂草    kongxincaoti.ttf空心草

#设置字体，如果没有，也可以不设置
font = ImageFont.truetype(word_css,word_size)

#分割得到数组
im1=Image.open(imageFile) #打开图片
draw = ImageDraw.Draw(im1)
print(font.getsize(strs))
print("竖向文字")
#
im1=Image.open(imageFile)
draw = ImageDraw.Draw(im1)
#draw.text((x, y),s.replace(",","\n"),(r,g,b),font=font) #设置位置坐标 文字 颜色 字体
right = 0   #往右位移量
down = 0    #往下位移量
w = 500     #文字宽度（默认值）
h = 500     #文字高度（默认值）
row_hight = 300 #行高设置（文字行距）
word_margin = 20 #文字间距
#一个一个写入文字
print(strs)
for k,s2 in enumerate(strs):
    if k == 0:
        w,h = font.getsize(s2)   #获取第一个文字的宽和高
    # if s2 == "," or s2 == "\n" :  #换行识别
    #     right = right + w  + row_hight
    #     down = 0
    #     continue
    # else :
    #     down = down+h + word_dir
    down = down + h + word_margin
    # print("序号-值",k,s2)
    # print("宽-高",w,h)
    # print("位移",right,down)
    # print("坐标",x+right, y+down)
    if s2==',':
        down -= int(h/2)  # 横向，纵向区别对待。是因为，纵向标点
        draw.text((x + right+int(w/4), y + down), s2, (255, 255, 0), font=font)  # 设置位置坐标 文字 颜色 字体
    else:
        draw.text((x+right, y+down),s2,(255,255,0),font=font) #设置位置坐标 文字 颜色 字体
#定义文件名 数字需要用str强转
new_filename = file_save_dir +  strs.replace(",","-").replace("\n","-") + ".jpg"
im1.save(new_filename)
del draw #删除画笔
im1.close()