## 文本合成入口

E:\lxd\text_renderer_lv\main.py

还需要做的事：
* bg还有点少，需要添加。（还可以背景拼接，模拟多色背景）
* 字体需要筛选，常规字体少一点，书名常用的艺术字体多一点
* 背景长宽的拓展与字的浮动及遮挡模拟
* 模糊，色彩，透视等数据增强


## 文本合成思路

### 1.形式上的模拟

* 竖着的中文与横着的英文

  ![bookseg_0430_183237_11_IMG20220430161927_00008](readme_lv.assets/bookseg_0430_183237_11_IMG20220430161927_00008.jpg)

  

* 中文和数字的组合形式

![bookseg_0427_185462_IMG_20220426_131153(1)_00022](readme_lv.assets/bookseg_0427_185462_IMG_20220426_131153(1)_00022.jpg)

![bookseg_0427_185462_IMG_20220426_131231(1)_00024](readme_lv.assets/bookseg_0427_185462_IMG_20220426_131231(1)_00024.jpg)

* 不同底色的文本拼接

![bookseg_0427_185462_IMG_20220426_131037_00005](readme_lv.assets/bookseg_0427_185462_IMG_20220426_131037_00005.jpg)

![bookseg_0427_185462_IMG_20220426_131055_00035](readme_lv.assets/bookseg_0427_185462_IMG_20220426_131055_00035.jpg)

![bookseg_0427_185462_IMG_20220426_131103_00026](readme_lv.assets/bookseg_0427_185462_IMG_20220426_131103_00026.jpg)

![bookseg_0427_185462_IMG_20220426_131116(1)_00053](readme_lv.assets/bookseg_0427_185462_IMG_20220426_131116(1)_00053.jpg)

* 汉语文字与标点（闭合标点是横向的）

![bookseg_0427_185462_IMG_20220426_131108_00078](readme_lv.assets/bookseg_0427_185462_IMG_20220426_131108_00078.jpg)

![bookseg_0427_185462_IMG_20220426_131108_00079](readme_lv.assets/bookseg_0427_185462_IMG_20220426_131108_00079.jpg)

* 数字与闭合标点

![bookseg_0427_185462_IMG_20220426_131128(1)_00019](readme_lv.assets/bookseg_0427_185462_IMG_20220426_131128(1)_00019.jpg)

* 带有间隔的文字

![bookseg_0427_185462_IMG_20220426_131116(1)_00001](readme_lv.assets/bookseg_0427_185462_IMG_20220426_131116(1)_00001.jpg)



* 纯英文，要么横向逐字符排列

  ![bookseg_0427_185462_IMG_20220426_131027_00027](readme_lv.assets/bookseg_0427_185462_IMG_20220426_131027_00027.jpg)

  要么，竖向 逐语义排列

  这两种情况的英文切出来都是横向的，只是与汉语一起横向排列的切出后是与竖向汉语混在一起的。

* 不同大小文字混合排版

  ![bookseg_0427_185462_IMG_20220426_131221(1)_00020](readme_lv.assets/bookseg_0427_185462_IMG_20220426_131221(1)_00020.jpg)

  ![bookseg_0430_183237_11_IMG20220430161904_00001](readme_lv.assets/bookseg_0430_183237_11_IMG20220430161904_00001.jpg)

  ![bookseg_0430_183237_11_IMG20220430161910_00006](readme_lv.assets/bookseg_0430_183237_11_IMG20220430161910_00006.jpg)

  ![bookseg_0430_183237_11_IMG20220430161931_00000](readme_lv.assets/bookseg_0430_183237_11_IMG20220430161931_00000.jpg)

### 2.逼近原始文本语义

## 程序意外暂停分析
不支持的字体，系统会重新计数，但是列表总数一定，每次都会pop一个，最后计数没到，但列表已空，导致卡呆呆的。

## 不合格字体删除
1.main.py 运行时，不支持的字体会在E:\lxd\OCR_project\OCR_SOURCE\font/font_not_suport.txt文件里生成。
2.运行，costum_utils.font_check_and_remove.py 删除对应字体。
