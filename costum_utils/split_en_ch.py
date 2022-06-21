# 对中英文文本进行分离，使用标识符标明其属性。
'''
需要划分到英文的部分：
    1. 英文
    2.闭合标点
非闭合标点，中心打印。
'''
s = "中文，英文 夹杂：China's Leg♥end Holdings will split its several business arms to go public on stock markets, 又一段中文，the group's president Zhu Linan said on Tuesday.该集团总裁朱利安周二表示，中国联想控股将分拆其多个业务部门在股市上市。"
str_list = []
cn_str = ''
en_str = ''
en_state = 0
cn_state = 0
for i in s:
    if ord(i)<256: # 英文
        en_str += i
        if cn_str and cn_str not in [' ','\n','\b','\t']:
            str_list.append(('ch',cn_str))
            cn_str = ''
    else:
        cn_str += i
        if en_str and en_str not in [' ','\n','\b','\t']:
            str_list.append(('en',en_str))
            en_str = ''

print(str_list)




# 这种简单的实现无法对混杂的中英文进行多段分离。
# import re
# s = "China's Legend Holdings will split its several business arms to go public on stock markets, the group's president Zhu Linan said on Tuesday.该集团总裁朱利安周二表示，中国联想控股将分拆其多个业务部门在股市上市。"
# uncn = re.compile(r'[\u0061-\u007a,\u0020]')
# en = "".join(uncn.findall(s.lower()))
# print(en)
#中文的编码范围是：\u4e00-\u9fa5，相应的[^\u4e00-\u9fa5]可匹配非中文。
# 匹配英文时，需要将空格[\u0020]加入，不然单词之间没空格了。