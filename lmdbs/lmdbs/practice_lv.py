# -*- coding: utf-8 -*-
import lmdb

# # map_size定义最大储存容量，单位是kb，以下定义1TB容量
# env = lmdb.open(r"D:\dataset\for_lmdb", map_size=1099511627776)
#
# txn = env.begin(write=True)
#
# # 添加数据和键值
# txn.put(key='1'.encode(), value='aaa'.encode())
# txn.put(key='2'.encode(), value='bbb'.encode())
# txn.put(key='3'.encode(), value='ccc'.encode())
#
# # 通过键值删除数据
# txn.delete(key='1'.encode())
#
# # 修改数据
# txn.put(key='3'.encode(), value='ddd'.encode())
#
# # 通过commit()函数提交更改
# txn.commit()
# env.close()
env = lmdb.open(r"E:\lxd_dataset\OCR_OUTPUT\effect_layout_image\effect_ensembleauthor", map_size=1099511)

txn = env.begin(write=False)
print(txn.get('id-000000445'.encode()))
    # print(i)
# print(txn.cursor())