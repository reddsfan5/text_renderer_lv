# -*- coding: utf-8 -*-
import lmdb
env_db = lmdb.open(r"D:\dataset\OCR\single_lmdb_num_1000")
txn = env_db.begin()

for key, value in txn.cursor():  # 遍历
    print(key)

env_db.close()
