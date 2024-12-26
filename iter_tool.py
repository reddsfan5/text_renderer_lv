from itertools import *
import time
import operator

def minute(a,b):
    return b-a



iter_tool = chain.from_iterable([[1, 2, 3, 4,5],[5,6,7,8]])
for i in iter_tool:
    print(i)
    time.sleep(.2)
