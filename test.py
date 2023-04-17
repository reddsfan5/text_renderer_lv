count : int
def increment():
    global count
    count = 0
    print("Count is now:", count)


def f():
    c =count + 1
    print(c)

increment()  # 输出：Count is now: 1
f()