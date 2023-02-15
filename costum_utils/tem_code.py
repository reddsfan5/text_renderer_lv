class EmptyFont(NameError):
    def __init__(self,info=None):
        self.info = info
def fun1():
    func2()
def func2():
    raise EmptyFont

if __name__ == '__main__':


    try:
        fun1()
    except EmptyFont:
        print(2)

    print(['32','3']*5)