class A:
    def __init__(self,
                 a=1,
                 b=2
                 ):
        self.a=a
        self.b=b


class B(A):
    def __init__(self,**para):
        super(B, self).__init__(**para)


if __name__ == '__main__':
    t = B(a=1,b=7)
    print(t.a, t.b)