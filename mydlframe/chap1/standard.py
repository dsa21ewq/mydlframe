import numpy as np
class Variable:
    def __init__(self,data):
        self.data=data
        self.grad=None
        self.creator=None
    def set_creator(self,func):
        self.creator=func
    def backward(self):
        f=self.creator
        if f is not None:
            x=f.input
            f.backward(self)
            x.backward()


class Func:
    def __call__(self,input):
        x=input.data
        y=self.forward(x)
        self.input=input
        output =Variable(y)
        self.output=output
        output.creator=self
        return output
    def forward(self,x):
        raise NotImplementedError()
    def backward(self,y):
        raise NotImplementedError()
class square(Func):
    def forward(self,x):
        return x**2
    def backward(self,y):
        x=self.input
        x.grad=y.grad*2*x.data
class exp(Func):
    def forward(self,x):
        return np.exp(x)
    def backward(self,y):
        x=self.input
        x.grad=y.grad*y.data
def numerical_diff(f,x,eps=1e-4):
    x0=Variable(x.data-eps)
    x1=Variable(x.data+eps)
    y0=f(x0)
    y1=f(x1)
    return (y1.data-y0.data)/(2*eps)


A=square()
B=exp()
C=square()
x=Variable(np.array(0.5))
a=A(x)
b=B(a)
y=C(b)
tail=y
y.grad=np.array(1)
y.backward()
assert y.creator==C
assert y.creator.input==b
assert y.creator.input.creator==B
assert y.creator.input.creator.input==a
assert y.creator.input.creator.input.creator==A
assert y.creator.input.creator.input.creator.input==x

print(x.grad)