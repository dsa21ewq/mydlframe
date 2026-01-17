import numpy as np
import unittest as ut
class Variable:
    def __init__(self,data):
        if data is not None:
            if not isinstance(data,np.ndarray):
                raise TypeError(f"{type(data)} is not supported.")
        self.data=data
        self.grad=None
        self.creator=None
    def set_creator(self,func):
        self.creator=func
    def backward(self):
        if self.grad is None:
            self.grad=np.ones_like(self.data)
        funcs=[self.creator]
        while(funcs):
            f=funcs.pop()
            x,y=f.input,f.output
            x.grad=f.backward(y.grad)
            if x.creator is not None:
                funcs.append(x.creator)
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x
class Func:
    def __call__(self,input):
        x=input.data
        y=self.forward(x)
        self.input=input
        output =Variable(as_array(y))
        self.output=output
        output.creator=self
        return output
    def forward(self,x):
        raise NotImplementedError()
    def backward(self,gy):
        raise NotImplementedError()
class Square(Func):
    def forward(self,x):
        return x**2
    def backward(self,gy):
        x=self.input.data
        return gy*2*x
class Exp(Func):
    def forward(self,x):
        return np.exp(x)
    def backward(self,gy):
        x=self.input.data
        return gy*np.exp(x)
def numerical_diff(f,x,eps=1e-4):
    x0=Variable(x.data-eps)
    x1=Variable(x.data+eps)
    y0=f(x0)
    y1=f(x1)
    return (y1.data-y0.data)/(2*eps)

def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)



x=Variable(np.array(0.5))
a=square(x)
b=exp(a)
y=square(b)
y.backward()
assert y.creator.input==b
assert y.creator.input.creator.input==a
assert y.creator.input.creator.input.creator.input==x

# print(x.grad)

class SquareTest(ut.TestCase):
    def test_forward(self):
        x=Variable(np.array(2.0))
        y=square(x)
        expected=np.array(4.0)
        self.assertEqual(y.data,expected)
    def test_backward(self):
        x=Variable(np.array(3.0))
        y=square(x)
        y.backward()
        expected=np.array(6.0)
        self.assertEqual(x.grad,expected)
    def test_check_gradient(self):
        x=Variable(np.random.rand(1))
        y=square(x)
        y.backward()
        num_grad=numerical_diff(square,x)
        flag=np.allclose(x.grad,num_grad)
        self.assertTrue(flag)