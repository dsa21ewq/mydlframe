import numpy as np
from sympy import false


class Variable:
    def __init__(self,data):
        if data is not None:
            if not isinstance(data,np.ndarray):
                raise TypeError(f"{type(data)} is not supported.")
        self.data=data
        self.grad=None
        self.creator=None
    def clear_grad(self):
        self.grad=None
    def set_creator(self,func):
        self.creator=func
    def backward(self):
        if self.grad is None:
            self.grad=np.ones_like(self.data)
        funcs=[self.creator]# 加个[]仅用于可遍历
        while(funcs):
            f=funcs.pop()
            gys=[output.grad for output in f.outputs]
            gxs=f.backward(*gys)
            if not isinstance(gxs,tuple):
                gxs=(gxs,)
            for x,gx in zip(f.inputs,gxs):
                if x.grad is None:
                    x.grad = gx
                else: x.grad+=gx
                if x.creator is not None:
                    func=x.creator
                    if(not func.is_visited):
                        func.is_visited=True
                        funcs.append(func)
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x
class Func:
    def __call__(self,*inputs):
        self.is_visited=False
        if len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
            inputs = inputs[0]
        xs=[x.data for x in inputs]
        ys=self.forward(*xs)
        if not isinstance(ys,tuple):
            ys=(ys,)
        self.inputs=inputs
        outputs =[Variable(as_array(y)) for y in ys]
        self.outputs=outputs
        for output in outputs:
            output.creator=self
        return outputs if len(outputs)>1 else outputs[0]
    def forward(self,xs):
        raise NotImplementedError()
    def backward(self,gys):
        raise NotImplementedError()
#     传参的时候unzip，具体函数里的穿入和传出都是真实的参数数量，不含*
class Add(Func):
    def forward(self,x0,x1):
        y=x0+x1
        return (y,)#(y,)? y is not iterable
    def backward(self,gy):
        return gy,gy
class Square(Func):
    def forward(self,x):
        return x**2
    def backward(self,gy):
        x=self.inputs[0].data
        return gy*2*x
class Exp(Func):
    def forward(self,x):
        return np.exp(x)
    def backward(self,gy):
        x=self.inputs[0].data
        return gy*np.exp(x)
class Real_io_function(Func):
    def forward(self,x,y):
        return (x**2)*y,(y**2)*x
    def backward(self,ga,gb):
        x=self.inputs[0].data
        y=self.inputs[1].data
        return [2*x*y*ga+y**2 * gb,x**2*ga+2*x*y*gb]
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

def add(*x):
    return Add()(*x)

def multiple_output_function(x):
    return Square()(x),Square()(x)
def multiple_io_function(x,y):
    return Square()(x),Square()(y)

xs=[Variable(np.array(3.0)),Variable(np.array(2.0))]
x,y=Variable(np.array(3.0)),Variable(np.array(2.0))
a,b = Real_io_function()(x,y)
c=add(x,x)
z=add(x,x)
print(z.data)
z.backward()
print(x.grad)


