import contextlib
import heapq
import weakref
import numpy as np
class Config:
    enable_backprop=True
@contextlib.contextmanager
def using_config(name,value):
    old_value=getattr(Config,'enable_backprop')
    setattr(Config,'enable_backprop',value)
    try:
        yield
    finally:
        setattr(Config,'enable_backprop',old_value)
def no_grad():
    return using_config('enable_backprop',False)
class Variable:
    __counter=0
    __array_priority__=200
    def __mul__(self, other):
        return mul(self,other)
    def __rmul__(self, other):
        return mul(self,other)
    def __add__(self, other):
        return add(self,other)
    def __radd__(self, other):
        return add(self,other)
    def __sub__(self, other):
        return sub(self,other)
    def __rsub__(self, other):
        return rsub(self,other)
    def __truediv__(self, other):
        return div(self,other)
    def __rtruediv__(self, other):
        return rdiv(self,other)
    def __neg__(self):
        return neg(self)
    def __pow__(self, power, modulo=None):
        return pow(self,power)
    def shape(self):
        return self.data.shape
    def ndim(self):
        return self.data.ndim
    def size(self):
        return self.data.size
    def dtype(self):
        return self.data.dtype
    def __len__(self):
        return len(self.data)
    def __repr__(self):
        if self.data is None:
            return 'Variable(None)'
        else:
            p=str(self.data).replace('\n','\n'+' ' * 9)
            return f'Variable({p})'
    def __init__(self,data,name=None):
        if data is not None:
            if not isinstance(data,np.ndarray):
                raise TypeError(f"{type(data)} is not supported.")
        self.data=data
        self.name=name
        self.grad=None
        self.creator=None
        self.generation=0
        self.id = Variable.__counter
        Variable.__counter += 1
    def clear_grad(self):
        self.grad=None
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1
    def backward(self,retain_grad=False):
        if self.grad is None:
            self.grad=np.ones_like(self.data)
        funcs=[]# 加个[]仅用于可遍历
        # 我并不理解seen_set 的作用，开个boolean似乎是等价的
        seen_set=set()
        def add_func(f):
            if f not in seen_set:
                heapq.heappush(funcs, (-f.generation, id(f), f))
                seen_set.add(f)

        add_func(self.creator)
        while funcs:
            _, _, f = heapq.heappop(funcs)
            gys=[output().grad for output in f.outputs]
            gxs=f.backward(*gys)
            if not isinstance(gxs,tuple):
                gxs=(gxs,)
            for x,gx in zip(f.inputs,gxs):
                if x.grad is None:
                    x.grad = gx
                else: x.grad=x.grad+gx
                if x.creator is not None:
                    add_func(x.creator)
            if not retain_grad:
                for y in f.outputs:
                    y().grad=None
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x
def as_variable(x):
    if isinstance(x,Variable):
        return x
    return Variable(x)
class Func:
    __counter = 0
    def __call__(self,*inputs,name=None):
        self.id = Func.__counter
        Func.__counter += 1
        class_name = self.__class__.__name__
        self.name = f"{name if name else class_name} (ID:{self.id})"
        if len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
            inputs = inputs[0]
        inputs=[as_variable(x) for x in inputs]
        xs=[x.data for x in inputs]
        ys=self.forward(*xs)
        if not isinstance(ys,tuple):
            ys=(ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        if Config.enable_backprop:
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]
            self.generation = max([input.generation for input in self.inputs])
            for output in outputs:
                output.set_creator(self)
        return outputs if len(outputs)>1 else outputs[0]
    def forward(self,xs):
        raise NotImplementedError()
    def backward(self,gys):
        raise NotImplementedError()
#     传参的时候unzip，具体函数forward和backward里的穿入和传出都是真实的参数数量的一些数值，不含*
class Add(Func):
    def forward(self,x0,x1):
        y=x0+x1
        return (y,)#(y,)? y is not iterable
    def backward(self,gy):
        return gy,gy
def add(x0,x1):
    x1=as_array(x1)
    return Add()(x0,x1)
class Square(Func):
    def forward(self,x):
        return x**2
    def backward(self,gy):
        x=self.inputs[0].data
        return gy*2*x
def square(x):
    return Square()(x)

class Exp(Func):
    def forward(self,x):
        return np.exp(x)
    def backward(self,gy):
        x=self.inputs[0].data
        return gy*np.exp(x)

def exp(x):
    return Exp()(x)

class Mul(Func):
    def forward(self,x1,x2):
        y=x1*x2
        return (y,)
    def backward(self,gy):
        x0=self.inputs[0].data
        x1=self.inputs[1].data
        return x1*gy,x0*gy
def mul(x0,x1):
    x1 = as_array(x1)
    return Mul()(x0,x1)

class Neg(Func):
    def forward(self,x):
        return -x
    def backward(self,gy):
        return -gy
def neg(x):
    return Neg()(x)
class Sub(Func):
    def forward(self,x0,x1):
        return x0-x1
    def backward(self,gy):
        return gy,-gy
def sub(x0,x1):
    x1=as_array(x1)
    return Sub()(x0,x1)
def rsub(x0,x1):
    x1=as_array(x1)
    return Sub()(x1,x0)

class Div(Func):
    def forward(self,x0,x1):
        return x0/x1
    def backward(self,gy):
        x0=self.inputs[0].data
        x1=self.inputs[1].data
        return gy/x1,gy*(-1)*(x0/x1**2)
def div(x0,x1):
    x1=as_array(x1)
    return Div()(x0,x1)
def rdiv(x0,x1):
    x1=as_array(x1)
    return Div()(x1,x0)

class Pow(Func):
    def __init__(self,c):
        self.c=c
    def forward(self,x):
        return x**self.c
    def backward(self,gy):
        c=self.c
        x=self.inputs[0].data
        return c*x**(c-1)*gy
def pow(x,c):
    return Pow(c)(x)


def numerical_diff(f,x,eps=1e-4):
    x0=Variable(as_array(x.data-eps))
    x1=Variable(as_array(x.data+eps))
    y0=f(x0)
    y1=f(x1)
    return (y1.data-y0.data)/(2*eps)
