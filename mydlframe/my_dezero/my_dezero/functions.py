from core_simple import *


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
        x=self.inputs[0]
        return gy*2*x
def square(x):
    return Square()(x)

class Exp(Func):
    def forward(self,x):
        return np.exp(x)
    def backward(self,gy):
        x=self.inputs[0]
        return gy*exp(x)

def exp(x):
    return Exp()(x)

class Mul(Func):
    def forward(self,x1,x2):
        y=x1*x2
        return (y,)
    def backward(self,gy):
        x0=self.inputs[0]
        x1=self.inputs[1]
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
        x0=self.inputs[0]
        x1=self.inputs[1]
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
        x=self.inputs[0]
        return c*x**(c-1)*gy
def pow(x,c):
    return Pow(c)(x)


class Sin(Func):
    def forward(self,x):
        y=np.sin(x)
        return y
    def backward(self,gy):
        x=self.inputs[0]
        gx=gy*cos(x)
        return gx
def sin(x):
    return Sin()(x)


class Cos(Func):
    def forward(self, x):
        return np.cos(x)
    def backward(self, gy):
        x = self.inputs[0]
        return gy * -sin(x)
def cos(x):
    return Cos()(x)

def numerical_diff(f,x,eps=1e-4):
    x0=Variable(as_array(x.data-eps))
    x1=Variable(as_array(x.data+eps))
    y0=f(x0)
    y1=f(x1)
    return (y1.data-y0.data)/(2*eps)


def newton_minimal(f,iters=10,x=Variable(np.array(1))):
    for i in range(iters):
        print(i,x)
        y = f(x)
        x.clear_grad()
        y.backward(create_graph=True)
        gx = x.grad
        x.clear_grad()
        gx.backward()
        gx2 = x.grad
        x.data -= gx.data / gx2.data

def n_order_diff(f,n,x):
    y=f(x)
    y.backward(create_graph=True)
    for i in range(n):
        gx=x.grad
        x.clear_grad()
        gx.backward(create_graph=True)
        print(x.grad)
