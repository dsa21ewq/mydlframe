import contextlib
import heapq
import os
import weakref
import numpy as np
from graphviz import Digraph


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
    def backward(self,retain_grad=False,create_graph=False):
        if self.grad is None:
            self.grad=Variable(np.ones_like(self.data))
        funcs=[]# 加个[]仅用于可遍历
        seen_set=set()
        def add_func(f):
            if f not in seen_set:
                heapq.heappush(funcs, (-f.generation, id(f), f))
                seen_set.add(f)

        add_func(self.creator)
        while funcs:
            _, _, f = heapq.heappop(funcs)
            gys=[output().grad for output in f.outputs]
            with using_config('enable_backprop',create_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)
                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx
                    if x.creator is not None:
                        add_func(x.creator)
                if not retain_grad:
                    for y in f.outputs:
                        y().grad = None
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


def get_dot_graph(output, verbose=True):
    dot = Digraph()
    dot.attr(rankdir='TB')
    dot.attr('node', fontname='Verdana', fontsize='10')

    funcs = []
    seen_set = set()

    # 辅助函数：确保 ID 是以字母开头的安全字符串，避免 Graphviz 语法解析错误
    def safe_id(obj):
        return "node_" + str(id(obj))

    def add_func(f):
        if f is not None and f not in seen_set:
            funcs.append(f)
            seen_set.add(f)

    # 1. 绘制起始变量节点 (输出节点)
    node_name = f"<B>{output.name}</B><BR/>" if output.name else ""
    var_label = f"<{node_name}Variable (ID:{output.id})<BR/>data: {output.data}>"
    dot.node(safe_id(output), var_label, color='orange', style='filled', shape='ellipse')

    add_func(output.creator)

    while funcs:
        f = funcs.pop()

        # 2. 绘制函数节点 (算子)
        func_label = f"{f.__class__.__name__} (ID:{f.id})\ngen: {f.generation}"
        dot.node(safe_id(f), func_label, shape='box', style='filled, bold', fillcolor='lightgrey')

        # 3. 绘制函数的所有输出 (通常是 output 本身)
        for wx in f.outputs:
            x = wx()
            if x is not None:
                x_name = f"<B>{x.name}</B><BR/>" if x.name else ""
                x_label = f"<{x_name}Variable (ID:{x.id})<BR/>data: {x.data}>"
                dot.node(safe_id(x), x_label, color='orange', style='filled', shape='ellipse')
                dot.edge(safe_id(f), safe_id(x))

        # 4. 绘制函数的所有输入
        for x in f.inputs:
            x_name = f"<B>{x.name}</B><BR/>" if x.name else ""
            # 如果有梯度，分行显示，注意：由于 gx 的 backward 也会调用此函数，这里会递归显示梯度数据
            grad_val = x.grad.data if (x.grad is not None and x.grad.data is not None) else "None"
            grad_str = f"<BR/>grad: {grad_val}" if x.grad is not None else ""

            var_label = f"<{x_name}Variable (ID:{x.id})<BR/>data: {x.data}{grad_str}>"

            dot.node(safe_id(x), var_label, color='lightblue', style='filled', shape='ellipse')
            dot.edge(safe_id(x), safe_id(f))

            if x.creator is not None:
                add_func(x.creator)

    return dot


def plot_graph(output, filename="graph.png"):
    dot = get_dot_graph(output)
    dot.attr(dpi='300')
    # 移除 filename 中的扩展名，由 format 参数决定
    base_name = filename.split('.')[0]
    dot.render(base_name, format='png', cleanup=True)
    print(f"计算图已成功保存至: {base_name}.png")