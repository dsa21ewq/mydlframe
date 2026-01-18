import heapq
import weakref

import numpy as np
import os
from graphviz import Digraph
import contextlib

from pandas.core.internals.construction import ndarray_to_mgr

class Config:
    enable_backprop=True
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

def add(x0,x1):
    x1=as_array(x1)
    return Add()(x0,x1)

def multiple_output_function(x):
    return Square()(x),Square()(x)
def multiple_io_function(x,y):
    return Square()(x),Square()(y)


a=Variable(np.array(2))
b=Variable(np.array(3))
print(3*a+1)
#
#
# def get_dot_graph(output, verbose=True):
#     os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'
#     dot = Digraph()
#     funcs = []
#     seen_set = set()
#
#     def add_func(f):
#         if f is not None and f not in seen_set:
#             funcs.append(f)
#             seen_set.add(f)
#
#     # 绘制起始输出节点
#     var_label = f"Variable (ID:{output.id})\ndata: {output.data}"
#     dot.node(str(id(output)), var_label, color='orange', style='filled')
#
#     add_func(output.creator)
#
#     while funcs:
#         f = funcs.pop()
#         # 绘制函数节点
#         func_label = f"{f.__class__.__name__} (ID:{f.id})\ngen: {f.generation}"
#         dot.node(str(id(f)), func_label, shape='record')
#
#         # --- 修改点 1: 处理弱引用输出 ---
#         for wx in f.outputs:
#             x = wx()  # 调用弱引用获取实例
#             if x is not None:
#                 # 只有当变量还存在时才连线
#                 dot.edge(str(id(f)), str(id(x)))
#
#         # 连线: 输入变量 -> 函数
#         for x in f.inputs:
#             var_label = f"Variable (ID:{x.id})\ndata: {x.data}\ngrad: {x.grad}"
#             dot.node(str(id(x)), var_label, color='lightblue', style='filled')
#             dot.edge(str(id(x)), str(id(f)))
#
#             if x.creator is not None:
#                 add_func(x.creator)
#
#     return dot
# def plot_graph(output, filename="graph.png"):
#     dot = get_dot_graph(output)
#     # 保存为图片
#     dot.render(filename.split('.')[0], format='png', cleanup=True)
#     print(f"计算图已保存至: {filename}")
#
# # --- 使用示例 ---
# # 确保你之前的变量 A 已经计算完成
# plot_graph(A, "my_computation_graph.png")