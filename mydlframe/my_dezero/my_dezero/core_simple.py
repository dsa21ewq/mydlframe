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
    def reshape(self,*shape):
        if len(shape)==1 and isinstance(shape[0],(tuple,list)):
            shape=shape[0]
        return reshape(self,shape)
    def transpose(self):
        return transpose(self)
    @property
    def T(self):
        return transpose(self)
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
        # if len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
        #     inputs = inputs[0]
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
#     调用传参的时候unzip，具体函数forward穿入和传出都是真实的参数数量的一些数值(not variable，\in nparray)，不含*
#     backward里的穿入和传出都是真实的参数数量的一些Variable(\in class Variable)
class Add(Func):
    def forward(self,x0,x1):
        self.x0_shape,self.x1_shape=x0.shape,x1.shape
        y=x0+x1
        # nparray 会自动实现正向的广播，我们只记录shape然后在反向传播中处理
        return (y,)#(y,)? y is not iterable
    def backward(self,gy):
        gx0=gy
        gx1=gy
        if self.x0_shape !=self.x1_shape:
            gx0=sum_to(gx0,self.x0_shape)
            gx1=sum_to(gx1,self.x1_shape)
        return gx0,gx1
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
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy * x1
        gx1 = gy * x0
        if self.x0_shape != self.x1_shape:
            gx0 = sum_to(gx0, self.x0_shape)
            gx1 = sum_to(gx1, self.x1_shape)
        return gx0, gx1
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
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        return x0-x1
    def backward(self,gy):
        gx0 = gy
        gx1 = gy
        if self.x0_shape != self.x1_shape:
            gx0 = sum_to(gx0, self.x0_shape)
            gx1 = sum_to(gx1, self.x1_shape)
        return gx0,-gx1
def sub(x0,x1):
    x1=as_array(x1)
    return Sub()(x0,x1)
def rsub(x0,x1):
    x1=as_array(x1)
    return Sub()(x1,x0)

class Div(Func):
    def forward(self,x0,x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape

        return x0/x1
    def backward(self,gy):
        x0, x1 = self.inputs

        gx0 = gy / x1
        gx1 = gy * (-x0 / (x1 ** 2))
        if self.x0_shape != self.x1_shape:
            gx0 = sum_to(gx0, self.x0_shape)
            gx1 = sum_to(gx1, self.x1_shape)
        return gx0,gx1
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

class Tanh(Func):
    def forward(self,x):
        return np.tanh(x)
    def backward(self,gy):
        y=self.outputs[0]()
        gx =gy*(1-y*y)
        return gx
def tanh(x):
    return Tanh()(x)



class Reshape(Func):
    def __init__(self,shape):
        self.shape=shape

    def forward(self,x):
        self.x_shape=x.shape
        y=x.reshape(self.shape)
        return y

    def backward(self,gy):
        return reshape(gy,self.x_shape)

def reshape(x,shape):
    if(x.shape==shape):
        return as_variable(x)
    return Reshape(shape)(x)

class Transpose(Func):
    def forward(self,x):
        return np.transpose(x)
    def backward(self,gy):
        return transpose(gy)

def transpose(x):
    return Transpose()(x)

class Matmul(Func):
    def forward(self,x,W):
        return x.dot(W)
    def backward(self,gy):
        x,W=self.inputs
        gx=matmul(gy,W.T)
        gW=matmul(x.T,gy)
        return gx,gW
def matmul(x,W):
    return Matmul()(x,W)



class Sum(Func):
    def __init__(self,axis,keepdims):
        self.axis=axis
        self.keepdims=keepdims
    def forward(self,x):
        self.x_shape=x.shape
        y=x.sum(axis=self.axis,keepdims=self.keepdims)
        return y
    def backward(self,gy):
        gy=reshape_sum_backward(gy,self.x_shape,self.axis,self.keepdims)
        gx=broadcast_to(gy,self.x_shape)
        return gx
    # def backward(self,gy):
    #     gx=broadcast_to(gy.data,self.x_shape)
    #     return as_variable(gx) 像这样的话就计算图就断了，无法进行求高阶导数等功能
#     需要利用Func的子类来保存图，一个Func的forward保证是Var进入Var出，并保存连接
def sum(x,axis=None,keepdims=False):
    return Sum(axis,keepdims)(x)

class BroadcastTo(Func):
    def __init__(self,shape):
        self.shape=shape
    def forward(self,x):
        self.x_shape=x.shape
        y=np.broadcast_to(x,self.shape)
        return y
    def backward(self,gy):
        gx=sum_to(gy,self.x_shape)
        return gx
def broadcast_to(x,shape):
    if x.shape== shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)

class SumTo(Func):
    def __init__(self,shape):
        self.shape=shape
    def forward(self,x):
        self.x_shape=x.shape
        return forward_sum_to(x,self.shape)
    def backward(self,gy):
        return broadcast_to(gy,self.x_shape)
def sum_to(x,shape):
    if x.shape==shape:
        return as_variable(x)
    return SumTo(shape)(x)



def forward_sum_to(x, shape):
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))  # 前置多出来的维度索引

    # 找出 shape 中长度为 1 的轴
    # 比如 (2, 3) -> (1, 3)，我们需要对第 0 轴求和
    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])

    # 合并所有需要求和的轴
    sum_axis = lead_axis + axis

    # 执行求和
    y = x.sum(axis=sum_axis, keepdims=True)

    # 如果有前置多出来的维度，sum 完之后是 (1, 1, 3)，需要去掉前面的 1 变成 (1, 3)
    if lead > 0:
        y = y.squeeze(axis=lead_axis)

    return y



class MeanSquaredError(Func):
    def forward(self,x0,x1):
        diff =x0-x1
        y=(diff**2).sum()/len(diff)
        return y
    def backward(self,gy):
        x0,x1=self.inputs
        diff=x0-x1
        gx0=gy*diff*(2./len(diff))
        gx1=-gx0
        return gx0,gx1
def mean_squared_error(x0,x1):
    return MeanSquaredError()(x0,x1)

# if don't like Func Linear
# also ok to do this:
def linear_simple(x,W,b=None):
    t=matmul(x,W)
    if b is None:
        return t
    y=t+b
    t.data=None #清除tdata，因为不再需要，只需要t用来存梯度
    return y



# my Linear
class Linear(Func):  # 假设你的基类是 Function
    def forward(self, x, W, b=None):
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        if len(self.inputs) == 3:
            x, W, b = self.inputs
        else:
            x, W = self.inputs
            b = None

        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)

        if b is not None:
            # 关键点：gb 需要对 batch 维度求和，保持和 b 形状一致
            gb = sum(gy, axis=0)
            return gx, gW, gb
        return gx, gW
def linear(x,W,b=None):
    return Linear()(x,W,b)


class Sigmoid(Func):
    def forward(self,x):
        return 1/(1+np.exp(-x))
    def backward(self,gy):
        y= self.outputs[0]()
        return gy*y*(1-y)
def sigmoid(x):
    return Sigmoid()(x)

class ReLU(Func):
    def forward(self,x):
        return np.maximum(0,x)
    def backward(self,gy):
            # ???
            # 从 inputs 获取原始输入 x
            x, = self.inputs
            # 创建一个掩码：x > 0 的地方为 1，否则为 0
            # 这里转换成和 gy 相同的类型
            mask = (x.data > 0).astype(gy.data.dtype)
            # 只有 x > 0 的位置，梯度才能传回去
            gx = gy * mask
            return gx
def relu(x):
    return ReLU()(x)




def numerical_diff(f,x,eps=1e-4):
    x0=Variable(as_array(x.data-eps))
    x1=Variable(as_array(x.data+eps))
    y0=f(x0)
    y1=f(x1)
    return (y1.data-y0.data)/(2*eps)


def reshape_sum_backward(gy, x_shape, axis, keepdims):
    """
    专门为 Sum 算子的 backward 服务的辅助函数。
    当 keepdims=False 时，将梯度 gy 重新 reshape 成对齐 x_shape 的形状（即把 axis 变回 1）。
    """
    if keepdims or axis is None:
        return gy

    # 统一将 axis 处理成元组，方便后续逻辑
    if isinstance(axis, int):
        axis = (axis,)

    # 构造新的形状
    # 比如 x_shape 是 (2, 3, 4)，axis=(1,)，gy 形状是 (2, 4)
    # 我们需要把 gy 变成 (2, 1, 4)
    actual_axis = [a if a >= 0 else a + len(x_shape) for a in axis]
    shape = list(x_shape)
    for a in actual_axis:
        shape[a] = 1

    return gy.reshape(tuple(shape))

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