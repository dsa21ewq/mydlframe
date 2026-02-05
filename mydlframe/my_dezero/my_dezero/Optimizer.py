import numpy as np
from sympy import gamma


class Optimizer:
    def __init__(self):
        self.target=None
        self.hooks=[]
    def setup(self,target):
        self.target=target
        return self
    def update(self):
        params=[p for p in self.target.params() if p.grad is not None]
        # hooks 是一些预处理
        for f in self.hooks:
            f(params)
        for param in params:
            self.update_one(param)
    def update_one(self,param):
        raise NotImplementedError
    def add_hook(self,f):
        self.hooks.append(f)
class SGD(Optimizer):
    def __init__(self,lr=0.01):
        super().__init__()
        self.lr=lr
    def update_one(self,param):
        param.data -= self.lr * param.grad.data

class Momentum(Optimizer):
    def __init__(self,lr=0.01,gamma=0.9):
        super().__init__()
        self.lr=lr
        self.gamma=gamma
        self.speed= {}
    def update_one(self,param):
        speed_id=id(param)
        if speed_id not in self.speed:
            self.speed[speed_id]=np.zeros_like(param.data)
        v=self.speed[speed_id]
        v*=self.gamma
        v+=self.lr*param.grad.data
        param.data-=v
class Nesterov_Momentum(Optimizer):
    def __init__(self,lr=0.01,gamma=0.9):
        super().__init__()
        self.lr=lr
        self.gamma=gamma
        self.speed= {}
    def update_one(self,param):
        # here are some tricks 1.consider theta t- gamma*vt=phi t 2. delta phi t is almost delta theta t , and delta phi t is gamma * v (t+1) + lr * grad
        speed_id=id(param)
        if speed_id not in self.speed:
            self.speed[speed_id]=np.zeros_like(param.data)
        v=self.speed[speed_id]
        v*=self.gamma
        v+=self.lr*param.grad.data
        param.data -= (self.gamma * v + self.lr * param.grad.data)


class AdaGrad(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr
        self.h = {}  # 存储每个参数的梯度平方和

    def update_one(self, param):
        h_id = id(param)
        if h_id not in self.h:
            self.h[h_id] = np.zeros_like(param.data)

        h = self.h[h_id]
        g = param.grad.data

        # 核心逻辑：累加梯度的平方
        h += g * g

        # 更新参数：学习率除以 sqrt(h)
        # 注意：这里的 1e-7 是为了防止除 0
        param.data -= (self.lr / (np.sqrt(h) + 1e-7)) * g

class Adam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0 # 记录迭代次数，用于偏差修正
        self.ms = {} # 一阶矩
        self.vs = {} # 二阶矩

    def update(self):
        self.t += 1 # 每次全局更新时时间步 +1
        super().update()

    def update_one(self, param):
        pid = id(param)
        if pid not in self.ms:
            self.ms[pid] = np.zeros_like(param.data)
            self.vs[pid] = np.zeros_like(param.data)

        m, v = self.ms[pid], self.vs[pid]
        g = param.grad.data

        # 1. 更新矩（注意这里也是修改 reference）
        m += (1 - self.beta1) * (g - m)  # 等价于 m = b1*m + (1-b1)*g
        v += (1 - self.beta2) * (g**2 - v) # 等价于 v = b2*v + (1-b2)*g^2

        # 2. 偏差修正
        m_hat = m / (1 - self.beta1**self.t)
        v_hat = v / (1 - self.beta2**self.t)

        # 3. 更新参数
        param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)