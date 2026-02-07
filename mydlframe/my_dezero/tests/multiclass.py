import numpy as np
from my_dezero.my_dezero.Model import *
from my_dezero.my_dezero.Optimizer import *
from my_dezero.my_dezero.core_simple import *



def to_onehot(t):
    onehot = np.zeros((t.size, 3), dtype=np.float32)
    for i, label in enumerate(t):
        if label == -1:
            onehot[i, 0] = 1
        elif label == 0:
            onehot[i, 1] = 1
        elif label == 1:
            onehot[i, 2] = 1
    return onehot
def make_spiral(n_samples_per_class=100, noise=0.2):
    X = []
    y = []

    labels = [-1, 0, 1]
    for i, label in enumerate(labels):
        r = np.linspace(0.0, 1, n_samples_per_class)
        t = np.linspace(i * 4, (i + 1) * 4, n_samples_per_class)

        x = r * np.sin(t) + np.random.randn(n_samples_per_class) * noise
        y_ = r * np.cos(t) + np.random.randn(n_samples_per_class) * noise

        X.append(np.c_[x, y_])
        y.append(np.full(n_samples_per_class, label))

    X = np.vstack(X)
    y = np.concatenate(y)

    return X, y
x, t = make_spiral(200,noise=0.1)
# print(x.shape)  # (600, 2)
# print(set(t))   # {-1, 0, 1}

label_map = {-1: 0, 0: 1, 1: 2}
t_oh=to_onehot(t)
lr = 0.01
max_iter = 10000

model = MLP([10, 10, 10,3], activation=relu)
optimizer = Momentum(lr)
optimizer.setup(model)

for i in range(max_iter):
    y = model(x)
    loss = softmax_cross_entropy(y, t_oh)

    model.cleargrads()
    loss.backward()
    optimizer.update()

    if i % 1000 == 0:
        print(loss)
import numpy as np
import matplotlib.pyplot as plt

# 假设 x, t 是你的螺旋数据，model 是训练好的 MLP
# x.shape = (N, 2), t.shape = (N,), model 输出 3 维 logits

# 把标签转成类别索引，用于对比
label_map = {-1:0, 0:1, 1:2}
t_cls = np.vectorize(label_map.get)(t)

# 生成网格
h = 0.01  # 网格步长
x_min, x_max = x[:,0].min() - 0.5, x[:,0].max() + 0.5
y_min, y_max = x[:,1].min() - 0.5, x[:,1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

grid = np.c_[xx.ravel(), yy.ravel()]

# MLP 预测
Z = model(grid).data.argmax(axis=1)
Z = Z.reshape(xx.shape)

# 画图
plt.figure(figsize=(6,6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

# 原数据点
plt.scatter(x[t==-1,0], x[t==-1,1], c='red', label='-1', s=10)
plt.scatter(x[t==0,0], x[t==0,1], c='green', label='0', s=10)
plt.scatter(x[t==1,0], x[t==1,1], c='blue', label='1', s=10)

plt.legend()
plt.axis("equal")
plt.title("MLP Decision Boundary on 3-class Spiral")
plt.show()
