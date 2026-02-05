import math

import numpy as np
from my_dezero.my_dezero.core_simple import *
# def my_sin(x,threshold=1e-150):
#     y=0
#     for i in range(0,10000):
#         c= (-1) **i/math.factorial(2*i+1)
#         t= c*x**(2*i+1)
#         y=y+t
#         if(abs(t.data)<threshold):
#             break
#     return y
# x=Variable(np.array(np.pi/4))
# y=my_sin(x)
# y.backward()
# print(y.data)
# print(x.grad)
# plot_graph(y)
# x = Variable(np.array(2.0), name="x")
# def f(x):
#     return x**4-2*x**2
# newton_minimal(f,iters=10,x=x)

# 1. 定义数据
# x = Variable(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
#
# # 2. 前向传播
# y = sum(x)
# print("y:")
# print(y)
# # 3. 第一次反向传播（计算一阶导数）
# y.backward(create_graph=True,retain_grad=True)# 必须开启 create_graph 以便计算二阶导
# print("一阶梯度 gx:")
# print(x.grad)
#
# # 4. 对梯度进行运算
# # 理论上 gx 的每个元素都是 1.0，平方后还是 1.0
# gx = x.grad
# z = gx * gx
# print(z)
# z = sum(z) # 如果 z 是矩阵，可以求个和变成标量再 backward
# # 清除之前的梯度（可选，视你的 Variable.clear_grad 实现而定）
# x.clear_grad()
# z.backward()
# print(f"gx 的生成器: {gx.creator}")
# print("\n梯度的导数 (gx * gx)' :")
# print(x.grad)
# x = Variable(np.array([[1.0, 2.0], [3.0, 4.0]]))
# y = sum(x * x)  # y = x1^2 + x2^2 + x3^2 + x4^2
#
# # 第一步：计算一阶导
# y.backward(create_graph=True)
# gx = x.grad
# print("一阶梯度 gx:\n", gx) # 应该是 [[2, 4], [6, 8]]
#
# # 第二步：准备计算二阶导
# # 重点：我们不需要对 gx*gx backward，我们直接对 gx 求和再 backward
# # 或者直接对 y 的一阶导数路径进行回溯
# x.clear_grad() # 清除一阶导，准备看二阶导
# gx.backward() # 让一阶梯度 gx 再次反向传播
#
# print("\n二阶梯度 (gx 对 x 的导数):")
# print(x.grad)
a=Variable(np.array([[1,2,3],[4,5,6]]))
indices=[0,0,1]
y=a[indices]
print(y)