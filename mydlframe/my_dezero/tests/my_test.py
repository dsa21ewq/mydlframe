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
# 1. 创建输入变量
x = Variable(np.array(2.0), name="x")

# 2. 正向传播 y = x^3
y = x ** 3
y.name = "y"

# 3. 计算一阶导数 (dy/dx)
# 注意：必须开启 create_graph=True 才能为导数计算构建计算图
y.backward(create_graph=True)
gx = x.grad
gx.name = "gx"
print(f"一阶导数 (x=2.0): {gx.data}") # 预期结果: 12.0

# 4. 清除之前的梯度，计算二阶导数 (d^2y/dx^2)
x.clear_grad()
gx.backward()
print(f"二阶导数 (x=2.0): {x.grad.data}") # 预期结果: 12.0

# 5. 可视化一阶导数的计算图
# 确保你已经修复了 safe_id 的问题
x = Variable(np.array(2.0), name="x")
y = x ** 4
y.name = "y"

# 计算一阶导并创建计算图
y.backward(create_graph=True)
gx = x.grad
gx.name = "gx"

# 绘制 gx 的计算图（这展现了导数是如何由 x 计算出来的）
plot_graph(gx, filename="y_x4_second_order_graph.png")