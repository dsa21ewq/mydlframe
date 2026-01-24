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

x=Variable(np.array([[1,2,3],[4,5,6]]))
y=Variable(np.array([[10,20,30],[40,50,60]]))
k=reshape(x,(6,))
print(k)
z=x+y
z.backward(retain_grad=True)
print(z)
print(z.grad)
print(x.grad)
print(y.grad)