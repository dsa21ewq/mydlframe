import numpy as np

from my_dezero.my_dezero.core_simple import *

np.random.seed(0)
x=np.random.rand(100,1)
y=5+2*x+np.random.rand(100,1)

W=Variable(np.zeros((1,1)))
b=Variable(np.zeros(1))
def predict(x):
    y=linear(x,W,b)
    return y

lr =00.1
iters=10000
for i in range(iters):
    y_next=predict(x)
    loss=mean_squared_error(y,y_next)

    W.clear_grad()
    b.clear_grad()
    loss.backward()

    W.data -=lr * W.grad.data
    b.data -=lr * b.grad.data
    print(W,b,loss)