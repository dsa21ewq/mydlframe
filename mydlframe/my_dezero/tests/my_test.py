import math

import numpy as np
from my_dezero.my_dezero.core_simple import *
def my_sin(x,threshold=1e-150):
    y=0
    for i in range(0,10000):
        c= (-1) **i/math.factorial(2*i+1)
        t= c*x**(2*i+1)
        y=y+t
        if(abs(t.data)<threshold):
            break
    return y
x=Variable(np.array(np.pi/4))
y=my_sin(x)
y.backward()
print(y.data)
print(x.grad)
plot_graph(y)