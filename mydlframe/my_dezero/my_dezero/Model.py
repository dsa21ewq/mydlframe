
from my_dezero.my_dezero.layers import *
from my_dezero.my_dezero.utils import *

class Model(Layer):
    def plot(self,*inputs, to_file='model.png'):
        y=self.forward(*inputs)
        return plot_graph(y,verbose=True,filename=to_file)

import my_dezero.my_dezero.layers as L
import my_dezero.my_dezero.functions as F
class MLP(Model):
    def __init__(self,fc_output_size,activation=F.sigmoid):
        super().__init__()
        self.activation=activation
        self.layers=[]

        for i ,out_size in enumerate(fc_output_size):
            layer=L.Linear(out_size)
            setattr(self,'l'+str(i),layer)
            self.layers.append(layer)
    def forward(self, x):
        for l in self.layers[:-1]:
            x=self.activation(l(x))
        return self.layers[-1](x)
