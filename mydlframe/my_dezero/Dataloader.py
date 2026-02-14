import math
import numpy as np


class Dataloader:
    def __init__(self,dataset,batch_size,shuffle=True):
        self.dataset=dataset
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.data_size=len(dataset)
        self.max_iter=math.ceil(self.data_size/batch_size)
        self.reset()
    def reset(self):
        self.iteration=0
        if self.shuffle:
            self.index=np.random.permutation(self.data_size)
        else:
            self.index=np.arange(self.data_size)
    def __iter__(self):
        return self
    def __next__(self):
        if  self.iteration>=self.max_iter:
            self.reset()
            raise StopIteration
        i,batch_size=self.iteration,self.batch_size
        batch_index=self.index[i*batch_size:(i+1)*batch_size]
        batch=[self.dataset[i] for i in batch_index]
        x=np.array([example[0] for example in batch])
        t=np.array([example[1] for example in batch])
        self.iteration +=1
        return x,t
    def next(self):
        return self.__next__()


def accuracy(y, t):
    """
    y: 模型的预测结果, 形状为 (N, C)
    t: 真实标签, 形状可以是 (N, C) 的 One-hot，也可以是 (N,) 的类别索引
    """
    # 如果 t 是 One-hot 编码，转换为类别索引
    if t.ndim == 2:
        t = np.argmax(t, axis=1)

    # 获取预测结果中概率最大的类别索引
    target_y = y.data if hasattr(y, 'data') else y
    pred = np.argmax(target_y, axis=1)
    # 计算正确的数量并取平均
    result = (pred == t)
    return np.mean(result)