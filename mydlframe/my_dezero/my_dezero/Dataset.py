import numpy as np


class Dataset:
    def __init__(self,train=True,transfrom=None,target_transform=None):
        self.train=train
        self.transform=transfrom
        self.target_transform=target_transform
        if self.transform is None:
            self.transform=lambda x:x
        if self.target_transform is None:
            self.transform=lambda x:x

        self.data=None
        self.label=None
        self.prepare()
    def __getitem__(self, index):
        assert index.isscalar(index)
        if self.label is None:
            return self.transform(self.data[index]),None
        else:
            return self.transform(self.data[index]),self.target_transform(self.label[index])
    def __len__(self):
        return len(self.data)
    def prepare(self):
        pass
class Spiral(Dataset):
    def prepare(self):
        self.data,self.label=get_spiral()
def get_spiral(noise=0.2,nums=300):
    x, t = make_spiral(nums, noise=noise)
    return x,to_onehot(t)
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

class BigData(Dataset):
    def __getitem__(self, index):
        x=np.load('data/{}.npy'.format(index))
        t=np.load('label/{}.npy'.format(index))
    def __len__(self):
        return 1000000