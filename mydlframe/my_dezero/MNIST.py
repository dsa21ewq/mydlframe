import os
import urllib.request
import gzip
import numpy as np
from my_dezero.Dataset import Dataset


class MNIST(Dataset):
    def __init__(self, train=True, transform=None, target_transform=None):
        self.url = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')  # 建议统一放在 data 文件夹

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        self.files = {
            'train_img': 'train-images-idx3-ubyte.gz',
            'train_label': 'train-labels-idx1-ubyte.gz',
            'test_img': 't10k-images-idx3-ubyte.gz',
            'test_label': 't10k-labels-idx1-ubyte.gz'
        }
        super().__init__(train, transform, target_transform)

    def prepare(self):
        img_name = self.files['train_img'] if self.train else self.files['test_img']
        label_name = self.files['train_label'] if self.train else self.files['test_label']

        img_path = os.path.join(self.data_dir, img_name)
        label_path = os.path.join(self.data_dir, label_name)

        # 如果文件不存在，自动下载
        self._download(img_name, img_path)
        self._download(label_name, label_path)

        self.data = self._load_img(img_path)
        self.label = self._load_label(label_path)

    def _download(self, file_name, file_path):
        if not os.path.exists(file_path):
            print(f"Downloading {file_name}...")
            urllib.request.urlretrieve(self.url + file_name, file_path)
            print("Done.")

    def _load_img(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            # MNIST 图像文件前 16 字节是魔数、数量、行、列信息
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 784).astype(np.float32) / 255.0

    def _load_label(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            # 标签文件前 8 字节是头信息
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        return self._to_onehot(labels)  # 别忘了你的 loss 需要 one-hot

    def _to_onehot(self, labels):
        onehot = np.zeros((labels.size, 10), dtype=np.float32)
        for i, label in enumerate(labels):
            onehot[i, label] = 1
        return onehot