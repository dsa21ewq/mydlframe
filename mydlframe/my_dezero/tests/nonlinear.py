from my_dezero.core_simple import *
# 生成数据
np.random.seed(0)

# 自变量
x_data = np.linspace(-np.pi, np.pi, 100).reshape(100, 1)

# sin + 噪声（关键点：这是散点）
noise = 0.1 * np.random.randn(100, 1)
y_data = np.sin(x_data) + noise

x = Variable(x_data, name="x")
y = Variable(y_data, name="y")

# 初始化参数
W1 = Variable(0.1 * np.random.randn(1, 10), name="W1")
b1 = Variable(np.zeros(10), name="b1")
W2 = Variable(0.1 * np.random.randn(10, 1), name="W2")
b2 = Variable(np.zeros(1), name="b2")

def model(x):
    h = tanh(linear(x, W1, b1))
    y = linear(h, W2, b2)
    return y

lr = 0.1
iters = 2000

for i in range(iters):
    y_pred = model(x)
    loss = mean_squared_error(y_pred, y)

    W1.clear_grad(); b1.clear_grad()
    W2.clear_grad(); b2.clear_grad()

    loss.backward()

    # SGD
    W1.data -= lr * W1.grad.data
    b1.data -= lr * b1.grad.data
    W2.data -= lr * W2.grad.data
    b2.data -= lr * b2.grad.data

    if i % 200 == 0:
        print(f"iter {i}, loss {loss.data}")

import matplotlib.pyplot as plt

with no_grad():
    y_test = model(x)

plt.figure()

# 🔵 原始数据：散点
plt.scatter(x_data, y_data, label="training data (noisy)", s=20)

# 🔴 模型预测：连续曲线
plt.plot(x_data, y_test.data, label="model prediction")

plt.legend()
plt.title("Fitting sin(x) from noisy points")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
