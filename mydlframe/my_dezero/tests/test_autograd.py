# tests/test_autograd.py
import numpy as np
from my_dezero.my_dezero.core_simple import *
import time
import pytest


def test_add_backward():
    x = Variable(np.array(2.0))
    y = Variable(np.array(3.0))
    z = x + y
    z.backward()
    assert x.grad == 1.0
    assert y.grad == 1.0

def test_gradient_check():
    """使用数值微分验证自动微分的准确性"""
    x = Variable(np.array(2.0))
    y = square(add(x, np.array(3.0)))  # y = (x + 3)^2
    y.backward()

    # 计算数值导数
    f = lambda x: square(add(x, np.array(3.0)))
    expected_grad = numerical_diff(f, x)

    # 允许极小的误差
    assert np.allclose(x.grad, expected_grad, atol=1e-6)


def test_operator_overload():
    x = Variable(np.array(2.0))
    y = 3 * x + 1.0  # 涉及 __rmul__ 和 __add__
    y.backward()

    assert y.data == 7.0
    assert x.grad == 3.0


def test_sub_and_div():
    x = Variable(np.array(10.0))
    y = Variable(np.array(2.0))

    z1 = x - y  # 8.0
    z2 = x / y  # 5.0

    z1.backward(retain_grad=True)
    z2.backward()

    assert x.grad == 1.0 + (1 / 2.0)  # z1对x导数是1，z2对x导数是1/y
    assert y.grad == -1.0 + (-10.0 / (2.0 ** 2))  # z1对y导数是-1，z2对y是-x/y^2


def test_variable_reuse():
    x = Variable(np.array(3.0))
    y = add(x, x)  # y = 2x
    y.backward()
    assert x.grad == 2.0

    x.clear_grad()
    z = add(add(x, x), x)  # z = 3x
    z.backward()
    assert x.grad == 3.0


def test_complex_graph():
    x = Variable(np.array(2.0))
    a = square(x)  # gen: 1
    b = square(a)  # gen: 2
    c = square(a)  # gen: 2
    y = add(b, c)  # gen: 3
    y.backward()

    # y = (x^2)^2 + (x^2)^2 = 2x^4
    # dy/dx = 8x^3 = 8 * (2^3) = 64
    assert x.grad == 64.0

def test_no_grad():
    with no_grad():
        x = Variable(np.array(2.0))
        y = square(x)
    assert y.creator is None # 不应该建立引用关系


def test_sphere_optimization():
    """测试 Sphere 函数的梯度下降收敛性"""
    # 初始化变量 (x, y) = (1.0, 1.0)
    x = Variable(np.array(1.0))
    y = Variable(np.array(1.0))
    lr = 0.1  # 学习率
    iters = 40

    for i in range(iters):
        # f(x, y) = x^2 + y^2
        z = x ** 2 + y ** 2

        x.clear_grad()
        y.clear_grad()
        z.backward()

        # 梯度下降更新: x = x - lr * grad
        x.data -= lr * x.grad
        y.data -= lr * y.grad

    # 20次迭代后，(x, y) 应该非常接近极小值 (0, 0)
    assert np.allclose(x.data, 0, atol=1e-2)
    assert np.allclose(y.data, 0, atol=1e-2)


def test_matyas_optimization():
    """测试 Matyas 函数的梯度下降"""
    x = Variable(np.array(1.0))
    y = Variable(np.array(1.0))
    lr = 0.2
    iters = 1000

    for i in range(iters):
        # f(x, y) = 0.26(x^2 + y^2) - 0.48xy
        z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y

        x.clear_grad()
        y.clear_grad()
        z.backward()

        x.data -= lr * x.grad
        y.data -= lr * y.grad

    # 极小值点在 def benchmark_speed():
    #     start = time.time()
    #     # 执行 10000 次 Sphere 函数的反向传播
    #     x = Variable(np.array(1.0))
    #     for _ in range(10000):
    #         z = x**2
    #         z.backward()
    #     end = time.time()
    #     print(f"10000次反向传播耗时: {end - start:.4f}s")
    assert np.allclose(x.data, 0, atol=1e-2)
    assert np.allclose(y.data, 0, atol=1e-2)

def test_benchmark_speed(): # 修改了名字
    start = time.time()
    x = Variable(np.array(1.0))
    for _ in range(10000):
        z = x**2
        z.backward()
    end = time.time()
    print(f"\n10000次反向传播耗时: {end - start:.4f}s")