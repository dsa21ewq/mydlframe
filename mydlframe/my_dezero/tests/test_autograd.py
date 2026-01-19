
import math

import numpy as np
from matplotlib import pyplot as plt

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

    # 40次迭代后，(x, y) 应该非常接近极小值 (0, 0)
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
    assert np.allclose(x.data, 0, atol=1e-2)
    assert np.allclose(y.data, 0, atol=1e-2)


def test_rosenbrock_optimization(plot_results=True):
    """
    测试 Rosenbrock 函数的梯度下降，并可选地绘制优化轨迹。
    f(x, y) = 100 * (y - x^2)^2 + (1 - x)^2
    """
    x = Variable(np.array(-1.0), name="x")
    y = Variable(np.array(1.0), name="y")

    lr = 0.001
    iters = 10000

    # 用于存储优化过程中的 x 和 y 值，以便绘图
    x_history = []
    y_history = []

    print("--- 开始测试 Rosenbrock 优化 ---")
    print(f"初始点: x={x.data.item():.4f}, y={y.data.item():.4f}")
    print(f"学习率: {lr}, 迭代次数: {iters}")

    for i in range(iters):
        # 记录当前状态
        x_history.append(x.data.item())
        y_history.append(y.data.item())

        z = 100 * (y - x ** 2) ** 2 + (1 - x) ** 2

        x.clear_grad()
        y.clear_grad()
        z.backward()

        x.data -= lr * x.grad
        y.data -= lr * y.grad

        if i % 2000 == 0 or i == iters - 1:
            print(f"Iteration {i}: loss={z.data:.4f}, x={x.data:.4f}, y={y.data:.4f}")

    # 绘制优化轨迹 (如果 plot_results 为 True)
    if plot_results:
        print("\n--- 绘制优化轨迹 ---")
        _plot_rosenbrock_trajectory(x_history, y_history)
        print("优化轨迹图已显示。")

    # 断言检查
    print("\n--- 验证结果 ---")
    final_x = x.data.item()
    final_y = y.data.item()
    print(f"最终点: x={final_x:.4f}, y={final_y:.4f}")

    # Rosenbrock 的全局最小值在 (1, 1)
    assert np.allclose(final_x, 1.0, atol=1e-2), f"x 预期 1.0, 实际 {final_x}"
    assert np.allclose(final_y, 1.0, atol=1e-2), f"y 预期 1.0, 实际 {final_y}"
    print("Rosenbrock 优化测试通过！")
    print("---------------------------------")


def _plot_rosenbrock_trajectory(x_history, y_history):
    """
    辅助函数：绘制 Rosenbrock 函数的等高线和优化轨迹。
    """
    X = np.linspace(-2, 2, 100)
    Y = np.linspace(-1, 3, 100)
    X_mesh, Y_mesh = np.meshgrid(X, Y)
    Z_mesh = 100 * (Y_mesh - X_mesh ** 2) ** 2 + (1 - X_mesh) ** 2

    plt.figure(figsize=(10, 8))
    # 绘制等高线，使用对数刻度来处理 Rosenbrock 宽广的值域
    cp = plt.contour(X_mesh, Y_mesh, Z_mesh, levels=np.logspace(-1, 3, 20), cmap='viridis', alpha=0.5)
    plt.clabel(cp, inline=True, fontsize=8, fmt='%.1f')

    # 绘制梯度下降路径
    plt.plot(x_history, y_history, 'r.-', label='Gradient Descent Path', markersize=2, linewidth=1)
    # 标记起点和终点
    plt.plot(x_history[0], y_history[0], 'go', markersize=8, label=f'Start ({x_history[0]:.2f}, {y_history[0]:.2f})')
    plt.plot(x_history[-1], y_history[-1], 'bo', markersize=8, label=f'End ({x_history[-1]:.2f}, {y_history[-1]:.2f})')
    plt.plot(1, 1, 'b*', markersize=15, alpha=0.8, label='Minimum (1, 1)')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Rosenbrock Function Optimization Trajectory')
    plt.legend()
    plt.grid(True)
    plt.show()
def test_plot():
    x = Variable(np.array(1.0))
    y = Variable(np.array(1.0))
    z= 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    plot_graph(z)


def test_benchmark_speed():
    start = time.time()
    x = Variable(np.array(1.0))
    for _ in range(10000):
        z = x**2
        z.backward()
    end = time.time()
    print(f"\n10000次反向传播耗时: {end - start:.4f}s")

