import sys

# # 创建一个列表对象，a 指向它
# a = [1, 2, 3]
# print(sys.getrefcount(a))  # 输出通常为 2 (一个是 a，一个是 getrefcount 的参数临时引用)
#
# b = a  # 引用计数 +1
# print(sys.getrefcount(a))  # 输出 3
#
# del b  # 引用计数 -1
# print(sys.getrefcount(a))  # 输出 2

import gc

# class Node:
#     def __init__(self):
#         self.cycle = None

# # 创建循环引用
# obj1 = Node()
# obj2 = Node()
# obj1.cycle = obj2
# obj2.cycle = obj1
#
# print(sys.getrefcount(obj1))
# print(sys.getrefcount(obj2))
# # 销毁外部变量
# del obj1
# del obj2
#
#
# # 此时 obj1 和 obj2 的引用计数都是 1 (互相引用)，不会被立即释放
# # 只有当 Python 运行 GC 的“标记-清除”时，发现它们是孤岛，才会回收
# gc.collect() # 手动触发垃圾回收
#

# 整数池 (Integer Interning)
x = 100
y = 100
print(x is y)  # True，它们指向内存中同一个地址

# 较大的对象
a = 100000000
b = 100000000
print(id(a))
print(id(b))
print(id(a) == id(b)) # 这等同于 a is b