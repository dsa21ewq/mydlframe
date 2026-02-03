from my_dezero.my_dezero.core_simple import *


def get_dot_graph(output, verbose=True):
    dot = Digraph()
    dot.attr(rankdir='TB')
    dot.attr('node', fontname='Verdana', fontsize='10')

    funcs = []
    seen_set = set()

    # 辅助函数：确保 ID 是以字母开头的安全字符串，避免 Graphviz 语法解析错误
    def safe_id(obj):
        return "node_" + str(id(obj))

    def add_func(f):
        if f is not None and f not in seen_set:
            funcs.append(f)
            seen_set.add(f)

    # 1. 绘制起始变量节点 (输出节点)
    node_name = f"<B>{output.name}</B><BR/>" if output.name else ""
    var_label = f"<{node_name}Variable (ID:{output.id})<BR/>data: {output.data}>"
    dot.node(safe_id(output), var_label, color='orange', style='filled', shape='ellipse')

    add_func(output.creator)

    while funcs:
        f = funcs.pop()

        # 2. 绘制函数节点 (算子)
        func_label = f"{f.__class__.__name__} (ID:{f.id})\ngen: {f.generation}"
        dot.node(safe_id(f), func_label, shape='box', style='filled, bold', fillcolor='lightgrey')

        # 3. 绘制函数的所有输出 (通常是 output 本身)
        for wx in f.outputs:
            x = wx()
            if x is not None:
                x_name = f"<B>{x.name}</B><BR/>" if x.name else ""
                x_label = f"<{x_name}Variable (ID:{x.id})<BR/>data: {x.data}>"
                dot.node(safe_id(x), x_label, color='orange', style='filled', shape='ellipse')
                dot.edge(safe_id(f), safe_id(x))

        # 4. 绘制函数的所有输入
        for x in f.inputs:
            x_name = f"<B>{x.name}</B><BR/>" if x.name else ""
            # 如果有梯度，分行显示，注意：由于 gx 的 backward 也会调用此函数，这里会递归显示梯度数据
            grad_val = x.grad.data if (x.grad is not None and x.grad.data is not None) else "None"
            grad_str = f"<BR/>grad: {grad_val}" if x.grad is not None else ""

            var_label = f"<{x_name}Variable (ID:{x.id})<BR/>data: {x.data}{grad_str}>"

            dot.node(safe_id(x), var_label, color='lightblue', style='filled', shape='ellipse')
            dot.edge(safe_id(x), safe_id(f))

            if x.creator is not None:
                add_func(x.creator)

    return dot


def plot_graph(output, verbose=True,filename="graph.png"):
    dot = get_dot_graph(output,verbose)
    dot.attr(dpi='300')
    # 移除 filename 中的扩展名，由 format 参数决定
    base_name = filename.split('.')[0]
    dot.render(base_name, format='png', cleanup=True)
    print(f"计算图已成功保存至: {base_name}.png")