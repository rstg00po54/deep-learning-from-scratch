from graphviz import Digraph

dot = Digraph(format='png')

# 添加输入层
dot.node('x1', 'x1', shape='circle')
dot.node('x2', 'x2', shape='circle')
dot.node('b', '1', shape='circle', style='filled', fillcolor='gray')

# 添加隐藏层
dot.node('h1', 'z1', shape='circle')
dot.node('h2', 'z2', shape='circle')
dot.node('h3', 'z3', shape='circle')

# 添加输出层
dot.node('y1', 'y1', shape='circle')
dot.node('y2', 'y2', shape='circle')

# 连接输入层到隐藏层
for i in ['x1', 'x2', 'b']:
    for j in ['h1', 'h2', 'h3']:
        dot.edge(i, j)

# 连接隐藏层到输出层
for i in ['h1', 'h2', 'h3']:
    for j in ['y1', 'y2']:
        dot.edge(i, j)

# 生成图像
dot.render('neural_network')
