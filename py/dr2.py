import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

def draw_node(ax, xy, label, radius=0.2, text_color='black', bold=False):
    circle = Circle(xy, radius, facecolor='none', edgecolor='black', lw=2, zorder=2)
    ax.add_patch(circle)
    fontweight = 'bold' if bold else 'normal'
    ax.text(*xy, label, color=text_color, ha='center', va='center', fontsize=12, weight=fontweight)

import numpy as np


def draw_arrow(ax, start, end, radius=0.3, label=None, label_offset=(0.02, 0.02), **kwargs):

    start = np.array(start)
    end = np.array(end)
    vec = end - start
    length = np.linalg.norm(vec)

    if length == 0:
        return

    direction = vec / length
    new_start = start + direction * radius
    new_end = end - direction * radius

    # 画箭头
    arrow = FancyArrowPatch(
        new_start, new_end,
        arrowstyle='->',
        color=kwargs.get('color', 'gray'),
        lw=kwargs.get('lw', 1.5),
        mutation_scale=kwargs.get('mutation_scale', 10),
    )
    ax.add_patch(arrow)

    # 加标签
    if label:
        label_pos = (new_start + new_end) / 2 + np.array(label_offset)
        ax.text(*label_pos, label, fontsize=16, ha='center', va='center')

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_facecolor('#222')   # 深背景保留
ax.axis('off')
ax.set_aspect('equal')
ax.set_xlim(-1, 7)  # 横坐标范围：略微比你所有节点宽
ax.set_ylim(-1, 3)  # 纵坐标范围：涵盖所有节点

# 坐标布局
input_layer = [(0, 2), (0, 1), (0, 0)]
hidden1_layer = [(2, 2), (2, 1), (2, 0)]
hidden2_layer = [(4, 1.5), (4, 0.5)]
output_layer = [(6, 1.5), (6, 0.5)]

# 输入层节点
draw_node(ax, input_layer[0], '1', text_color='black', bold=True)  # bias
draw_node(ax, input_layer[1], 'x₁', text_color='black')
draw_node(ax, input_layer[2], 'x₂', text_color='black')

# 第一隐藏层
draw_node(ax, hidden1_layer[0], 'a₁⁽¹⁾', text_color='black')
draw_node(ax, hidden1_layer[1], 'a₂⁽¹⁾', text_color='black')
draw_node(ax, hidden1_layer[2], 'a₃⁽¹⁾', text_color='black')

# 第二隐藏层
for i in range(2):
    draw_node(ax, hidden2_layer[i], '', text_color='black')

# 输出层
draw_node(ax, output_layer[0], 'y₁', text_color='black')
draw_node(ax, output_layer[1], 'y₂', text_color='black')

# 输入层 -> 第一隐藏层（添加标签）
draw_arrow(ax, input_layer[0], hidden1_layer[0], label='b₁⁽¹⁾')
draw_arrow(ax, input_layer[1], hidden1_layer[0], label='w₁₁⁽¹⁾')
draw_arrow(ax, input_layer[2], hidden1_layer[0], label='w₁₂⁽¹⁾')

# 其余输入层 -> 第一隐藏层（无标签）
for i in range(1, 3):
    for j in range(1, 3):
        draw_arrow(ax, input_layer[i], hidden1_layer[j])

# 第一隐藏层 -> 第二隐藏层
for h1 in hidden1_layer:
    for h2 in hidden2_layer:
        draw_arrow(ax, h1, h2)

# 第二隐藏层 -> 输出层
for h2 in hidden2_layer:
    for out in output_layer:
        draw_arrow(ax, h2, out)

plt.tight_layout()
plt.show()
