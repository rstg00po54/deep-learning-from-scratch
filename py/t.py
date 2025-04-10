import numpy as np
import matplotlib.pyplot as plt

# 生成网格点
x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x, y)

# 定义矢量场 (U, V)
U = X  # x 方向上的分量
V = Y  # y 方向上的分量

# 绘制矢量场
plt.figure(figsize=(6,6))
plt.quiver(X, Y, U, V, color='gray', angles='xy', scale_units='xy', scale=5)

# 设置坐标轴
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.xlabel(r"$x_0$")
plt.ylabel(r"$x_1$")
plt.grid(True, linestyle=":", linewidth=0.5)

# 显示图像
plt.show()
