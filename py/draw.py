import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 生成 x0 和 x1 的网格数据
x0 = np.linspace(-3, 3, 50)
x1 = np.linspace(-3, 3, 50)
X0, X1 = np.meshgrid(x0, x1)
Z = X0**3 + X1**3  # 计算 f(x0, x1)

# 创建 3D 图形
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 绘制曲面
ax.plot_wireframe(X0, X1, Z, color='black', linewidth=0.8)

# 设置标签
ax.set_xlabel(r"$x_0$")
ax.set_ylabel(r"$x_1$")
ax.set_zlabel(r"$f(x)$")

# 显示图像
plt.show()
