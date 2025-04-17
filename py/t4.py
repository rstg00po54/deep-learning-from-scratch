import numpy as np
import matplotlib.pyplot as plt

# 激活函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

# 数据：一个输入 + 期望输出
x = 1.0
y = 0.0

# 参数初始化
w = 0.5
b = 0.0
lr = 0.1

# 存储训练过程
epochs = 50
loss_history = []

# 训练过程
for epoch in range(epochs):
    # 前向传播
    z = w * x + b
    y_hat = sigmoid(z)
    loss = 0.5 * (y_hat - y)**2
    loss_history.append(loss)

    # 反向传播
    dL_dyhat = y_hat - y
    dyhat_dz = sigmoid_derivative(z)
    dL_dw = dL_dyhat * dyhat_dz * x
    dL_db = dL_dyhat * dyhat_dz * 1

    # 参数更新
    w -= lr * dL_dw
    b -= lr * dL_db

# 绘制训练过程的损失图
plt.figure(figsize=(6, 4))
plt.plot(loss_history, label="Loss")
plt.title(f"Final Loss: {loss_history[-1]:.6f}")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
