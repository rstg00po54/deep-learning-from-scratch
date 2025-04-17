import numpy as np

# 激活函数及其导数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

# 初始化参数
w = 0.5
b = 0.0

# 输入和标签（x 是输入，y 是期望输出）
x = 1.0
y = 0.0

# 学习率
lr = 0.1

# 训练过程
for epoch in range(10):
    # ----- 前向传播 -----
    z = w * x + b
    y_hat = sigmoid(z)
    loss = 0.5 * (y_hat - y) ** 2

    # ----- 反向传播 -----
    dL_dyhat = y_hat - y                    # ∂L/∂ŷ
    dyhat_dz = sigmoid_derivative(z)        # ∂ŷ/∂z
    dz_dw = x                               # ∂z/∂w
    dz_db = 1                               # ∂z/∂b

    dL_dw = dL_dyhat * dyhat_dz * dz_dw     # ∂L/∂w
    dL_db = dL_dyhat * dyhat_dz * dz_db     # ∂L/∂b

    # ----- 参数更新 -----
    w -= lr * dL_dw
    b -= lr * dL_db

    # ----- 打印 -----
    print(f"Epoch {epoch+1}: loss = {loss:.6f}, w = {w:.4f}, b = {b:.4f}")
