import numpy as np

# 假设输入大小，隐藏层大小和输出大小
input_size = 2
hidden_size = 3
output_size = 1

# 模拟一些网络参数
W1 = np.random.randn(input_size, hidden_size)  # 权重 W1
b1 = np.zeros(hidden_size)  # 偏置 b1
W2 = np.random.randn(hidden_size, output_size)  # 权重 W2
b2 = np.zeros(output_size)  # 偏置 b2

# 将所有参数打包成一个一维数组
params = np.concatenate([W1.flatten(), b1.flatten(), W2.flatten(), b2.flatten()])

# 使用 np.split 将参数拆分回原来的结构
W1_flat, b1_flat, W2_flat, b2_flat = np.split(
    params, [input_size * hidden_size, input_size * hidden_size + hidden_size, input_size * hidden_size + hidden_size + hidden_size]
)

# 恢复原始形状
W1_recovered = W1_flat.reshape(input_size, hidden_size)
b1_recovered = b1_flat.reshape(hidden_size)
W2_recovered = W2_flat.reshape(hidden_size, output_size)
b2_recovered = b2_flat.reshape(output_size)

# 打印结果确认
print("W1 recovered:", W1_recovered)
print("b1 recovered:", b1_recovered)
print("W2 recovered:", W2_recovered)
print("b2 recovered:", b2_recovered)
