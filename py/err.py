import numpy as np

# 模拟神经网络输出
np.random.seed(0)
Y_pred = np.random.rand(5, 1)  # 模拟神经网络输出，形状 (5,1)

# 正确的标签(列向量)
# 5行1列
Y_true_correct = np.array([1, 0, 1, 0, 1], dtype=np.int32).reshape(-1, 1)
# reshape改变数组的形状（维度） 的方法

# 错误的标签(一维)
# 1行5列
Y_true_wrong = np.array([1, 0, 1, 0, 1], dtype=np.int32)
np.set_printoptions(precision=2)
# 均方误差
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
print("Y_pred", Y_pred)
print("正确", Y_true_correct)
print("错误", Y_true_wrong)
# 打印比较结果
print("=== LOSS 正确 reshape ===")
print("Y_true shape:", Y_true_correct.shape)
print("Y_pred shape:", Y_pred.shape)
print("MSE loss:", Y_true_correct + Y_pred)
print("MSE loss:", mse(Y_true_correct, Y_pred))

print("\n=== LOSS 错误未 reshape ===")
print("Y_true shape:", Y_true_wrong.shape)
print("Y_pred shape:", Y_pred.shape)
print("MSE loss:", Y_true_wrong + Y_pred)
try:
    print("MSE loss:", mse(Y_true_wrong, Y_pred))
except Exception as e:
    print("计算失败！", e)


import numpy as np

# 假设输出是 softmax 之后的概率
y_pred = np.array([
    [0.1, 0.7, 0.2],
    [0.8, 0.1, 0.1],
    [0.3, 0.2, 0.5]
], dtype=np.float32)

# 正确分类索引(整数标签)
y_true_int = np.array([1, 0, 2])

# 将整数标签转换为 one-hot 编码(float 类型)
y_true_onehot = np.zeros_like(y_pred)
y_true_onehot[np.arange(len(y_true_int)), y_true_int] = 1.0

# 损失函数：均方误差
def mean_squared_error(y_true, y_pred):
    return 0.5 * np.mean((y_true - y_pred) ** 2)

# 损失函数：交叉熵(标签为 one-hot)
def cross_entropy_onehot(y_true, y_pred):
    delta = 1e-7  # 防止 log(0)
    return -np.sum(y_true * np.log(y_pred + delta)) / y_true.shape[0]

# 损失函数：交叉熵(标签为整数)
def cross_entropy_int(y_true_idx, y_pred):
    delta = 1e-7
    batch_size = y_pred.shape[0]
    return -np.sum(np.log(y_pred[np.arange(batch_size), y_true_idx] + delta)) / batch_size

# ========== 打印结果 ==========
print("== MSE(float) ==")
print("loss:", mean_squared_error(y_true_onehot, y_pred))

print("\n== Cross Entropy(one-hot float) ==")
print("loss:", cross_entropy_onehot(y_true_onehot, y_pred))

print("\n== Cross Entropy(int 标签) ==")
print("loss:", cross_entropy_int(y_true_int, y_pred))


print("\n=== 广播=1==")
# 2D 数组 (3, 4)
arr1 = np.array([[1, 2, 3, 4],
                 [5, 6, 7, 8],
                 [9, 10, 11, 12]])

# 1D 数组 (4,)
arr2 = np.array([1, 1, 1, 1])
# 广播机制使得 arr2 会自动扩展为 (3, 4) 来与 arr1 进行逐元素相加
# 沿着行计算均值 (axis=0)
# 沿着列计算均值 (axis=1)
result = np.mean(arr1 + arr2, axis=0)
print(result)


print("\n=== 广播=2==")
arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([1, 1, 1])

# 广播机制会把 arr2 从 (3,) 扩展为 (2, 3)，然后做加法
result = arr1 + arr2
print(result)