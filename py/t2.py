import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
'''
输入层：2个神经元

隐藏层：3个神经元

输出层：1个神经元


两行三列
matrix = np.array([[-0.20212614, 0.06943245, 0.41022952],
                   [-0.1731148 , 0.61708178, 0.75795003]])
 W1   W2
x   x   x
    x   x   x
x   x   x
 b1   b2

'''
import numpy as np
import numpy.typing as npt
Array4x2 = npt.NDArray[np.float64]
Array4x1 = npt.NDArray[np.float64]



# 激活函数 ReLU 和它的导数 max(0,x)
def relu(x):
    return np.maximum(0, x)

# 导数
def relu_deriv(x):
    return (x > 0).astype(float)

# 损失函数 (均方误差)
def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_true - y_pred) ** 2)

# 神经网络前向传播
def forward(x: Array4x2, W1, b1, W2, b2):
    z1 = np.dot(x, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    return z2, a1

# 数值梯度计算
# 计算了 一个标量函数 f(x) 相对于 x 的 数值梯度（numerical gradient），适用于 x 是 一维数组 的情况。
def _numerical_gradient_no_batch(f, x):
    h = 1e-4  # 0.0001，微小的扰动值
    grad = np.zeros_like(x)  # 初始化梯度数组，形状和 x 一致

    for idx in range(x.size):  # 遍历 x 的每个元素
        tmp_val = x[idx]  # 备份原来的值

        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # 计算 f(x + h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # 计算 f(x - h)

        grad[idx] = (fxh1 - fxh2) / (2 * h)  # 计算中心差分近似的数值梯度

        x[idx] = tmp_val  # 还原 x 的值
    
    return grad  # 返回梯度数组

# 计算数值梯度（numerical gradient），即通过 微小扰动 来估计函数的梯度。它可以处理 单个数据点 或 批量数据（矩阵）。
def numerical_gradient(f, X):
    # 如果 X 是 一维数组（即单个样本输入），直接调用 _numerical_gradient_no_batch 计算梯度。
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        #处理批量输入（矩阵）
        # X.ndim > 1 时，说明 X 可能是一个 二维矩阵，代表多个样本的输入。
        # 创建一个 形状与 X 相同 的 grad 矩阵用于存储梯度。
        # 遍历 X 的每一行（每个样本 x），对每个 x 计算梯度，并存入 grad[idx]。
        grad = np.zeros_like(X) # 创建一个和 X 形状相同的全 0 数组，数据类型与 X 一致
        
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)
        
        return grad
'''
def 点餐工厂(菜名, 口味):
    def 点单(数量):
        return f"{菜名} × {数量}，口味：{口味}"
    return 点单
    
点辣的宫保鸡丁 = 点餐工厂("宫保鸡丁", "微辣")
print(点辣的宫保鸡丁(2))  # 输出：宫保鸡丁 × 2，口味：微辣

'''
def loss_fn_factory(X, Y, input_size, hidden_size, output_size):
    # 数值梯度计算
    def loss_fn(params):
        W1_flat, b1_flat, W2_flat, b2_flat = np.split(
            params,
            [input_size * hidden_size, 
             input_size * hidden_size + hidden_size, 
             input_size * hidden_size + hidden_size + hidden_size]
        )
        W1 = W1_flat.reshape(input_size, hidden_size)
        b1 = b1_flat
        W2 = W2_flat.reshape(hidden_size, output_size)
        b2 = b2_flat
        output, _ = forward(X, W1, b1, W2, b2)
        return mean_squared_error(Y, output)
    return loss_fn

# 神经网络训练
# X 4,2 Y 4,1
def train_neural_network(X:Array4x2, Y:Array4x1, epochs=1000, learning_rate=1e-2):
    # 随机初始化权重和偏置
    input_size = X.shape[1] # 2
    hidden_size = 3
    output_size = 1
    # 生成一个形状为 (input_size, hidden_size) 的矩阵，矩阵的每个元素是从标准正态分布（均值为0，标准差为1）中随机抽取的值。
    W1 = np.random.randn(input_size, hidden_size)
    b1 = np.zeros(hidden_size)#创建一个元素全部为零的数组
    W2 = np.random.randn(hidden_size, output_size)
    b2 = np.zeros(output_size)
    print(type(X))
    '''
    W1 2x3
    b1 3
    W2 3x1
    b2 1
    '''
    loss_history = []
    W1_history = []
    W2_history = []

    loss_fn = loss_fn_factory(X, Y, input_size, hidden_size, output_size)
    # 训练过程
    for epoch in range(epochs):
        # 前向传播
        output, a1 = forward(X, W1, b1, W2, b2)
        # 计算损失
        loss = mean_squared_error(Y, output)
# flatten() 是 NumPy 中用于 将多维数组展平为一维数组 的方法
        # 合并参数并计算梯度
        params = np.concatenate([W1.flatten(), b1, W2.flatten(), b2])
        grads = numerical_gradient(loss_fn, params)
         
        # 解包梯度
        W1_grad, b1_grad, W2_grad, b2_grad = np.split(
            grads,
            [input_size * hidden_size, input_size * hidden_size + hidden_size, 
             input_size * hidden_size + hidden_size + hidden_size]
        )
        W1_grad = W1_grad.reshape(input_size, hidden_size)
        W2_grad = W2_grad.reshape(hidden_size, output_size)
        # 更新权重和偏置
        W1 -= learning_rate * W1_grad
        b1 -= learning_rate * b1_grad
        W2 -= learning_rate * W2_grad
        b2 -= learning_rate * b2_grad
        # 每 100 步打印一次损失和权重
        if epoch % 100 == 0:
            loss_history.append(loss)
            W1_history.append(W1.copy())
            W2_history.append(W2.copy())
            print(W1)

    return W1, b1, W2, b2, loss_history, W1_history, W2_history

def visualize_training_process(loss_history, W1_history, W2_history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.plot(loss_history)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')

    plt.subplot(1, 3, 2)
    W1_vals = np.array([W1.flatten() for W1 in W1_history])
    W2_vals = np.array([W2.flatten() for W2 in W2_history])

    for i in range(W1_vals.shape[1]):
        plt.plot(W1_vals[:, i], label=f'W1_{i}')
    for i in range(W2_vals.shape[1]):
        plt.plot(W2_vals[:, i], label=f'W2_{i}')

    plt.xlabel('Epochs')
    plt.ylabel('Weight Values')
    plt.title('Weight Changes Over Epochs')
    plt.legend()
    plt.tight_layout()
    # plt.show()

# 可视化损失函数
def visualize_loss_surface():
    W1_vals = np.linspace(-2, 2, 100)
    W2_vals = np.linspace(-2, 2, 100)
    W1_grid, W2_grid = np.meshgrid(W1_vals, W2_vals)
    
    # 创建一个简单的损失函数（以 W1 和 W2 为输入）
    def loss_function(W1, W2):
        return np.sin(W1) + np.cos(W2) + (W1 ** 2 + W2 ** 2) * 0.1
    
    Loss_grid = loss_function(W1_grid, W2_grid)

    # 绘制 3D 图
    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(111, projection='3d')#111 表示“1 行 1 列的网格，选第 1 个子图”。
    ax = plt.subplot(1, 3, 3, projection='3d')
    ax.plot_surface(W1_grid, W2_grid, Loss_grid, cmap='viridis')
    
    ax.set_xlabel("W1")
    ax.set_ylabel("W2")
    ax.set_zlabel("Loss")
    ax.set_title("Loss Function Landscape")
    
    plt.show()
    

def main():
    X: Array4x2 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) # 4行2列
    Y: Array4x1 = np.array([[0], [1], [1], [0]])
    W1, b1, W2, b2, loss_history, W1_history, W2_history = train_neural_network(X, Y, epochs=5000, learning_rate=1e-2)
    visualize_training_process(loss_history, W1_history, W2_history)
    visualize_loss_surface()

if __name__ == "__main__":
    main()
