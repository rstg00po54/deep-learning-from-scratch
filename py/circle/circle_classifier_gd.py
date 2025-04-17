import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from logger import Logger
import pygetwindow as gw
import my_logger as log
import win32gui
import win32con
import sys
import os
logger = Logger(enabled=True, level='info')
'''
输入层：2个神经元

隐藏层：3个神经元

输出层：1个神经元


两行三列
matrix = np.array([[-0.20212614, 0.06943245, 0.41022952],
                   [-0.1731148 , 0.61708178, 0.75795003]])
2x3  3x1
 W1   W2
x   x   
    x   x
x   x   
 b1   b2

'''
import numpy as np
import numpy.typing as npt
Array4x2 = npt.NDArray[np.float64]
Array4x1 = npt.NDArray[np.float64]
Array2x3 = npt.NDArray[np.float64]
Array3x1 = npt.NDArray[np.float64]
Array3 = npt.NDArray[np.float64]
Array1 = npt.NDArray[np.float64]

Array4x3 = npt.NDArray[np.float64]
Array0x4 = npt.NDArray[np.float64]

# 创建全局变量
fig, axs = None, None
running = True
# 激活函数 ReLU 和它的导数 max(0,x)
def leaky_relu(x):
    return np.where(x > 0, x, 0.01 * x)
def leaky_relu_deriv(x):
    return np.where(x > 0, 1.0, 0.01)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def relu(x):
    return np.maximum(0, x)
def activation_fn(x):

    # return sigmoid(x)

    return relu(x)
def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)  # 防止溢出
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# 导数
def relu_deriv(x):
    return (x > 0).astype(float)

def cross_entropy(y_pred, y_true):
    eps = 1e-9
    return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))
# 损失函数 (均方误差)
def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r = np.mean((y_true - y_pred) ** 2)
    # print(y_true)
    # print(y_pred)
    # print(r)
    ss = y_true - y_pred
    return r
# def mean_squared_error(pred, label):
#     epsilon = 1e-7
#     return -np.mean(label * np.log(pred + epsilon) + (1 - label) * np.log(1 - pred + epsilon))

# def forward(X, W1, b1, W2, b2):
#     z1 = np.dot(X, W1) + b1
#     a1 = np.tanh(z1)  # 隐藏层激活（你也可以用 ReLU 或 sigmoid）
#     z2 = np.dot(a1, W2) + b2
#     y = softmax(z2)   # 添加 softmax 输出
#     print("89")
#     return y, a1, z2
# 神经网络前向传播
def forward( x: Array4x2, 
            W1: Array2x3, 
            b1: Array3, 
            W2: Array3x1, 
            b2: Array1) -> tuple[Array4x1, Array4x3]:
    logger.debug("forward")
    z1: Array4x3 = np.dot(x, W1) + b1
    a1: Array4x3 = np.tanh(z1)
    # logger.warning(a1)
    z2: Array4x1 = np.dot(a1, W2) + b2
    y = softmax(z2) 
    # print("100")
    return  z2, z1, a1

# 数值梯度计算
# 计算了 一个标量函数 f(x) 相对于 x 的 数值梯度（numerical gradient），适用于 x 是 一维数组 的情况。
def _numerical_gradient_no_batch(f, x):
    logger.debug("一个标量函数 f(x) 相对于 x 的 数值梯度")
    h = 1e-3  # 0.0001，微小的扰动值
    grad = np.zeros_like(x)  # 初始化梯度数组，形状和 x 一致
    logger.debug(f"x.size {x.size}")
    for idx in range(x.size):  # 遍历 x 的每个元素

        logger.debug(f"idx ----{idx}")
        tmp_val = x[idx]  # 备份原来的值

        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # 计算 f(x + h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # 计算 f(x - h)

        grad[idx] = (fxh1 - fxh2) / (2 * h)  # 计算中心差分近似的数值梯度
        logger.debug(f"grad {grad[idx]:.2f}")
        # if idx == 0:
            # logger.warning(f"idx {idx}, fxh1={fxh1:.6f}, fxh2={fxh2:.6f}, diff={(fxh1 - fxh2):.6e}, grad={grad[idx]:.6e}")

        x[idx] = tmp_val  # 还原 x 的值
    logger.debug(f"return {grad}")
    return grad  # 返回梯度数组

# 计算数值梯度（numerical gradient），即通过 微小扰动 来估计函数的梯度。它可以处理 单个数据点 或 批量数据（矩阵）。
def numerical_gradient(f, X):
    logger.debug(f"计算数值梯度, X 是 {X.ndim} 维的数组")
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
def cross_entropy_error(y, t):
    delta = 1e-7  # 避免 log(0)
    return -np.sum(t * np.log(y + delta)) / y.shape[0]

'''
def 点餐工厂(菜名, 口味):
    def 点单(数量):
        return f"{菜名} x {数量}，口味：{口味}"
    return 点单
    
点辣的宫保鸡丁 = 点餐工厂("宫保鸡丁", "微辣")
logger.debug(点辣的宫保鸡丁(2))  # 输出：宫保鸡丁 × 2，口味：微辣

'''
def loss_fn_factory(X, Y, input_size2, hidden_size3, output_size1):
    # 数值梯度计算
    logger.debug('数值梯度计算')
    def loss_fn(params):
        logger.debug("loss_fn")
        W1_flat, b1_flat, W2_flat, b2_flat = np.split(
            params,
            [input_size2 * hidden_size3, 
             input_size2 * hidden_size3 + hidden_size3, 
             input_size2 * hidden_size3 + hidden_size3 + hidden_size3]
        )
        W1 = W1_flat.reshape(input_size2, hidden_size3)
        b1 = b1_flat
        W2 = W2_flat.reshape(hidden_size3, output_size1)
        b2 = b2_flat
        output, a1, z2 = forward(X, W1, b1, W2, b2)
        return mean_squared_error(Y, output)
        # return cross_entropy_error(Y, output)
    return loss_fn
def on_close(event):
    global running
    running = False
    print("窗口关闭，退出")
    # plt.close('all')
    # sys.exit(0)
    # sys.exit(0)
# 神经网络训练
# X 4,2 Y 4,1
def train_neural_network(X:Array4x2, Y:Array4x1, epochs=1000, learning_rate=1e-2):
    # 随机初始化权重和偏置
    input_size2 = X.shape[1] # 2
    hidden_size3 = 3
    output_size1 = 1
    
    np.set_printoptions(precision=2)
    # 生成一个形状为 (input_size2, hidden_size3) 的矩阵，矩阵的每个元素是从标准正态分布（均值为0，标准差为1）中随机抽取的值。
    W1: Array2x3 = np.random.randn(input_size2, hidden_size3)
    b1: Array3   = np.zeros(hidden_size3)#创建一个元素全部为零的数组
    W2: Array3x1 = np.random.randn(hidden_size3, output_size1)
    b2: Array1   = np.zeros(output_size1)
    logger.debug(type(X))
    # -----------------------------------------------------
    W1 = np.array([
        [0.5,  0.3, 0.4],   # 对应输入特征 x1 的权重
        [0.2,  0.7, 0.5]   # 对应输入特征 x2 的权重
    ])  # shape: (2, 3)
    W2 = np.array([
        [0.1],   # 对应输入特征 x1 的权重
        [0.4],   # 对应输入特征 x2 的权重
        [0.6]
    ])  # shape: (2, 3)
    W1 = np.random.randn(input_size2, hidden_size3) * np.sqrt(2. / input_size2)
    W2 = np.random.randn(hidden_size3, output_size1) * np.sqrt(2. / hidden_size3)

    # log.warning_array(W1, tag="W1")
    # W1 = np.random.randn(input_size2, hidden_size3) * np.sqrt(2. / hidden_size3)
    # W2 = np.random.randn(hidden_size3, output_size1) * np.sqrt(2. / output_size1)
    '''
    W1 2x3
    b1 3
    W2 3x1
    b2 1
    '''
    loss_history = []
    W1_history = []
    W2_history = []
    a1_history = []
    z1_history = []

    loss_fn = loss_fn_factory(X, Y, input_size2, hidden_size3, output_size1)
    # 训练过程
    for epoch in range(epochs):
        if running  == False:
            print("退出")
            sys.exit(0)
        logger.debug("------------------------------------------")
        # 前向传播
        output, z1, a1 = forward(X, W1, b1, W2, b2)
        logger.debug(output)
        # axis=0 → 沿着“竖直方向”聚合（跨样本、按列求平均）
        # axis=1 → 沿着“水平方向”聚合（每个样本内部求平均）
        a1_mean = np.mean(a1, axis=0)
        z1_mean = np.mean(z1, axis=0)
        a1_history.append(a1_mean)
        z1_history.append(z1_mean)
        # 计算损失
        loss: float = mean_squared_error(Y, output)
# flatten() 是 NumPy 中用于 将多维数组展平为一维数组 的方法
        # 合并参数并计算梯度
        params = np.concatenate([W1.flatten(), b1, W2.flatten(), b2])
        grads = numerical_gradient(loss_fn, params)
         
        # 解包梯度
        W1_grad_2x3, b1_grad_3, W2_grad_3x1, b2_grad_3 = np.split(
            grads,
            [input_size2 * hidden_size3, input_size2 * hidden_size3 + hidden_size3, 
             input_size2 * hidden_size3 + hidden_size3 + hidden_size3]
        )
        W1_grad_2x3 = W1_grad_2x3.reshape(input_size2, hidden_size3)
        W2_grad_3x1 = W2_grad_3x1.reshape(hidden_size3, output_size1)
        # 更新权重和偏置
        W1 -= learning_rate * W1_grad_2x3
        b1 -= learning_rate * b1_grad_3
        W2 -= learning_rate * W2_grad_3x1
        b2 -= learning_rate * b2_grad_3
        # 每 100 步打印一次损失和权重
        loss_history.append(loss)
        W1_history.append(W1.copy())
        W2_history.append(W2.copy())
        if epoch % 200 == 0:
            logger.warning(f"运行: {epoch}, loss {loss:.3f}")
            log.warning_array(W1, tag="输出向量：")
            log.warning_array(b1, tag="输出向量：")
            log.warning_array(W2, tag="输出向量：")
            log.warning_array(b2, tag="输出向量：")
            log.warning_array(grads, tag="输出向量：grads")
            loss = mean_squared_error(Y, output)
            print(f"Epoch {epoch}, Loss: {loss:.6f}")
            print("a1 dead neurons:", (a1 == 0).sum(), "/", a1.size)
            # log.warning_array(grads, tag="grads")
        global fig, axs  # 使用 global 声明全局变量
        if epoch % 10 == 0:
            axs[0].clear()
            a1_array = np.array(a1_history)
            for i in range(3):  # 隐藏层 3 个神经元
                y_values = [a[i] for a in a1_array]
                axs[0].plot(y_values)
                # axs[0].plot(y_values, label=f'Neuron {i+1}')

                # 显示当前值
                axs[0].text(len(y_values)+5, y_values[-1], f'{y_values[-1]:.2f}', fontsize=8, ha='left', va='center', color=axs[0].get_lines()[-1].get_color())
                # 也可以加个小圆点
                axs[0].plot(len(y_values)-1, y_values[-1], 'o', color=axs[0].get_lines()[-1].get_color())
            loss_array = np.array(loss_history)
            loss_values = np.array(loss_history)
            axs[0].plot(loss_values, label="loss")
            axs[0].text(5, loss_values[-1], f'loss:{loss_values[-1]:.2f}', fontsize=8, ha='left', va='center', color=axs[0].get_lines()[-1].get_color())
            axs[0].set_title('Hidden Layer Activation (a1)')
            axs[0].set_xlabel('Epoch')
            axs[0].set_ylabel('Activation')
            axs[0].legend()

            axs[1].clear()
            z1_array = np.array(z1_history)
            for i in range(3):  # 隐藏层 3 个神经元
                # ax.plot(a1_array[:, i], label=f'Neuron {i+1}')
                y_values = [z[i] for z in z1_array]
                axs[1].plot(y_values, label=f'Neuron {i+1}')
                # 显示当前值
                axs[1].text(len(y_values)+5, y_values[-1], f'{y_values[-1]:.2f}', fontsize=8, ha='left', va='center', color=axs[0].get_lines()[-1].get_color())
                # 也可以加个小圆点
                axs[1].plot(len(y_values)-1, y_values[-1], 'o', color=axs[0].get_lines()[-1].get_color())
            axs[1].plot(np.array(loss_history), label="loss")
            axs[1].set_title('Hidden Layer Activation (a1)')
            axs[1].set_xlabel('Epoch')
            axs[1].set_ylabel('Activation')
            axs[1].legend()
            if running :
                fig.canvas.draw()
                fig.canvas.flush_events()

            # plt.pause(0.01)
            # plt.show(block=False)
    # plt.ioff()
    # plt.show()
    # plt.show(block=False)
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
    
    # plt.show()
    
def calc():
    width = 2  # 正方形的边长
    num_samples = 100  # 生成的点的数量

    # 生成在 [0, width/2] 范围内的随机点
    points = np.random.uniform(0, width / 2, (num_samples, 2))

    # 计算使圆面积为正方形面积的一半的半径
    radius = math.sqrt(2 / math.pi)  # 半径 = sqrt(0.5 / π)

    # 计算每个点到原点的距离
    distances = np.linalg.norm(points, axis=1)

    # 判断点是否在圆内
    # inside_circle = (distances <= radius).astype(int)
    # inside_circle = (distances <= radius).astype(np.float32).reshape(-1, 1)
    inside_circle = (distances <= radius).astype(int).reshape(-1, 1)
    # 统计在圆内的点的数量
    num_points_inside = np.sum(inside_circle)


    print(f"有 {num_points_inside} 个点在圆内，共{num_samples}个点。（理论期望约为 50%）")

    # 打印前十个点的信息
    print("inside_circle[:10]:", inside_circle[:10])
    # print("points[:10]:", points[:10])
    print("distances[:10]:", distances[:10])
    W1, b1, W2, b2, loss_history, W1_history, W2_history = train_neural_network(points, inside_circle, epochs=3000, learning_rate=1e-1)
    # 测试
    points = np.random.uniform(0, 2, (20, 2))  # 随机生成点坐标
    # 圆心 (0, 0)，半径 1
    distances = np.linalg.norm(points, axis=1)  # 计算每个点到原点的距离
    # 判断哪些点在圆内
    inside_circle = (distances <= radius).astype(int)   # 点是否在圆内
    print("inside_circle[:10]:", inside_circle[:10])
    # print("points[:10]:", points[:10])
    print("distances[:10]:", distances[:10])
    print("radius:", radius)
    pred, a1, z2 = forward(points, W1, b1, W2, b2)
    r = mean_squared_error(pred, inside_circle)
    logger.warning(f"预测结果：{r}")
    log.warning_array(pred, tag="输出向量：")


def tran_xy() :
    X: Array4x2 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) # 4行2列
    Y: Array4x1 = np.array([[0], [1], [1], [0]])
    W1, b1, W2, b2, loss_history, W1_history, W2_history = train_neural_network(X, Y, epochs=10000, learning_rate=1e-3)

    pred, _ = forward(X, W1, b1, W2, b2)
    logger.warning("预测结果：")
    log.warning_array(pred, tag="输出向量：")
    visualize_training_process(loss_history, W1_history, W2_history)
    visualize_loss_surface()
def main():
    # tran_xy()
    try:
        # while True:
            # 主程序逻辑，比如训练、绘图、循环等
        plt.ion()           # 打开交互模式
        plt.show()          # 显示窗口（只调用一次）
        # 获取窗口句柄并取消置顶
        hwnd = win32gui.GetForegroundWindow()
        win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, 0, 0, 0, 0,
                            win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
    # fig, ax = plt.subplots()
        global fig, axs  # 使用 global 声明全局变量
        fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        fig.tight_layout()  # 添加这一行，自动调整间距避免重叠
        plt.subplots_adjust(hspace=0.3) 
        fig.canvas.mpl_connect('close_event', on_close)

        calc()
        # pass
        plt.ioff()
        plt.show()
    except KeyboardInterrupt:
        print("\n已手动终止程序(Ctrl+C), 正常退出。")
if __name__ == "__main__":
    main()
