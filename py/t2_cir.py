import numpy as np
import logging
import math

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("NN")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))

def binary_cross_entropy(pred, label):
    epsilon = 1e-7
    return -np.mean(label * np.log(pred + epsilon) + (1 - label) * np.log(1 - pred + epsilon))

def generate_data(n=5000):
    # points = np.random.uniform(0, 2, (n, 2))
    # radius = 1
    # distances = np.linalg.norm(points, axis=1)
    # labels = (distances <= radius).astype(np.float32).reshape(-1, 1)
    width = 2  # 正方形的边长
    num_samples = 100  # 生成的点的数量

    # 生成在 [0, width/2] 范围内的随机点
    points = np.random.uniform(0, width / 2, (n, 2))

    # 计算使圆面积为正方形面积的一半的半径
    radius = math.sqrt(2 / math.pi)  # 半径 = sqrt(0.5 / π)

    # 计算每个点到原点的距离
    distances = np.linalg.norm(points, axis=1)

    # 判断点是否在圆内
    inside_circle = (distances <= radius).astype(int)

    # 统计在圆内的点的数量
    num_points_inside = np.sum(inside_circle)

    print(f"有 {num_points_inside} 个点在圆内，共{num_samples}个点。（理论期望约为 50%）")
    inside_circle = (distances <= radius).astype(np.float32).reshape(-1, 1)
    print(inside_circle)
    return points, inside_circle

def forward(X, W1, b1, W2, b2):
    z1 = np.dot(X, W1) + b1
    a1 = np.tanh(z1)  # ✅ 用 tanh 代替 ReLU
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)  # ✅ 输出用 sigmoid
    return a2, (X, z1, a1, z2, a2)

def backward(Y, cache, W2):
    X, z1, a1, z2, a2 = cache
    dz2 = a2 - Y
    dW2 = np.dot(a1.T, dz2) / len(X)
    db2 = np.mean(dz2, axis=0, keepdims=True)

    dz1 = np.dot(dz2, W2.T) * (1 - np.tanh(z1)**2)
    dW1 = np.dot(X.T, dz1) / len(X)
    db1 = np.mean(dz1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2

def train():
    X, Y = generate_data()
    input_dim = 2
    hidden_dim = 16
    output_dim = 1

    W1 = np.random.randn(input_dim, hidden_dim) * 0.1
    b1 = np.zeros((1, hidden_dim))
    W2 = np.random.randn(hidden_dim, output_dim) * 0.1
    b2 = np.zeros((1, output_dim))

    learning_rate = 0.01

    for epoch in range(5000):
        pred, cache = forward(X, W1, b1, W2, b2)
        loss = binary_cross_entropy(pred, Y)
        if epoch % 300 == 0:
            logger.warning(f"Epoch {epoch}, Loss: {loss:.4f}")

        dW1, db1, dW2, db2 = backward(Y, cache, W2)

        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

    return W1, b1, W2, b2

def test(W1, b1, W2, b2):
    X, Y = generate_data(20)
    pred, _ = forward(X, W1, b1, W2, b2)
    logger.warning("预测结果：")
    for i in range(len(pred)):
        logger.warning(f"点: {X[i]}, 实际: {int(Y[i][0])}, 预测: {pred[i][0]:.2f}")

if __name__ == "__main__":
    W1, b1, W2, b2 = train()
    test(W1, b1, W2, b2)
