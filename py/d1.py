import numpy as np
import matplotlib.pyplot as plt
import struct

# 读取 MNIST 图像文件 (.ubyte)
def read_mnist_images(file_path):
    with open(file_path, 'rb') as f:
        # 读取文件头信息 (魔法数, 图像数量, 图像行数, 图像列数)
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        
        # 读取图像数据
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
        
    return images

# 读取 MNIST 标签文件 (.ubyte)
def read_mnist_labels(file_path):
    with open(file_path, 'rb') as f:
        # 读取文件头信息 (魔法数, 标签数量)
        magic, num_labels = struct.unpack(">II", f.read(8))
        
        # 读取标签数据
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        
    return labels

# 读取图像和标签
train_images = read_mnist_images("../../TensorFlow-MNIST/mnist/data/train-images.idx3-ubyte")
train_labels = read_mnist_labels("../../TensorFlow-MNIST/mnist/data/train-labels.idx1-ubyte")

# 显示第一张图像及其标签
plt.imshow(train_images[0], cmap="gray")
plt.title(f"Label: {train_labels[0]}")
plt.show()

# 显示前 5 张图像及其标签
plt.figure(figsize=(10, 2))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(train_images[i], cmap="gray")
    plt.title(f"Label: {train_labels[i]}")
    plt.axis('off')
plt.show()









# train_images = read_mnist_images("../../TensorFlow-MNIST/mnist/data/train-images.idx3-ubyte")
