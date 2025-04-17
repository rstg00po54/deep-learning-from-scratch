import matplotlib.pyplot as plt
import win32gui
import win32con
import numpy as np


# 初始化窗口，只创建一次
fig, ax = plt.subplots()
line, = ax.plot([], [], label='Neuron Output')
ax.set_xlim(0, 100)
ax.set_ylim(-1, 1)
ax.legend()
plt.ion()           # 打开交互模式
plt.show()          # 显示窗口（只调用一次）
# 获取窗口句柄并取消置顶
hwnd = win32gui.GetForegroundWindow()
win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, 0, 0, 0, 0,
                     win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
# 更新图内容
for i in range(1000):
    y = np.sin(np.linspace(0, 2*np.pi, 100) + i * 0.1)
    line.set_ydata(y)
    line.set_xdata(np.arange(len(y)))
    ax.relim()       # 重新计算坐标范围
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()
    # plt.pause(0.01)  # 短暂暂停，不阻塞

plt.ioff()           # 可选：关闭交互模式
