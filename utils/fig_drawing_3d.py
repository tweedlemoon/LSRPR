import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def generate_3d_fig(side_lenth=28, z=0):
    plt.rcParams['font.sans-serif'] = ['STKAITI']
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.rcParams['axes.facecolor'] = '#cc00ff'
    # fig = plt.figure(figsize=(10, 8), facecolor='#cc00ff')
    fig = plt.figure(figsize=(16, 9))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    delta = 1
    # 生成代表X轴数据的列表
    x = np.arange(-side_lenth / 2, side_lenth / 2, delta)
    # 生成代表Y轴数据的列表
    y = np.arange(-side_lenth / 2, side_lenth / 2, delta)
    # 对x、y数据执行网格化
    x, y = np.meshgrid(x, y)
    z1 = z
    # z1 = np.exp(-x ** 2 - y ** 2)
    z2 = np.zeros_like(z1)
    # 计算Z轴数据（高度数据）
    # Z = (Z1 - Z2) * 2
    # # 绘制3D图形
    # ax.plot_surface(X, Y, Z,
    #                 rstride=1,  # rstride（row）指定行的跨度
    #                 cstride=1,  # cstride(column)指定列的跨度
    #                 cmap=plt.get_cmap('rainbow'))  # 设置颜色映射
    ax.plot_surface(x, y, z1,
                    cmap=plt.get_cmap('OrRd'))  # 设置颜色映射
    ax.plot_surface(x, y, z2, alpha=0.5,
                    cmap=plt.get_cmap('hot'))
    plt.xlabel('X axis', fontsize=15)
    plt.ylabel('Y axis', fontsize=15)
    ax.set_zlabel('Z axis', fontsize=15)
    ax.set_title('', y=1.02, fontsize=25, color='gold')
    # # 设置Z轴范围
    # ax.set_zlim(-2, 2)
    plt.show()


if __name__ == '__main__':
    generate_3d_fig()
