from pro.PFRecAlg3d import *

import numpy as np
import matplotlib.pyplot as plt


k = 1
M = 3
n = M + k - 1


def g(x, k=k, m=M):
    return 100 * (
        k + sum(
            (x[m-1:] - 0.5) ** 2 - np.cos(20 * np.pi * (x[m-1:] - 0.5))
        )
    )


def f(x, k=k, m=M):
    f1 = 0.5 * x[0] * x[1] * (1 + g(x, k, m))
    f2 = 0.5 * x[0] * (1 - x[1]) * (1 + g(x, k, m))
    f3 = 0.5 * (1 - x[0]) * (1 + g(x, k, m))
    return f1, f2, f3


def to_index(x, x_min, x_max, res):
    return int((x - x_min) / (x_max - x_min) * (res - 1) + 0.5)


def index_list(
        data, y_range=((0., 1.), (0., 1.), (0., 1.)), res=(128, 128, 128)
):
    y1 = np.clip(data[0], *y_range[0])
    y2 = np.clip(data[1], *y_range[1])
    y3 = np.clip(data[2], *y_range[2])

    y_list = [[] for _ in range(3)]
    for i in range(len(y1)):
        y_list[0].append(to_index(y1[i], *y_range[0], res[0]))
        y_list[1].append(to_index(y2[i], *y_range[1], res[1]))
        y_list[2].append(to_index(y3[i], *y_range[2], res[2]))

    return y_list


def samples(num=10, x_range=(0., 1.)):
    x_line_space = list(np.linspace(*x_range, num + 1))
    x_index = [0 for _ in range(n + 1)]
    x_list = []
    y_list = [[] for _ in range(3)]

    for dec_index in range((num + 1)**n):

        temp = [x_line_space[t] for t in x_index[1:]]
        # temp[-1] = 0.5
        x_list.append(temp)
        y1, y2, y3 = f(np.array(temp, np.float32), k=k, m=M)
        y_list[0].append(y1)
        y_list[1].append(y2)
        y_list[2].append(y3)

        add_flag = -1
        while x_index[add_flag] == num:
            x_index[add_flag] = 0
            add_flag -= 1
        x_index[add_flag] += 1

    return x_list, y_list, x_index


def show3d(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*data, marker='o', c='r', s=40)
    plt.show()
    return 0


def to_matrix(data, res=(128, 128, 128)):
    matrix = np.zeros(res, np.float32)
    for i in range(len(data[0])):
        matrix[data[0][i], data[1][i], data[2][i]] = 1.
    return matrix


x, y, index = samples(100, (0, 1))

resolution = 3 * (256,)
# show3d(y)
y_list = index_list(y, ((0., 1.), (0., 1.), (0., 1.)), resolution)
# show3d(y_list)
input = to_matrix(y_list, resolution)

f_rec = PFRecAlg3d(u=2)
f_rec.get_input(input)
f_rec.run()
f_rec.show()
