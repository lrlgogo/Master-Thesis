import numpy as np
import matplotlib.pyplot as plt


class TestPro:
    """
    this is the class of 3-Dim multiple objective optimization problem
    provides basic utils for sampling, drawing and others
    """
    def __init__(
            self,
            moo_pro,
            sample_num=10,
            resolution=(128, 128, 128),
            y_clip_range=((0., 1.), (0., 1.), (0., 1.))
    ):
        self.moo_pro = moo_pro
        self.res = resolution
        self.y_range = y_clip_range
        self.M = 3  # this class founded on 3-D test problem
        # self.k = moo_pro.k
        self.n = moo_pro.n
        self.sample_num = sample_num

        self.y_samples = [[] for _ in range(self.M)]
        self.x_samples = None  # unnecessary
        self.y_index_list = None
        self.y_matrix = None

    @staticmethod
    def _to_index(x, x_min, x_max, res):

        return int((x - x_min) / (x_max - x_min) * (res - 1) + 0.5)

    def _index_list(self):

        y_temp = []
        for i in range(self.M):
            y_temp.append(np.clip(self.y_samples[i], *self.y_range[i]))
        self.y_index_list = [[] for _ in range(self.M)]
        for i in range(self.M):
            for j in range(len(y_temp[i])):
                temp = self._to_index(
                        y_temp[i][j],
                        *self.y_range[i],
                        self.res[i]
                    )
                if temp not in self.y_index_list:
                    self.y_index_list[i].append(temp)

        return 0

    def samples(self):
        x_range = self.moo_pro.x_range
        x_line_space = []
        for i in range(self.n):
            x_line_space.append(
                np.linspace(*x_range[i], self.sample_num + 1)
            )
        x_index = [0 for _ in range(self.n + 1)]
        x_list = []  # unnecessary
        y_list = [[] for _ in range(self.M)]

        for dec_index in range((self.sample_num + 1) ** self.n):

            temp = [x_line_space[i][x_index[-1 - i]] for i in range(self.n)]
            x_list.append(temp)
            y_obj = self.moo_pro.fun_obj(np.array(temp))

            for i in range(self.M):
                y_list[i].append(y_obj[i])

            add_flag = -1
            while x_index[add_flag] == self.sample_num:
                x_index[add_flag] = 0
                add_flag -= 1
            x_index[add_flag] += 1

        print('x code index: ', x_index)  # unnecessary, just for test

        self.y_samples = y_list
        self.x_samples = x_list
        self._index_list()

        return 0

    def to_matrix(self):
        data = self.y_index_list
        res = self.res
        matrix = np.zeros(res, np.float32)
        for i in range(len(data[0])):
            matrix[data[0][i], data[1][i], data[2][i]] = 1.
        self.y_matrix = matrix
        return 0

    def run(self):
        self.samples()
        self.to_matrix()
        return 0

    @staticmethod
    def show3d(data, size=40):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(*data, marker='o', c='r', s=size)
        plt.show()

        return 0


class DTLZ1:
    """
    test problem DTLZ1
    """
    def __init__(self, x_range, k=1):
        self.x_range = x_range
        self.M = 3
        self.k = k
        self.n = self.M + self.k - 1

    @staticmethod
    def g(x, k, m):
        return 100 * (
            k + sum(
                (x[m - 1:] - 0.5) ** 2 - np.cos(20 * np.pi * (x[m - 1:] - 0.5))
            )
        )

    def fun_obj(self, x):
        k = self.k
        m = self.M
        g_value = self.g(x, k, m)
        f1 = 0.5 * x[0] * x[1] * (1 + g_value)
        f2 = 0.5 * x[0] * (1 - x[1]) * (1 + g_value)
        f3 = 0.5 * (1 - x[0]) * (1 + g_value)
        return f1, f2, f3


class DTLZ2:
    """
    test problem DTLZ2
    """
    def __init__(self, x_range, k=1):
        self.x_range = x_range
        self.M = 3
        self.k = k
        self.n = self.M + self.k - 1

    @staticmethod
    def g(x, m):
        return sum((x[m - 1:] - 0.5) ** 2)

    def fun_obj(self, x):
        m = self.M
        k = self.k
        g_value = self.g(x, m)
        theta = [0.5 * np.pi * x[i] for i in range(m - k)]
        f1 = np.cos(theta[0]) * np.cos(theta[1]) * (1 + g_value)
        f2 = np.cos(theta[0]) * np.sin(theta[1]) * (1 + g_value)
        f3 = np.sin(theta[0]) * (1 + g_value)
        return f1, f2, f3


class DTLZ3:
    """
    test problem DTLZ3
    """
    def __init__(self, x_range, k=1):
        self.x_range = x_range
        self.M = 3
        self.k = k
        self.n = self.M + self.k - 1

    @staticmethod
    def g(x, k, m):
        return 100 * (
            k + sum(
                (x[m - 1:] - 0.5) ** 2 - np.cos(20 * np.pi * (x[m - 1:] - 0.5))
            )
        )

    def fun_obj(self, x):
        k = self.k
        m = self.M
        g_value = self.g(x, k, m)
        theta = [0.5 * np.pi * x[i] for i in range(m - k)]
        f1 = np.cos(theta[0]) * np.cos(theta[1]) * (1 + g_value)
        f2 = np.cos(theta[0]) * np.sin(theta[1]) * (1 + g_value)
        f3 = np.sin(theta[0]) * (1 + g_value)
        return f1, f2, f3


class DTLZ4:
    """
    test problem DTLZ4
    """

    def __init__(self, x_range, k=1, alpha=100):
        self.x_range = x_range
        self.M = 3
        self.k = k
        self.n = self.M + self.k - 1
        self.alpha = alpha

    @staticmethod
    def g(x, m):
        return sum((x[m - 1:] - 0.5) ** 2)

    def fun_obj(self, x):
        m = self.M
        k = self.k
        g_value = self.g(x, m)
        theta = [0.5 * np.pi * (x[i] ** self.alpha) for i in range(m - k)]
        f1 = np.cos(theta[0]) * np.cos(theta[1]) * (1 + g_value)
        f2 = np.cos(theta[0]) * np.sin(theta[1]) * (1 + g_value)
        f3 = np.sin(theta[0]) * (1 + g_value)
        return f1, f2, f3


class DTLZ5:
    """
    test problem DTLZ5
    """

    def __init__(self, x_range, k=1, alpha=0.1):
        self.x_range = x_range
        self.M = 3
        self.k = k
        self.n = self.M + self.k - 1
        self.alpha = alpha

    def g(self, x, m):
        return np.sum(x[m - 1:] ** self.alpha)

    def fun_obj(self, x):
        m = self.M
        # k = self.k
        g_value = self.g(x, m)
        theta = [
            0.5 * np.pi * x[0],
            np.pi * (1 + 2 * g_value * x[1]) / (4 * (1 + g_value))
        ]
        f1 = np.cos(theta[0]) * np.cos(theta[1]) * (1 + g_value)
        f2 = np.cos(theta[0]) * np.sin(theta[1]) * (1 + g_value)
        f3 = np.sin(theta[0]) * (1 + g_value)
        return f1, f2, f3


class DTLZ6:
    """
    test problem DTLZ6
    """
    def __init__(self, x_range, k=1):
        self.x_range = x_range
        self.M = 3
        self.k = k
        self.n = self.M + self.k - 1

    @staticmethod
    def g(x, k, m):
        return 1 + 9 * np.sum(x[m - 1:]) / k

    def fun_obj(self, x):
        m = self.M
        k = self.k
        g_value = self.g(x, k, m)
        f1 = x[0]
        f2 = x[1]
        h_value = m - sum([fi * (1 + np.sin(3 * np.pi * fi)) / (1 + g_value)
                           for fi in [f1, f2]])
        f3 = (1 + g_value) * h_value

        return f1, f2, f3
