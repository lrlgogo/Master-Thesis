"""
this contains the classes, methods and utils etc. to generate the examples
...
"""
import math as mt
import numpy as np
import matplotlib.pyplot as plt


class Problem:
    """
    this is a class to manage and store the state, functions of a MOO problem
    """
    def __init__(self, fun, x_range, y_range):
        self.fun = fun
        self.x_range = x_range
        self.y_range = y_range
        self.x_dim = len(x_range)


class ExhaustionSampleImage:
    """
    this is the class to manage and store the image of a problem
    """
    def __init__(
            self,
            prob_class: Problem,
            y_resolution=(256, 256),
            init_samples=10000
    ):
        self.prob = prob_class
        self.y1_res, self.y2_res = y_resolution
        self.x_range = prob_class.x_range
        self.y_range = prob_class.y_range
        self.x_sample_range = self.x_range
        self.y_sample_range = self.y_range
        self.x_dim = prob_class.x_dim
        self.sample_num = init_samples
        self.y_image = None
        self.y_list = [[], []]
        # self.x_list = None

    def run(self, y_res=None, samples=None, x_range=None, y_range=None):
        if y_res is not None:
            self.y1_res, self.y2_res = y_res
        if samples is not None:
            self.sample_num = samples
        if x_range is not None:
            self.x_sample_range = x_range
        if y_range is not None:
            self.y_sample_range = y_range
        x_list = []
        for dim in range(self.x_dim):
            x_list.append(
                list(np.linspace(*self.x_range[dim], self.sample_num + 1))
            )
        sample_index = [0] * (self.x_dim + 1)
        # self.x_list = [[] for _ in range(self.x_dim)]
        for dec_index in range((self.sample_num + 1) ** self.x_dim):
            x_input = [
                x_list[dim][sample_index[-dim - 1]] for dim in range(self.x_dim)
            ]
            """
            for index in range(self.x_dim):
                self.x_list[index].append(x_input[index])
            """
            y1, y2 = self.prob.fun(x_input)
            y1 = np.clip(y1, *self.y_sample_range[0])
            y2 = np.clip(y2, *self.y_sample_range[1])
            self.y_list[0].append(y1)
            self.y_list[1].append(y2)
            add_flag = -1
            while sample_index[add_flag] == self.sample_num:
                sample_index[add_flag] = 0
                add_flag -= 1
            sample_index[add_flag] += 1

        self._to_image((self.y1_res, self.y2_res), self.y_sample_range)

        return sample_index

    def plot(self):
        plt.scatter(*self.y_list, s=1, c='blue')
        plt.show()
        return 0

    def show(self):
        plt.imshow(self.y_image, cmap='Greys')
        plt.show()
        return 0

    def image(self):
        return self.y_image

    @staticmethod
    def _to_index(x, x_range, res):
        return int(
            (x - x_range[0]) / (x_range[1] - x_range[0]) * (res - 1) + 0.5
        )

    def _to_image(self, y_res=None, y_range=None):
        if y_res is not None:
            self.y1_res, self.y2_res = y_res
        if y_range is not None:
            self.y_range = y_range
        n = len(self.y_list[0])
        image = np.zeros((self.y1_res, self.y2_res), dtype=np.int8)
        for i in range(n):
            index1 = self._to_index(
                self.y_list[0][i], self.y_range[0], self.y1_res
            )
            index2 = self._to_index(
                self.y_list[1][i], self.y_range[1], self.y2_res
            )
            if index1 < self.y1_res and index2 < self.y2_res:
                image[index2, index1] = 1
        self.y_image = image[::-1, :]
        return 0


valN = 2
valInitPar = [[0.5, 0, 9], [2, 0, 9], [0.5, 10, 9],
              [0.5, 10., 4.], [6., 4., 4., 2., 9., 0.25]]
valInitName = ['ZDT1', 'ZDT2', 'ZDT3', 'ZDT4', 'ZDT6']


def g1(x, n):
    return 1 + 9 * sum(x[1:]) / (n - 1)


def funZDT1(x, n=valN):
    x1 = x[0]
    f1 = x1
    gx = g1(x, n)
    f2 = gx * (1 - (x1 / gx) ** 0.5)
    return f1, f2


def funZDT123(x, par_list, n=valN):
    a, b, c = par_list
    x1 = x[0]
    f1 = x1
    g = lambda t: 1 + c * sum(t[1:]) / (n - 1)
    gx = g(x)
    f2 = gx * (1 - (x1 / gx) ** a - x1 * mt.sin(b * mt.pi * x1) / gx)
    return f1, f2


def funZDT4(x, par_list=valInitPar[3], n=valN):
    a, b, c = par_list
    x1 = x[0]
    f1 = x1
    g = lambda x: 1 + b * (n - 1) + sum(
        [xi**2 - b * mt.cos(c * mt.pi * xi) for xi in x[1:]])
    gx = g(x)
    f2 = gx * (1 - (x1 / gx)**a)
    return f1, f2


def funZDT6(x, par_list=valInitPar[4], n=valN):
    a, b, c, d, e, f = par_list
    f1 = 1 - ((mt.sin(b * mt.pi * x[0])) ** a) * mt.exp(-1 * c * x[0])
    g = lambda x: 1 + e * ((sum(x[1:]) / (n - 1)) ** f)
    gx = g(x)
    f2 = gx * (1 - (f1 / gx) ** d)
    return f1, f2


def funZDT2(x, par_list=valInitPar[1], n=valN):
    return funZDT123(x, par_list, n)


def funZDT3(x, par_list=valInitPar[2], n=valN):
    return funZDT123(x, par_list, n)


def funSCH(x):
    return x[0] ** 2, (x[0] - 2) ** 2


def funFON(x):
    f1 = 1 - np.exp(-np.sum((np.array(x) - 1 / 3 ** 0.5) ** 2))
    f2 = 1 - np.exp(-np.sum((np.array(x) + 1 / 3 ** 0.5) ** 2))
    return f1, f2


pro_zdt1 = Problem(funZDT1, ((0., 1.), (0., 0.01)), ((0, 1), (0, 1)))
zdt1 = ExhaustionSampleImage(pro_zdt1, (256, 256), 100)

pro_zdt2 = Problem(funZDT2, ((0., 1.), (0., 1.)), ((0, 1), (0, 1)))
zdt2 = ExhaustionSampleImage(pro_zdt2, (256, 256), 1000)

pro_zdt3 = Problem(funZDT3, ((0., 1.), (0., 1.)), ((0, 0.5), (0, 1)))
zdt3 = ExhaustionSampleImage(pro_zdt3, (256, 256), 1000)

pro_zdt4 = Problem(funZDT4, ((0., 1.), (0., 1.)), ((0, 1), (0, 1)))
zdt4 = ExhaustionSampleImage(pro_zdt4, (256, 256), 1000)

pro_zdt6 = Problem(funZDT6, ((0., 1.), (0., 0.001)), ((0, 1), (0, 1)))
zdt6 = ExhaustionSampleImage(pro_zdt6, (256, 256), 1000)

pro_sch = Problem(funSCH, ((-10 ** 1, 10 ** 1),), ((0, 5), (0, 5)))
sch = ExhaustionSampleImage(pro_sch, (512, 512), 10000)

pro_fon = Problem(funFON, ((-4, 4), (-4, 4), (-4, 4)), ((0, 1), (0, 1)))
fon = ExhaustionSampleImage(pro_fon, (256, 256), 100)
