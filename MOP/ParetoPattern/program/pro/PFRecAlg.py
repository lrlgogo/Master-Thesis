import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import initializers


for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
del gpu


def kernel_initial(u=2):

    kernel = np.zeros(2 * (2*u - 1,), dtype='float32')
    kernel[u-1:, :u] = np.ones(2 * (u,), dtype='float32')

    return kernel


def activation(x, input_mat):

    a = np.where(x > 1., 0, 1)
    a = 0.5 * (a + input_mat)

    return np.where(a > 0.5, 1, 0)


def funConv(input, kernel):

    x = input.reshape((1,) + input.shape + (1,))

    conv = Conv2D(1, kernel.shape[0], padding='same',
                  kernel_initializer=initializers.zeros,
                  use_bias=False)
    conv.build(x.shape)
    conv.kernel = kernel
    y = np.reshape(conv(x), input.shape)
    y = np.where(y > 1., 2, y)

    return y


def funPFRec(input_mat, u):

    if input_mat.dtype != 'float32':
        input_mat = np.asarray(input_mat, dtype='float32')
    k = input_mat.shape

    kernel = kernel_initial(u)
    kernel = kernel.reshape(kernel.shape + (1, 1))

    w = 0
    A = input_mat
    while w * (u - 1) < (k[0] - 1):
        C = funConv(A, kernel)
        A = C
        w += 1

    y = activation(A, input_mat)

    return y


class PFRecAlgII:
    """
    this is the algorithm of Pareto front recognizing, here is the algorithm I
    ...
    """
    def __init__(self, u=2):
        self.input = None
        self.output = None
        self.k = 0
        self.u = u

    def get_input(self, input):
        self.input = input
        return 0

    def run(self):
        self.output = funPFRec(self.input, self.u)
        return 0

    def reset(self, u=2):
        self.u = u
        return 0

    def show(self):
        plt.imshow(self.output, cmap='binary')
        plt.show()
        return 0
