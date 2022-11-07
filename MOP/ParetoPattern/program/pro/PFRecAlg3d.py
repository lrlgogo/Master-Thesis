import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv3D
from tensorflow.keras import initializers
import matplotlib.pyplot as plt


for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
del gpu


def kernel_initial(u=2):

    kernel = np.zeros(3 * (2 * u - 1,), dtype=np.float32)
    kernel[: u, : u, : u] = np.ones(3 * (u,), dtype=np.float32)

    return kernel


def activation(x, input_mat):

    a = np.where(x > 1.1, 0, 1)
    a = 0.5 * (a + input_mat)

    return np.where(a > 0.8, 1, 0)


def funPFRec(input_mat, u=2):

    if input_mat.dtype != np.float32:
        input_mat = np.asarray(input_mat, dtype=np.float32)
    input_shape = input_mat.shape
    k = max(input_shape)

    kernel = kernel_initial(u)
    kernel = kernel.reshape(kernel.shape + (1, 1))

    A = input_mat.reshape((1,) + input_shape + (1,))

    conv = Conv3D(
        1, 2 * u - 1, padding='same',
        kernel_initializer=initializers.zeros,
        use_bias=False
    )
    conv.build(A.shape[1:])
    conv.kernel = kernel

    w = 0
    while w * (u - 1) < (k - 1):
        C = conv(A)
        A = C
        A = tf.where(A > 1.1, 2., A)
        w += 1

    A = np.reshape(A, input_shape)
    y = activation(A, input_mat)

    return y


class PFRecAlg3d:
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

    def show(self, size=10):
        x_list, y_list, z_list = [[], [], []]
        x_dim, y_dim, z_dim = self.output.shape
        for i in range(x_dim):
            for j in range(y_dim):
                for k in range(z_dim):
                    if self.output[i, j, k] > 0.5:
                        x_list.append(i)
                        y_list.append(j)
                        z_list.append(k)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_list, y_list, z_list, marker='o', s=size)
        return 0


"""
f = PFRecAlg3d(u=11)
input = np.zeros((128, 128, 128), np.float32)
for i in range(128):
    for j in range(128):
        for k in range(128):
            if i + j + k > 50:
                input[i, j, k] = 1

u, v, w = [[], [], []]
for i in range(128):
    for j in range(128):
        for k in range(128):
            if input[i, j, k]:
                u.append(i)
                v.append(j)
                w.append(k)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(u, v, w, marker='o')
plt.show()
"""
